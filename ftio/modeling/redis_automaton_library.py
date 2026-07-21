"""
RedisAutomatonLibrary: Redis-backed alternative to AutomatonLibrary.

Same interface and merge semantics as AutomatonLibrary (load, save, seed,
get_rank_behavior, available_apps, available_rank_keys), but backed by a Redis
server instead of the filesystem -- for sharing one reference library
across multiple processes or nodes without a shared filesystem.

It also closes a race AutomatonLibrary has: its save() does a plain
load -> merge -> write with no locking, so two concurrent writers to the
same (app_name, rank_key) can interleave and one's contribution is lost.
Here, save()/seed() hold a Redis distributed lock (SET NX PX, released on
exit) around that critical section.

Key layout (all under one `redis_prefix`, default "ftio:models"):
    <prefix>:apps                    Set of app names.
    <prefix>:<app_name>:keys         Set of rank_keys for that app.
    <prefix>:<app_name>:<rank_key>   JSON blob (ReferenceAutomaton.to_dict()).
    <prefix>:<app_name>:<rank_key>:lock   Distributed lock for save()/seed().

Requires the optional `redis` package (`pip install redis`, or
`pip install .[redis-libs]`). Not imported at module load time -- only
when RedisAutomatonLibrary is actually instantiated -- so importing this
module is safe even without redis-py installed.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Licensed under the BSD 3-Clause License.
"""

from __future__ import annotations

import importlib.util
import json
import time

from ftio.modeling.reference_automaton import NodeBehavior, ReferenceAutomaton


class RedisAutomatonLibrary:
    """
    Redis-backed library of compiled reference automata.

    Parameters
    ----------
    redis_client : "redis.Redis" | None
        An existing redis-py client (sync). If None, one is created from
        host/port/db.
    host, port, db : connection parameters, used only if redis_client is None.
    prefix : str
        Key namespace, so multiple libraries (or apps unrelated to FTIO) can
        share one Redis instance without colliding.
    lock_timeout : float
        Seconds a save()/seed() lock is held before it auto-expires -- a
        safety net against a crashed holder leaving the lock stuck, not a
        tuning knob for normal operation (one JSON blob write is fast).
    """

    def __init__(
        self,
        redis_client=None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "ftio:models",
        lock_timeout: float = 10.0,
    ):
        if importlib.util.find_spec("redis") is None:
            raise ImportError(
                "RedisAutomatonLibrary requires the optional 'redis' package: "
                "pip install redis  (or: pip install .[redis-libs])"
            )
        import redis as _redis

        self._redis = redis_client or _redis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )
        self._prefix = prefix
        self._lock_timeout = lock_timeout

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def _apps_key(self) -> str:
        return f"{self._prefix}:apps"

    def _keys_key(self, app_name: str) -> str:
        return f"{self._prefix}:{app_name}:keys"

    def _ref_key(self, app_name: str, rank_key: str) -> str:
        return f"{self._prefix}:{app_name}:{rank_key}"

    def _lock_key(self, app_name: str, rank_key: str) -> str:
        return f"{self._prefix}:{app_name}:{rank_key}:lock"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def available_apps(self) -> list[str]:
        return sorted(self._redis.smembers(self._apps_key()))

    def available_rank_keys(self, app_name: str) -> list[str]:
        return sorted(self._redis.smembers(self._keys_key(app_name)))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, app_name: str, rank_key: str) -> ReferenceAutomaton | None:
        """Load reference for app + rank_key.

        Falls back to the nearest available rank configuration (by initial
        rank count) if an exact match is not found -- same policy as
        AutomatonLibrary.load().
        """
        ref = self._load_key(app_name, rank_key)
        if ref is not None:
            return ref

        available = self.available_rank_keys(app_name)
        if not available:
            return None

        try:
            target_initial = int(rank_key.split("_")[0])
        except ValueError:
            return None

        def _initial(k: str) -> int:
            try:
                return int(k.split("_")[0])
            except ValueError:
                return -1

        nearest_key = min(available, key=lambda k: abs(_initial(k) - target_initial))
        ref = self._load_key(app_name, nearest_key)
        if ref is not None:
            print(
                f"[RedisAutomatonLibrary] No exact match for {app_name}:{rank_key}; "
                f"using nearest: {nearest_key}"
            )
        return ref

    def _load_key(self, app_name: str, rank_key: str) -> ReferenceAutomaton | None:
        raw = self._redis.get(self._ref_key(app_name, rank_key))
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(
                f"[RedisAutomatonLibrary] JSON parse error for {app_name}:{rank_key}: {exc}"
            )
            return None

        # Detect format: our compact reference dict vs a raw PhaseAutomaton export.
        first_state = (data.get("states") or [{}])[0]
        if "period_mean" in first_state:
            return ReferenceAutomaton.from_dict(data)
        return ReferenceAutomaton.from_automaton_dict(data, app_name, rank_key)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _write(self, app_name: str, rank_key: str, ref: ReferenceAutomaton) -> None:
        pipe = self._redis.pipeline()
        pipe.set(self._ref_key(app_name, rank_key), json.dumps(ref.to_dict()))
        pipe.sadd(self._apps_key(), app_name)
        pipe.sadd(self._keys_key(app_name), rank_key)
        pipe.execute()

    def save(self, automaton, app_name: str, rank_key: str) -> None:
        """Save a PhaseAutomaton to the library.

        Same merge semantics as AutomatonLibrary.save() (matching-topology
        pools, mismatched-topology saves the new path under a versioned key
        while still pooling shared configurations into the node table), but
        the whole load -> merge -> write critical section is held under a
        Redis lock keyed on (app_name, rank_key), so two concurrent writers
        can't interleave and silently lose one side's contribution.
        """
        with self._redis.lock(
            self._lock_key(app_name, rank_key), timeout=self._lock_timeout
        ):
            new_ref = ReferenceAutomaton.from_automaton_dict(
                automaton.to_dict(), app_name, rank_key
            )

            existing = self.load(app_name, rank_key)
            if existing is not None and existing.rank_key == rank_key:
                merged = existing.merge(new_ref)
                if merged is existing:
                    # Topology mismatch — existing.nodes was just enriched in
                    # place with any shared configurations; persist that, then
                    # save the new run's own path under a versioned key.
                    self._write(app_name, rank_key, existing)

                    versioned_key = f"{rank_key}_v{int(time.time())}"
                    self._write(app_name, versioned_key, new_ref)
                    print(
                        f"[RedisAutomatonLibrary] Topology mismatch for {app_name}:{rank_key}; "
                        f"saved new run as {versioned_key} "
                        f"(shared configurations pooled into {rank_key})"
                    )
                    return
            else:
                merged = new_ref

            self._write(app_name, rank_key, merged)
            print(
                f"[RedisAutomatonLibrary] Saved {app_name}:{rank_key} "
                f"({merged.run_count} run(s), {merged.n_states} states)"
            )

    def seed(self, app_name: str, node_estimates: dict[int, dict]) -> None:
        """Write a user-supplied early estimate as a starting reference.

        Same semantics as AutomatonLibrary.seed() -- see
        ReferenceAutomaton.from_node_seed for the dilution/survival rule.
        Does nothing if a reference already exists for the derived
        rank_key, so a seed never clobbers real profiling data; the
        existence check and the write happen under the same lock as
        save() to avoid racing a concurrent save().
        """
        ref = ReferenceAutomaton.from_node_seed(app_name, node_estimates)
        with self._redis.lock(
            self._lock_key(app_name, ref.rank_key), timeout=self._lock_timeout
        ):
            if self.load(app_name, ref.rank_key) is not None:
                print(
                    f"[RedisAutomatonLibrary] Seed skipped — {app_name}:{ref.rank_key} "
                    "already exists"
                )
                return
            self._write(app_name, ref.rank_key, ref)
            print(
                f"[RedisAutomatonLibrary] Seeded {app_name}:{ref.rank_key} from user "
                f"estimate ({len(node_estimates)} configuration(s))"
            )

    # ------------------------------------------------------------------
    # Node lookup (configuration-level, cross-path)
    # ------------------------------------------------------------------

    def get_rank_behavior(
        self,
        app_name: str,
        ranks: int,
        at_time: float | None = None,
        at_cycle: float | None = None,
    ) -> list[NodeBehavior]:
        """Look up what's known about one configuration, across every stored
        path -- same semantics as AutomatonLibrary.get_rank_behavior()."""
        nodes: dict[int, list[NodeBehavior]] = {}
        for key in self.available_rank_keys(app_name):
            ref = self._load_key(app_name, key)
            if ref is None:
                continue
            for behavior in ref.get_rank_behavior(ranks):
                ReferenceAutomaton._fold_behavior(nodes, ranks, behavior)

        merged_ref = ReferenceAutomaton(
            app_name=app_name,
            rank_key=str(ranks),
            n_states=0,
            state_stats=[],
            transition_causes=[],
            run_count=0,
            nodes=nodes,
        )
        return merged_ref.get_rank_behavior(ranks, at_time=at_time, at_cycle=at_cycle)
