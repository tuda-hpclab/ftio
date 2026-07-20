"""
ReferenceAutomaton: compiled reference built from one or more profiling runs.

Unlike PhaseAutomaton (which records a single live run), ReferenceAutomaton
stores per-state distribution statistics and merges new runs using pooled
variance — the topology (number of states, rank sequence) is the stable part;
timing distributions improve with each additional run.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Licensed under the BSD 3-Clause License.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Same statistical philosophy as ftio.prediction.change_detection.ksigma:
# self-calibrating k-sigma with a relative noise floor. Reused here (rather
# than a fixed ratio threshold) so "is this a new behavior or the same one
# with more variance" answers consistently with how FTIO already decides
# "is this a new phase" online.
_MERGE_K = 3.0
_MERGE_SIGMA_REL_FLOOR = 0.02
_WINDOW_FUZZ_Z = 1.0  # how many std a window's edges are widened by


@dataclass
class StateStats:
    """Distribution statistics for a single reference state, aggregated across runs.

    dwell_mean / dwell_std: seconds spent continuously in this state before the
    run transitioned out of it (a PhaseState's ``duration``), pooled across every
    historical occurrence of this configuration -- i.e. "how long does the
    application typically stay here once it enters this configuration." It is
    not the gap between two predictions; it spans the whole entry-to-exit time.
    """

    period_mean: float
    period_std: float
    dwell_mean: float  # seconds spent in this state
    dwell_std: float
    ranks: int = 0
    n_samples: int = 1


@dataclass
class NodeBehavior:
    """One distinct regime for a configuration, indexed on two axes.

    A configuration (rank count) does not always behave the same way for its
    whole dwell -- e.g. ranks=32 might do 3s-period checkpoints for its first
    few bursts, then drift to an 8s period afterwards. Each NodeBehavior is
    one such regime, located on:

    cycle window (c_start/c_end)
        Bursts/predictions observed since *this occurrence* of the node
        began. The authoritative axis for "which behavior is this" -- it is
        driven by the application's own control flow (iteration count,
        checkpoint number), so it is unaffected by a run simply being faster
        or slower for unrelated reasons (cluster contention, filesystem
        load, a bigger checkpoint that run). Two occurrences of a node that
        take a different number of *seconds* to reach the same behavior
        still agree on which *cycle* that behavior starts at.

    time window (t_start/t_end)
        Wall-clock seconds since entering this occurrence of the node.
        Kept alongside cycles because it answers a genuinely different
        question -- "how many seconds until this transitions" -- which
        TransitionPredictor still needs for scheduling/ETA purposes. It is
        NOT used to decide whether two observations are the same behavior;
        see ``overlaps``.

    Window edges are themselves mean +/- std -- pooled across occurrences
    just like period is -- so two behaviors adjacent on an axis naturally end
    up with overlapping windows once the boundary varies between
    occurrences. That overlap is not resolved away; it is the honest answer
    when asked "what applies here" and the query falls in the fuzzy boundary.
    """

    period_mean: float
    period_std: float
    t_start_mean: float
    t_start_std: float
    t_end_mean: float
    t_end_std: float
    c_start_mean: float = np.nan  # bursts since entering this occurrence
    c_start_std: float = 0.0
    c_end_mean: float = np.nan
    c_end_std: float = 0.0
    ranks: int = 0
    n_samples: int = 1

    def covers_time(self, t: float, z: float = _WINDOW_FUZZ_Z) -> bool:
        lo = self.t_start_mean - z * self.t_start_std
        hi = self.t_end_mean + z * self.t_end_std
        return lo <= t <= hi

    def covers_cycle(self, c: float, z: float = _WINDOW_FUZZ_Z) -> bool:
        if np.isnan(self.c_start_mean) or np.isnan(self.c_end_mean):
            return False
        lo = self.c_start_mean - z * self.c_start_std
        hi = self.c_end_mean + z * self.c_end_std
        return lo <= c <= hi

    def overlaps(self, other: NodeBehavior, z: float = _WINDOW_FUZZ_Z) -> bool:
        """Whether two behaviors describe the same window -- cycle count
        first (authoritative) when both sides have it, wall-clock time as
        the fallback when one side doesn't (e.g. a hand-authored seed, which
        has no burst counts to offer)."""
        if not (np.isnan(self.c_start_mean) or np.isnan(other.c_start_mean)):
            a_lo = self.c_start_mean - z * self.c_start_std
            a_hi = self.c_end_mean + z * self.c_end_std
            b_lo = other.c_start_mean - z * other.c_start_std
            b_hi = other.c_end_mean + z * other.c_end_std
            return a_lo <= b_hi and b_lo <= a_hi
        a_lo = self.t_start_mean - z * self.t_start_std
        a_hi = self.t_end_mean + z * self.t_end_std
        b_lo = other.t_start_mean - z * other.t_start_std
        b_hi = other.t_end_mean + z * other.t_end_std
        return a_lo <= b_hi and b_lo <= a_hi


class ReferenceAutomaton:
    """
    Compiled reference automaton built from one or more PhaseAutomaton exports.

    Two views of the same data coexist:

    ``state_stats`` (the path)
        Ordered list, one entry per position in *this* rank_key's sequence.
        Powers StateTracker/TransitionPredictor's positional matching and
        "next period" lookahead. Two runs only pool into the same path when
        their full rank sequences match exactly (see ``merge``).

    ``nodes`` (the configuration table)
        Dict keyed by rank count -> list[NodeBehavior]. Pooled across *every*
        occurrence of that configuration, including occurrences reached via a
        different path -- but NOT blindly averaged: a new observation only
        folds into an existing behavior when its window overlaps that
        behavior's (cycle count first, wall-clock time as fallback) AND its
        period is within k-sigma of it (see ``_fold_behavior``). Otherwise it
        starts a new behavior entry, so a third, genuinely different regime
        doesn't corrupt the first two. This is what makes a configuration's
        stats reusable across malleable runs that resize in a different
        order, and what a user seeds with an early estimate
        (``AutomatonLibrary.seed`` / ``from_node_seed``) without having to
        know the full path up front.

    Library key: derived from the rank sequence across states, e.g. "128" for
    a fixed-rank run, "16_32_128" for a malleable run that scaled up twice.
    """

    def __init__(
        self,
        app_name: str,
        rank_key: str,
        n_states: int,
        state_stats: list[StateStats],
        transition_causes: list[str],
        run_count: int = 1,
        nodes: dict[int, list[NodeBehavior]] | None = None,
    ):
        self.app_name = app_name
        self.rank_key = rank_key
        self.n_states = n_states
        self.state_stats = state_stats
        self.transition_causes = transition_causes
        self.run_count = run_count
        self.nodes = (
            nodes if nodes is not None else self._nodes_from_state_stats(state_stats)
        )

    @staticmethod
    def _nodes_from_state_stats(
        state_stats: list[StateStats],
    ) -> dict[int, list[NodeBehavior]]:
        """Fallback node builder used when no richer source is available
        (e.g. ``from_node_seed``, or any direct construction from StateStats
        alone). StateStats carries no burst count, only wall-clock dwell, so
        behaviors built here have a valid time window but an unknown
        (NaN) cycle window -- see ``from_automaton_dict`` for the richer,
        cycle-aware builder used when the raw PhaseAutomaton export (which
        does carry per-state burst counts) is available.
        """
        nodes: dict[int, list[NodeBehavior]] = {}
        occurrence_elapsed = 0.0
        prev_ranks: int | None = None
        for s in state_stats:
            if s.ranks != prev_ranks:
                occurrence_elapsed = 0.0
            dwell = 0.0 if np.isnan(s.dwell_mean) else s.dwell_mean
            if s.ranks > 0:
                behavior = NodeBehavior(
                    period_mean=s.period_mean,
                    period_std=s.period_std,
                    t_start_mean=occurrence_elapsed,
                    t_start_std=0.0,
                    t_end_mean=occurrence_elapsed + dwell,
                    t_end_std=0.0,
                    ranks=s.ranks,
                    n_samples=s.n_samples,
                )
                ReferenceAutomaton._fold_behavior(nodes, s.ranks, behavior)
            occurrence_elapsed += dwell
            prev_ranks = s.ranks
        return nodes

    @staticmethod
    def pool_behavior(a: NodeBehavior, b: NodeBehavior) -> NodeBehavior:
        """Pool two NodeBehaviors describing the same regime into one.

        ``_pool`` already treats NaN as "no information on this side" and
        adopts the other side's value, so pooling a seed's behavior (no
        cycle window) with a real one (has a cycle window) correctly yields
        the real cycle window rather than corrupting it.
        """
        if a.ranks != b.ranks:
            raise ValueError(
                f"cannot pool behavior for different configurations: {a.ranks} vs {b.ranks}"
            )
        pm, ps = _pool(
            a.period_mean,
            a.period_std,
            a.n_samples,
            b.period_mean,
            b.period_std,
            b.n_samples,
        )
        ts_m, ts_s = _pool(
            a.t_start_mean,
            a.t_start_std,
            a.n_samples,
            b.t_start_mean,
            b.t_start_std,
            b.n_samples,
        )
        te_m, te_s = _pool(
            a.t_end_mean, a.t_end_std, a.n_samples, b.t_end_mean, b.t_end_std, b.n_samples
        )
        cs_m, cs_s = _pool(
            a.c_start_mean,
            a.c_start_std,
            a.n_samples,
            b.c_start_mean,
            b.c_start_std,
            b.n_samples,
        )
        ce_m, ce_s = _pool(
            a.c_end_mean, a.c_end_std, a.n_samples, b.c_end_mean, b.c_end_std, b.n_samples
        )
        return NodeBehavior(
            pm,
            ps,
            ts_m,
            ts_s,
            te_m,
            te_s,
            cs_m,
            cs_s,
            ce_m,
            ce_s,
            a.ranks,
            a.n_samples + b.n_samples,
        )

    @staticmethod
    def _fold_behavior(
        nodes: dict[int, list[NodeBehavior]],
        ranks: int,
        new_behavior: NodeBehavior,
        k: float = _MERGE_K,
        sigma_rel_floor: float = _MERGE_SIGMA_REL_FLOOR,
    ) -> None:
        """Fold one observation into node ``ranks``'s behavior list in place.

        Pools into the first existing behavior whose window overlaps (cycle
        count first, time as fallback -- see NodeBehavior.overlaps) AND whose
        period is within k-sigma (self-calibrating, same rule as
        ftio.prediction.change_detection.ksigma). Otherwise appends a new,
        independent behavior -- this is how a third, unrelated regime avoids
        corrupting the first two instead of being blended into their mean.
        """
        lst = nodes.setdefault(ranks, [])
        for i, existing in enumerate(lst):
            if not existing.overlaps(new_behavior):
                continue
            sigma_eff = max(
                existing.period_std, sigma_rel_floor * existing.period_mean, 1e-9
            )
            if abs(new_behavior.period_mean - existing.period_mean) <= k * sigma_eff:
                lst[i] = ReferenceAutomaton.pool_behavior(existing, new_behavior)
                return
        lst.append(new_behavior)

    def node(
        self, ranks: int, at_time: float | None = None, at_cycle: float | None = None
    ) -> list[NodeBehavior]:
        """Look up this reference's own knowledge of one configuration.

        Any combination of at_time / at_cycle may be given:

        neither: every known behavior -- "all possible combinations."

        at_cycle=C only: behaviors whose cycle window covers C (the
        authoritative, speed-independent axis).

        at_time=T only: behaviors whose wall-clock window covers T -- usable,
        but remember this axis drifts across runs of different speed.

        both: behaviors satisfying both constraints at once (the intersection)
        -- a behavior missing one axis (e.g. a hand-authored seed has no
        cycle data) is not disqualified on that axis alone, since "unknown"
        is not the same as "doesn't match."
        """
        behaviors = self.nodes.get(ranks, [])
        if at_time is None and at_cycle is None:
            return list(behaviors)
        matches = []
        for b in behaviors:
            time_ok = at_time is None or b.covers_time(at_time)
            cycle_ok = (
                at_cycle is None or np.isnan(b.c_start_mean) or b.covers_cycle(at_cycle)
            )
            if time_ok and cycle_ok:
                matches.append(b)
        return matches

    @property
    def edges(self) -> dict[tuple[int, int], dict]:
        """Aggregated (from_ranks -> to_ranks) transitions along this path.

        Purely observational: an edge only exists once a transition has
        actually been recorded, unlike a node, which can be seeded.
        """
        out: dict[tuple[int, int], dict] = {}
        for i, cause in enumerate(self.transition_causes):
            if i + 1 >= len(self.state_stats):
                continue
            key = (self.state_stats[i].ranks, self.state_stats[i + 1].ranks)
            entry = out.setdefault(key, {"cause": cause, "count": 0})
            entry["count"] += 1
        return out

    @classmethod
    def from_node_seed(
        cls, app_name: str, node_estimates: dict[int, dict]
    ) -> ReferenceAutomaton:
        """Build a reference purely from user-supplied per-configuration guesses.

        ``node_estimates`` maps rank count -> {"period": seconds, "dwell": seconds}.
        No order/path is required -- states are laid out in ascending rank order
        as a reasonable default topology so StateTracker/TransitionPredictor can
        use the seed immediately. Each seeded node carries n_samples=1, so a
        real profiling run *within k-sigma of the guess* outweighs it 1:1 and
        the estimate converges on the real one -- the "early estimate FTIO
        overwrites" mechanism, but only within the same k-sigma tolerance that
        decides whether any two observations count as the same behavior (see
        ``_fold_behavior``). A guess far from reality is NOT quietly
        corrected: it survives as a separate, permanent behavior alongside
        the real one, because the clustering rule can't distinguish "stale
        wrong seed" from "a second real regime that happens to share this
        rank count." A bad-enough guess needs manual cleanup, not dilution.
        A seed only ever describes one behavior per configuration (a user
        can't guess a split they've never observed, nor a cycle count --
        only FTIO can count real bursts); FTIO splits it into more
        behaviors itself, with real cycle windows, once real data disagrees
        with the guess closely enough to matter but not closely enough to pool.
        """
        ordered_ranks = sorted(node_estimates)
        stats = [
            StateStats(
                period_mean=float(node_estimates[r].get("period", np.nan)),
                period_std=0.0,
                dwell_mean=float(node_estimates[r].get("dwell", np.nan)),
                dwell_std=0.0,
                ranks=r,
                n_samples=1,
            )
            for r in ordered_ranks
        ]
        return cls(
            app_name=app_name,
            rank_key=cls.rank_key_from_sequence(ordered_ranks),
            n_states=len(stats),
            state_stats=stats,
            transition_causes=["rank_change"] * max(0, len(stats) - 1),
            run_count=0,  # a seed is not a profiled run
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def period_sequence(self) -> list[float]:
        return [s.period_mean for s in self.state_stats]

    @property
    def rank_sequence(self) -> list[int]:
        return [s.ranks for s in self.state_stats]

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @staticmethod
    def rank_key_from_sequence(ranks: list[int]) -> str:
        """Derive a library key from a sequence of observed rank counts.

        Consecutive duplicates are collapsed so the key reflects distinct
        rank configurations: [16, 16, 32, 32, 128] → "16_32_128".
        """
        if not ranks:
            return "0"
        unique: list[int] = []
        for r in ranks:
            if not unique or r != unique[-1]:
                unique.append(r)
        return "_".join(str(r) for r in unique)

    @classmethod
    def from_automaton_dict(
        cls,
        data: dict,
        app_name: str,
        rank_key: str,
    ) -> ReferenceAutomaton:
        """Build from a single PhaseAutomaton.to_dict() export (one run, std = 0).

        Builds ``nodes`` directly from the raw states here (rather than
        letting __init__ derive it from ``state_stats``) because the raw
        export carries ``n_predictions`` -- the burst count per state -- which
        StateStats does not. That is what makes the cycle-count axis
        (NodeBehavior.c_start/c_end) available at all.
        """
        raw_states = data.get("states", [])
        raw_transitions = data.get("transitions", [])

        stats: list[StateStats] = []
        nodes: dict[int, list[NodeBehavior]] = {}
        occurrence_elapsed = 0.0
        occurrence_cycle = 0.0
        prev_ranks: int | None = None

        for s in raw_states:
            period = s.get("period")
            if period is None and s.get("dominant_freq"):
                period = 1.0 / s["dominant_freq"]
            period = float(period) if period is not None else np.nan
            dur = s.get("duration")
            dwell = float(dur) if dur is not None else 0.0
            n_pred = s.get("n_predictions")
            cycles = float(n_pred) if n_pred is not None else 0.0
            ranks = int(s.get("ranks", 0))

            stats.append(
                StateStats(
                    period_mean=period,
                    period_std=0.0,
                    dwell_mean=dwell if dur is not None else np.nan,
                    dwell_std=0.0,
                    ranks=ranks,
                    n_samples=1,
                )
            )

            if ranks != prev_ranks:
                occurrence_elapsed = 0.0
                occurrence_cycle = 0.0
            if ranks > 0:
                behavior = NodeBehavior(
                    period_mean=period,
                    period_std=0.0,
                    t_start_mean=occurrence_elapsed,
                    t_start_std=0.0,
                    t_end_mean=occurrence_elapsed + dwell,
                    t_end_std=0.0,
                    c_start_mean=occurrence_cycle,
                    c_start_std=0.0,
                    c_end_mean=occurrence_cycle + cycles,
                    c_end_std=0.0,
                    ranks=ranks,
                    n_samples=1,
                )
                cls._fold_behavior(nodes, ranks, behavior)
            occurrence_elapsed += dwell
            occurrence_cycle += cycles
            prev_ranks = ranks

        causes = [t.get("cause", "frequency") for t in raw_transitions]

        return cls(
            app_name=app_name,
            rank_key=rank_key,
            n_states=len(raw_states),
            state_stats=stats,
            transition_causes=causes,
            run_count=1,
            nodes=nodes,
        )

    @staticmethod
    def _stats_from_dict(s: dict) -> StateStats:
        pm = s.get("period_mean")
        ps = s.get("period_std") or 0.0
        dm = s.get("dwell_mean")
        ds = s.get("dwell_std") or 0.0
        return StateStats(
            period_mean=float(pm) if pm is not None else np.nan,
            period_std=float(ps),
            dwell_mean=float(dm) if dm is not None else np.nan,
            dwell_std=float(ds),
            ranks=int(s.get("ranks", 0)),
            n_samples=int(s.get("n_samples", 1)),
        )

    @staticmethod
    def _behavior_from_dict(b: dict) -> NodeBehavior:
        pm = b.get("period_mean")
        cs = b.get("c_start_mean")
        ce = b.get("c_end_mean")
        return NodeBehavior(
            period_mean=float(pm) if pm is not None else np.nan,
            period_std=float(b.get("period_std") or 0.0),
            t_start_mean=float(b.get("t_start_mean") or 0.0),
            t_start_std=float(b.get("t_start_std") or 0.0),
            t_end_mean=float(b.get("t_end_mean") or 0.0),
            t_end_std=float(b.get("t_end_std") or 0.0),
            c_start_mean=float(cs) if cs is not None else np.nan,
            c_start_std=float(b.get("c_start_std") or 0.0),
            c_end_mean=float(ce) if ce is not None else np.nan,
            c_end_std=float(b.get("c_end_std") or 0.0),
            ranks=int(b.get("ranks", 0)),
            n_samples=int(b.get("n_samples", 1)),
        )

    @classmethod
    def from_dict(cls, data: dict) -> ReferenceAutomaton:
        """Load from a previously saved reference JSON (our own compact format)."""
        raw = data.get("states", [])
        stats = [cls._stats_from_dict(s) for s in raw]

        # "nodes" carries cross-path clustering accumulated by previous merges
        # (see merge()); older files or a hand-authored seed may not have it,
        # in which case __init__ derives it from state_stats as before.
        raw_nodes = data.get("nodes")
        nodes = (
            {
                int(ranks): [cls._behavior_from_dict(b) for b in behaviors]
                for ranks, behaviors in raw_nodes.items()
            }
            if raw_nodes
            else None
        )

        return cls(
            app_name=data.get("app_name", "unknown"),
            rank_key=data.get("rank_key", "0"),
            n_states=data["n_states"],
            state_stats=stats,
            transition_causes=data.get("transition_causes", []),
            run_count=data.get("run_count", 1),
            nodes=nodes,
        )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_nodes_in_place(self, other_nodes: dict[int, list[NodeBehavior]]) -> None:
        for ranks, behaviors in other_nodes.items():
            for behavior in behaviors:
                self._fold_behavior(self.nodes, ranks, behavior)

    def merge(self, other: ReferenceAutomaton) -> ReferenceAutomaton:
        """Merge another run into this reference using pooled statistics.

        The ordered path only pools when both automata have the same number
        of states. If topologies differ (the application changed phase
        structure, or resized in a different order), the path itself is left
        alone and the caller should save the new run under a different key --
        but any configuration the two runs have in common is still folded
        into this reference's node table in place (clustered, not blindly
        averaged -- see ``_fold_behavior``), so that knowledge isn't
        discarded just because the full paths don't match.

        The node table itself is always built purely by folding self.nodes
        and other.nodes together (never re-derived from the merged path),
        because only the source references' own node tables carry cycle-count
        windows -- state_stats does not.
        """
        if other.n_states != self.n_states:
            self._merge_nodes_in_place(other.nodes)
            return self

        merged: list[StateStats] = []
        for a, b in zip(self.state_stats, other.state_stats, strict=True):
            pm, ps = _pool(
                a.period_mean,
                a.period_std,
                a.n_samples,
                b.period_mean,
                b.period_std,
                b.n_samples,
            )
            dm, ds = _pool(
                a.dwell_mean,
                a.dwell_std,
                a.n_samples,
                b.dwell_mean,
                b.dwell_std,
                b.n_samples,
            )
            merged.append(
                StateStats(
                    period_mean=pm,
                    period_std=ps,
                    dwell_mean=dm,
                    dwell_std=ds,
                    ranks=a.ranks,
                    n_samples=a.n_samples + b.n_samples,
                )
            )

        result = ReferenceAutomaton(
            app_name=self.app_name,
            rank_key=self.rank_key,
            n_states=self.n_states,
            state_stats=merged,
            transition_causes=self.transition_causes,
            run_count=self.run_count + other.run_count,
            nodes={},
        )
        result._merge_nodes_in_place(self.nodes)
        result._merge_nodes_in_place(other.nodes)
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        def _f(v: float) -> float | None:
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return None
            return float(v)

        return {
            "app_name": self.app_name,
            "rank_key": self.rank_key,
            "n_states": self.n_states,
            "run_count": self.run_count,
            "states": [
                {
                    "period_mean": _f(s.period_mean),
                    "period_std": _f(s.period_std),
                    "dwell_mean": _f(s.dwell_mean),
                    "dwell_std": _f(s.dwell_std),
                    "ranks": s.ranks,
                    "n_samples": s.n_samples,
                }
                for s in self.state_stats
            ],
            "transition_causes": self.transition_causes,
            "nodes": {
                str(ranks): [
                    {
                        "period_mean": _f(b.period_mean),
                        "period_std": _f(b.period_std),
                        "t_start_mean": _f(b.t_start_mean),
                        "t_start_std": _f(b.t_start_std),
                        "t_end_mean": _f(b.t_end_mean),
                        "t_end_std": _f(b.t_end_std),
                        "c_start_mean": _f(b.c_start_mean),
                        "c_start_std": _f(b.c_start_std),
                        "c_end_mean": _f(b.c_end_mean),
                        "c_end_std": _f(b.c_end_std),
                        "ranks": b.ranks,
                        "n_samples": b.n_samples,
                    }
                    for b in behaviors
                ]
                for ranks, behaviors in self.nodes.items()
            },
        }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _pool(
    m1: float,
    s1: float,
    n1: int,
    m2: float,
    s2: float,
    n2: int,
) -> tuple[float, float]:
    """Merge two (mean, std, n) triplets into a pooled (mean, std)."""
    if np.isnan(m1):
        return m2, s2
    if np.isnan(m2):
        return m1, s1
    n = n1 + n2
    m = (n1 * m1 + n2 * m2) / n
    var = (n1 * (s1**2 + (m1 - m) ** 2) + n2 * (s2**2 + (m2 - m) ** 2)) / n
    return m, float(np.sqrt(var))
