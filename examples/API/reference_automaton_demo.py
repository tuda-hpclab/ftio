"""
ReferenceAutomaton / node-model demo — malleability across runs.

Where phase_automaton_demo.py shows a single live run's state machine, this
demo shows what gets learned ACROSS runs: the configuration table (`nodes`)
that lets a later run (or a user-supplied guess) ask "what should I expect
at ranks=32?" before it has finished measuring it.

  1. Malleability      — rank changes segment phases; each rank count
                          becomes its own node, independent of position.
  2/3. Same node, two speeds
                        — a configuration can behave differently over its
                          own lifetime (no rank change involved), and the
                          axis you index that by matters: wall-clock time
                          drifts across runs of different speed, cycle count
                          (bursts observed) does not.
  4. Seeding            — a user can seed an early estimate; it dilutes away
                          when a real run agrees closely, but survives
                          (uncorrected) as a separate entry when it doesn't.
  5. Cross-path sharing — a configuration reached via two different
                          malleability paths shares its node-level stats.

See docs/phase_automaton.md ("Configuration table (nodes)") for the full
design writeup this demo backs.

Run:
    python examples/API/reference_automaton_demo.py
"""

from pathlib import Path

import numpy as np

from ftio.freq.prediction import Prediction
from ftio.modeling.automaton_library import AutomatonLibrary
from ftio.modeling.phase_automaton import PhaseAutomaton
from ftio.modeling.reference_automaton import ReferenceAutomaton

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def mock_prediction(freq: float, t_start: float, t_end: float, ranks: int) -> Prediction:
    pred = Prediction(transformation="dft", t_start=t_start, t_end=t_end, ranks=ranks)
    pred.dominant_freq = np.array([freq])
    pred.conf = np.array([0.95])
    pred.amp = np.array([1.0])
    return pred


def build_stream(
    phases: list[tuple[int, float, int]], t0: float = 0.0
) -> list[Prediction]:
    """phases: list of (ranks, period, n_bursts)."""
    stream, t = [], t0
    for ranks, period, n in phases:
        for _ in range(n):
            stream.append(mock_prediction(1.0 / period, t, t + period, ranks))
            t += period
    return stream


def automaton(
    phases: list[tuple[int, float, int]],
    method: str | None = "ksigma",
    rank_trigger: bool = True,
) -> PhaseAutomaton:
    aut = PhaseAutomaton(method=method, rank_changes_trigger=rank_trigger)
    aut.build(build_stream(phases))
    return aut


def _bar(label: str, width: int = 78) -> None:
    print(f"\n{'=' * width}")
    print(label)
    print("=" * width)


# ──────────────────────────────────────────────────────────────────────
# Demo 1: malleability -- 4 phases, different rank counts
# ──────────────────────────────────────────────────────────────────────


def demo_malleability():
    _bar("DEMO 1 — malleability: ranks 8 -> 16 -> 32 -> 64")

    aut = automaton(
        [(8, 5.0, 6), (16, 10.0, 6), (32, 3.0, 6), (64, 20.0, 6)], method=None
    )
    aut.print_graph()

    ref = ReferenceAutomaton.from_automaton_dict(aut.to_dict(), "app_A", "8_16_32_64")
    print("\n  node table (one behavior per rank count, as expected):")
    for ranks in (8, 16, 32, 64):
        b = ref.get_rank_behavior(ranks)[0]
        print(
            f"    ranks={ranks:3d}  period={b.period_mean:5.1f}s  "
            f"cycles=[{b.c_start_mean:.0f},{b.c_end_mean:.0f}]"
        )


# ──────────────────────────────────────────────────────────────────────
# Demo 2/3: same node, two runs at different speeds -- time vs. cycle count
# ──────────────────────────────────────────────────────────────────────


def demo_speed_confound():
    _bar("DEMO 2/3 — same node (ranks=32), two runs at different speeds")
    print(
        "  Both runs: 4 bursts of period=3s, THEN switch to period=8s (no rank\n"
        "  change). FAST run is unthrottled. SLOW run has the SAME 4-burst-then-\n"
        "  switch behavior, but contention stretches every burst by 20%.\n"
    )

    ref_fast = ReferenceAutomaton.from_automaton_dict(
        automaton([(32, 3.0, 4), (32, 8.0, 4)]).to_dict(), "run", "32"
    )
    ref_slow = ReferenceAutomaton.from_automaton_dict(
        automaton([(32, 3.0 * 1.2, 4), (32, 8.0 * 1.2, 4)]).to_dict(), "run", "32"
    )

    for label, ref in (("FAST", ref_fast), ("SLOW", ref_slow)):
        print(f"  {label} run behaviors:")
        for b in ref.get_rank_behavior(32):
            print(
                f"    period={b.period_mean:.2f}s  "
                f"time=[{b.t_start_mean:.1f},{b.t_end_mean:.1f}]s  "
                f"cycle=[{b.c_start_mean:.0f},{b.c_end_mean:.0f}]"
            )

    probe_t = (
        ref_fast.get_rank_behavior(32)[0].t_end_mean
        + ref_slow.get_rank_behavior(32)[0].t_end_mean
    ) / 2
    print(f"\n  Querying BOTH runs at t={probe_t:.1f}s (between the two switch times):")
    print(
        f"    fast -> {[f'{b.period_mean:.2f}s' for b in ref_fast.get_rank_behavior(32, at_time=probe_t)]}  (already switched)"
    )
    print(
        f"    slow -> {[f'{b.period_mean:.2f}s' for b in ref_slow.get_rank_behavior(32, at_time=probe_t)]}  (hasn't switched yet)"
    )
    print(
        "    -> wall-clock time gives DIFFERENT, inconsistent answers for the same logical point."
    )

    probe_c = 3  # still inside the first 4-burst behavior on BOTH runs
    print(f"\n  Querying BOTH runs at cycle={probe_c} (bursts observed so far):")
    print(
        f"    fast -> {[f'{b.period_mean:.2f}s' for b in ref_fast.get_rank_behavior(32, at_cycle=probe_c)]}"
    )
    print(
        f"    slow -> {[f'{b.period_mean:.2f}s' for b in ref_slow.get_rank_behavior(32, at_cycle=probe_c)]}"
    )
    print(
        "    -> cycle count gives the SAME behavior identified, regardless of run speed."
    )


# ──────────────────────────────────────────────────────────────────────
# Demo 4: seeding -- dilutes when close, survives (doesn't corrupt) when far off
# ──────────────────────────────────────────────────────────────────────


def demo_seeding(lib: AutomatonLibrary):
    _bar("DEMO 4 — seed: dilutes when close, survives when far off")

    lib.seed("seed_demo_close", {128: {"period": 10.3, "dwell": 20.0}})
    lib.save(automaton([(128, 10.0, 8)]), "seed_demo_close", "128")
    close_result = lib.get_rank_behavior("seed_demo_close", 128)
    print("  seed=10.3s, real=10.0s (3% off, within ~6% tolerance):")
    print(f"    -> {[f'{b.period_mean:.2f}s (n={b.n_samples})' for b in close_result]}")

    lib.seed("seed_demo_far", {128: {"period": 100.0, "dwell": 100.0}})
    lib.save(automaton([(128, 10.0, 8)]), "seed_demo_far", "128")
    far_result = lib.get_rank_behavior("seed_demo_far", 128)
    print("  seed=100.0s, real=10.0s (10x off, outside tolerance):")
    print(f"    -> {[f'{b.period_mean:.2f}s (n={b.n_samples})' for b in far_result]}")
    print(
        "    -> the bad seed is NOT silently corrected; it lingers as a permanent, wrong entry."
    )


# ──────────────────────────────────────────────────────────────────────
# Demo 5: cross-path node sharing
# ──────────────────────────────────────────────────────────────────────


def demo_cross_path(lib: AutomatonLibrary):
    _bar("DEMO 5 — cross-path: ranks=128 reached via two different malleability paths")

    lib.save(
        automaton([(8, 5.0, 4), (128, 10.0, 4)], method=None), "cross_path_demo", "8_128"
    )
    lib.save(
        automaton([(32, 3.0, 4), (128, 10.0, 4)], method=None),
        "cross_path_demo",
        "32_128",
    )

    shared = lib.get_rank_behavior("cross_path_demo", 128)
    print("  path A: 8->128, path B: 32->128, both reach 128 at period=10s")
    print(
        f"  get_rank_behavior('cross_path_demo', 128) -> {len(shared)} behavior(s), "
        f"n_samples={shared[0].n_samples} (pooled from BOTH paths)"
    )


if __name__ == "__main__":
    import shutil
    import tempfile

    library_dir = tempfile.mkdtemp(prefix="ftio_modeling_demo_")
    try:
        demo_malleability()
        demo_speed_confound()
        demo_seeding(AutomatonLibrary(library_dir))
        demo_cross_path(AutomatonLibrary(library_dir))
    finally:
        shutil.rmtree(library_dir, ignore_errors=True)
