"""
PhaseAutomaton demo — inspecting how the state machine builds up.

How it works
------------
The automaton consumes a stream of FTIO Prediction objects one at a time.
Each Prediction carries a dominant frequency (detected by DFT/wavelet).

  step 1  Prediction(freq=0.20)  →  bootstrap: open State 0 (freq=0.20)
  step 2  Prediction(freq=0.20)  →  same phase, State 0 grows
  …
  step N  Prediction(freq=0.05)  →  detector fires: close State 0,
                                     open State 1 (freq=0.05), log Transition

Four transition triggers (all optional, combinable):
  1. rank_change    — explicit configuration boundary; highest priority
  2. period_ratio   — max(T_new/T_cur, T_cur/T_new) > threshold; no warm-up
  3. cusum/ph/adwin — accumulate frequency sequence; fire on distribution shift
  4. ksigma         — state-adaptive k-sigma; robust to within-phase noise

Run:
    python examples/API/phase_automaton_demo.py
"""

from pathlib import Path

import numpy as np

from ftio.freq.prediction import Prediction
from ftio.prediction.phase_automaton import PhaseAutomaton

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def mock_prediction(
    freq: float, t_start: float, t_end: float, ranks: int = 1
) -> Prediction:
    pred = Prediction(transformation="dft", t_start=t_start, t_end=t_end, ranks=ranks)
    pred.dominant_freq = np.array([freq])
    pred.conf = np.array([0.95])
    pred.amp = np.array([1.0])
    return pred


def _bar(label: str, width: int = 65) -> None:
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print(f"{'─' * width}")


# ──────────────────────────────────────────────────────────────────────
# Demo 1: step-by-step trace with 3 phases (CUSUM)
# ──────────────────────────────────────────────────────────────────────


def demo_step_by_step():
    _bar("DEMO 1 — step-by-step  (3 phases, CUSUM)")

    stream = (
        [mock_prediction(0.20, t * 5, (t + 1) * 5) for t in range(5)]
        + [mock_prediction(0.05, 25 + t * 20, 25 + (t + 1) * 20) for t in range(5)]
        + [mock_prediction(0.50, 125 + t * 2, 125 + (t + 1) * 2) for t in range(5)]
    )

    automaton = PhaseAutomaton(method="cusum")

    print(
        f"\n{'Pred':>5}  {'freq (Hz)':>10}  {'period (s)':>11}  {'#states':>7}  {'#trans':>6}  {'event'}"
    )
    print(f"{'─'*5}  {'─'*10}  {'─'*11}  {'─'*7}  {'─'*6}  {'─'*20}")
    for i, pred in enumerate(stream, start=1):
        freq = pred.dominant_freq[0]
        transitioned = automaton.step(pred)
        event = "→ TRANSITION" if transitioned else ""
        print(
            f"{i:>5}  {freq:>10.3f}  {1/freq:>11.2f}  "
            f"{len(automaton.states):>7}  {len(automaton.transitions):>6}  {event}"
        )

    automaton.print_summary()
    automaton.plot(title="Demo 1 — CUSUM (3 phases)", show=False)


# ──────────────────────────────────────────────────────────────────────
# Demo 2: small within-phase variations — CUSUM vs k-sigma
# ──────────────────────────────────────────────────────────────────────


def demo_small_variations():
    _bar("DEMO 2 — small within-phase variations  (CUSUM vs k-sigma)")

    # Scenario
    # --------
    # Phase A (preds  1-10):  true freq = 0.20 Hz, noise ± 5%  (0.19–0.21 Hz)
    #   Within-phase jitter that should NOT trigger a transition.
    # Phase B (preds 11-20):  true freq = 0.26 Hz, noise ± 5%  (0.247–0.273 Hz)
    #   A genuine 30% upward shift that SHOULD be detected.
    # Phase C (preds 21-28):  true freq = 0.10 Hz, noise ± 5%  (0.095–0.105 Hz)
    #   A large 62% drop; both detectors should catch it quickly.
    #
    # k-sigma self-calibration:
    #   Phase A builds σ ≈ 0.006 Hz → 3σ threshold = 0.018 Hz.
    #   First Phase B obs (≈0.26 Hz): z = (0.26−0.20)/0.006 ≈ 10 → fires ✓
    #   Max within-phase noise (0.01 Hz): z ≈ 1.7 < 3 → no false alarm ✓

    rng = np.random.default_rng(42)

    def _noisy(freq, n, noise_frac=0.05, t0=0.0, period=5.0):
        raw = freq + rng.uniform(-noise_frac * freq, noise_frac * freq, size=n)
        return [
            mock_prediction(float(f), t0 + i * period, t0 + (i + 1) * period)
            for i, f in enumerate(raw)
        ]

    stream = (
        _noisy(0.20, 10, t0=0.0, period=5.0)
        + _noisy(0.26, 10, t0=50.0, period=5.0)
        + _noisy(0.10, 8, t0=100.0, period=10.0)
    )

    automata = {
        "cusum": PhaseAutomaton(method="cusum", rank_changes_trigger=False),
        "ksigma": PhaseAutomaton(method="ksigma", rank_changes_trigger=False),
    }

    labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 8
    true_freq = [0.20] * 10 + [0.26] * 10 + [0.10] * 8

    print(f"\n{'Pred':>5}  {'phase':>5}  {'true Hz':>8}  {'actual Hz':>10}  ", end="")
    for name in automata:
        print(f"  {name:>8} (#s/#t)", end="")
    print()
    print(f"{'─'*5}  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*22}  {'─'*22}")

    for i, (pred, lbl, tf) in enumerate(
        zip(stream, labels, true_freq, strict=False), start=1
    ):
        freq = pred.dominant_freq[0]
        print(f"{i:>5}  {lbl:>5}  {tf:>8.3f}  {freq:>10.4f}  ", end="")
        for _name, aut in automata.items():
            tr = aut.step(pred)
            marker = " ←" if tr else "  "
            print(f"  {len(aut.states):>8}/{len(aut.transitions):<3}{marker}", end="")
        print()

    print()
    for name, aut in automata.items():
        print(f"\n[{name}]")
        aut.print_summary()

    automata["ksigma"].plot(
        title="Demo 2 — k-sigma: small within-phase noise", show=False
    )
    automata["cusum"].plot(title="Demo 2 — CUSUM: small within-phase noise", show=False)


# ──────────────────────────────────────────────────────────────────────
# Demo 3: all detectors side-by-side (clean 3-phase signal)
# ──────────────────────────────────────────────────────────────────────


def demo_detector_comparison():
    _bar("DEMO 3 — all detectors side-by-side  (clean 3-phase signal)")

    stream = (
        [mock_prediction(0.20, t * 5, (t + 1) * 5) for t in range(5)]
        + [mock_prediction(0.05, 25 + t * 20, 25 + (t + 1) * 20) for t in range(5)]
        + [mock_prediction(0.50, 125 + t * 2, 125 + (t + 1) * 2) for t in range(5)]
    )
    stream_adwin = (
        [mock_prediction(5.0, t * 0.2, (t + 1) * 0.2) for t in range(5)]
        + [mock_prediction(0.05, 1 + t * 20, 1 + (t + 1) * 20) for t in range(5)]
        + [mock_prediction(50.0, 101 + t * 0.02, 101 + (t + 1) * 0.02) for t in range(5)]
    )

    automata = {
        "cusum": PhaseAutomaton(method="cusum"),
        "ph": PhaseAutomaton(method="ph"),
        "ksigma": PhaseAutomaton(method="ksigma"),
        "adwin": PhaseAutomaton(method="adwin"),
    }

    print(f"\n{'Pred':>5}  {'freq':>7}  ", end="")
    for name in automata:
        print(f"  {name:>10} (#s/#t)", end="")
    print()

    for i, (pred, pred_adwin) in enumerate(
        zip(stream, stream_adwin, strict=False), start=1
    ):
        freq = pred.dominant_freq[0]
        print(f"{i:>5}  {freq:>7.3f}  ", end="")
        for name, aut in automata.items():
            p = pred_adwin if name == "adwin" else pred
            aut.step(p)
            print(f"  {len(aut.states):>10}/{len(aut.transitions):<5}", end="")
        print()

    print()
    for _name, aut in automata.items():
        aut.print_summary()

    automata["ksigma"].plot(title="Demo 3 — k-sigma (large shifts)", show=False)


# ──────────────────────────────────────────────────────────────────────
# Demo 4: rank change triggers a new state (same frequency)
# ──────────────────────────────────────────────────────────────────────


def demo_rank_change():
    _bar("DEMO 4 — rank change trigger  (same freq, ranks 4 → 8 → 4)")

    stream = (
        [mock_prediction(0.20, t * 5.0, (t + 1) * 5.0, ranks=4) for t in range(5)]
        + [
            mock_prediction(0.20, 25.0 + t * 5.0, 25.0 + (t + 1) * 5.0, ranks=8)
            for t in range(5)
        ]
        + [
            mock_prediction(0.20, 50.0 + t * 5.0, 50.0 + (t + 1) * 5.0, ranks=4)
            for t in range(5)
        ]
    )

    aut = PhaseAutomaton(method=None, rank_changes_trigger=True)

    print(
        f"\n{'Pred':>5}  {'freq (Hz)':>10}  {'ranks':>6}  {'#states':>7}  {'#trans':>6}  {'event'}"
    )
    print(f"{'─'*5}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*20}")
    for i, pred in enumerate(stream, start=1):
        freq = pred.dominant_freq[0]
        transitioned = aut.step(pred)
        event = "→ RANK CHANGE" if transitioned else ""
        print(
            f"{i:>5}  {freq:>10.3f}  {int(pred.ranks):>6}  "
            f"{len(aut.states):>7}  {len(aut.transitions):>6}  {event}"
        )

    aut.print_summary()
    print("\n  Cause breakdown:")
    for t in aut.transitions:
        print(f"    {t}")
    aut.plot(title="Demo 4 — Rank-change trigger (freq constant)", show=False)


# ──────────────────────────────────────────────────────────────────────
# Demo 5: period-ratio threshold (no statistical detector)
# ──────────────────────────────────────────────────────────────────────


def demo_period_ratio():
    _bar("DEMO 5 — period-ratio threshold  (method=None, threshold=1.5)")

    stream = (
        [mock_prediction(0.20, t * 5.0, (t + 1) * 5.0) for t in range(4)]
        + [
            mock_prediction(0.05, 20.0 + t * 20.0, 20.0 + (t + 1) * 20.0)
            for t in range(4)
        ]
        + [
            mock_prediction(0.50, 100.0 + t * 2.0, 100.0 + (t + 1) * 2.0)
            for t in range(4)
        ]
    )

    aut_ratio = PhaseAutomaton(
        method=None, period_ratio_threshold=1.5, rank_changes_trigger=False
    )
    aut_ksigma = PhaseAutomaton(method="ksigma", rank_changes_trigger=False)
    aut_cusum = PhaseAutomaton(method="cusum", rank_changes_trigger=False)

    print(
        f"\n{'Pred':>5}  {'freq (Hz)':>10}  {'period (s)':>10}  "
        f"{'ratio-only':>12}  {'ksigma':>8}  {'cusum':>8}  {'event'}"
    )
    print(f"{'─'*5}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*20}")
    for i, pred in enumerate(stream, start=1):
        freq = pred.dominant_freq[0]
        tr_r = aut_ratio.step(pred)
        tr_k = aut_ksigma.step(pred)
        tr_c = aut_cusum.step(pred)
        event = "period_ratio" if tr_r else "ksigma" if tr_k else "cusum" if tr_c else ""
        print(
            f"{i:>5}  {freq:>10.3f}  {1/freq:>10.2f}  "
            f"{len(aut_ratio.states):>5}/{len(aut_ratio.transitions):<5}  "
            f"{len(aut_ksigma.states):>3}/{len(aut_ksigma.transitions):<3}  "
            f"{len(aut_cusum.states):>3}/{len(aut_cusum.transitions):<3}  "
            f"{event}"
        )

    print()
    aut_ratio.print_summary()
    aut_ratio.plot(title="Demo 5 — Period-ratio threshold (method=None)", show=False)


# ──────────────────────────────────────────────────────────────────────
# Demo 6: real HPC trace files  (JSONL + IOR + HACC-IO)
# ──────────────────────────────────────────────────────────────────────


def demo_real_files():
    _bar("DEMO 6 — real HPC trace files  (JSONL · IOR · HACC-IO)")

    # Six real trace files ordered by increasing rank count.
    # Each call to main() yields one Prediction summarising the full trace.
    # rank_changes_trigger=True fires whenever ranks differ between predictions,
    # creating explicit configuration-boundary states.
    candidates = [
        (_REPO_ROOT / "examples/tmio/JSONL/8.jsonl", "JSONL 8-rank"),
        (_REPO_ROOT / "examples/tmio/ior/parallel/384.msgpack", "IOR parallel 384-rank"),
        (
            _REPO_ROOT / "examples/tmio/ior/collective/1536_old.json",
            "IOR coll. 1536-rank (old run)",
        ),
        (
            _REPO_ROOT / "examples/tmio/ior/collective/1536_new.json",
            "IOR coll. 1536-rank (new run)",
        ),
        (_REPO_ROOT / "examples/tmio/HACC-IO/9216.json", "HACC-IO 9216-rank"),
        (
            _REPO_ROOT / "examples/tmio/HACC-IO/9216_new.json",
            "HACC-IO 9216-rank (new run)",
        ),
    ]

    from ftio.cli.ftio_core import main as ftio_main

    preds: list = []
    print(f"\n  {'Label':<37}  {'ranks':>6}  {'freq (Hz)':>10}  {'period (s)':>10}")
    print(f"  {'─'*37}  {'─'*6}  {'─'*10}  {'─'*10}")
    for path, label in candidates:
        try:
            p_list, _ = ftio_main(["ftio", str(path), "-e", "no"])
            valid = [p for p in p_list if not p.is_empty()]
            for p in valid:
                freq = p.dominant_freq[0]
                print(f"  {label:<37}  {p.ranks:>6}  {freq:>10.5f}  {1/freq:>10.2f}")
                preds.append(p)
        except Exception as exc:
            print(f"  {label:<37}  SKIP ({exc})")

    if len(preds) < 2:
        print("\n  Not enough valid predictions — skipping.")
        return

    print(f"\n  Total predictions loaded: {len(preds)}")

    # Three automata with different trigger configurations
    configs = {
        "rank + ksigma": PhaseAutomaton(method="ksigma", rank_changes_trigger=True),
        "rank + period-ratio": PhaseAutomaton(
            method=None, rank_changes_trigger=True, period_ratio_threshold=1.5
        ),
        "ksigma only": PhaseAutomaton(method="ksigma", rank_changes_trigger=False),
    }
    for aut in configs.values():
        aut.build(preds)

    print(
        f"\n  {'Config':<22}  {'#states':>7}  {'#trans':>7}  {'States (period s, ranks)'}"
    )
    print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*50}")
    for name, aut in configs.items():
        summary = "  →  ".join(f"{s.period:.1f}s/{s.ranks}r" for s in aut.states)
        print(f"  {name:<22}  {len(aut.states):>7}  {len(aut.transitions):>7}  {summary}")

    print()
    for name, aut in configs.items():
        print(f"\n[{name}]")
        aut.print_summary()

    configs["rank + ksigma"].plot(
        title="Demo 6 — Real HPC traces: rank + k-sigma", show=False
    )
    configs["rank + period-ratio"].plot(
        title="Demo 6 — Real HPC traces: rank + period-ratio", show=False
    )


# ──────────────────────────────────────────────────────────────────────
# Demo 7: multi-rank mock  (same 3-phase pattern, ranks=1 vs ranks=16)
# ──────────────────────────────────────────────────────────────────────


def demo_multi_rank():
    _bar("DEMO 7 — multi-rank mock  (ranks=1 vs ranks=16, k-sigma)")

    def three_phase_stream(scale: float, ranks: int):
        return (
            [mock_prediction(0.20 * scale, t * 5, (t + 1) * 5, ranks) for t in range(5)]
            + [
                mock_prediction(0.05 * scale, 25 + t * 20, 25 + (t + 1) * 20, ranks)
                for t in range(5)
            ]
            + [
                mock_prediction(0.50 * scale, 125 + t * 2, 125 + (t + 1) * 2, ranks)
                for t in range(5)
            ]
        )

    automata = {}
    for label, (ranks, scale) in [("ranks=1", (1, 1.0)), ("ranks=16", (16, 4.0))]:
        aut = PhaseAutomaton(method="ksigma")
        aut.build(three_phase_stream(scale, ranks))
        automata[label] = aut

    print(f"\n{'Config':<12}  {'#states':>7}  {'#trans':>6}  {'State periods (s)'}")
    print(f"{'─'*12}  {'─'*7}  {'─'*6}  {'─'*30}")
    for label, aut in automata.items():
        periods = "  →  ".join(f"{s.period:.2f}" for s in aut.states)
        print(f"{label:<12}  {len(aut.states):>7}  {len(aut.transitions):>6}  {periods}")

    for label, aut in automata.items():
        aut.print_summary()
        aut.plot(title=f"Demo 7 — {label}", show=False)


# ──────────────────────────────────────────────────────────────────────
# Show all queued matplotlib figures
# ──────────────────────────────────────────────────────────────────────


def _show_all():
    try:
        import matplotlib.pyplot as plt

        plt.show()
    except Exception:
        pass


def demo_warmup_artifact():
    """Why the FIRST prediction of a run must not be modelled.

    A DFT cannot resolve a period longer than the window it ran on, and FTIO trims
    that window to the data seen so far. Early in a run the window therefore holds
    a single I/O phase, and the transform reports that one burst's *width* as the
    "period" -- the k=1 bin, one cycle across the whole window. That is not
    evidence of periodicity.

    Measured on a real LAMMPS GLASS run (8 x 1 GB restarts, true period 41.8 s),
    replaying the trace the way the online predictor sees it:

        window   period   conf   cycles in window
          5.0 s    5.0 s   1.00     1.01   <- one burst; "period" = its width
         44.5 s   44.4 s   0.67     1.00   <- right answer, still only one cycle
         84.8 s   42.4 s   0.59     2.00   <- two cycles: now it is measurable

    Note confidence is 1.00 on the WRONG answer and 0.67 on the right one, and the
    wrong answer is stable for four rounds -- so neither a confidence gate nor a
    stability gate can filter it. Only the cycle count can. That is what
    `min_cycles` does: it refuses to model a period the window has not held twice.
    """
    print("\n" + "=" * 70)
    print("DEMO: a period needs two periods before it can be believed")
    print("=" * 70)

    # LAMMPS as measured: four warm-up rounds (a 5 s burst in a 5 s window),
    # then the steady state once a second checkpoint is in the window.
    warmup = [mock_prediction(1 / 5.0, 0.0, 5.0, ranks=2) for _ in range(4)]
    steady = [mock_prediction(1 / 42.0, 0.0, 85.0 + 10 * i, ranks=2) for i in range(8)]

    for label, min_cycles in (("min_cycles=1  (off)", 1.0), ("min_cycles=2  (on)", 2.0)):
        aut = PhaseAutomaton(method="cusum", min_cycles=min_cycles)
        for pred in warmup + steady:
            aut.step(pred)
        states = ", ".join(f"{s.period:.0f}s" for s in aut.states)
        print(
            f"\n  {label}:  {len(aut.states)} state(s), "
            f"{len(aut.transitions)} transition(s)  ->  [{states}]"
        )

    print(
        "\n  Off: the run 'learns' a 5 s phase and then a false transition to 42 s."
        "\n  On : only the real 42 s phase is modelled; no transition ever happened."
        "\n"
        "\n  Cost: one extra period before modelling starts."
        "\n  Note: min_cycles only makes sense when a Prediction's t_start/t_end span"
        "\n  the whole run so far, as they do online. It is off by default for callers"
        "\n  that hand in a Prediction per short slice."
    )


def demo_build_a_model_from_a_real_run():
    """How to create a phase model from an actual JIT run.

    A JIT run writes `bandwidth.json` ({"t": [...], "b": [...]}) into its log dir,
    e.g. logs_nodes1_Jobid0_DF_26/bandwidth.json. Replay it the way the online
    predictor sees it -- a growing window -- and step the automaton:

        import json
        import numpy as np
        from ftio.cli.ftio_core import core
        from ftio.modeling import PhaseAutomaton
        from ftio.parse.args import parse_args

        d = json.load(open("logs_nodes1_Jobid0_DF_26/bandwidth.json"))
        t, b = np.asarray(d["t"]), np.asarray(d["b"])
        args = parse_args(["-m", "write", "-e", "no", "--freq", "10"], "ftio")

        aut = PhaseAutomaton(method="cusum")
        now = t[0] + 10.0                      # one prediction round every 10 s
        while now <= t[-1]:
            m = t <= now                       # everything observed so far
            sim = {
                "time": t[m],
                "bandwidth": b[m],
                "total_bytes": int(np.trapezoid(b[m], t[m])),
                "ranks": 2,
            }
            prediction, _ = core(sim, args)
            aut.step(prediction)
            now += 10.0

        aut.drop_warmup_states()               # the first state is a DFT artifact
        for s in aut.states:
            print(f"phase: period={s.period:.1f}s  conf={s.confidence:.2f}")

    To persist it for the NEXT run of the same application, hand it to the
    library -- ModelManager.save_run() calls drop_warmup_states() for you:

        from ftio.modeling.model_manager import ModelManager
        ModelManager("./ftio_models", "lammps", "greedy").save_run(aut)

    The next run loads `./ftio_models/lammps/ranks_<key>.json` and starts from the
    known period instead of rediscovering it -- which on the measured traces takes
    40 s for LAMMPS (3 checkpoints) and 140 s for WRF (4 checkpoints).
    """
    print("\n" + "=" * 70)
    print("DEMO: building a phase model from a real JIT run")
    print("=" * 70)
    print(demo_build_a_model_from_a_real_run.__doc__)


if __name__ == "__main__":
    demo_step_by_step()
    demo_small_variations()
    demo_detector_comparison()
    demo_rank_change()
    demo_period_ratio()
    demo_real_files()
    demo_multi_rank()
    demo_warmup_artifact()
    demo_build_a_model_from_a_real_run()
    _show_all()
