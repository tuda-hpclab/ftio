# Phase Automaton

The phase automaton models I/O behaviour as a finite state machine where each **state** represents a stable frequency regime and **transitions** are fired when the regime changes.  It is designed for online prediction scenarios where the application's I/O pattern may shift — for example, between compute phases, I/O phases, or checkpointing bursts.

- [1. Concept](#1-concept)
- [2. Enabling the automaton](#2-enabling-the-automaton)
- [3. Transition triggers](#3-transition-triggers)
  - [3.1 Rank-count trigger](#31-rank-count-trigger)
  - [3.2 Period-ratio trigger](#32-period-ratio-trigger)
  - [3.3 Statistical detector](#33-statistical-detector)
- [4. Combining triggers](#4-combining-triggers)
- [5. Export](#5-export)
- [6. Offline STFT](#6-offline-stft)
  - [6.1 Worked example: a real malleable run (HACC-IO, ranks 4 → 8 → 4)](#61-worked-example-a-real-malleable-run-hacc-io-ranks-4--8--4)
- [7. Examples](#7-examples)
- [8. Example output](#8-example-output)
  - [8.1 Three-phase signal — CUSUM detector](#81-three-phase-signal--cusum-detector)
  - [8.2 Rank-change trigger — same frequency, different rank counts](#82-rank-change-trigger--same-frequency-different-rank-counts)
- [9. Output JSON format](#9-output-json-format)
- [10. Automaton Profiles](#10-automaton-profiles)
  - [10.1 What each piece does](#101-what-each-piece-does)
  - [10.2 Example output](#102-example-output)
  - [10.3 Application identity](#103-application-identity)
    - [10.3.1 First run — cold start](#1031-first-run--cold-start)
    - [10.3.2 Subsequent runs — warm start](#1032-subsequent-runs--warm-start)
  - [10.4 Malleable applications](#104-malleable-applications)
  - [10.5 Matching strategies](#105-matching-strategies)
  - [10.6 AutomatonProfile file format](#106-automatonprofile-file-format)
- [11. AutomatonLibrary](#11-automatonlibrary)
  - [11.1 Configuration table (nodes)](#111-configuration-table-nodes)
    - [11.1.1 Path view vs. node view](#1111-path-view-vs-node-view)
    - [11.1.2 Multiple behaviors per configuration](#1112-multiple-behaviors-per-configuration)
    - [11.1.3 Scoping the behavior using wall time or phase cycles](#1113-scoping-the-behavior-using-wall-time-or-phase-cycles)
    - [11.1.4 User-provided input before the run](#1114-user-provided-input-before-the-run)
    - [11.1.5 Cross-path sharing](#1115-cross-path-sharing)
  - [11.2 Walkthrough](#112-walkthrough)
  - [11.3 Redis-backed library](#113-redis-backed-library)
- [12. Combining with other flags](#12-combining-with-other-flags)

---

## 1. Concept

Each time `predictor` produces a new prediction, the automaton checks whether the new prediction belongs to the current state or signals a phase transition:

```
prediction_n  ──►  automaton  ──► same state  (no action)
                      │
                      └──► transition  ──► new state  (window reset / output)
```

States accumulate a history of predictions.  When a transition fires, a new state is opened and the analysis window can be reset to focus on the new regime.

---

## 2. Enabling the automaton

```bash
predictor live.jsonl -f 100 --phase-automaton
```

The automaton also works offline with `ftio --transformation stft` — see [6. Offline STFT](#6-offline-stft) below.

---

## 3. Transition triggers

Three independent triggers can fire a transition.  They can be used individually or combined (any one trigger firing is sufficient).

### 3.1 Rank-count trigger

**Default: enabled.**  Fires when the number of active I/O ranks in the new prediction differs from the current state's rank count.

The rank count itself comes from one of two places:

- **Authoritative, when available** — TMIO's msgpack trace format reports `total_number_of_ranks` directly (the size of the run's own communicator) on every sample. When present, FTIO uses it verbatim, and the trigger fires the moment it changes — there is nothing to guess.
- **Inferred, otherwise** — older traces, or formats without that field, only give `number_of_ranks`, a grouping key that reflects however many ranks' messages had been received into the current window. A single window where a straggler rank hadn't reported yet can look exactly like a rank change even though nothing changed. To guard against that, an inferred rank difference must repeat for `--pa-rank-confirm` consecutive predictions (default: **2**) before it is accepted:

    ```bash
    predictor live.jsonl -f 100 --phase-automaton --pa-rank-confirm 3
    ```

    Set it to `1` to fire immediately on the first differing window (the old behaviour). There is one exception even before confirmation: if the period-ratio or statistical detector *also* fires on that same first differing window, that is strong corroborating evidence the rank change is real (a stray straggler wouldn't also move the frequency estimate), so the transition fires immediately and is still labelled `rank_change` rather than `frequency`.

Disable the trigger entirely with `--pa-no-rank-trigger`:
```bash
predictor live.jsonl -f 100 --phase-automaton --pa-no-rank-trigger
```

### 3.2 Period-ratio trigger

**Default: disabled.**  Fires when the ratio between the new and current dominant period exceeds a threshold:

```
max(T_new / T_cur,  T_cur / T_new)  >  RATIO
```

`RATIO = 1.5` means a 50 % change in period length triggers a transition.

```bash
predictor live.jsonl -f 100 --phase-automaton --pa-period-ratio 1.5
```

No warm-up period is needed; the check activates from the second prediction onwards.

### 3.3 Statistical detector

**Default: `ksigma`.**  A statistical change-point test applied to the series of dominant-frequency values.  Choose with `--pa-method`:

| Method | Description | Characteristics |
|--------|-------------|-----------------|
| `ksigma` | State-adaptive k-sigma | Recommended.  Adapts threshold to within-state variance; robust to noise. |
| `cusum` | Adaptive-variance CUSUM | Fast reaction to sustained shifts; sensitive to variance changes. |
| `ph` | Page-Hinkley | Sequential test for monotonic drift; good for gradual changes. |
| `adwin` | Adaptive Windowing | Needs many samples or large frequency ratios to fire. |
| `none` | Disabled | Use only rank and/or period-ratio triggers. |

```bash
# Default: ksigma
predictor live.jsonl -f 100 --phase-automaton

# CUSUM
predictor live.jsonl -f 100 --phase-automaton --pa-method cusum

# No statistical detection, period-ratio only
predictor live.jsonl -f 100 --phase-automaton --pa-method none --pa-period-ratio 1.5
```

---

## 4. Combining triggers

Any combination is valid.  A transition fires when **any** enabled trigger activates:

```bash
# All three triggers active
predictor live.jsonl -f 100 --phase-automaton \
    --pa-method ksigma \
    --pa-period-ratio 1.5
    # rank trigger is on by default

# Period-ratio and statistical only (no rank trigger)
predictor live.jsonl -f 100 --phase-automaton \
    --pa-method cusum \
    --pa-period-ratio 2.0 \
    --pa-no-rank-trigger
```

---

## 5. Export

When `predictor` exits, the full automaton state (all states, transitions, and configuration) is written as JSON:

```bash
predictor live.jsonl -f 100 --phase-automaton --pa-export /tmp/my_automaton.json
```

Default path: `./phase_automaton.json`.

---

## 6. Offline STFT

The automaton is not limited to the online predictor. `--transformation stft` already slides a window across the *entire* trace in a single call and reports a dominant frequency per window (see `ftio_stft` in `ftio/freq/_stft_workflow.py`) — that per-window sequence is exactly what the automaton needs, so one offline run reconstructs the same state/transition history the online predictor would have produced by polling ZMQ the whole time:

```bash
ftio trace.jsonl --transformation stft --phase-automaton
```

All the `--pa-*` flags described above apply the same way (`--pa-method`, `--pa-period-ratio`, `--pa-rank-confirm`, `--pa-export`, ...). On exit, `ftio` prints the same summary/graph the online predictor logs and writes the same `phase_automaton.json`.

Only STFT produces a window sequence — DFT and wavelet each return one global result for the whole trace, so `--phase-automaton` has nothing to build from with those transformations and is silently skipped.

Using the Python API directly (what the CLI does under the hood):

```python
from ftio.freq._stft_workflow import ftio_stft
from ftio.modeling.phase_automaton import PhaseAutomaton, windows_from_stft_prediction
from ftio.parse.args import parse_args

args = parse_args(["-tr", "stft", "-e", "no"], "ftio")
prediction, _ = ftio_stft(args, bandwidth, time_stamps, ranks=8)

windows = windows_from_stft_prediction(prediction)   # one Prediction per STFT window
aut = PhaseAutomaton(method="ksigma")
aut.build(windows)
aut.print_graph()
```

See `demo_offline_stft()` in `examples/API/phase_automaton_demo.py` for a runnable version (a synthetic 5 Hz → 10 Hz trace, one STFT call, two states, one transition).

### 6.1 Worked example: a real malleable run (HACC-IO, ranks 4 → 8 → 4)

This is a real trace, not a synthetic signal — HACC-IO run under [DMR](https://github.com/bsc-pm/dmr) (Dynamic MPI Resources) + DLB, scaling from 4 to 8 ranks and back to 4 mid-run. TMIO logged one JSONL line per flush, each carrying its own `number_of_ranks` at the time of that flush, so the rank change is genuinely recorded in the trace, flush by flush.

```bash
ftio all_MPI.jsonl -f 10 --transformation stft --phase-automaton -e no
```

```
Identified segments
┏━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Window ┃ Freq (Hz) ┃ Period (s) ┃ Conf. (%) ┃ Time Range (s) ┃
┡━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│      0 │ 7.895e-01 │     1.2667 │    100.00 │   [0.82, 8.42] │
│      1 │ 2.632e-01 │     3.8000 │    100.00 │  [6.52, 14.12] │
│      2 │ 3.947e-01 │     2.5333 │    100.00 │ [12.22, 19.82] │
│      3 │ 3.947e-01 │     2.5333 │    100.00 │ [17.92, 25.52] │
│      4 │ 7.895e-01 │     1.2667 │    100.00 │ [23.62, 31.22] │
└────────┴───────────┴────────────┴───────────┴────────────────┘

PhaseAutomaton  method='ksigma'  rank_sensitive=True (confirm=2)  states=2  transitions=1
  State(0: freq=0.7895 Hz, period=1.27 s, ranks=4, n_phases=3, duration=19.0 s)
  State(1: freq=0.3947 Hz, period=2.53 s, ranks=8, n_phases=2, duration=11.4 s)
  Transition(0→1 at t=19.82 s, 0.7895→0.3947 Hz, cause='rank_change', pred #3)
```

![Phase automaton on a real HACC-IO malleability trace: two states (4 ranks, 8 ranks) and one rank_change transition](images/phase_automaton_malleability.png)

Two things worth noticing here:

- **The transition is labelled `rank_change`, not `frequency`**, even though the statistical detector would have fired on the period shift alone (0.79 Hz → 0.26/0.39 Hz is a large jump). The rank count differing on the very same window the frequency shifts is exactly the corroboration case described above — the automaton attributes the transition to the more specific, more actionable cause.
- **The run's final return to 4 ranks (window 4, the rightmost dot in the plot) does not open a third state.** The trace ends immediately after that one window, so `--pa-rank-confirm`'s default of 2 never gets a second window to confirm the reading. This is the debounce working as intended, not a bug — a single trailing window is exactly the kind of reading it's designed not to trust on its own (see [3.1 Rank-count trigger](#31-rank-count-trigger)). A longer trailing segment (or `--pa-rank-confirm 1`) would have caught it.

#### 6.1.1 Where this trace comes from

HACC-IO's malleability support lives in the `HACC-IO` repo's `Makefile`, gated by `make MALLEABLE=1` (needs `DMR_PATH`/`DLB_HOME` pointing at DMR/DLB installs). Under the hood it's real MPI rank resizing via DMR's `dmr_wrapper`, not a simulated rank count:

```bash
# build against DMR/DLB
make MALLEABLE=1 run_malleable_with_include2

# submitted via sbatch, which launches through dmr_wrapper:
$DMR_PATH/scripts/dmr_wrapper mpirun --host "$NODELIST_WITH_COUNTS" ./HACC_ASYNC_IO 1000000 test_run/mpi
```

There are two ways to attach the TMIO tracer to the same DMR-driven run, which is where the `_MPI`/`_Libc` naming in the example trace files (`4_MPI.jsonl`, `4_Libc.jsonl`, ...) comes from — both scale the same 4→8→4, they just differ in how TMIO gets linked in:

| Target | TMIO attached via |
|---|---|
| `run_malleable_with_include2` (→ `_MPI` traces) | linked in at compile time (`-ltmio`) |
| `run_malleable_with_lib` (→ `_Libc` traces) | `LD_PRELOAD=./libtmio.so`, no recompile needed |

---

## 7. Examples

```bash
# Minimal: default triggers (rank + ksigma)
predictor live.jsonl -f 100 --phase-automaton -e no

# Detect period changes of 50 % or more, disable statistical trigger
predictor live.jsonl -f 100 --phase-automaton \
    --pa-method none --pa-period-ratio 1.5 -e no

# Maximum sensitivity: all three triggers, Page-Hinkley for slow drifts
predictor live.jsonl -f 100 --phase-automaton \
    --pa-method ph --pa-period-ratio 1.3 -e no

# Export automaton, then inspect
predictor live.jsonl -f 100 --phase-automaton --pa-export run_automaton.json -e no
cat run_automaton.json
```

---

## 8. Example output

The following examples use the demo script (`examples/API/phase_automaton_demo.py`) and can be reproduced via the `PhaseAutomaton` API directly.

### 8.1 Three-phase signal — CUSUM detector

A synthetic trace with three distinct I/O periods fed one prediction at a time.  Each row shows the state count and whether a transition fired:

```
 Pred   freq (Hz)   period (s)  #states  #trans  event
─────  ──────────  ───────────  ───────  ──────  ────────────────────
    1       0.200         5.00        1       0
    2       0.200         5.00        1       0
    3       0.200         5.00        1       0
    4       0.200         5.00        1       0
    5       0.200         5.00        1       0
    6       0.050        20.00        2       1  → TRANSITION
    7       0.050        20.00        2       1
    8       0.050        20.00        2       1
    9       0.050        20.00        2       1
   10       0.050        20.00        2       1
   11       0.500         2.00        3       2  → TRANSITION
   12       0.500         2.00        3       2
   13       0.500         2.00        3       2
   14       0.500         2.00        3       2
   15       0.500         2.00        3       2
```

Calling `automaton.print_graph()` on the resulting automaton prints the state graph:

```
=================================================================
PhaseAutomaton graph  method='cusum'  states=3  transitions=2
─────────────────────────────────────────────────────────────────
  ┌──────────────────────────────┐
  │ S0                           │
  │ f = 0.2000 Hz                │
  │ T = 5.00 s                   │
  │ ranks = 1                    │
  │ dur   = 45.0 s               │
  └──────────────────────────────┘
                  │
                  ├─ freq shift (statistical)  @t=45.0 s
                  │  T: 5.00 s → 20.00 s
                  ▼
  ┌──────────────────────────────┐
  │ S1                           │
  │ f = 0.0500 Hz                │
  │ T = 20.00 s                  │
  │ ranks = 1                    │
  │ dur   = 82.0 s               │
  └──────────────────────────────┘
                  │
                  ├─ freq shift (statistical)  @t=127.0 s
                  │  T: 20.00 s → 2.00 s
                  ▼
  ┌──────────────────────────────┐
  │ S2                           │
  │ f = 0.5000 Hz                │
  │ T = 2.00 s                   │
  │ ranks = 1                    │
  │ dur   = 8.0 s                │
  └──────────────────────────────┘
=================================================================
```

### 8.2 Rank-change trigger — same frequency, different rank counts

When the dominant period stays constant but the rank count shifts (e.g. checkpoint scaling), the rank trigger fires regardless of the statistical detector:

```
=================================================================
PhaseAutomaton graph  method=None  states=3  transitions=2
─────────────────────────────────────────────────────────────────
  ┌──────────────────────────────┐
  │ S0                           │
  │ f = 0.2000 Hz                │
  │ T = 5.00 s                   │
  │ ranks = 4                    │
  │ dur   = 30.0 s               │
  └──────────────────────────────┘
                  │
                  ├─ rank change  @t=30.0 s
                  │  T: 5.00 s → 5.00 s
                  ▼
  ┌──────────────────────────────┐
  │ S1                           │
  │ f = 0.2000 Hz                │
  │ T = 5.00 s                   │
  │ ranks = 8                    │
  │ dur   = 25.0 s               │
  └──────────────────────────────┘
                  │
                  ├─ rank change  @t=55.0 s
                  │  T: 5.00 s → 5.00 s
                  ▼
  ┌──────────────────────────────┐
  │ S2                           │
  │ f = 0.2000 Hz                │
  │ T = 5.00 s                   │
  │ ranks = 4                    │
  │ dur   = 20.0 s               │
  └──────────────────────────────┘
=================================================================
```

Three states are created even though the frequency never changes, because each rank count (4 → 8 → 4) defines a distinct operational regime.

---

## 9. Output JSON format

The exported file contains:

```json
{
  "config": {
    "method": "ksigma",
    "period_ratio": null,
    "rank_trigger": true
  },
  "states": [
    {
      "id": 0,
      "predictions": [...],
      "dominant_freq_median": 0.092,
      "ranks": 384
    },
    {
      "id": 1,
      "predictions": [...],
      "dominant_freq_median": 0.045,
      "ranks": 192
    }
  ],
  "transitions": [
    {
      "from_state": 0,
      "to_state": 1,
      "trigger": "rank_change",
      "timestamp": 145.3
    }
  ]
}
```

`states` lists every identified regime; `transitions` records what caused each regime change and when.

---

## 10. Automaton Profiles

Automaton profiles extend the phase automaton with a second purpose: **predicting future transitions** by comparing a live run against statistics pooled from past runs of the same application.

Two kinds of prediction remain explicitly separate:

| | Source | Answers |
|---|---|---|
| **Frequency prediction** | DFT / wavelet (existing) | What is the dominant period *right now*? |
| **Transition prediction** | Automaton profile (new) | *When* will the period change, and to *what*? |

Enable with `--pa-library`:

```bash
predictor live.jsonl -f 100 --pa-library ./ftio_models --pa-app-name ior
```

`--pa-library` implies `--phase-automaton`; you do not need to pass both.

---

### 10.1 What each piece does

Two objects hold the data:

| Object | One instance per... | Holds |
|---|---|---|
| `PhaseAutomaton` | one live run | that run's actual states/transitions: real, observed data points |
| `AutomatonProfile` | one `(app, rank_key)` | statistics *pooled* across every `PhaseAutomaton` run seen for that config (mean ± std per state, `run_count`) |

`AutomatonProfile` already makes a prediction on its own: `get_rank_behavior(ranks)` returns the expected period/dwell for a configuration, pooled from every past run: a history-based estimate, independent of any live run. What it *cannot* do alone is the live version of that same question: "where is *this run*, right now, and when will it transition?" That combines the live `PhaseAutomaton` with an `AutomatonProfile` through two more objects:

- **`StateTracker`**: matches the live run's current position against the profile's states.
- **`TransitionPredictor`**: uses that position plus the profile's dwell-time distributions to predict *when* the next transition fires and *what* period follows.

Where an `AutomatonProfile` is loaded from and saved back to on disk is a separate, storage-level concern. See [11. AutomatonLibrary](#11-automatonlibrary).

---

### 10.2 Example output

The examples below follow a 3-state IOR run (write → read → checkpoint) at 128 ranks through its history: cold start, the first warm run, then a later run with several prior runs already pooled.

In the logs, lines marked `+` (green) come from `AutomatonProfile`/`AutomatonLibrary`, through `ModelManager`, `StateTracker`, or `TransitionPredictor`: loading or saving a profile, matching position against it, or its transition/timing estimate. Unmarked lines come from the live `PhaseAutomaton` and the frequency prediction underneath it, and would appear the same way with `--phase-automaton` alone, no profile involved.

#### 10.2.1 Run 1 — cold start

No reference exists yet.  FTIO builds the automaton from scratch and saves it on exit.

```diff
+[ModelManager] Cold start — no reference for ior/ranks_128. Building automaton from this run.

 [PREDICTOR] (#1): Started
 [PREDICTOR] (#1): Dominant freq 0.518 Hz (1.93 sec)
 [PREDICTOR] (#1): Freq candidates (1 found):
 [PREDICTOR] (#1):    0) 0.52 Hz -- conf 0.87
 [PREDICTOR] (#1): Time window 2.000 sec ([0.000,2.000] sec)
 [PREDICTOR] (#1): Total bytes 512 MB
 [PREDICTOR] (#1): Phase automaton: State 0 — freq=0.5181 Hz, period=1.93 s, ranks=128, n_preds=1
+[PREDICTOR] (#1): Reference (greedy): cold start — no library match. Learning from this run.
 [PREDICTOR] (#1): Ended
 ...
 ^C
 [PhaseAutomaton] Saved to ./phase_automaton.json  (3 states, 2 transitions)
+[AutomatonLibrary] Saved ior/ranks_128 → ./ftio_models/ior/ranks_128.json (1 run(s), 3 states)
```

#### 10.2.2 Run 2 — first warm start, no timing bounds yet

Run 1's profile (1 run pooled, `std = 0`) is loaded. The next-period prediction is available but ETA bounds are not yet:

**Startup:**
```diff
+[ModelManager] Loaded ior/ranks_128 (3 states, 1 run(s))
```

```diff
+[PREDICTOR] (#7): Reference (greedy): state 1/3, pos=0%, tracking=0.93
+[PREDICTOR] (#7):   Next period ≈ 0.96s (timing improves after ≥2 runs in library)
```

**Exit:**
```diff
 [PhaseAutomaton] Saved to ./phase_automaton.json  (3 states, 2 transitions)
+[AutomatonLibrary] Saved ior/ranks_128 → ./ftio_models/ior/ranks_128.json (2 run(s), 3 states)
```

#### 10.2.3 Run 5 — mature warm start, full timing bounds

This example skips ahead to a run where the profile already has four prior runs pooled into it, so real timing bounds are available. A reference is loaded. Each prediction now shows a transition forecast alongside the frequency result.

The predictions quoted below (`#7`, `#22`, `#24`, ...) are snapshots along this run's actual timeline: state 0 (write) from t=0 to the transition at prediction `#24` (t=48s), state 1 (read) from there to the transition at `#72` (t=145s), then state 2 (checkpoint):

```
period(s)
 1.93 │████████████████                                ██████████
 0.96 │                ████████████████████████████████
      ┬───────────────┬───────────────────────────────┬─────────
      0              48                             145  t(s)

      state 0 (write)   state 1 (read)                 state 2 (checkpoint)
                      ▲                               ▲
                     #24                             #72  (transition fires)
```

**Startup:**
```diff
+[ModelManager] Loaded ior/ranks_128 (3 states, 4 run(s))
```

**Early in state 1 — write phase (prediction #7, t=14s):**
```diff
 [PREDICTOR] (#7): Started
 [PREDICTOR] (#7): Dominant freq 0.518 Hz (1.93 sec)
 [PREDICTOR] (#7): Freq candidates (1 found):
 [PREDICTOR] (#7):    0) 0.52 Hz -- conf 0.87
 [PREDICTOR] (#7): Time window 14.000 sec ([0.000,14.000] sec)
 [PREDICTOR] (#7): Total bytes 3 GB
 [PREDICTOR] (#7): Bytes transferred since last time 3 GB
 [PREDICTOR] (#7): Phase automaton: State 0 — freq=0.5181 Hz, period=1.93 s, ranks=128, n_preds=7
+[PREDICTOR] (#7): Reference (greedy): state 1/3, pos=0%, tracking=0.94
+[PREDICTOR] (#7):   Transition in ~31.2s [28.1s–34.3s] → next period ≈ 0.96s
 [PREDICTOR] (#7): Ended
```

**Approaching the first transition (prediction #22, t=44s):**
```diff
 [PREDICTOR] (#22): Dominant freq 0.521 Hz (1.92 sec)
 [PREDICTOR] (#22): Phase automaton: State 0 — freq=0.5181 Hz, period=1.93 s, ranks=128, n_preds=22
+[PREDICTOR] (#22): Reference (greedy): state 1/3, pos=0%, tracking=0.96
+[PREDICTOR] (#22):   Transition in ~1.2s [0.0s–3.3s] → next period ≈ 0.96s
 [PREDICTOR] (#22): Ended
```

**Transition fires (prediction #24, t=48s):**
```diff
 [PREDICTOR] (#24): Dominant freq 1.042 Hz (0.96 sec)
 [PREDICTOR] (#24): Phase automaton: State 1 — freq=1.0417 Hz, period=0.96 s, ranks=128, n_preds=1
 [PREDICTOR] (#24):   → TRANSITION: State 0 → 1  (0.5181 → 1.0417 Hz, cause='frequency')
+[PREDICTOR] (#24): Reference (greedy): state 2/3, pos=50%, tracking=0.98
+[PREDICTOR] (#24):   Transition in ~62.0s [56.8s–67.2s] → next period ≈ 1.93s
 [PREDICTOR] (#24): Ended
```

**Mid read phase (prediction #40, t=80s):**
```diff
 [PREDICTOR] (#40): Dominant freq 1.038 Hz (0.96 sec)
 [PREDICTOR] (#40): Phase automaton: State 1 — freq=1.0417 Hz, period=0.96 s, ranks=128, n_preds=17
+[PREDICTOR] (#40): Reference (greedy): state 2/3, pos=50%, tracking=0.97
+[PREDICTOR] (#40):   Transition in ~30.0s [24.8s–35.2s] → next period ≈ 1.93s
 [PREDICTOR] (#40): Ended
```

**Final state — checkpoint phase (prediction #72, t=145s):**
```diff
 [PREDICTOR] (#72): Dominant freq 0.519 Hz (1.93 sec)
 [PREDICTOR] (#72): Phase automaton: State 2 — freq=0.5181 Hz, period=1.93 s, ranks=128, n_preds=12
 [PREDICTOR] (#72):   → TRANSITION: State 1 → 2  (1.0417 → 0.5181 Hz, cause='frequency')
+[PREDICTOR] (#72): Reference (greedy): state 3/3, pos=100%, tracking=0.95
+[PREDICTOR] (#72):   → APPLICATION IN FINAL REFERENCE STATE
 [PREDICTOR] (#72): Ended
```

**Exit:**
```diff
 [PhaseAutomaton] Saved to ./phase_automaton.json  (3 states, 2 transitions)
+[AutomatonLibrary] Saved ior/ranks_128 → ./ftio_models/ior/ranks_128.json (5 run(s), 3 states)
```

#### 10.2.4 Run 6 — behavior changed, a new state appears

Continuing from Run 5's 3-state, 5-run profile. This run's application now does something new after checkpointing, an extra phase the profile has never seen, so a 4th state opens.

**Startup:** identical to Run 5, the 3-state profile is loaded as usual:
```diff
+[ModelManager] Loaded ior/ranks_128 (3 states, 5 run(s))
```

States 0 through 2 track normally, same as Run 5 above, until the new phase begins:

**A genuinely new phase starts (prediction #78):**
```diff
 [PREDICTOR] (#78): Dominant freq 3.333 Hz (0.30 sec)
 [PREDICTOR] (#78): Phase automaton: State 3 — freq=3.333 Hz, period=0.30 s, ranks=128, n_preds=1
 [PREDICTOR] (#78):   → TRANSITION: State 2 → 3  (0.5181 → 3.333 Hz, cause='frequency')
+[PREDICTOR] (#78): Reference (greedy): state 3/3, pos=100%, tracking=0.16
+[PREDICTOR] (#78):   → APPLICATION IN FINAL REFERENCE STATE
```

The live `PhaseAutomaton` correctly opens its own 4th state, it observes reality directly and does not need the profile's permission to do so. But the loaded profile only ever saw 3 states, so `StateTracker` has nowhere further to go: it stays pinned at state 2 (the last one it knows), keeps reporting `state 3/3`, and keeps printing `APPLICATION IN FINAL REFERENCE STATE`, exactly as it did at the real final state in Run 5. There is no dedicated "unrecognized state" message. The only signal that something is actually wrong is `tracking` dropping from ~0.95 in Run 5 to `0.16` here, `1.0 - abs(observed_period - reference_period) / reference_period`, reflecting how far the new 0.30s period is from the 1.93s the pinned reference state expects.

**Exit:**
```diff
 [PhaseAutomaton] Saved to ./phase_automaton.json  (4 states, 3 transitions)
+[AutomatonLibrary] Topology mismatch for ior/ranks_128; saved new run as ranks_128_v1735229000 (shared configurations pooled into ranks_128)
```

The existing 3-state file is left untouched (still `5 run(s)`), this run's own 4-state path is saved separately, and no `Saved` line is printed for the untouched file since it wasn't rewritten. See [11. AutomatonLibrary](#11-automatonlibrary) for what this does on disk.

---

### 10.3 Application identity

The library is organised as `<library_dir>/<app_name>/ranks_<key>.json`.  The `app_name` subdirectory separates different applications that happen to run at the same rank count.

```bash
--pa-app-name ior           # → ftio_models/ior/ranks_128.json
--pa-app-name hacc-io       # → ftio_models/hacc-io/ranks_9216.json
```

If `--pa-app-name` is omitted, the stem of the monitored filename is used
(e.g. `ior_write` from `ior_write.jsonl`).

#### 10.3.1 First run — cold start

On the first run for an app+config, no reference exists.  FTIO logs:

```
[ModelManager] Cold start — no reference for ior/ranks_128. Building automaton from this run.
```

The automaton is built normally.  On exit it is saved to the library as the first reference (std = 0 for all distributions; timing bounds require at least two runs).

#### 10.3.2 Subsequent runs — warm start

Once a reference exists, FTIO loads it and tracks position in it:

```
[ModelManager] Loaded ior/ranks_128 (3 states, 4 run(s))

[PREDICTOR] (#7): Reference (greedy): state 2/3, pos=50%, tracking=0.94
[PREDICTOR] (#7):   Transition in ~8.0s [5.5s–10.5s] → next period ≈ 0.96s
```

After the run completes, the new timing is merged into the library distributions using pooled statistics; estimates improve with each run.

When only one run is in the library (std = 0), the next-period prediction is still shown but no timing bounds are available:

```
[PREDICTOR] (#7):   Next period ≈ 0.96s (timing improves after ≥2 runs in library)
```

When the tracker reaches the last reference state:

```
[PREDICTOR] (#12):  → APPLICATION IN FINAL REFERENCE STATE
```

---

### 10.4 Malleable applications

Rank changes mid-run are already captured by the automaton as state transitions (each state stores its `ranks`).  The library key encodes the full rank sequence, so a malleable run is stored under its own **path** file, separate from a fixed-rank run:

```bash
# fixed-rank run  → ftio_models/ior/ranks_128.json
# malleable run   → ftio_models/ior/ranks_16_32_128.json
```

The tracker uses rank count as a secondary matching signal alongside period, so a mid-run rank change in a live malleable run is a strong position cue against a malleable reference.

That per-path separation is deliberate: it is what lets `--pa-match` replay a *specific* rank sequence, but it means two runs that reach the same configuration via *different* paths (`8→128` vs. `32→128`) don't share knowledge at the path level. They do at the **configuration level**: see [11.1 Configuration table (nodes)](#111-configuration-table-nodes) below.

---

### 10.5 Matching strategies

`StateTracker` decides which state of the loaded profile the live run is currently in. On every new prediction, it compares the observed period, and secondarily the observed rank count, against each reference state's stored `period_mean` and `ranks`, considering only the current state or a later one, never an earlier one. That forward-only rule matches the physical reality that an application moves through its own phases in a fixed order; it never returns to an earlier one.

Three strategies are available via `--pa-match`, differing in how much observation history they use and how they weigh a state's own timing uncertainty:

| Strategy | Matches on | Cost per update |
|---|---|---|
| `greedy` (default) | one observation, nearest `period_mean` plus a rank penalty | O(n) |
| `dtw` | a trailing window of observations, aligned against forward suffixes of the reference | O(n²) |
| `viterbi` | one observation, HMM forward pass with Gaussian emission around each state's `period_mean`/`period_std` | O(n) |

**`greedy`**: at each step, computes `abs(period - state.period_mean) / state.period_mean`, plus a fixed 0.3 penalty when the observed rank count differs from the state's expected one, for every reference state at or after the current position, and jumps straight to whichever is closest. Reacts immediately to a real period change since it only needs one sample, but a single noisy reading can pull it to the wrong state too.

**`dtw`**: instead of one sample, takes the trailing window of the last N observations (N = number of reference states) and aligns it, using dynamic time warping, against every possible forward starting point in the reference's period sequence, picking whichever alignment has the lowest total cost. Averaging over a window smooths out noise that would fool `greedy`, at the cost of needing enough samples to fill that window first, and of being quadratic instead of linear.

**`viterbi`**: models the state sequence as a hidden Markov chain. Each reference state "emits" periods from a Gaussian centered at its `period_mean`, with the emission's spread taken from the state's own `period_std` where available, or a synthetic 10% of the mean when it isn't, since a single-run profile has `std = 0`, which would otherwise collapse every probability to zero. Transitions are left to right only: from each state the tracker can stay or advance exactly one state, never skip ahead or move back. This is the most statistically grounded option, and the one that most directly benefits from `period_std` once at least two runs have pooled into the profile.

```bash
predictor live.jsonl -f 100 --pa-library ./ftio_models --pa-app-name ior --pa-match viterbi
```

---

### 10.6 AutomatonProfile file format

Each file in the library is one `AutomatonProfile`, serialized as compact JSON: per-state distribution statistics (the **path**, used by the tracker), plus a `nodes` table (the **configuration table**, used by `get_rank_behavior`, see below). Intentionally much smaller than the full `--pa-export` single-run snapshot.

```json
{
  "app_name": "ior",
  "rank_key": "128",
  "n_states": 3,
  "run_count": 4,
  "states": [
    {"period_mean": 1.93, "period_std": 0.08, "dwell_mean": 45.2, "dwell_std": 3.1, "ranks": 128, "n_samples": 4},
    {"period_mean": 0.96, "period_std": 0.04, "dwell_mean": 62.0, "dwell_std": 5.2, "ranks": 128, "n_samples": 4},
    {"period_mean": 1.93, "period_std": 0.07, "dwell_mean": 38.1, "dwell_std": 2.9, "ranks": 128, "n_samples": 4}
  ],
  "transition_causes": ["frequency", "frequency"],
  "nodes": {
    "128": [
      {
        "period_mean": 1.93, "period_std": 0.08,
        "t_start_mean": 0.0, "t_start_std": 0.0, "t_end_mean": 45.2, "t_end_std": 3.1,
        "c_start_mean": 0.0, "c_start_std": 0.0, "c_end_mean": 24.0, "c_end_std": 1.2,
        "ranks": 128, "n_samples": 4
      }
    ]
  }
}
```

`states` and `nodes` describe the same underlying data from two angles: `states` is the ordered path this specific rank sequence took; `nodes` is keyed by configuration (rank count) and pooled across *every* occurrence of that configuration, including ones reached by a different path. See [11.1 Configuration table (nodes)](#111-configuration-table-nodes).

---

## 11. AutomatonLibrary

`AutomatonLibrary` is what gathers the different `AutomatonProfile` files together: a directory-backed store, one file per `(app_name, rank_key)`.

```
./ftio_models/                        <- one AutomatonLibrary
├── ior/
│   ├── ranks_128.json                <- one AutomatonProfile (fixed-rank path)
│   └── ranks_16_32_128.json          <- another (malleable path, same app)
└── hacc-io/
    └── ranks_9216.json               <- different app, its own file
```

Concretely: an `AutomatonProfile` **is** the deserialized content of one file. `AutomatonLibrary` **is** the folder plus the code that lists, loads, and writes those files.

Gathering the profiles together is what makes [11.1 Configuration table (nodes)](#111-configuration-table-nodes) possible: no single profile file knows about its siblings, so a question like "what have we ever seen at `ranks=128`, across every stored path for this app" can only be answered by something that can see every file at once. `AutomatonLibrary.get_rank_behavior()` does exactly that, folding the node table from every stored `rank_key` file for an app into one answer; see [11.1](#111-configuration-table-nodes) for the full mechanics.

Unlike `StateTracker` and `TransitionPredictor`, which run on every single live prediction, `AutomatonLibrary` does almost nothing automatically at runtime. `ModelManager.step()` calls `AutomatonLibrary.load()` exactly once at the start of a run, and again only if the rank count changes mid-run (malleability); `save()` is called exactly once, at the end (`model_manager.py`). Its query methods (`get_rank_behavior()`, `available_apps()`, ...) are not part of that per-step loop at all: they are plain function calls a caller can make at any time, live during a run or entirely offline, but nothing invokes them automatically.

On a run after the first, this load/save sequence has three effects, in order:

1. **Load.** At startup, `ModelManager` asks `AutomatonLibrary` to load the profile for this `(app_name, rank_key)`. If the exact key has no file yet, it falls back to the nearest available rank configuration by initial rank count; if nothing exists at all, this is a cold start (see [10.3.1](#1031-first-run--cold-start)) and the run proceeds with no profile to track against.
2. **Track.** While the run is live, `StateTracker` matches each new prediction against the loaded profile, and `TransitionPredictor` turns that position into an ETA and next-period estimate (see [10.5](#105-matching-strategies)). Neither writes anything back; the loaded profile is read-only for the whole run.
3. **Save.** On exit, `AutomatonLibrary.save()` is called with this run's finished `PhaseAutomaton`. If a profile already exists and its topology matches (same number of states), `AutomatonProfile.merge()` pools the new run's statistics into it and `run_count` increments by one. If the topology does not match, meaning the application's behavior genuinely changed (see [10.2.4](#1024-run-6--behavior-changed-a-new-state-appears)), the existing file is left untouched and the new run is saved under a versioned key instead, so a real change is never silently blended into the old numbers.

`save()` does no statistics itself: it is a `load` → `AutomatonProfile.merge()` (the pooling math lives there, not here) → `write` sequence. What the library adds is exactly what a single profile object has no way to do for itself: know that sibling files exist (`available_apps()`, `available_rank_keys()`), find one on disk, fall back when the exact rank key doesn't exist yet, and write the result back.

### 11.1 Configuration table (nodes)

Everything above (the `states` path, `--pa-match` tracking, ETA forecasts) answers questions about *one specific rank sequence*. The **configuration table** (`nodes` in the library file, `AutomatonProfile.nodes` in Python) answers a narrower but more reusable question: *"what do we know about `ranks=32`, regardless of how the run got there?"*

It is reached through two Python APIs, not (yet) a CLI flag:

```python
from ftio.modeling import AutomatonLibrary, ModelManager

lib = AutomatonLibrary("./ftio_models")
mgr = ModelManager("./ftio_models", "ior")

mgr.get_rank_behavior(32)              # every known behavior for ranks=32
lib.get_rank_behavior("ior", 32)        # same thing, called directly on the library
```

Both work even during cold start, or for a rank count the current run's own path hasn't reached yet, as long as *some* stored path (or a seed, below) has seen that configuration.

#### 11.1.1 Path view vs. node view

Three runs of the same `ior` benchmark, each following the automaton's usual `states` → `transitions` path, but scaling to `ranks=128` from a different starting point:

```
Run A, ranks "8_128":    [ranks=8,  period=3.0s] --rank_change--> [ranks=128, period=10.1s]
Run B, ranks "32_128":   [ranks=32, period=5.0s] --rank_change--> [ranks=128, period=9.9s]
Run C, ranks "128":                                                [ranks=128, period=9.8s]
```

Each run is stored under its own **path** file, all three under the same `ior/` app directory (`ior/ranks_8_128.json`, `ior/ranks_32_128.json`, `ior/ranks_128.json`), because `--pa-match` replays one *specific* rank sequence, so a run that scaled up from 8 must stay separate from one that scaled up from 32.

But at the moment all three reach `ranks=128`, they are describing the *same underlying configuration*, whichever path got them there. The node table pools exactly that:

```
                        ranks = 128
                 ┌──────────────────────────┐
  Run A  ───────▶│  period_mean ≈ 9.93 s     │
  Run B  ───────▶│  period_std  ≈ 0.13 s     │   ◀── mgr.get_rank_behavior(128)
  Run C  ───────▶│  n_samples = 3            │       lib.get_rank_behavior("ior", 128)
                 └──────────────────────────┘
```

The box above highlights the three numbers most relevant to this example (pooling), not the whole object. The call actually returns the complete `NodeBehavior`, every field:

```python
mgr = ModelManager("./ftio_models", "ior")
mgr.get_rank_behavior(128)
# -> [NodeBehavior(period_mean=9.93, period_std=0.13,
#                   t_start_mean=0.0, t_start_std=0.0, t_end_mean=45.2, t_end_std=3.1,
#                   c_start_mean=0.0, c_start_std=0.0, c_end_mean=4.5, c_end_std=0.3,
#                   ranks=128, n_samples=3)]
```

One call answers "what happens at `ranks=128`, across every run we've ever seen", independent of whether that run's own `states` path ever reached 128 itself. That is the difference to hold onto: **`states`/`--pa-match` are about one run's trajectory; the node table is about a configuration, pooled over every trajectory that ever visited it.**

#### 11.1.2 Multiple behaviors per configuration

A configuration does not have to mean one fixed period for its whole dwell. If the statistical detector sees the frequency shift *without* a rank change, that becomes a second, distinct `NodeBehavior` for the same node, not blended into a misleading average of the two:

```python
>>> ref.get_rank_behavior(32)
[NodeBehavior(period_mean=3.0, ...),   # first few bursts
 NodeBehavior(period_mean=8.0, ...)]   # after that
```

Calling `get_rank_behavior()` (on `AutomatonProfile`, `AutomatonLibrary`, or `ModelManager`; same method name on all three) with no further arguments returns every behavior ever seen for that configuration. Give a specific point (see below) and it narrows to whichever behavior(s) actually apply there: one if unambiguous, several if the query falls in a genuine overlap between two behaviors, zero if nothing has ever been observed to cover it.

#### 11.1.3 Scoping the behavior using wall time or phase cycles

Every `NodeBehavior` carries **two** windows describing when it applies, because they answer two different questions and only one of them survives runs of different speed:

| Axis | What it is | Reliable across runs? |
|---|---|---|
| `c_start` / `c_end` | bursts observed since entering this configuration | **yes**, driven by the app's own control flow |
| `t_start` / `t_end` | wall-clock seconds since entering this configuration | no, drifts with contention, load, checkpoint size |

Duration isn't a separate field: it's `t_end_mean - t_start_mean` (or the cycle-axis equivalent, `c_end_mean - c_start_mean`), computed from the window edges above. It describes the whole pooled behavior, not anything relative to a query point: `get_rank_behavior(ranks, at_time=x)` uses `x` only to select which behavior covers it, the resulting window edges (and the duration derived from them) are the same regardless of where exactly `x` fell inside that window.

Concretely, two runs of the same 4-burst → 8s-period behavior, one 20% slower than the other for reasons unrelated to which behavior is active:

```
FAST run (ranks=32):
  period=3.00s  time=[ 0.0,20.0]s  cycle=[0,5]
  period=8.00s  time=[20.0,44.0]s  cycle=[5,8]

SLOW run (ranks=32, 20% slower throughout):
  period=3.60s  time=[ 0.0,24.0]s  cycle=[0,5]
  period=9.60s  time=[24.0,52.8]s  cycle=[5,8]

Query both runs at t=22s (a fixed wall-clock instant):
  fast run -> ['8.00s']   already switched
  slow run -> ['3.60s']   hasn't switched yet
  -> SAME instant, OPPOSITE answers. Wall-clock time is unreliable here.

Query both runs at cycle=3 (a fixed logical point):
  fast run -> ['3.00s']
  slow run -> ['3.60s']
  -> different periods (as expected -- the runs really do run at different
     speeds), but the SAME behavior is identified both times, correctly.
```

```python
ref.get_rank_behavior(32, at_cycle=3)        # the authoritative axis
ref.get_rank_behavior(32, at_time=22.0)      # drifts across runs of different speed
ref.get_rank_behavior(32, at_time=22.0, at_cycle=3)   # both given -> intersection
```

`at_cycle` is the axis to reach for when you want to know *which regime is active*. `at_time` stays useful for a different question, "how many seconds until this transitions", which is what the ETA forecast (`Transition in ~Xs [...] → next period ≈ Ys`, shown earlier) still uses it for.

#### 11.1.4 User-provided input before the run

A user can seed a configuration's expected behavior before any profiling run exists:

```python
AutomatonLibrary("./ftio_models").seed(
    "ior", {32: {"period": 3.0, "dwell": 40.0}, 64: {"period": 20.0, "dwell": 100.0}}
)
```

The next real run that reaches that configuration folds in as a normal observation. **If the guess was close** (within ~6% by default, the same k-sigma tolerance that decides whether any two observations are "the same behavior"), it pools and the seed is diluted away:

```
seed=10.3s, real=10.0s  (3% off, within tolerance)
  -> 1 behavior: 10.15s (n=2)        # seed successfully overwritten
```

**If the guess was far off**, it is *not* silently corrected: it survives as a separate, permanent entry, because the same rule that protects two genuinely distinct real behaviors from being averaged together can't tell "stale wrong guess" apart from "a second real regime that happens to share this rank count":

```
seed=100.0s, real=10.0s  (10x off, outside tolerance)
  -> 2 behaviors: 100.00s (n=1), 10.00s (n=1)   # bad seed just sits there
```

A guess this far off needs to be corrected by hand (edit or remove the library file); don't rely on real data to quietly fix it.

#### 11.1.5 Cross-path sharing

The node table is pooled across every malleability path stored for an app, not just the one matching your exact rank sequence:

```
path A:  8   -> 128  (period=10s)
path B:  32  -> 128  (period=10s)

get_rank_behavior("ior", 128) -> 1 behavior, n_samples=2   # pooled from BOTH paths
```

This is the mechanism referenced in [10.4 Malleable applications](#104-malleable-applications) above: paths are stored separately, but a configuration common to two paths still shares its stats.

---

### 11.2 Walkthrough

A minimal, non-CLI walkthrough of what `save()` actually does across three runs of the same app:

**Run 1 — no file exists yet for `ior/ranks_128`:**

```python
from ftio.modeling import AutomatonLibrary

lib = AutomatonLibrary("./ftio_models")
lib.save(automaton_run_1, "ior", "128")
```

FTIO prints one line:

```
[AutomatonLibrary] Saved ior/ranks_128 → ./ftio_models/ior/ranks_128.json (1 run(s), 3 states)
```

That single line doesn't say *what* landed in the file. Annotated below (this diff block is added here for this walkthrough, it is not something FTIO prints) using git-diff-style markers: `+` for something that did not exist before this save, `~` for something that existed and changed:

```diff
+ state 0   period=1.930s  dwell=12.00s  ranks=128   (new — no prior file)
+ state 1   period=0.960s  dwell=31.20s  ranks=128   (new)
+ state 2   period=1.930s  dwell=47.00s  ranks=128   (new)
```

**Run 2 — same config, timings differ slightly:**

```python
lib.save(automaton_run_2, "ior", "128")
```
```
[AutomatonLibrary] Saved ior/ranks_128 → ./ftio_models/ior/ranks_128.json (2 run(s), 3 states)
```
```diff
~ state 0   period 1.930s → 1.929s   dwell 12.00s → 12.15s   n_samples 1 → 2
~ state 1   period 0.960s → 0.960s   dwell 31.20s → 31.10s   n_samples 1 → 2
~ state 2   period 1.930s → 1.929s   dwell 47.00s → 46.90s   n_samples 1 → 2
```

Every state is `~` this time, not `+`: `save()` loaded the existing file, called `AutomatonProfile.merge()` (the pooled-variance math lives there, not in `AutomatonLibrary`), and wrote the result back. The file's *shape* didn't change (still 3 states), only the pooled distributions moved and `n_samples` / `run_count` incremented.

**Run 3 — a `PhaseAutomaton` with a *different* number of states (say 4, not 3):**

This hits the topology-mismatch path: the existing 3-state path is left untouched, the new run is saved under a versioned key so nothing is lost, and only whatever rank configuration the two runs happen to share still gets pooled into the node table (see [11.1 Configuration table (nodes)](#111-configuration-table-nodes)):

```
[AutomatonLibrary] Topology mismatch for ior/ranks_128; saved new run as ranks_128_v1735142400 (shared configurations pooled into ranks_128)
```
```diff
! ranks_128.json        : untouched (3 states) — different topology, path not merged
+ ranks_128_v1735142400 : new file (4 states) — this run's own path, saved separately
~ config ranks=128      : node-table entry still pooled from both, despite the path split
```

---

### 11.3 Redis-backed library

`AutomatonLibrary` is a directory of JSON files: fine for one machine, but it assumes a shared filesystem if multiple predictor processes (different nodes, different jobs) need to read and write the *same* library, and it has no locking: `save()` is a plain load → merge → write, so two concurrent writers to the same `(app_name, rank_key)` can interleave and one's contribution is silently lost.

`RedisAutomatonLibrary` (`ftio/modeling/redis_automaton_library.py`) is a drop-in alternative with the exact same methods (`load`, `save`, `seed`, `get_rank_behavior`, `available_apps`, `available_rank_keys`) and the exact same merge/dilution/clustering semantics; only the storage backend changes. `save()`/`seed()` additionally hold a Redis lock around their critical section, so concurrent writers can't race:

```python
from ftio.modeling.redis_automaton_library import RedisAutomatonLibrary

lib = RedisAutomatonLibrary(host="redis.cluster.local", port=6379)
lib.seed("ior", {32: {"period": 3.0, "dwell": 40.0}})
lib.save(automaton, "ior", "8_32")   # locked, race-safe
lib.get_rank_behavior("ior", 32)             # same semantics as AutomatonLibrary
```

Requires the optional `redis` package (`pip install redis`, or `pip install .[redis-libs]`); importing `ftio.modeling` never requires it; only instantiating `RedisAutomatonLibrary` does, and it fails with a clear message if it's missing. Tests (`TestRedisAutomatonLibrary` in `test/test_modeling.py`) run against an in-memory `fakeredis` server, no real Redis instance needed; install with `pip install .[development-libs]`.

There is no `--pa-redis-*` CLI flag yet; use `RedisAutomatonLibrary` from Python, or swap it in where `ModelManager` constructs an `AutomatonLibrary` internally if you need it wired into `predictor`.

---

## 12. Combining with other flags

All existing `--pa-*` flags work alongside `--pa-library`:

```bash
predictor live.jsonl -f 100 \
    --pa-library ./ftio_models \
    --pa-app-name ior \
    --pa-method ksigma \
    --pa-period-ratio 1.5 \
    --pa-export ./this_run.json \
    --pa-match viterbi
```

`--pa-export` writes the single-run full snapshot as before; `--pa-library` additionally merges distributions into the library.
