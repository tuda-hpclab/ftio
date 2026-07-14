"""Tests for the flush decision and the new-prediction handling in the JIT trigger.

Two defects starved the stage-out in the WarpX run (DF_17): only 3 of 10
checkpoints were flushed while the app ran, the other 7 fell to the post-app
sweep.

1. `predictor_gekko_zmq` reports probability = -1 when the dominant frequency
   fits no probability bin. The trigger compared that sentinel with `> 0.5`, so
   "no estimate" was read as "not periodic" and no flush was issued at all.
2. In the countdown loop the cancel branch sat in the `else` of
   `if not sync_trigger.empty()`, so it ran on an *empty* queue and blocked in
   `Queue.get()` until the next prediction arrived.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import inspect

from ftio.api.gekkoFs.stage_data import should_flush, strategy_avoid_interference


def _prediction(probability: float) -> dict:
    return {
        "t_wait": 0.0,
        "t_end": 20.0,
        "t_start": 0.0,
        "t_flush": 0.0,
        "freq": 0.0847,
        "conf": 0.76,
        "probability": probability,
        "total_bytes": 1_400_000_000,
        "source": "#3",
    }


# ── should_flush: the -1 sentinel is "unknown", not "not periodic" ────────────


def test_no_probability_bin_still_flushes():
    # WarpX predictions #3 and #7 arrived as -1 and staged nothing.
    assert should_flush(_prediction(-1)) is True


def test_high_probability_flushes():
    assert should_flush(_prediction(1.0)) is True


def test_low_probability_does_not_flush():
    # A real estimate below the threshold must still suppress the flush.
    assert should_flush(_prediction(0.25)) is False


def test_threshold_is_still_one_half():
    assert should_flush(_prediction(0.51)) is True
    assert should_flush(_prediction(0.5)) is False


def test_sentinel_is_not_treated_as_a_low_probability():
    # The bug in one line: -1 < 0.5, so the sentinel used to lose the comparison.
    assert should_flush(_prediction(-1)) != should_flush(_prediction(0.0))


# ── the cancel branch must never run on an empty queue ────────────────────────


def test_cancel_only_consumes_the_queue_when_it_is_non_empty():
    """A blocking get() on an empty queue stalls the countdown.

    Guard the structure directly: every `sync_trigger.get()` inside the wait loop
    has to be reached through a non-empty check, never through its `else`.
    """
    src = inspect.getsource(strategy_avoid_interference)
    # The countdown loop starts at the `while time.time() < countdown:` line.
    loop = src.split("while time.time() < countdown:", 1)[1]
    loop = loop.split("if condition and should_flush", 1)[0]

    assert "sync_trigger.get()" in loop, "cancel no longer drains the queue"
    for line in loop.splitlines():
        if "sync_trigger.get()" in line:
            indent = len(line) - len(line.lstrip())
            guard = [
                ln
                for ln in loop.splitlines()
                if "not sync_trigger.empty()" in ln
                and (len(ln) - len(ln.lstrip())) < indent
            ]
            assert guard, "sync_trigger.get() is not guarded by a non-empty check"


def test_cancel_is_reached_on_its_own_branch_not_an_else():
    src = inspect.getsource(strategy_avoid_interference)
    assert 'elif "cancel" in handle_new_prediction:' in src
