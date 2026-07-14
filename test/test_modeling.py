"""
Tests for ftio.modeling: ReferenceAutomaton, AutomatonLibrary,
StateTracker, TransitionPredictor, ModelManager.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Licensed under the BSD 3-Clause License.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ftio.freq.prediction import Prediction
from ftio.modeling import (
    AutomatonLibrary,
    MatchStrategy,
    ModelManager,
    PhaseAutomaton,
    ReferenceAutomaton,
    StateTracker,
    TransitionForecast,
    TransitionPredictor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(freq: float, t: float, ranks: int = 128, dt: float = 2.0) -> Prediction:
    p = Prediction(transformation="dft")
    p.dominant_freq = np.array([freq])
    p.conf = np.array([0.9])
    p.amp = np.array([1.0])
    p.phi = np.array([0.0])
    p.t_start = t
    p.t_end = t + dt
    p.ranks = ranks
    return p


def _build_automaton(freqs_and_counts: list[tuple[float, int, int]]) -> PhaseAutomaton:
    """Build an automaton from (freq, n_predictions, ranks) triples."""
    aut = PhaseAutomaton(method="ksigma")
    t = 0.0
    for freq, n, ranks in freqs_and_counts:
        for _ in range(n):
            aut.step(_pred(freq, t, ranks=ranks))
            t += 2.0
    return aut


def _simple_ref(n_states: int = 2) -> ReferenceAutomaton:
    """Two-state reference: 0.5 Hz → 1.5 Hz, both at ranks=128."""
    aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
    return ReferenceAutomaton.from_automaton_dict(aut.to_dict(), "app", "128")


# ---------------------------------------------------------------------------
# ReferenceAutomaton
# ---------------------------------------------------------------------------


class TestReferenceAutomaton:
    def test_from_automaton_dict_basic(self):
        aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
        ref = ReferenceAutomaton.from_automaton_dict(aut.to_dict(), "app", "128")
        assert ref.n_states == 2
        assert ref.app_name == "app"
        assert ref.rank_key == "128"
        assert ref.run_count == 1
        assert len(ref.state_stats) == 2

    def test_period_sequence(self):
        ref = _simple_ref()
        seq = ref.period_sequence
        assert len(seq) == 2
        assert abs(seq[0] - 2.0) < 0.01  # 1/0.5
        assert abs(seq[1] - 0.67) < 0.01  # 1/1.5

    def test_rank_sequence(self):
        ref = _simple_ref()
        assert ref.rank_sequence == [128, 128]

    def test_rank_key_from_sequence_fixed(self):
        assert ReferenceAutomaton.rank_key_from_sequence([128, 128, 128]) == "128"

    def test_rank_key_from_sequence_malleable(self):
        assert (
            ReferenceAutomaton.rank_key_from_sequence([16, 16, 32, 32, 128])
            == "16_32_128"
        )

    def test_rank_key_empty(self):
        assert ReferenceAutomaton.rank_key_from_sequence([]) == "0"

    def test_merge_updates_run_count(self):
        ref1 = _simple_ref()
        ref2 = _simple_ref()
        merged = ref1.merge(ref2)
        assert merged.run_count == 2
        assert merged.n_states == 2

    def test_merge_pools_dwell_means(self):
        ref1 = _simple_ref()
        ref2 = _simple_ref()
        merged = ref1.merge(ref2)
        # Same automaton merged → means unchanged, std → 0
        assert (
            abs(merged.state_stats[0].period_mean - ref1.state_stats[0].period_mean)
            < 1e-9
        )

    def test_merge_topology_mismatch_returns_self(self):
        ref1 = _simple_ref()
        aut3 = _build_automaton([(0.5, 8, 128), (1.5, 8, 128), (0.5, 8, 128)])
        ref3 = ReferenceAutomaton.from_automaton_dict(aut3.to_dict(), "app", "128")
        result = ref1.merge(ref3)
        assert result is ref1  # unchanged

    def test_round_trip_serialization(self):
        ref = _simple_ref()
        d = ref.to_dict()
        ref2 = ReferenceAutomaton.from_dict(d)
        assert ref2.n_states == ref.n_states
        assert ref2.run_count == ref.run_count
        assert (
            abs(ref2.state_stats[0].period_mean - ref.state_stats[0].period_mean) < 1e-9
        )


# ---------------------------------------------------------------------------
# AutomatonLibrary
# ---------------------------------------------------------------------------


class TestAutomatonLibrary:
    def test_save_and_load(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
        lib.save(aut, "ior", "128")
        loaded = lib.load("ior", "128")
        assert loaded is not None
        assert loaded.n_states == 2
        assert loaded.app_name == "ior"

    def test_merge_on_second_save(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
        lib.save(aut, "ior", "128")
        lib.save(aut, "ior", "128")
        loaded = lib.load("ior", "128")
        assert loaded.run_count == 2

    def test_load_nonexistent_returns_none(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        assert lib.load("no_such_app", "999") is None

    def test_nearest_fallback(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
        lib.save(aut, "ior", "128")
        # Request ranks=256 — should get nearest (128)
        loaded = lib.load("ior", "256")
        assert loaded is not None
        assert loaded.rank_key == "128"

    def test_available_apps(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        aut = _build_automaton([(0.5, 8, 128)])
        lib.save(aut, "ior", "128")
        lib.save(aut, "hacc", "9216")
        apps = lib.available_apps()
        assert "ior" in apps
        assert "hacc" in apps

    def test_available_rank_keys(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        aut = _build_automaton([(0.5, 8, 128)])
        lib.save(aut, "ior", "128")
        lib.save(aut, "ior", "256")
        keys = lib.available_rank_keys("ior")
        assert "128" in keys
        assert "256" in keys

    def test_malleable_key_stored_separately(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        aut_fixed = _build_automaton([(0.5, 8, 128)])
        aut_malleable = _build_automaton([(0.5, 8, 16), (1.5, 8, 128)])
        lib.save(aut_fixed, "ior", "128")
        lib.save(aut_malleable, "ior", "16_128")
        keys = lib.available_rank_keys("ior")
        assert "128" in keys
        assert "16_128" in keys

    def test_load_raw_automaton_export(self, tmp_path):
        """Loading a raw PhaseAutomaton JSON (not our compact format) should work."""
        aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
        app_dir = tmp_path / "ior"
        app_dir.mkdir()
        path = app_dir / "ranks_128.json"

        with open(path, "w") as fh:
            json.dump(aut.to_dict(), fh)
        lib = AutomatonLibrary(str(tmp_path))
        loaded = lib.load("ior", "128")
        assert loaded is not None
        assert loaded.n_states == 2


# ---------------------------------------------------------------------------
# StateTracker
# ---------------------------------------------------------------------------


class TestStateTracker:
    def test_greedy_starts_at_zero(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        assert t.current_state_index == 0

    def test_greedy_advances_on_freq_change(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        t.update(0.5, 1.0, 128)  # state 0 (period=2s)
        t.update(0.5, 3.0, 128)
        t.update(1.5, 5.0, 128)  # state 1 (period=0.67s)
        assert t.current_state_index == 1

    def test_greedy_never_goes_backward(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        t.update(1.5, 0.0, 128)  # jumps to state 1
        t.update(0.5, 2.0, 128)  # should stay at 1, not go back
        assert t.current_state_index >= 1

    def test_position_property(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        assert t.position == 0.0
        t._current_idx = 1
        assert t.position == 1.0

    def test_is_final_state(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        assert not t.is_final_state
        t._current_idx = 1
        assert t.is_final_state

    def test_elapsed_in_state(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        t.update(0.5, 0.0, 128)
        t.update(0.5, 5.0, 128)
        assert abs(t.elapsed_in_state - 5.0) < 0.01

    def test_dtw_strategy_runs(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.DTW)
        for i in range(4):
            t.update(0.5, float(i * 2), 128)
        assert t.current_state_index == 0

    def test_viterbi_strategy_runs(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.VITERBI)
        for i in range(4):
            t.update(0.5, float(i * 2), 128)
        assert t.current_state_index == 0

    def test_rank_mismatch_penalty(self):
        """A rank mismatch should increase the distance but not crash."""
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY, rank_mismatch_weight=1.0)
        # Period matches state 0 but ranks differ — still should resolve to state 0
        t.update(0.5, 0.0, ranks=64)
        assert t.current_state_index == 0

    def test_malleable_rank_sequence(self):
        """Tracker should handle predictions with changing rank counts."""
        aut = _build_automaton([(0.5, 8, 16), (1.5, 8, 128)])
        ref = ReferenceAutomaton.from_automaton_dict(aut.to_dict(), "app", "16_128")
        t = StateTracker(ref, MatchStrategy.GREEDY)
        t.update(0.5, 0.0, ranks=16)
        t.update(1.5, 16.0, ranks=128)
        assert t.current_state_index == 1


# ---------------------------------------------------------------------------
# TransitionPredictor
# ---------------------------------------------------------------------------


class TestTransitionPredictor:
    def test_predict_returns_forecast(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        p = TransitionPredictor(ref, t)
        fc = p.predict(0.5, 128)
        assert isinstance(fc, TransitionForecast)

    def test_forecast_at_start(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        p = TransitionPredictor(ref, t)
        fc = p.predict(0.5, 128)
        assert fc.current_state_idx == 0
        assert fc.n_states == 2
        assert not fc.at_end

    def test_forecast_at_end(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        t._current_idx = 1  # force to last state
        p = TransitionPredictor(ref, t)
        fc = p.predict(1.5, 128)
        assert fc.at_end
        assert np.isnan(fc.eta_seconds)
        assert np.isnan(fc.next_period)

    def test_tracking_quality_perfect(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        p = TransitionPredictor(ref, t)
        # Observe exactly the reference period
        ref_period = ref.state_stats[0].period_mean
        fc = p.predict(1.0 / ref_period, 128)
        assert fc.tracking_quality == pytest.approx(1.0, abs=1e-6)

    def test_tracking_quality_poor(self):
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        p = TransitionPredictor(ref, t)
        # Observe a very different period
        fc = p.predict(10.0, 128)
        assert fc.tracking_quality < 0.5

    def test_no_timing_on_first_run(self):
        """With only one run, dwell_std = 0 and dwell_mean may produce eta."""
        ref = _simple_ref()  # single run: dwell_std = 0
        t = StateTracker(ref, MatchStrategy.GREEDY)
        p = TransitionPredictor(ref, t)
        fc = p.predict(0.5, 128)
        # dwell_mean is set from the automaton duration; eta is computable
        # dwell_std = 0 so bounds equal eta
        if not np.isnan(fc.eta_seconds):
            assert fc.eta_lower <= fc.eta_seconds <= fc.eta_upper + 1e-9

    def test_eta_decreases_over_time(self):
        """ETA should decrease as time passes in the current state."""
        ref = _simple_ref()
        t = StateTracker(ref, MatchStrategy.GREEDY)
        p = TransitionPredictor(ref, t)
        t.update(0.5, 0.0, 128)
        fc1 = p.predict(0.5, 128)
        t.update(0.5, 10.0, 128)
        fc2 = p.predict(0.5, 128)
        if not np.isnan(fc1.eta_seconds) and not np.isnan(fc2.eta_seconds):
            assert fc2.eta_seconds <= fc1.eta_seconds


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class TestModelManager:
    def test_cold_start_returns_none(self, tmp_path):
        mgr = ModelManager(str(tmp_path), "ior")
        p = _pred(0.5, 0.0, ranks=128)
        result = mgr.step(p)
        assert result is None
        assert mgr.cold_start

    def test_warm_start_returns_forecast(self, tmp_path):
        # First run: build library
        aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
        lib = AutomatonLibrary(str(tmp_path))
        lib.save(aut, "ior", "128")

        mgr = ModelManager(str(tmp_path), "ior")
        p = _pred(0.5, 0.0, ranks=128)
        fc = mgr.step(p)
        assert fc is not None
        assert isinstance(fc, TransitionForecast)
        assert not mgr.cold_start

    def test_save_run_creates_library_entry(self, tmp_path):
        mgr = ModelManager(str(tmp_path), "ior")
        aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
        mgr.save_run(aut)
        lib = AutomatonLibrary(str(tmp_path))
        loaded = lib.load("ior", "128")
        assert loaded is not None

    def test_save_run_none_does_not_crash(self, tmp_path):
        mgr = ModelManager(str(tmp_path), "ior")
        mgr.save_run(None)  # must not raise

    def test_step_empty_prediction_returns_none(self, tmp_path):
        mgr = ModelManager(str(tmp_path), "ior")
        p = Prediction()  # no source set → is_empty() = True
        assert mgr.step(p) is None

    def test_app_name_property(self, tmp_path):
        mgr = ModelManager(str(tmp_path), "my_app", strategy="dtw")
        assert mgr.app_name == "my_app"

    def test_malleable_rank_key_derived_from_run(self, tmp_path):
        """save_run should use the rank sequence from the automaton, not a fixed key."""
        aut = _build_automaton([(0.5, 8, 16), (1.5, 8, 128)])
        mgr = ModelManager(str(tmp_path), "ior")
        mgr.save_run(aut)
        lib = AutomatonLibrary(str(tmp_path))
        keys = lib.available_rank_keys("ior")
        assert "16_128" in keys

    def test_all_strategies_accepted(self, tmp_path):
        for strat in ["greedy", "dtw", "viterbi"]:
            mgr = ModelManager(str(tmp_path), "app", strategy=strat)
            assert mgr._strategy == MatchStrategy(strat)

    def test_invalid_strategy_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ModelManager(str(tmp_path), "app", strategy="bad_strategy")


# ---------------------------------------------------------------------------
# Shim backward compatibility
# ---------------------------------------------------------------------------


def test_shim_import():
    from ftio.modeling.phase_automaton import PhaseAutomaton as PA2
    from ftio.prediction.phase_automaton import PhaseAutomaton as PA

    assert PA is PA2


def test_shim_all_exports():
    from ftio.prediction import phase_automaton as shim

    assert hasattr(shim, "PhaseAutomaton")
    assert hasattr(shim, "PhaseState")
    assert hasattr(shim, "Transition")


# ---------------------------------------------------------------------------
# min_cycles — do not model a period the window has not held twice
# ---------------------------------------------------------------------------
def _windowed_pred(freq: float, t_start: float, t_end: float) -> Prediction:
    """A Prediction whose window is the whole run observed so far (as online)."""
    p = Prediction(transformation="dft")
    p.dominant_freq = np.array([freq])
    p.conf = np.array([1.0])  # the warm-up artifact comes with FULL confidence
    p.amp = np.array([1.0])
    p.phi = np.array([0.0])
    p.t_start = t_start
    p.t_end = t_end
    p.ranks = 2
    return p


def test_min_cycles_ignores_a_window_holding_a_single_period():
    """The measured LAMMPS warm-up: one 5 s burst in a 5 s window.

    The DFT reports "period = 5.0 s" -- that is the burst's *width*, one cycle
    across the whole window (the k=1 bin), not evidence of periodicity. It comes
    with confidence 1.00 and stays stable for several rounds, so only the cycle
    count can reject it.
    """
    aut = PhaseAutomaton(method="cusum", min_cycles=2.0)
    for _ in range(4):
        aut.step(_windowed_pred(1 / 5.0, 0.0, 5.0))  # 1.0 cycle in the window
    assert aut.states == [], "a single-cycle window must not open a state"


def test_min_cycles_admits_the_window_once_it_holds_two_periods():
    aut = PhaseAutomaton(method="cusum", min_cycles=2.0)
    aut.step(_windowed_pred(1 / 5.0, 0.0, 5.0))  # 1.0 cycle -> ignored
    assert not aut.states
    aut.step(_windowed_pred(1 / 42.0, 0.0, 85.0))  # 2.02 cycles -> modelled
    assert len(aut.states) == 1
    assert aut.states[0].period == pytest.approx(42.0, rel=1e-3)


def test_min_cycles_prevents_the_false_transition_entirely():
    """Without the guard the run learns 5 s, then 'transitions' to the real 42 s."""
    warmup = [_windowed_pred(1 / 5.0, 0.0, 5.0) for _ in range(4)]
    steady = [_windowed_pred(1 / 42.0, 0.0, 85.0 + 10 * i) for i in range(8)]

    off = PhaseAutomaton(method="cusum", min_cycles=1.0)
    for p in warmup + steady:
        off.step(p)
    assert len(off.states) == 2
    assert len(off.transitions) == 1, "the artifact fakes a phase change"

    on = PhaseAutomaton(method="cusum", min_cycles=2.0)
    for p in warmup + steady:
        on.step(p)
    assert len(on.states) == 1
    assert not on.transitions, "no phase change ever happened"
    assert on.states[0].period == pytest.approx(42.0, rel=1e-3)


def test_min_cycles_off_by_default_so_short_window_callers_are_unaffected():
    # A caller may hand in Predictions whose window is one slice, not the whole
    # run; the guard must not silently discard those.
    aut = PhaseAutomaton(method="cusum")
    assert aut.min_cycles == 1.0
    aut.step(_windowed_pred(0.5, 0.0, 2.0))  # 1.0 cycle
    assert len(aut.states) == 1


def test_min_cycles_keeps_genuinely_fast_periodic_io():
    # A real 2 s period observed over a 60 s window is 30 cycles: never a warm-up.
    aut = PhaseAutomaton(method="cusum", min_cycles=2.0)
    aut.step(_windowed_pred(0.5, 0.0, 60.0))
    assert len(aut.states) == 1
    assert aut.states[0].period == pytest.approx(2.0, rel=1e-3)
