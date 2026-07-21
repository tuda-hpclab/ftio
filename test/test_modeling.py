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


class TestReferenceAutomatonNodes:
    """The configuration table: node lookup, seeding, cross-path merging."""

    def test_single_mode_config_returns_one_behavior(self):
        """A configuration seen with only one stable period gets a
        single-element behavior list -- no special-casing needed for the
        common, unambiguous case."""
        ref = ReferenceAutomaton.from_automaton_dict(
            _build_automaton([(0.5, 8, 128)]).to_dict(), "app", "128"
        )
        behaviors = ref.get_rank_behavior(128)
        assert len(behaviors) == 1
        assert behaviors[0].period_mean == pytest.approx(2.0)

    def test_nodes_split_into_distinct_behaviors_when_periods_differ(self):
        """ranks=128 shows two genuinely different periods back to back ->
        kept as two distinct behaviors, NOT blended into one misleading mean."""
        ref = _simple_ref()  # ranks=128: period 2.0s, then period 0.667s
        behaviors = ref.get_rank_behavior(128)
        assert len(behaviors) == 2
        periods = sorted(b.period_mean for b in behaviors)
        assert periods[0] == pytest.approx(0.6666666, rel=1e-3)
        assert periods[1] == pytest.approx(2.0)

    def test_node_at_time_picks_the_right_behavior(self):
        ref = _simple_ref()
        early = ref.get_rank_behavior(
            128, at_time=1.0
        )  # within the first behavior's window
        late = ref.get_rank_behavior(128, at_time=25.0)  # within the second's
        assert len(early) == 1 and early[0].period_mean == pytest.approx(2.0)
        assert len(late) == 1 and late[0].period_mean == pytest.approx(
            0.6666666, rel=1e-3
        )

    def test_node_at_time_outside_any_window_returns_empty(self):
        ref = _simple_ref()
        assert ref.get_rank_behavior(128, at_time=9999.0) == []

    def test_node_lookup_missing_returns_empty_list(self):
        ref = _simple_ref()
        assert ref.get_rank_behavior(9999) == []

    def test_from_automaton_dict_populates_cycle_windows(self):
        """from_automaton_dict has the raw per-state burst count available,
        so unlike the generic StateStats-only fallback, its behaviors carry
        a real (non-NaN) cycle window."""
        ref = _simple_ref()
        b = ref.get_rank_behavior(128)[0]
        assert not np.isnan(b.c_start_mean)
        assert b.c_start_mean == pytest.approx(0.0)
        assert b.c_end_mean > 0

    def test_node_at_cycle_picks_the_right_behavior(self):
        """The cycle axis, not wall-clock time, distinguishes behaviors --
        this is the axis that stays valid when a run speeds up or slows
        down for reasons unrelated to which behavior is active."""
        ref = _simple_ref()
        first_end = ref.get_rank_behavior(128)[0].c_end_mean
        early = ref.get_rank_behavior(128, at_cycle=first_end - 1)
        late = ref.get_rank_behavior(128, at_cycle=first_end + 1)
        assert len(early) == 1 and early[0].period_mean == pytest.approx(2.0)
        assert len(late) == 1 and late[0].period_mean == pytest.approx(
            0.6666666, rel=1e-3
        )

    def test_node_at_time_and_cycle_both_given_is_an_intersection(self):
        ref = _simple_ref()
        b0, b1 = ref.get_rank_behavior(128)
        # A time that matches b0 but a cycle that matches b1 -> no behavior
        # satisfies both at once.
        result = ref.get_rank_behavior(
            128, at_time=b0.t_start_mean, at_cycle=b1.c_start_mean + 1
        )
        assert result == []
        # Consistent time+cycle from the same behavior -> matches.
        result = ref.get_rank_behavior(
            128, at_time=b0.t_start_mean, at_cycle=b0.c_start_mean
        )
        assert result == [b0]

    def test_seed_has_no_cycle_window(self):
        """A user can't guess a burst count they've never observed -- a
        seed's cycle window is unknown (NaN), not a fabricated zero."""
        ref = ReferenceAutomaton.from_node_seed(
            "demo", {32: {"period": 3.0, "dwell": 20.0}}
        )
        b = ref.get_rank_behavior(32)[0]
        assert np.isnan(b.c_start_mean)

    def test_nodes_pool_matching_repeated_occurrence(self):
        """ranks=16 appears twice (shrink then regrow) with the SAME period
        both times -> folds into one behavior, not two."""
        aut = _build_automaton([(0.5, 8, 16), (1.5, 8, 128), (0.5, 8, 16)])
        ref = ReferenceAutomaton.from_automaton_dict(aut.to_dict(), "app", "16_128_16")
        behaviors = ref.get_rank_behavior(16)
        assert len(behaviors) == 1
        assert behaviors[0].n_samples == 2

    def test_edges_from_transitions(self):
        """A frequency-drift transition without a rank change is a self-loop
        at the node level (128 -> 128) -- edges only carry real information
        when the configuration actually changes, unlike a rank-change trigger."""
        ref = _simple_ref()
        edges = ref.edges
        assert edges == {(128, 128): {"cause": "frequency", "count": 1}}

    def test_edges_across_rank_change(self):
        aut = _build_automaton([(0.5, 8, 16), (1.5, 8, 128)])
        ref = ReferenceAutomaton.from_automaton_dict(aut.to_dict(), "app", "16_128")
        assert ref.edges == {(16, 128): {"cause": "rank_change", "count": 1}}

    def test_from_node_seed_builds_usable_reference(self):
        ref = ReferenceAutomaton.from_node_seed(
            "demo",
            {64: {"period": 20.0, "dwell": 100.0}, 8: {"period": 5.0, "dwell": 40.0}},
        )
        assert ref.rank_key == "8_64"  # sorted ascending regardless of dict order
        assert ref.run_count == 0  # a seed is not a profiled run
        assert ref.get_rank_behavior(8)[0].period_mean == pytest.approx(5.0)
        assert ref.get_rank_behavior(64)[0].t_end_mean == pytest.approx(
            100.0
        )  # "dwell" -> window length

    def test_merge_topology_mismatch_pools_shared_node(self):
        """Two runs with different paths that both touch ranks=128 at the
        SAME period should share that node's stats (one pooled behavior)
        even though the paths can't merge."""
        ref_a = ReferenceAutomaton.from_automaton_dict(
            _build_automaton([(0.5, 8, 128)]).to_dict(), "app", "128"
        )
        ref_b = ReferenceAutomaton.from_automaton_dict(
            _build_automaton([(0.2, 8, 16), (0.5, 8, 128)]).to_dict(), "app", "16_128"
        )
        result = ref_a.merge(ref_b)
        assert result is ref_a  # topology mismatch -> identity preserved
        assert 16 in result.nodes  # picked up from ref_b even though topology differs
        assert (
            len(result.get_rank_behavior(128)) == 1
        )  # same period both times -> one behavior
        assert result.get_rank_behavior(128)[0].n_samples == 2  # pooled from both runs

    def test_merge_keeps_genuinely_different_behaviors_separate(self):
        """If the two paths reach ranks=128 with DIFFERENT periods, the merge
        must not average them into a meaningless midpoint."""
        ref_a = ReferenceAutomaton.from_automaton_dict(
            _build_automaton([(0.5, 8, 128)]).to_dict(), "app", "128"  # period 2.0s
        )
        ref_b = ReferenceAutomaton.from_automaton_dict(
            _build_automaton([(0.2, 8, 16), (0.1, 8, 128)]).to_dict(),
            "app",
            "16_128",  # period 10s
        )
        result = ref_a.merge(ref_b)
        behaviors = result.get_rank_behavior(128)
        assert len(behaviors) == 2
        periods = sorted(b.period_mean for b in behaviors)
        assert periods[0] == pytest.approx(2.0)
        assert periods[1] == pytest.approx(10.0)

    def test_nodes_survive_dict_round_trip(self):
        ref = _simple_ref()
        ref2 = ReferenceAutomaton.from_dict(ref.to_dict())
        assert set(ref2.nodes) == set(ref.nodes)
        assert len(ref2.get_rank_behavior(128)) == len(ref.get_rank_behavior(128))
        p1 = sorted(b.period_mean for b in ref.get_rank_behavior(128))
        p2 = sorted(b.period_mean for b in ref2.get_rank_behavior(128))
        assert p1 == pytest.approx(p2)


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


class TestAutomatonLibraryNodes:
    """Cross-path configuration lookup and user-supplied early estimates."""

    def test_get_rank_behavior_across_different_paths(self, tmp_path):
        """ranks=128 is reachable via two different malleability paths;
        get_rank_behavior should pool both without needing an exact path match."""
        lib = AutomatonLibrary(str(tmp_path))
        lib.save(_build_automaton([(0.2, 8, 16), (0.5, 8, 128)]), "app", "16_128")
        lib.save(_build_automaton([(0.3, 8, 32), (0.5, 8, 128)]), "app", "32_128")

        behaviors = lib.get_rank_behavior("app", 128)
        assert len(behaviors) == 1  # same period both times -> one behavior
        assert behaviors[0].n_samples == 2  # pooled from both paths

    def test_get_rank_behavior_missing_configuration(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        lib.save(_build_automaton([(0.5, 8, 128)]), "app", "128")
        assert lib.get_rank_behavior("app", 9999) == []

    def test_seed_then_load(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        lib.seed(
            "app",
            {8: {"period": 5.0, "dwell": 40.0}, 64: {"period": 20.0, "dwell": 100.0}},
        )
        loaded = lib.load("app", "8_64")
        assert loaded is not None
        assert loaded.run_count == 0
        assert loaded.get_rank_behavior(64)[0].period_mean == pytest.approx(20.0)

    def test_seed_does_not_clobber_existing(self, tmp_path):
        lib = AutomatonLibrary(str(tmp_path))
        lib.save(_build_automaton([(0.5, 8, 128)]), "app", "128")
        lib.seed(
            "app", {128: {"period": 999.0, "dwell": 1.0}}
        )  # wrong guess, real data exists
        loaded = lib.load("app", "128")
        assert loaded.run_count == 1  # untouched by the seed

    def test_seed_close_to_real_gets_diluted(self, tmp_path):
        """A seed within k-sigma of the real value pools -- the seed is
        outweighed and the estimate converges on the real one. The default
        tolerance is tight (k=3, 2% relative floor -> ~6% by default), so
        "close" here means close, not just same order of magnitude."""
        lib = AutomatonLibrary(str(tmp_path))
        lib.seed(
            "app", {128: {"period": 10.3, "dwell": 20.0}}
        )  # 3% off -> within tolerance
        real = _build_automaton([(0.1, 8, 128)])  # true period = 10s
        lib.save(real, "app", "128")

        after = lib.get_rank_behavior("app", 128)
        assert len(after) == 1  # close enough -> pooled into one behavior
        assert after[0].period_mean == pytest.approx(10.15, rel=0.01)
        assert after[0].n_samples == 2

    def test_seed_wildly_wrong_is_not_silently_overwritten(self, tmp_path):
        """A seed far from reality does NOT get quietly corrected -- it is
        preserved as a separate, permanent (wrong) behavior instead of being
        blended into the real one. This is the flip side of not corrupting
        two genuinely distinct real behaviors with a blind average: the
        clustering rule can't tell "unrelated seed" apart from "a second,
        real regime that happens to share this rank count." A guess this far
        off needs to be corrected by hand (or the library file edited/removed),
        not by hoping real data will dilute it away."""
        lib = AutomatonLibrary(str(tmp_path))
        lib.seed("app", {128: {"period": 100.0, "dwell": 100.0}})  # bad guess
        real = _build_automaton([(0.1, 8, 128)])  # true period = 10s -- 10x off
        lib.save(real, "app", "128")

        after = lib.get_rank_behavior("app", 128)
        assert len(after) == 2  # NOT pooled -- both survive
        periods = sorted(b.period_mean for b in after)
        assert periods[0] == pytest.approx(10.0)
        assert periods[1] == pytest.approx(100.0)  # the bad seed, still there


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

    def test_get_rank_behavior_from_seed_before_any_step(self, tmp_path):
        """get_rank_behavior should answer from a seed alone, without step() ever
        having been called -- it's a pure library lookup, independent of the
        live tracker's state."""
        lib = AutomatonLibrary(str(tmp_path))
        lib.seed("ior", {64: {"period": 20.0, "dwell": 100.0}})

        mgr = ModelManager(str(tmp_path), "ior")
        guess = mgr.get_rank_behavior(64)
        assert len(guess) == 1
        assert guess[0].period_mean == pytest.approx(20.0)

    def test_get_rank_behavior_true_cold_start_returns_empty(self, tmp_path):
        """No library entry at all for this app -- nothing to guess from."""
        mgr = ModelManager(str(tmp_path), "never_profiled_app")
        assert mgr.get_rank_behavior(64) == []


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


# ---------------------------------------------------------------------------
# RedisAutomatonLibrary — same semantics as AutomatonLibrary, Redis-backed.
# Skipped unless the optional `redis` + `fakeredis` packages are installed
# (pip install .[redis-libs] and .[development-libs]).
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_client():
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("redis")
    return fakeredis.FakeRedis(decode_responses=True)


class TestRedisAutomatonLibrary:
    def test_save_and_load(self, redis_client):
        from ftio.modeling.redis_automaton_library import RedisAutomatonLibrary

        lib = RedisAutomatonLibrary(redis_client=redis_client)
        aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
        lib.save(aut, "ior", "128")
        loaded = lib.load("ior", "128")
        assert loaded is not None
        assert loaded.n_states == 2
        assert loaded.app_name == "ior"

    def test_merge_on_second_save(self, redis_client):
        from ftio.modeling.redis_automaton_library import RedisAutomatonLibrary

        lib = RedisAutomatonLibrary(redis_client=redis_client)
        aut = _build_automaton([(0.5, 8, 128), (1.5, 8, 128)])
        lib.save(aut, "ior", "128")
        lib.save(aut, "ior", "128")
        assert lib.load("ior", "128").run_count == 2

    def test_nearest_fallback(self, redis_client):
        from ftio.modeling.redis_automaton_library import RedisAutomatonLibrary

        lib = RedisAutomatonLibrary(redis_client=redis_client)
        lib.save(_build_automaton([(0.5, 8, 128)]), "ior", "128")
        loaded = lib.load("ior", "256")
        assert loaded is not None
        assert loaded.rank_key == "128"

    def test_seed_then_guess_before_any_run(self, redis_client):
        from ftio.modeling.redis_automaton_library import RedisAutomatonLibrary

        lib = RedisAutomatonLibrary(redis_client=redis_client)
        lib.seed("app", {32: {"period": 3.0, "dwell": 20.0}})
        behaviors = lib.get_rank_behavior("app", 32)
        assert len(behaviors) == 1
        assert behaviors[0].period_mean == pytest.approx(3.0)

    def test_get_rank_behavior_pools_across_paths(self, redis_client):
        from ftio.modeling.redis_automaton_library import RedisAutomatonLibrary

        lib = RedisAutomatonLibrary(redis_client=redis_client)
        lib.save(_build_automaton([(0.2, 8, 16), (0.5, 8, 128)]), "app", "16_128")
        lib.save(_build_automaton([(0.3, 8, 32), (0.5, 8, 128)]), "app", "32_128")
        behaviors = lib.get_rank_behavior("app", 128)
        assert len(behaviors) == 1
        assert behaviors[0].n_samples == 2

    def test_concurrent_saves_do_not_lose_a_contribution(self, redis_client):
        """The file-based AutomatonLibrary has no locking around its
        load-merge-write critical section; RedisAutomatonLibrary does. Fire
        several saves for the same key and confirm every run is accounted
        for in the final run_count -- none silently dropped by a race."""
        from ftio.modeling.redis_automaton_library import RedisAutomatonLibrary

        lib = RedisAutomatonLibrary(redis_client=redis_client)
        n_runs = 5
        for _ in range(n_runs):
            lib.save(_build_automaton([(0.5, 8, 128), (1.5, 8, 128)]), "ior", "128")
        assert lib.load("ior", "128").run_count == n_runs

    def test_missing_redis_package_raises_clear_error(self, monkeypatch):
        """Without redis-py installed, instantiation fails fast with a
        clear message rather than an opaque ImportError deep in a call."""
        import ftio.modeling.redis_automaton_library as mod

        monkeypatch.setattr(mod.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(ImportError, match="redis"):
            mod.RedisAutomatonLibrary()
