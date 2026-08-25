"""
ftio.modeling — automaton profile store and transition prediction.

PhaseAutomaton (moved here from ftio.prediction) models a single live run as
a finite state machine — one run's actual trajectory through its I/O phases.
The rest of this package adds an aggregation layer on top: pooling many runs'
trajectories into one statistical profile per (app, rank config), tracking a
live run's position against that profile, and forecasting transitions from it.

Public API
----------
PhaseAutomaton, PhaseState, Transition
    Live I/O phase state machine (original home: ftio.prediction.phase_automaton,
    kept there as a backward-compatibility shim). One instance per run.

AutomatonProfile, StateStats
    One (app, rank config)'s aggregated statistics, pooled from every
    PhaseAutomaton run seen for it so far; stores per-state distributions
    (mean ± std) rather than any single run's raw observations. This is the
    in-memory form of one file in an AutomatonLibrary.

AutomatonLibrary
    Directory-backed collection of AutomatonProfile files, one per
    (app_name, rank_key): <library>/<app_name>/ranks_<key>.json. Owns no
    statistics itself — load()/save() dispatch to AutomatonProfile.merge()
    for the actual pooling, and add directory listing (available_apps(),
    available_rank_keys()) that a single profile has no way to do for itself.

StateTracker, MatchStrategy
    Tracks a live run's position within an AutomatonProfile.
    Strategies: greedy, dtw, viterbi.

TransitionPredictor, TransitionForecast
    Predicts time-to-next-transition and next-state period, using the
    AutomatonProfile's dwell-time distributions for uncertainty bounds.

ModelManager
    Top-level entry point — wires AutomatonLibrary + StateTracker +
    TransitionPredictor together for the online pipeline.
"""

from ftio.modeling.automaton_library import AutomatonLibrary
from ftio.modeling.automaton_profile import AutomatonProfile, StateStats
from ftio.modeling.model_manager import ModelManager
from ftio.modeling.phase_automaton import PhaseAutomaton, PhaseState, Transition
from ftio.modeling.state_tracker import MatchStrategy, StateTracker
from ftio.modeling.transition_predictor import TransitionForecast, TransitionPredictor

__all__ = [
    "PhaseAutomaton",
    "PhaseState",
    "Transition",
    "AutomatonProfile",
    "StateStats",
    "AutomatonLibrary",
    "StateTracker",
    "MatchStrategy",
    "TransitionPredictor",
    "TransitionForecast",
    "ModelManager",
]
