"""
ModelManager: top-level entry point for the reference-automaton modeling layer.

Wires AutomatonLibrary, StateTracker, and TransitionPredictor together and
exposes a single step() method for the online prediction pipeline.

Cold start behaviour: if no reference exists for the current app + rank
configuration, the manager returns None (signalling learning mode) and builds
nothing itself — the PhaseAutomaton running in parallel handles that.  On
shutdown, save_run() merges the completed automaton into the library.

Only the compact mutable tracker state is written to the multiprocessing
Manager dict on every step; the read-only reference is loaded once and kept
on this object.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Licensed under the BSD 3-Clause License.
"""

from __future__ import annotations

import numpy as np

from ftio.freq.prediction import Prediction
from ftio.modeling.automaton_library import AutomatonLibrary
from ftio.modeling.reference_automaton import NodeBehavior, ReferenceAutomaton
from ftio.modeling.state_tracker import MatchStrategy, StateTracker
from ftio.modeling.transition_predictor import TransitionForecast, TransitionPredictor


class ModelManager:
    """
    Manages reference-automaton matching for a live prediction session.

    Parameters
    ----------
    library_dir : str
        Root directory of the automaton library (e.g. "./ftio_models").
        Created automatically if it does not exist.
    app_name : str
        Application identifier used as the library subdirectory.
        Different applications at the same rank count are kept separate.
    strategy : str
        Matching strategy: "greedy" (default), "dtw", or "viterbi".
    """

    def __init__(
        self,
        library_dir: str,
        app_name: str,
        strategy: str = "greedy",
    ):
        self._library = AutomatonLibrary(library_dir)
        self._app_name = app_name
        self._strategy = MatchStrategy(strategy)

        self._reference: ReferenceAutomaton | None = None
        self._tracker: StateTracker | None = None
        self._predictor: TransitionPredictor | None = None
        self._cold_start = True
        self._last_rank_key: str = ""
        self._observed_ranks: list[int] = []  # rank sequence seen so far this run

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cold_start(self) -> bool:
        return self._cold_start

    @property
    def app_name(self) -> str:
        return self._app_name

    @property
    def library_dir(self) -> str:
        return self._library.directory

    # ------------------------------------------------------------------
    # Live step
    # ------------------------------------------------------------------

    def step(self, prediction: Prediction) -> TransitionForecast | None:
        """Feed one live Prediction.

        Returns a TransitionForecast when a reference is loaded, or None on
        cold start (no matching reference) or an invalid prediction.
        """
        if prediction.is_empty():
            return None
        freq, _conf = prediction.get_dominant_freq_and_conf()
        if freq <= 0 or np.isnan(freq):
            return None

        ranks = max(0, int(prediction.ranks))

        # Accumulate the observed rank sequence (for library key on save)
        if not self._observed_ranks or self._observed_ranks[-1] != ranks:
            self._observed_ranks.append(ranks)

        rank_key = ReferenceAutomaton.rank_key_from_sequence(self._observed_ranks)

        # Load (or reload) the reference when the rank configuration changes.
        # This handles both the first call and mid-run rank changes (malleability).
        if self._reference is None or rank_key != self._last_rank_key:
            self._last_rank_key = rank_key
            ref = self._library.load(self._app_name, rank_key)
            if ref is not None:
                self._reference = ref
                self._cold_start = False
                self._tracker = StateTracker(ref, self._strategy)
                self._predictor = TransitionPredictor(ref, self._tracker)
                print(
                    f"[ModelManager] Loaded {self._app_name}/ranks_{ref.rank_key} "
                    f"({ref.n_states} states, {ref.run_count} run(s))"
                )
            else:
                self._cold_start = True
                self._tracker = None
                self._predictor = None
                # Only print the cold-start message once per rank_key
                if rank_key != self._last_rank_key or self._reference is None:
                    print(
                        f"[ModelManager] Cold start — no reference for "
                        f"{self._app_name}/ranks_{rank_key}. "
                        f"Building automaton from this run."
                    )

        if self._cold_start or self._tracker is None or self._predictor is None:
            return None

        self._tracker.update(freq, prediction.t_end, ranks)
        return self._predictor.predict(freq, ranks)

    # ------------------------------------------------------------------
    # Configuration-level lookup (independent of path/position)
    # ------------------------------------------------------------------

    def guess_node(
        self,
        ranks: int,
        at_time: float | None = None,
        at_cycle: float | None = None,
    ) -> list[NodeBehavior]:
        """Early-estimate lookup for one configuration, independent of path.

        Works even during cold start (no matching path loaded yet) or for a
        rank count this run's own path hasn't reached -- folds every stored
        path for this app that has seen ``ranks``, so a seed
        (AutomatonLibrary.seed) or a run that reached this configuration via
        a different malleability sequence still contributes a guess.

        at_time / at_cycle: either, both, or neither may be given -- see
        ReferenceAutomaton.node() for exact matching semantics. at_cycle
        (bursts since entering this configuration) is the axis that stays
        valid across runs of different speed; at_time (wall-clock seconds)
        can drift for reasons unrelated to which behavior is active.
        """
        return self._library.load_node(self._app_name, ranks, at_time, at_cycle)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def save_run(self, automaton) -> None:
        """Persist the current run's automaton to the library.

        Derives the rank key from the automaton's state rank sequence so
        malleable runs are stored under the correct key automatically.
        """
        if automaton is None or not automaton.states:
            return
        rank_seq = [s.ranks for s in automaton.states]
        rank_key = ReferenceAutomaton.rank_key_from_sequence(rank_seq)
        self._library.save(automaton, self._app_name, rank_key)
