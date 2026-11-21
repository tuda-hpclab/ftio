"""
AssignmentMatcher: group phases of several runs into phase families by solving
an assignment problem (Hungarian algorithm) between a run's phases and the
known families.

Unlike AlignmentMatcher, the order of phases is not used: any phase may join
any family, but each family takes at most one phase per run. A phase that fits
no family better than the threshold starts a new one, and a family that gets
no phase is simply not seen in this run.

Author: Albani Alvarado Mauriello
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: November 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ftio.modeling.phase_matcher import (
    FEATURES,  # default descriptors, shared with AlignmentMatcher
    NEW_FAMILY,
    NEW_RANKS,
    SAME_RANKS,
    Match,
    PhaseFamily,
    PhaseMatcher,
    phase_distance,
)
from ftio.modeling.phase_vector import PhaseVector


class AssignmentMatcher(PhaseMatcher):
    """Matches a run by a one-to-one assignment of its phases to known families."""

    def __init__(self, threshold: float = 0.5, features: dict = FEATURES):
        super().__init__(threshold)
        self.features = features

    def _reference(self, fam: PhaseFamily, ranks: int) -> dict:
        """Descriptors of the family's member at the closest rank count."""
        nearest = min(fam.members, key=lambda m: abs(np.log(m.ranks / ranks)))
        return nearest.as_dict()

    def cost_matrix(self, run: list[PhaseVector], fids: list[int]) -> np.ndarray:
        """phases x (families + one "new family" slot per phase).

        The slots cost the threshold, so a phase joins a family only if that is
        cheaper overall than starting a new one. They also keep the assignment
        feasible when a distance is inf (no common descriptor).
        Each phase is compared at its own rank count (malleability).
        """
        cost = np.full((len(run), len(fids) + len(run)), self.threshold)
        for i, v in enumerate(run):
            p = v.as_dict()
            for j, f in enumerate(fids):
                ref = self._reference(self.families[f], v.ranks)
                cost[i, j] = phase_distance(p, ref, self.features)
        return cost

    def match_run(self, run: list[PhaseVector]) -> list[Match]:
        """Assign the run's phases to the known families, then add them."""
        fids = list(self.families)  # fixed before this run adds new families
        cost = self.cost_matrix(run, fids)
        rows, cols = linear_sum_assignment(cost)  # rows come back sorted

        matches = []
        for i, j in zip(rows, cols):
            v = run[i]
            if j < len(fids):
                fid = fids[j]
                case = SAME_RANKS if v.ranks in self.families[fid].ranks else NEW_RANKS
            else:  # "new family" slot
                fid = len(self.families)
                self.families[fid] = PhaseFamily(fid)
                case = NEW_FAMILY
            self.families[fid].members.append(v)
            matches.append(Match(v, fid, case, float(cost[i, j])))
        return matches
