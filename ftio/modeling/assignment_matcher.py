"""
AssignmentMatcher: group phases of several runs into phase families by solving
an assignment problem (Hungarian algorithm) between a run's phases and the
known families.

Unlike AlignmentMatcher, the order of phases is not used: any phase may join
any family, but each family takes at most one phase per run. A phase that fits
no family better than the threshold starts a new one, and a family that gets
no phase is simply not seen in this run. With recurrences=True, a phase that
repeats an earlier phase of its run (PhaseVector.same_as, A ... A) is not
assigned itself but joins that phase's family, so the limit of one phase per
family and run applies to first visits only.

A family is compared through a reference built from all its members, per
descriptor: shares that do not change with the rank count (e.g. read_fraction,
or relative_volume, a share of the run) take the median, so one bad member
does not move it; rank-dependent ones (e.g. bytes_per_rank, total_bytes)
follow a robust power-law trend (Theil-Sen in log-log), evaluated at the new
phase's rank count.

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
from scipy.stats import theilslopes

from ftio.modeling.phase_matcher import (
    FEATURES,  # default descriptors, shared with AlignmentMatcher
    MIN_TREND_RANKS,
    NEW_FAMILY,
    NEW_RANKS,
    RANK_DEPENDENT,
    SAME_RANKS,
    Match,
    PhaseFamily,
    PhaseMatcher,
    phase_distance,
)
from ftio.modeling.phase_vector import PhaseVector


def median_of(points: list[tuple[int, float]]) -> float:
    """Median of a descriptor over (ranks, value) points (NaN skipped)."""
    values = [y for _, y in points if np.isfinite(y)]
    return float(np.median(values)) if values else np.nan


def trend_at(points: list[tuple[int, float]], at: int) -> float:
    """Fit value = c * ranks^b over (ranks, value) points and evaluate it at `at`.

    The fit is Theil-Sen in log-log: the slope is the median of all pairwise
    slopes, so one bad point barely moves it (unlike least squares, where the
    point at the largest rank count dominates an extrapolation). If the slope's
    confidence interval includes 0, the values are flat within their noise (e.g.
    phase boundaries that move between runs), and their median is returned
    instead of a trend. Only positive values are used; NaN if they cover fewer
    than MIN_TREND_RANKS rank counts.
    """
    pts = [(r, y) for r, y in points if np.isfinite(y) and y > 0]
    if len({r for r, _ in pts}) < MIN_TREND_RANKS:
        return np.nan
    log_r = np.log([r for r, _ in pts])
    log_y = np.log([y for _, y in pts])
    slope, intercept, low, high = theilslopes(log_y, log_r)
    if low <= 0 <= high:  # no clear trend
        return float(np.exp(np.median(log_y)))
    return float(np.exp(intercept + slope * np.log(at)))


class AssignmentMatcher(PhaseMatcher):
    """Matches a run by a one-to-one assignment of its phases to known families."""

    def __init__(
        self,
        threshold: float = 0.5,
        features: dict = FEATURES,
        recurrences: bool = False,
    ):
        super().__init__(threshold)
        self.features = features
        # follow PhaseVector.same_as: a phase that repeats an earlier phase of
        # its run joins that phase's family instead of being assigned itself
        self.recurrences = recurrences

    def _reference(self, fam: PhaseFamily, ranks: int) -> dict:
        """Descriptors the family is expected to have at `ranks`.

        Scale-invariant descriptors: median over all members. Rank-dependent
        ones (RANK_DEPENDENT): Theil-Sen trend evaluated at `ranks`; with fewer
        than MIN_TREND_RANKS rank counts, the member at the closest rank count,
        as a median would mix values from different scales.
        """
        members = [(m.ranks, m.as_dict()) for m in fam.members]
        nearest = min(members, key=lambda m: abs(np.log(m[0] / ranks)))[1]
        ref = {}
        for name in self.features:
            points = [(r, d.get(name, np.nan)) for r, d in members]
            if name in RANK_DEPENDENT:
                value = trend_at(points, ranks)
                ref[name] = value if np.isfinite(value) else nearest.get(name, np.nan)
            else:
                ref[name] = median_of(points)
        return ref

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
        """Assign the run's phases to the known families, then add them.

        With recurrences, only first visits are assigned (at most one per
        family, as before); a repeat (same_as set) joins the family of the
        phase it repeats, at the cost of its distance to that phase.
        """
        is_repeat = [self.recurrences and v.same_as is not None for v in run]
        repeats = [v for v, r in zip(run, is_repeat, strict=True) if r]
        assigned = [v for v, r in zip(run, is_repeat, strict=True) if not r]
        fids = list(self.families)  # fixed before this run adds new families
        cost = self.cost_matrix(assigned, fids)
        rows, cols = linear_sum_assignment(cost)  # rows come back sorted

        found = {}  # phase_id -> Match
        for i, j in zip(rows, cols, strict=True):
            v = assigned[i]
            if j < len(fids):
                fid = fids[j]
                case = SAME_RANKS if v.ranks in self.families[fid].ranks else NEW_RANKS
            else:  # "new family" slot
                fid = len(self.families)
                self.families[fid] = PhaseFamily(fid)
                case = NEW_FAMILY
            self.families[fid].members.append(v)
            found[v.phase_id] = Match(v, fid, case, float(cost[i, j]))
        for v in repeats:  # in run order, so a repeat's first visit is known
            first = found[v.same_as].phase
            fid = found[v.same_as].family_id
            cost = phase_distance(v.as_dict(), first.as_dict(), self.features)
            self.families[fid].members.append(v)
            found[v.phase_id] = Match(v, fid, SAME_RANKS, float(cost))
        return [found[v.phase_id] for v in run]
