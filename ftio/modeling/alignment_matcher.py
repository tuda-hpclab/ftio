"""
AlignmentMatcher: group phases of several runs into phase families by aligning
each run to the ordered list of known families (Needleman-Wunsch).

Unlike PhaseMatcher, runs may have different numbers of phases: a phase with
no fitting family starts a new one (gap in the profile), and a family missing
in a run is skipped (gap in the run). Only the order of phases is assumed to
be the same in every run.

Author: Albani Alvarado Mauriello
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: November 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import numpy as np

from ftio.modeling.phase_matcher import (
    NEW_FAMILY,
    NEW_RANKS,
    SAME_RANKS,
    Match,
    PhaseFamily,
    PhaseMatcher,
)
from ftio.modeling.phase_vector import PhaseVector

# descriptor -> (comparison, weight)
# we assume that all scale-free descriptors are equally important. 
# "log": |log(a/b)| for positive ratios,
# "lin": |a-b| for values in [0, 1].
FEATURES = {
    "period_ratio": ("log", 1.0),
    "relative_duration": ("lin", 1.0),
    "relative_volume": ("lin", 1.0),
    "read_fraction": ("lin", 1.0),
    "relative_position": ("lin", 0.5),  # index-based, shifts with extra phases
}

#used for the Needleman-Wunsch algorithm
DIAG, UP, LEFT = 0, 1, 2  # match, family missing in run, phase without family

MIN_TREND_RANKS = 3  # rank counts a family needs before its trend is extrapolated


def power_law_at(points: list[tuple[int, float]], at: int) -> float:
    """Fit y = c * ranks^b over (ranks, y) points and evaluate it at `at`.

    The fit is a line in log-log, so only positive values are used. Returns NaN
    if they cover fewer than MIN_TREND_RANKS rank counts.
    """
    pts = [(r, y) for r, y in points if np.isfinite(y) and y > 0]
    if len({r for r, _ in pts}) < MIN_TREND_RANKS:
        return np.nan
    log_r = np.log([r for r, _ in pts])
    log_y = np.log([y for _, y in pts])
    slope, intercept = np.polyfit(log_r, log_y, 1)
    return float(np.exp(intercept + slope * np.log(at)))


def phase_distance(a: dict, b: dict, features: dict = FEATURES) -> float:
    """Weighted mean distance over the descriptors both phases have (NaN skipped)."""
    total = weight = 0.0
    for name, (kind, w) in features.items():
        x, y = a.get(name, np.nan), b.get(name, np.nan)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if kind == "log":
            if x <= 0 or y <= 0:
                continue
            d = abs(np.log(x / y))
        else:
            d = abs(x - y)
        total += w * d
        weight += w
    return total / weight if weight > 0 else np.inf


def align(dist: np.ndarray, gap: float):
    """Global alignment of two phase sequences (Needleman-Wunsch).

    Args:
        dist: n x m matrix, dist[i, j] = cost of aligning family i with phase j.
        gap: cost of leaving a family or a phase unaligned.

    Returns a list of (family index | None, phase index | None, cost) in order.
    """
    n, m = dist.shape
    score = np.zeros((n + 1, m + 1))
    move = np.zeros((n + 1, m + 1), dtype=int)
    score[1:, 0], move[1:, 0] = gap * np.arange(1, n + 1), UP
    score[0, 1:], move[0, 1:] = gap * np.arange(1, m + 1), LEFT
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                score[i - 1, j - 1] + dist[i - 1, j - 1],
                score[i - 1, j] + gap,
                score[i, j - 1] + gap,
            )
            move[i, j] = int(np.argmin(options))  # ties prefer DIAG
            score[i, j] = options[move[i, j]]

    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if move[i, j] == DIAG:
            pairs.append((i - 1, j - 1, float(dist[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif move[i, j] == UP:
            pairs.append((i - 1, None, gap))
            i -= 1
        else:
            pairs.append((None, j - 1, gap))
            j -= 1
    return pairs[::-1]


class AlignmentMatcher(PhaseMatcher):
    """Matches a run by aligning it to the ordered list of known families."""

    def __init__(self, threshold: float = 0.5, trend: bool = True):
        super().__init__(threshold)
        self.gap = threshold / 2  # a pair is matched only if its distance < threshold
        self.trend = trend  # extrapolate families beyond their observed rank counts
        self.order: list[int] = []  # family ids in run order

    def _reference(self, fam: PhaseFamily, ranks: int) -> dict:
        """Descriptors the family is expected to have at `ranks`, to compare new phases with.

        Within the family's observed rank counts, or with fewer than
        MIN_TREND_RANKS of them, this is the member at the closest rank count.
        Beyond them, each descriptor is extrapolated with a power law fit over
        all members, as descriptors can still drift with scale; a descriptor
        that cannot be fit keeps the closest member's value.
        """
        # closest rank count
        nearest = min(fam.members, key=lambda m: abs(np.log(m.ranks / ranks)))
        ref = nearest.as_dict()
        observed = fam.ranks
        if (
            not self.trend
            or len(observed) < MIN_TREND_RANKS
            or min(observed) <= ranks <= max(observed)
        ):
            return ref
        members = [(m.ranks, m.as_dict()) for m in fam.members]
        for name, (kind, _) in FEATURES.items():
            # extrapolate the descriptor's trend to the new rank count
            value = power_law_at([(r, d.get(name, np.nan)) for r, d in members], ranks)
            if np.isfinite(value):
                ref[name] = min(value, 1.0) if kind == "lin" else value  # fractions
        return ref

    def match_run(self, run: list[PhaseVector]) -> list[Match]:
        """Align one run to the known families, then add its phases to them.

        Each phase is compared at its own rank count, as it can change within a
        run (malleability).
        """
        dist = np.array(
            [
                [
                    phase_distance(
                        self._reference(self.families[f], v.ranks), v.as_dict()
                    )
                    for v in run
                ]
                for f in self.order
            ]
        ).reshape(len(self.order), len(run))
        pairs = align(dist, self.gap)

        matches, new_order = [], []
        for i, j, cost in pairs:
            if j is None:  # family not seen in this run
                new_order.append(self.order[i])
                continue
            v = run[j]
            if i is None:  # no family fits
                fid = len(self.families)
                self.families[fid] = PhaseFamily(fid)
                case = NEW_FAMILY
            else:
                fid = self.order[i]
                case = SAME_RANKS if v.ranks in self.families[fid].ranks else NEW_RANKS
            self.families[fid].members.append(v)
            new_order.append(fid)
            matches.append(Match(v, fid, case, cost))
        self.order = new_order
        return matches
