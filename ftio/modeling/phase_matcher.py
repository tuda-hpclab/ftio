"""
PhaseMatcher: group the phases of several runs into phase families.

A phase family is one behavior observed in one or more runs, possibly at
different rank counts. PhaseMatcher itself matches by position: phases with
the same phase_id belong to the same family, which assumes that every run goes
through the same phases in the same order. AlignmentMatcher and
AssignmentMatcher build on it and compare descriptors instead.

This module also holds what they share: the descriptors (DESCRIPTOR_KINDS),
which of them are clean (CLEAN, NOT_CLEAN) or change with the rank count
(RANK_DEPENDENT), the default features (FEATURES) and phase_distance.

Author: Albani Alvarado Mauriello
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: October 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ftio.modeling.phase_vector import PhaseVector

SAME_RANKS = "same_ranks"  # known family, already seen at this rank count
NEW_RANKS = "new_ranks"  # known family, not yet seen at this rank count
NEW_FAMILY = "new_family"  # first time this phase_id is seen

MIN_TREND_RANKS = 3  # rank counts a family needs before a trend is fitted

# descriptors a matcher can compare (keys of PhaseVector.as_dict()) -> comparison:
# "log": |log(a/b)| for positive ratios, "lin": |a-b| for values in [0, 1]
DESCRIPTOR_KINDS = {
    "read_fraction": "lin",
    "relative_volume": "lin",
    "total_bytes": "log",
    "bytes_per_rank": "log",
    "relative_position": "lin",
    "relative_duration": "lin",
    "intensity": "log",  # bytes / duration
    "duration": "log",
    "period": "log",
    "period_ratio": "log",  # period / shortest period of the run
    "volume_per_period": "log",  # intensity x period, bytes per I/O burst
    "n_bursts": "log",  # duration / period
}

# descriptors that change with the rank count. AssignmentMatcher follows their
# trend over a family's members; all others are taken as scale-invariant.
RANK_DEPENDENT = {
    "total_bytes",
    "bytes_per_rank",  # constant in weak scaling, ~1/p in strong scaling
    "intensity",
    "duration",
    "period",
    "volume_per_period",
}

# descriptors that are safe to match on before the period is modeled: bytes
# only, no time and no order. All others are for ablations only (NOT_CLEAN).
CLEAN = {"read_fraction", "relative_volume", "total_bytes", "bytes_per_rank"}

# duration = n_bursts x period for any number of bursts: a number that varies
# with the rank count only adds noise, the period stays in it
_IN_DURATION = "contains the period (duration = n_bursts x period)"

# descriptor -> why matching on it is not clean. The period is what
# phase_model fits (period, frequency): matching on it makes the fitted models
# look better than they are, as the matcher then picks the phases that agree
# with the trend. Update if the targets change.
NOT_CLEAN = {
    "period": "is the modeled period",
    "period_ratio": "contains the period",
    "volume_per_period": "contains the period",
    "n_bursts": "contains the period",
    "duration": _IN_DURATION,
    "relative_duration": _IN_DURATION,
    "intensity": "contains the period (bytes / duration = bytes per burst / period)",
    "relative_position": "uses the phase order, which assignment is meant to "
    "ignore, and splits recurring phases",
}

# default descriptors of AlignmentMatcher and AssignmentMatcher:
# descriptor -> (comparison, weight). A subset of CLEAN. Both use the same
# defaults, so their results differ only by the matching method.
FEATURES = {
    "read_fraction": ("lin", 1.0),
    "relative_volume": ("lin", 1.0),
    "bytes_per_rank": ("log", 1.0),
}


def parse_features(specs: list[str]) -> dict:
    """Descriptors to match on, from "NAME" or "NAME:WEIGHT" strings (weight 1.0).

    e.g. ["period_ratio", "relative_position:0.5"]
        -> {"period_ratio": ("log", 1.0), "relative_position": ("lin", 0.5)}

    Raises:
        ValueError: unknown descriptor, descriptor given twice, or a weight
            that is not a number > 0.
    """
    features = {}
    for spec in specs:
        name, _, weight = spec.partition(":")
        if name not in DESCRIPTOR_KINDS:
            raise ValueError(
                f"unknown descriptor '{name}', known: {', '.join(DESCRIPTOR_KINDS)}"
            )
        if name in features:
            raise ValueError(f"descriptor '{name}' given twice")
        try:
            w = float(weight) if weight else 1.0
        except ValueError:
            raise ValueError(f"weight of '{name}' is not a number: '{weight}'") from None
        if not w > 0:  # also rejects nan
            raise ValueError(f"weight of '{name}' must be > 0, got {weight}")
        features[name] = (DESCRIPTOR_KINDS[name], w)
    return features


def phase_distance(a: dict, b: dict, features: dict) -> float:
    """Weighted mean distance over the descriptors both phases have (NaN skipped).

    features: descriptor -> (comparison, weight); "log": |log(a/b)| for
    positive ratios, "lin": |a-b| for values in [0, 1].
    Returns inf if the phases share no descriptor.
    """
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


@dataclass
class PhaseFamily:
    """One behavior, observed in one or more runs."""

    family_id: int
    members: list[PhaseVector] = field(default_factory=list)

    @property
    def ranks(self) -> set[int]:
        return {v.ranks for v in self.members}


@dataclass
class Match:
    """Result of matching one phase."""

    phase: PhaseVector
    family_id: int
    case: str  # SAME_RANKS, NEW_RANKS or NEW_FAMILY
    cost: float = 0.0  # distance to the family


class PhaseMatcher:
    """Groups phases of several runs into phase families."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold  # max distance to join a family (not used yet)
        self.families: dict[int, PhaseFamily] = {}

    def match_phase(self, v: PhaseVector) -> Match:
        """Match one phase, then add it to its family."""
        family_id = v.phase_id
        if family_id not in self.families:
            case = NEW_FAMILY
            self.families[family_id] = PhaseFamily(family_id)
        elif v.ranks in self.families[family_id].ranks:
            case = SAME_RANKS
        else:
            case = NEW_RANKS
        self.families[family_id].members.append(v)
        return Match(v, family_id, case)

    def match_run(self, run: list[PhaseVector]) -> list[Match]:
        """Match all phases of one run, then add them to their families."""
        return [self.match_phase(v) for v in run]
