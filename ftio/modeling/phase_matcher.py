"""
PhaseMatcher: group the phases of several runs into phase families.

A phase family is one behavior observed in one or more runs, possibly at
different rank counts. Phases are currently matched by position: phases with
the same phase_id belong to the same family, which assumes that every run goes
through the same phases in the same order.

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

# descriptors a matcher can compare (keys of PhaseVector.as_dict()) -> comparison:
# "log": |log(a/b)| for positive ratios, "lin": |a-b| for values in [0, 1]
DESCRIPTOR_KINDS = {
    "period_ratio": "log",
    "relative_duration": "lin",
    "relative_volume": "lin",
    "read_fraction": "lin",
    "relative_position": "lin",
}

# default descriptors of AlignmentMatcher and AssignmentMatcher:
# descriptor -> (comparison, weight). All scale-free descriptors are assumed
# equally important, except the position, which is index-based and shifts
# when a run has extra or missing phases.
FEATURES = {
    "period_ratio": ("log", 1.0),
    "relative_duration": ("lin", 1.0),
    "relative_volume": ("lin", 1.0),
    "read_fraction": ("lin", 1.0),
    "relative_position": ("lin", 0.5),
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
