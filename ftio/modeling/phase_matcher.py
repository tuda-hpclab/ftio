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

from ftio.modeling.phase_vector import PhaseVector

SAME_RANKS = "same_ranks"  # known family, already seen at this rank count
NEW_RANKS = "new_ranks"  # known family, not yet seen at this rank count
NEW_FAMILY = "new_family"  # first time this phase_id is seen


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
