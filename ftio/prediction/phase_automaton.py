"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

# Backward-compatibility shim — module moved to ftio.modeling.phase_automaton.
from ftio.modeling.phase_automaton import PhaseAutomaton, PhaseState, Transition

__all__ = ["PhaseAutomaton", "PhaseState", "Transition"]
