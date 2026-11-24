"""
FTIO Command Line Interface (CLI) module.

This module exposes the main entry points for the FTIO CLI tools,
including the core analysis and the predictor.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: January 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from ftio.cli.ftio_core import core, main
from ftio.cli.predictor import main as predictor_main

__all__ = ["core", "main", "predictor_main"]
