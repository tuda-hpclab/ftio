"""
FTIO GUI Dashboard for real-time prediction visualization.

This module provides a Dash-based web dashboard for visualizing FTIO predictions
and change point detection results in real-time.

Author: Amine Aherbil
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: January 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

__all__ = [
    "FTIODashApp",
    "PredictionData",
    "ChangePoint",
    "PredictionDataStore",
    "SocketListener",
]


def __getattr__(name):
    """Lazy import to avoid requiring dash unless actually used."""
    if name == "FTIODashApp":
        from ftio.gui.dashboard import FTIODashApp

        return FTIODashApp
    elif name == "SocketListener":
        from ftio.gui.socket_listener import SocketListener

        return SocketListener
    elif name in ("PredictionData", "ChangePoint", "PredictionDataStore"):
        from ftio.gui import data_models

        return getattr(data_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
