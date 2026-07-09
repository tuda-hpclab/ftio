"""
Import-smoke tests for the GekkoFS/GLASS entry points.

These guard two regressions that broke `jit` / `predictor_jit` at import time:

1. Python 3.14 made "forkserver" the default start method on Linux. The GekkoFS
   stack builds a module-level Manager()-backed FileQueue at import; under
   forkserver that raises during import. file_queue.py restores the fork start
   method, so importing these modules must succeed.

2. Commit 8d14a32 renamed processes_zmq.bind_socket -> setup_socket but left
   predictor_gekko_zmq importing the old name, so predictor_jit failed with
   ImportError. The import must resolve now.

Each import runs in a fresh interpreter so the process-wide start method is not
already fixed by pytest itself (which would mask regression 1).

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: July 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import subprocess
import sys

import pytest

ENTRY_POINTS = [
    "ftio.api.gekkoFs.jit.jit",
    "ftio.api.gekkoFs.predictor_gekko_zmq",
    "ftio.api.gekkoFs.ftio_gekko",
]


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_gekko_entrypoint_imports_in_fresh_interpreter(module):
    """Each GekkoFS entry point must import in a clean interpreter (any start method)."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{module} failed to import:\n{result.stderr}"


def test_predictor_uses_setup_socket_not_bind_socket():
    """predictor_gekko_zmq must reference the renamed setup_socket, not bind_socket.

    Run in a fresh interpreter: importing the predictor in-process under pytest
    trips a pre-existing circular import unrelated to this check.
    """
    check = (
        "import ftio.api.gekkoFs.predictor_gekko_zmq as p;"
        "from ftio.prediction.processes_zmq import setup_socket;"
        "assert p.setup_socket is setup_socket;"
        "assert not hasattr(p, 'bind_socket');"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
