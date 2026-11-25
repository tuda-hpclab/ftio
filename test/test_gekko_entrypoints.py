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


_DRAIN = """
import sys, threading, time
import zmq
import ftio.api.gekkoFs.predictor_gekko_zmq  # import order, see above
from ftio.prediction.processes_zmq import receive_messages

ctx = zmq.Context.instance()
pull = ctx.socket(zmq.PULL)
pull.bind("inproc://batch-window")
push = ctx.socket(zmq.PUSH)
push.connect("inproc://batch-window")
poller = zmq.Poller()
poller.register(pull, zmq.POLLIN)

def send():
    for _ in range(15):
        push.send(b"x")
        time.sleep(0.2)

sender = threading.Thread(target=send)
sender.start()
start = time.time()
msgs, _ranks, _recv = receive_messages(pull, poller, float(sys.argv[1]))
print(len(msgs), time.time() - start)
sender.join()
"""


def _drain_while_sending(max_batch_s: float) -> tuple[int, float]:
    """Push one message every 0.2 s for 3 s and time one receive_messages call.

    Fresh interpreter for the same circular-import reason as above.
    """
    result = subprocess.run(
        [sys.executable, "-c", _DRAIN, str(max_batch_s)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    n, took = result.stdout.split()
    return int(n), float(took)


def test_receive_messages_max_batch_window_closes_a_continuous_stream():
    """Senders that push more often than once per second never leave the 1 s gap
    that ends a batch, so the unbounded loop returns only when they stop (the
    S3D-IO METRICS_FLUSH_INTERVAL=1 runs got no prediction at all). With a cap
    the batch closes about that many seconds after its first message.
    """
    n, took = _drain_while_sending(0.0)
    assert n == 15 and took >= 2.8
    n, took = _drain_while_sending(1.0)
    assert 4 <= n <= 7 and took < 1.6
