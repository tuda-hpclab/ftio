"""
Map the ``--zmq_socket`` pattern choice to the ZMQ socket type FTIO binds on
the receiving side: ``push-pull`` -> ``PULL`` (reliable, PUSH sender blocks on
back-pressure), ``pub-sub`` -> ``SUB`` (lossy, PUB sender never blocks).

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Sep 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import zmq

PUSH_PULL = "push-pull"
PUB_SUB = "pub-sub"


def recv_socket_type(pattern: str) -> int:
    """``zmq.SUB`` for ``pub-sub``, ``zmq.PULL`` otherwise."""
    return zmq.SUB if pattern == PUB_SUB else zmq.PULL


def subscribe_all(socket) -> None:
    """Subscribe a SUB socket to every topic; no-op for other socket types."""
    if socket.getsockopt(zmq.TYPE) == zmq.SUB:
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
