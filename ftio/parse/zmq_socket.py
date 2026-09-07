"""
Helpers to translate the ``--zmq_socket`` pattern choice into the concrete
ZMQ socket type used on the FTIO (receiving) side.

Two patterns are supported:

* ``push-pull`` -- FTIO binds a ``zmq.PULL`` socket and senders ``connect`` a
  ``zmq.PUSH`` socket. Delivery is reliable (messages are queued), but a PUSH
  sender enters the mute state and *blocks* on ``send()`` once the high-water
  mark is reached with no puller draining. That back-pressure can stall the
  monitored application.

* ``pub-sub`` -- FTIO binds a ``zmq.SUB`` socket (subscribed to every topic)
  and senders ``connect`` a ``zmq.PUB`` socket. A PUB sender *never* blocks: if
  no subscriber is connected, or the high-water mark is reached, it silently
  drops. This is the appropriate choice when the sender is a running HPC
  application that must not be perturbed -- FTIO tolerates the occasional gap
  because it resamples onto a uniform time grid before the transform.

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
    """ZMQ socket type FTIO should bind for the incoming metric stream.

    Args:
        pattern: value of ``--zmq_socket`` (``"push-pull"`` or ``"pub-sub"``).

    Returns:
        ``zmq.PULL`` for push-pull, ``zmq.SUB`` for pub-sub.
    """
    return zmq.SUB if pattern == PUB_SUB else zmq.PULL


def subscribe_all(socket) -> None:
    """Subscribe a SUB socket to every topic; a no-op for any other type.

    A ``zmq.SUB`` socket receives nothing until it is subscribed, so this must
    be called before the first poll. ``""`` (empty prefix) matches all topics,
    matching the "no topic: get all" behaviour of the reference FlexMPI
    listener.
    """
    if socket.getsockopt(zmq.TYPE) == zmq.SUB:
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
