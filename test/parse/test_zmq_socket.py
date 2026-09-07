"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Sep 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import zmq

from ftio.parse.zmq_socket import PUB_SUB, PUSH_PULL, recv_socket_type, subscribe_all


def test_recv_socket_type_maps_pattern_to_zmq_constant():
    assert recv_socket_type(PUSH_PULL) == zmq.PULL
    assert recv_socket_type(PUB_SUB) == zmq.SUB
    assert recv_socket_type("anything-else") == zmq.PULL  # safe default


def test_subscribe_all_only_touches_sub_sockets():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    pull = ctx.socket(zmq.PULL)
    try:
        # neither call raises; SUB gets an empty-prefix subscription
        subscribe_all(sub)
        subscribe_all(pull)
        assert sub.getsockopt(zmq.TYPE) == zmq.SUB
    finally:
        sub.close()
        pull.close()
