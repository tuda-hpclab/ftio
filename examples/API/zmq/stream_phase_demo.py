"""
Streams synthetic I/O bursts to `predictor` over ZMQ -- no trace file needed.

Unlike docs/zmq.md's single-message example, this sends a whole *sequence*
of bursts over time, with a period change partway through, so you can watch
the phase automaton pick up a real transition live.

Uses the "direct" ZMQ format (see docs/zmq.md#generic-zmq-format): each
message is a MessagePack map with "ranks", "b", "ts", "te" -- exactly what
`predictor --zmq` (the default --zmq_format) expects. No TMIO, no trace
file, no compiled sender required.

Run (two terminals):

    # terminal 1 -- start predictor listening on the default port (5555)
    predictor --zmq -f 10 --phase-automaton --pa-method ksigma -e no

    # terminal 2 -- stream synthetic bursts
    python examples/API/zmq/stream_phase_demo.py

Phase A (checkpoints every ~2s) runs for 10 bursts, then Phase B
(checkpoints every ~1s, roughly double the frequency) runs for another 10 --
predictor's phase automaton should report one state for each phase and a
transition around t=20s.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import argparse
import time

import msgpack
import zmq

ADDRESS = "tcp://127.0.0.1:5555"
RANKS = 8
BANDWIDTH = 5e8  # bytes/s during a burst


def send_burst(sock, t_start: float, duration: float) -> None:
    """Push one burst as a "direct"-format ZMQ message: {ranks, b, ts, te}."""
    msg = {
        "ranks": RANKS,
        "b": [BANDWIDTH],
        "ts": [t_start],
        "te": [t_start + duration],
    }
    sock.send(msgpack.packb(msg))


def stream(address: str, speed: float) -> None:
    """Send Phase A (period ~2s) then Phase B (period ~1s), 10 bursts each.

    `speed` scales the real sleep time (1.0 = real-time; use e.g. 0.1 for a
    quick smoke test).
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(address)
    print(f"[sender] connected to {address}")

    t = 0.0
    phases = [
        ("A", 10, 2.0, 0.3),  # label, n_bursts, period, burst_duration
        ("B", 10, 1.0, 0.15),
    ]
    for label, n_bursts, period, duration in phases:
        for i in range(n_bursts):
            send_burst(sock, t, duration)
            print(f"[sender] t={t:6.2f}s  phase {label}  burst {i + 1}/{n_bursts}")
            time.sleep(period * speed)
            t += period

    print("[sender] done streaming")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address",
        default=ADDRESS,
        help=f"ZMQ address to connect to (default: {ADDRESS})",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Scales the delay between bursts (1.0 = real-time, 0.1 = 10x faster). Default: 1.0",
    )
    args = parser.parse_args()
    stream(args.address, args.speed)


if __name__ == "__main__":
    main()
