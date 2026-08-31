"""
Online prediction over ZMQ.

test_api.py calls core(data, args) once on an array in memory. Here predictor
reads the data from a ZMQ socket instead: it runs a prediction on every window
and, because --zmq_port_reply is set, sends each result back on a second socket.

argv is the argument list you would type after `predictor` on the command line:
the normal FTIO args (-f, -e, -m, ...) plus the ZMQ ports. Any process that
pushes MessagePack messages to --zmq_port is a sender. docs/zmq.md and
docs/api.md have a sender snippet.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Aug 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from ftio.cli.predictor import main

cli = "predictor --zmq -e no -f 10 -m write"
cli += " --zmq_port 5555 --zmq_port_reply 5556 --zmq_reply_format msgpack"
argv = cli.split()

if __name__ == "__main__":
    main(argv)  # read messages, predict, reply. Stop with Ctrl-C.
