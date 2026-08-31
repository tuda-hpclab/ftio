# ZMQ Interface

FTIO supports ZeroMQ (ZMQ) as a live data source, avoiding the need to write intermediate trace files to disk.  Instead of reading a file, `ftio` or `predictor` listens on a ZMQ socket for incoming bandwidth data and analyses it as it arrives.

- [Overview](#overview)
- [Flags](#flags)
- [Generic ZMQ format](#generic-zmq-format)
- [Reply formats](#reply-formats)
- [ZMQ with TMIO](#zmq-with-tmio)
- [Returning frequency predictions to TMIO](#returning-frequency-predictions-to-tmio)

---

## Overview

In ZMQ mode:

- The sender (application, TMIO, or any custom producer) pushes bandwidth data to a ZMQ socket.
- `ftio` or `predictor` receives messages, deserialises them, and analyses the bandwidth data.
- Predictions are printed to the console and, optionally, sent back over a reply socket.

Use `predictor` for continuous online analysis (re-runs on every new message); use `ftio` with `--zmq` for a single-shot analysis of one incoming batch.

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--zmq` | off | Enable ZMQ input mode (suppresses opening the HTML output). |
| `--zmq_format` | `direct` | Encoding of the ZMQ payload: `direct` (generic) or `tmio`. `--zmq_source` is a legacy alias of this flag. Not to be confused with `--source`, which selects the on-disk file format. |
| `--zmq_address` | `*` | ZMQ bind address. `*` binds to all interfaces; use `127.0.0.1` for localhost only. |
| `--zmq_port` | `5555` | ZMQ port for incoming data messages. |
| `--zmq_port_reply` | `5556` | ZMQ port for outgoing predictions. Passing this flag turns the reply on. |
| `--zmq_reply_format` | `msgpack` | Encoding of the reply: `msgpack`, `struct`, or `raw`. See [Reply formats](#reply-formats). |

---

## Generic ZMQ format

A sender pushes MessagePack messages to `predictor`. Each message is a map. FTIO
picks the shape from the keys that are present, there is no format field:

| Keys | Meaning |
|------|---------|
| `b`, `ts`, `te`, `ranks?` | Per rank I/O intervals. FTIO overlaps concurrent ranks. |
| `b`, `ts`, `ranks?` | Same, `te[i]` defaults to `ts[i+1]` (the last one keeps its start). |
| `b`, `t`, `ranks?` | An already overlapped signal (`b` at times `t`), used as is. This is the `b`, `t` pair `ftio_core.core()` takes. |

| Key | Type | Description |
|-----|------|-------------|
| `b` | float[] | Bandwidth (bytes/s), one value per rank interval or per sample. |
| `ts`, `te` | float[] | Rank interval start and end (seconds). Rank level only. |
| `t` | float[] | Sample times (seconds). App level only. |
| `ranks` | int | Number of I/O ranks. Optional, defaults to 0. |

Send one shape per run. A run that mixes shapes still works, but the merge takes
its keys from the first message and can drop data of the other shape.

For a send and receive example see
[API, Online prediction over ZMQ](api.md#online-prediction-over-zmq)
([`examples/API/zmq/predictor_zmq_api.py`](/examples/API/zmq/predictor_zmq_api.py)).

**Start the receiver:**

```bash
predictor --zmq -e no -f 100
```

**Example C++ sender** (using `zmq.hpp` and `msgpack.hpp`):

```cpp
#include <iostream>
#include <zmq.hpp>
#include <msgpack.hpp>

int main() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_PUSH);

    socket.bind("tcp://127.0.0.1:5555");

    // Create a MessagePack object to hold the data
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> packer(&buffer);

    // Pack the data into the MessagePack buffer
    packer.pack_map(4);
    packer.pack("ranks");
    packer.pack(8);

    // Pack the arrays
    packer.pack("b");
    packer.pack_array(5);
    packer.pack(3.0);
    packer.pack(0.0);
    packer.pack(3.0);
    packer.pack(0.0);
    packer.pack(3.0);

    packer.pack("ts");
    packer.pack_array(5);
    packer.pack(1.0);
    packer.pack(2.0);
    packer.pack(3.0);
    packer.pack(4.0);
    packer.pack(5.0);

    packer.pack("te");
    packer.pack_array(5);
    packer.pack(5.0);
    packer.pack(6.0);
    packer.pack(7.0);
    packer.pack(8.0);
    packer.pack(9.0);

    zmq::message_t message(buffer.size());
    memcpy(message.data(), buffer.data(), buffer.size());
    socket.send(message, zmq::send_flags::none);

    return 0;
}
```

**Example Python sender:**

```python
import zmq
import msgpack

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect("tcp://127.0.0.1:5555")

data = {
    "ranks": 8,
    "b":  [3.0, 0.0, 3.0, 0.0, 3.0],
    "ts": [1.0, 2.0, 3.0, 4.0, 5.0],
    "te": [5.0, 6.0, 7.0, 8.0, 9.0],
}
sock.send(msgpack.dumps(data))
```

**Streaming multiple messages (phase automaton demo):** the single-message example above is enough to check the wire format works, but it can't show the phase automaton doing anything — that needs a *sequence* of bursts over time. `examples/API/zmq/stream_phase_demo.py` sends 10 bursts with a ~2s period, then 10 more with a ~1s period, so you can watch a state open, a period change get detected, and a transition fire — with no trace file, no TMIO, no compiled sender:

```bash
# terminal 1
predictor --zmq -f 10 --phase-automaton --pa-method ksigma -e no

# terminal 2
python examples/API/zmq/stream_phase_demo.py
```

Expect `predictor` to log a state at ~2.1s period for the first ~20s, then a `TRANSITION: State 0 → 1 (0.48 → 1.00 Hz)` once the faster phase kicks in. Use `--speed 0.1` for a 10x-faster smoke test.

---

## Reply formats

When `--zmq_port_reply` is set, `predictor` pushes one message back per
prediction. `--zmq_reply_format` picks the encoding:

| Format | Payload |
|--------|---------|
| `msgpack` (default) | The whole prediction as a map: every `Prediction.to_dict()` field with numpy arrays turned into lists, plus `period` (`1/dominant_freq`, s) and `duty_cycle`. `dominant_freq` and `conf` are the single strongest value. See [API, the message you get back](api.md#the-message-you-get-back) for the field list. |
| `struct` or `raw` | `struct.pack("dd", freq, conf)`, 16 bytes, the format the TMIO prefetcher reads. |

```python
import zmq, msgpack

ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.bind("tcp://127.0.0.1:5556")          # predictor connects its reply socket here

while True:
    p = msgpack.unpackb(sock.recv())
    print(f"period {p['period']:.2f}s  conf {p['conf']:.2f}")
```

---

## ZMQ with TMIO

[TMIO](https://github.com/tuda-parallel/TMIO) can stream bandwidth data directly to `ftio` or `predictor` without writing trace files.

> **Note:** ZMQ support in TMIO is still under active development.

**Setup:**

1. Compile TMIO with ZMQ support:
   ```bash
   cd <tmio-build-dir>
   make zmq
   ```

2. The sender side writes the ZMQ address to a file called `ftio_port`:
   ```
   tcp://127.0.0.1:5555
   ```

3. Run the TMIO application normally:
   ```bash
   mpirun -np 8 ./test_run
   ```

4. In a separate terminal, launch `predictor` or `ftio`:
   ```bash
   # Online prediction (re-runs on each incoming batch)
   predictor --zmq --zmq_source tmio -m write_async -f 100

   # Single-shot analysis
   ftio --zmq --zmq_source tmio -m write_async -f 100
   ```

---

## Returning frequency predictions to TMIO

`predictor` can send the detected dominant frequency back to TMIO's I/O prefetcher over a second ZMQ socket. TMIO reads the legacy 16-byte format, so pass `--zmq_reply_format struct`:

```bash
predictor --zmq --zmq_source tmio --zmq_port_reply 5556 --zmq_reply_format struct -m read_sync
```

| Port | Direction | Purpose |
|------|-----------|---------|
| `--zmq_port` (5555) | TMIO → predictor | Incoming bandwidth data |
| `--zmq_port_reply` (5556) | predictor → TMIO | Outgoing dominant frequency (`struct.pack("dd", freq, conf)`) |

Any other consumer should use the default `msgpack` reply (see [Reply formats](#reply-formats)).

<p align="right"><a href="#zmq-interface">⬆</a></p>
