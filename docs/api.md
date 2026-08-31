# API

The API allows interacting with `ftio` directly, rather than using the command line interface provided in the [
`cli`](/ftio/cli/) folder.
Below are several examples of this.

- [API](#api)
    - [General](#general)
    - [Other ways to call FTIO](#other-ways-to-call-ftio)
    - [Metric Proxy](#metric-proxy)
    - [GekkoFS with Msgpack/JSON support](#gekkofs-with-msgpackjson-support)
    - [Online prediction over ZMQ](#online-prediction-over-zmq)

## General

The file [`ftio_api.py`](/examples/API/test_api.py) provides an example how to directly call use `ftio`:

```python
import numpy as np
from ftio.cli.ftio_core import core
from ftio.parse.args import parse_args
from ftio.processing.print_output import display_prediction
from ftio.parse.bandwidth import overlap

ranks = 10
total_bytes = 100

# Set up data
## 1) overlap for rank level metrics
b_rank = [0.0, 0.0, 1000.0, 1000.0, 0.0, 0.0, 1000.0, 1000.0, 0.0, 0.0, 1000.0, 1000.0, 0.0, 0.0]
t_rank_s = [0.5, 0.0, 10.5, 10.0, 20.5, 20.0, 30.5, 30.0, 40.5, 40.0, 50.5, 50.0, 60.5, 60]
t_rank_e = [5.0, 4.5, 15.0, 14.5, 25.0, 24.5, 35.0, 34.5, 45.0, 44.5, 55.0, 54.5, 65.0, 64.5]
b, t = overlap(b_rank, t_rank_s, t_rank_e)

## 2) or directly specify the app level metrics
# t = [10.0, 20.1, 30.0, 40.2, 50.3, 60, 70, 80.0,]
# b = [10, 0, 10, 0, 10, 0, 10, 0]


# command line arguments
argv = ["-e", "no"]  # ["-e", "mat"]

# set up data
data = {
    "time": np.array(t),
    "bandwidth": np.array(b),
    "total_bytes": total_bytes,
    "ranks": ranks
}

# parse args
args = parse_args(argv, "ftio")

# perform prediction
prediction, analysis_figures = core(data, args)

# plot and print info
analysis_figures.show()
display_prediction(args, prediction)
```

<p align="right"><a href="#api">⬆</a></p>

## Other ways to call FTIO

The `core()` call above runs in the same process. FTIO can also be reached over a
socket:

| Interface | Entry point | Transport | Per call | Use |
|-----------|-------------|-----------|----------|-----|
| in process | `ftio_core.core()` / `main()` | Python call | one prediction | this page, [Prediction API](prediction_api.md) |
| HTTP server | `server_ftio` | HTTP POST `:5000` | one prediction | [FTIO server](ftio_server.md) |
| ZMQ predictor | `predictor --zmq` | ZMQ PULL, optional PUSH reply | keeps running | [below](#online-prediction-over-zmq) |
| Metric Proxy | `admire_proxy_zmq` | ZMQ request/reply | one prediction | [Metric Proxy ZMQ](metric_proxy_zmq.md) |

See [Predictor](predictor.md) for a comparison of these.

<p align="right"><a href="#api">⬆</a></p>

## Metric Proxy

This API allows interaction with the [metric proxy](https://github.com/besnardjb/proxy_v2). Once executed, the proxy
outputs a JSON file which can be directly used with this API.
The file [`proxy.py`](/ftio/api/metric_proxy/proxy.py) provides an example. To execute it, simply call:

```sh
python proxy.py
```

The following line in [`proxy.py`](/ftio/api/metric_proxy/proxy.py) can be changed to specify the needed metric and the
path to the JSON file:

```py
b, t = parse("some_location/filename.json", "metric")
```

To suppress the output, the function `display_prediction` can be commented out. Moreover, `argv = ["-e", "plotly"]` can
be changed to `["-e", "no"]` to disable the plots.

Furthermore, at the end of [`proxy.py`](/ftio/api/metric_proxy/proxy.py), postprocessing occurs to label the phases
according to the function `label_phases` from [`post_processing.py`](/ftio/processing/post_processing.py).

### Dewrapped bandwidth (`--dewrap`)

The proxy's exporters account a call's bytes and duration only when the call **returns**, so in the sampled trace a
long I/O burst appears as a spike at its completion sample. With `--dewrap` (available in
[`parallel_proxy.py`](/ftio/api/metric_proxy/parallel_proxy.py) and understood by the ZMQ server
[`proxy_zmq.py`](/ftio/api/metric_proxy/proxy_zmq.py) — the proxy forwards it via its FTIO *custom arguments*), FTIO
reconstructs the wall-clock bandwidth from each cumulative `___size___<fn>` / `___time___<fn>` counter pair: every
burst is spread backwards from its completion time over its estimated span (Δtime ÷ `proxy_mpi_ranks`), i.e. **the
points are created in the past**, so the burst start point — and therefore the predicted phase — is correct. Each
value is the rate of the interval starting at its timestamp (sample-and-hold); the integral equals the transferred
bytes exactly (see `dewrap_bandwidth` in [`parse_proxy.py`](/ftio/api/metric_proxy/parse_proxy.py)).

The reconstructed signals are analyzed **in addition to** the regular metrics and stored under the
`___bandwidth_dewrap___<fn>` name — matching the virtual metric the proxy trace UI offers, so the FTIO overlay in the
UI lines up with the plotted metric. Both the size and time counters (and ideally `proxy_mpi_ranks`) must be present
in the payload; the proxy sends all of them by default. If the time counter is missing for a pair, that metric simply
keeps its normal completion-attributed derivative.

<p align="right"><a href="#api">⬆</a></p>

## GekkoFS with Msgpack/JSON support

The file [`ftio_gekko.py`](/ftio/api/gekkoFs/ftio_gekko.py) provides an example for the adapted `ftio` api for gekkoFs.
The file [`predictor_gekko.py`](/ftio/api/gekkoFs/predictor_gekko.py) provides an example for `predictor`.

For [`ftio_gekko.py`](/ftio/api/gekkoFs/ftio_gekko.py), the path to the files needs to be specified in the code:

```python
import glob
from ftio.api.gekkoFs.ftio_gekko import run

if __name__ == "__main__":
    # absolute path to search all text files inside a specific folder
    # path=r'/d/github/FTIO/examples/API/gekkoFs/JSON/*.json' # For JSON
    path = r"/d/github/FTIO/examples/API/gekkoFs/MSGPACK/write*.msgpack"  # For MSGPCK
    matched_files = glob.glob(path)
    run(matched_files)
```

Similarly, for [`predictor_gekko.py`](/ftio/api/gekkoFs/predictor_gekko.py), the following lines can be adjusted:

```python
import glob

def main(args: list[str] = []) -> None:
    n_buffers = 4  # number of buffers 
    args = ["-e", "plotly", "-f", "0.01"]  # arguments for ftio
    # path=r'/d/github/FTIO/examples/API/gekkoFs/JSON/*.json' # For JSON
    path = r"/d/github/FTIO/examples/API/gekkoFs/MSGPACK/write*.msgpack"  # For MSGPCK
    matched_files = glob.glob(path)
```

## Online prediction over ZMQ

Instead of calling `core()` on a fixed array, the data can be sent to `predictor`
over a ZMQ socket while the application runs. `predictor` reads each message,
runs a prediction, and (when `--zmq_port_reply` is set) sends the result back on
a second socket. No trace file is written.

### The message you send

A MessagePack map. Pick one of these shapes:

| Keys | Meaning |
|------|---------|
| `b`, `ts`, `te` (`ranks` optional) | rank level I/O intervals. FTIO overlaps concurrent ranks. |
| `b`, `ts` (`ranks` optional) | same, `te[i]` defaults to `ts[i+1]` |
| `b`, `t` (`ranks` optional) | an already overlapped signal, analysed as is |

| Key | Type | Meaning |
|-----|------|---------|
| `b` | float[] | bandwidth in bytes/s, one value per interval or per sample |
| `ts`, `te` | float[] | start and end time of each interval, seconds (rank level) |
| `t` | float[] | sample time, seconds (job level) |
| `ranks` | int | number of I/O ranks, optional, defaults to 0 |

Send one shape per run. Several rank level messages that arrive together are
merged before the overlap.

### The message you get back

One message per prediction, on `--zmq_port_reply`. With `--zmq_reply_format
msgpack` (the default) it is the whole prediction as a MessagePack map, the same
fields as [`Prediction.to_dict()`](prediction_api.md) with the numpy arrays
turned into lists:

| Key | Type | Meaning |
|-----|------|---------|
| `dominant_freq` | float | strongest frequency in Hz, `0.0` if none was found |
| `conf` | float | confidence of that frequency, `0.0` to `1.0` |
| `period` | float | `1 / dominant_freq` in seconds, `0.0` if none was found |
| `duty_cycle` | float | fraction of the period spent in I/O, `NaN` without `-bw` |
| `burst_widths` | float[] | width of each burst in seconds, empty without `-bw` |
| `source` | str | method that found the frequency (`dft`, `wave_disc`, ...) |
| `t_start`, `t_end` | float | the time window that was analysed, seconds |
| `total_bytes` | int | bytes in that window |
| `ranks` | int | rank count |
| `freq` | float | sampling rate used, Hz |
| `amp`, `phi` | float[] | amplitude and phase per dominant frequency |
| `periodicity` | float[] | periodicity score per frequency (from `-p`) |
| `candidates` | float[] | autocorrelation candidate periods (from `-au`) |
| `top_freqs` | map | top-N frequency candidates with metadata |
| `n_samples` | int | number of samples in the resampled signal |

With `--zmq_reply_format struct` (or `raw`) it is `struct.pack("dd", freq, conf)`,
16 bytes, the format the TMIO prefetcher reads.

### The predictor side

Run it with the same FTIO arguments as on the command line, plus the ZMQ ports.
`ftio.cli.predictor.main` takes that argument list. See
[`examples/API/zmq/predictor_zmq_api.py`](/examples/API/zmq/predictor_zmq_api.py).

```python
from ftio.cli.predictor import main

argv = (
    "predictor --zmq -e no -f 10 -m write "
    "--zmq_port 5555 --zmq_port_reply 5556 --zmq_reply_format msgpack"
).split()
main(argv)   # reads messages, predicts, replies. Stop with Ctrl-C.
```

### The sender side

Any process can be a sender. It packs its bandwidth data as one of the maps
above and pushes it to `--zmq_port`.

The predictor moves its analysis window forward as it runs, so each example
below assumes a predictor that was just started. The socket setup is the same
for all of them:

```python
import zmq
import msgpack

ctx = zmq.Context()
push = ctx.socket(zmq.PUSH)
push.connect("tcp://127.0.0.1:5555")
pull = ctx.socket(zmq.PULL)          # predictor connects its reply socket here
pull.bind("tcp://127.0.0.1:5556")
```

**Rank level, one message.** The `b_rank`, `t_rank_s`, `t_rank_e` arrays from
the `core()` example go straight into the message as `b`, `ts`, `te`. The three
lists line up by position, exactly like the arguments of `overlap()`. FTIO does
the overlap.

```python
b_rank   = [0.0, 0.0, 1000.0, 1000.0, 0.0, 0.0, 1000.0, 1000.0, 0.0, 0.0, 1000.0, 1000.0, 0.0, 0.0]
t_rank_s = [0.5, 0.0, 10.5, 10.0, 20.5, 20.0, 30.5, 30.0, 40.5, 40.0, 50.5, 50.0, 60.5, 60.0]
t_rank_e = [5.0, 4.5, 15.0, 14.5, 25.0, 24.5, 35.0, 34.5, 45.0, 44.5, 55.0, 54.5, 65.0, 64.5]

push.send(msgpack.packb({"ranks": 2, "b": b_rank, "ts": t_rank_s, "te": t_rank_e}))

r = msgpack.unpackb(pull.recv())   # blocks until the prediction comes back
print(r)
```

```python
{
  "phase": 0, "metric": "", "source": "dft",
  "dominant_freq": 0.046153846153846156, "conf": 0.8024962918003531,
  "period": 21.666666666666664,
  "periodicity": [], "amp": [230311.49], "phi": [-3.136],
  "t_start": 0.0, "t_end": 65.0, "total_bytes": 0, "freq": 10.0, "ranks": 0,
  "n_samples": 650, "top_freqs": {}, "candidates": [],
  "burst_widths": [], "duty_cycle": NaN
}
```

`total_bytes` and `ranks` are `0` on the ZMQ path: the wire format carries no
byte count and the rank count is not passed through. `burst_widths` is empty and
`duty_cycle` is `NaN` without `-bw`. The other two examples only print
`r["period"]` and `r["conf"]`.

**Rank level, one message per rank.** Each rank sends only its own I/O phases.
FTIO merges the messages that arrive in the same poll window, then overlaps.
This is the same 14 entries as above, split by rank.

```python
rank0 = {"b": [0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0],
         "ts": [0.5, 10.5, 20.5, 30.5, 40.5, 50.5, 60.5],
         "te": [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]}
rank1 = {"b": [0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0],
         "ts": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
         "te": [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5]}

for rank in (rank0, rank1):
    push.send(msgpack.packb({"ranks": 2, **rank}))
r = msgpack.unpackb(pull.recv())
print("period", r["period"], "s, confidence", r["conf"])
```

```
period 21.666666666666664 s, confidence 0.8024962918003531
```

**Job level.** Send an already overlapped signal as one `{"b": b, "t": t}`
message. FTIO skips the overlap and analyses it as is.

```python
b = [1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0]
t = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]

push.send(msgpack.packb({"b": b, "t": t}))
r = msgpack.unpackb(pull.recv())
print("period", r["period"], "s, confidence", r["conf"])
```

```
period 10.0 s, confidence 1.0
```

See [ZMQ](zmq.md#generic-zmq-format) for the wire format. For the GekkoFS/GLASS
setup (Cargo staging, the GekkoFS 9 field message) see
[`predictor_gekko_zmq.py`](https://github.com/tuda-parallel/FTIO/blob/main/ftio/api/gekkoFs/predictor_gekko_zmq.py).

<p align="right"><a href="#api">⬆</a></p>
