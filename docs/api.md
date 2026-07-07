# API

The API allows interacting with `ftio` directly, rather than using the command line interface provided in the [
`cli`](/ftio/cli/) folder.
Below are several examples of this.

- [API](#api)
    - [General](#general)
    - [Metric Proxy](#metric-proxy)
    - [GekkoFS with Msgpack/JSON support](#gekkofs-with-msgpackjson-support)
    - [GekkoFS with ZMQ](#gekkofs-with-zmq)

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

## GekkoFS with ZMQ

The file [`predictor_gekko_zmq.py`](https://github.com/tuda-parallel/FTIO/blob/main/ftio/api/gekkoFs/predictor_gekko_zmq.py) deploys
this functionality.

Download and compile the file [`test_mpi.cxx`](https://github.com/tuda-parallel/TMIO/blob/main/test/zmq/test_mpi.cxx)
from the `TMIO` repo. Then either execute it with a single rank (interactive) or with multiple ranks (with sleep).
Simply execute the [Makefile](https://github.com/tuda-parallel/TMIO/blob/main/test/zmq) in the folder:

```sh
make test_mpi

make run_mpi # or run_single_mpi
```

for `ftio`, first navigate to the script and then execute it:

```sh
cd ftio/api/gekkoFs
python3 predictor_gekko_zmq.py 
```

<p align="right"><a href="#api">⬆</a></p>
