# Predictor (online prediction)

The `predictor` runs FTIO continuously on a live stream instead of a static
trace: it listens for bandwidth data (a file it re-reads, or a ZMQ socket),
runs a prediction whenever new data arrives, and emits the dominant period /
frequency and confidence. It is the online counterpart to the offline `ftio`
command.

- [Predictors at a glance](#predictors-at-a-glance)
- [Running](#running)
- [Concurrency of predictions](#concurrency-of-predictions)
- [GLASS: GekkoFS ingestion and parallel fan-out](#glass-gekkofs-ingestion-and-parallel-fan-out)

## Predictors at a glance

| Predictor | Entry point | Source | Typical use |
|-----------|-------------|--------|-------------|
| Generic ZMQ / file | `predictor` (`ftio.cli.predictor`) | file or ZMQ (TMIO/msgpack) | standalone online prediction |
| GLASS / GekkoFS | `ftio.api.gekkoFs.predictor_gekko_zmq` | ZMQ from GekkoFS servers | ad-hoc FS + Cargo stage-out, launched by the JIT script |
| Metric Proxy | `ftio.api.metric_proxy.proxy_zmq` | ZMQ request/reply | stateless, per-request prediction |

See also [ZMQ interface](zmq.md), [FTIO server](ftio_server.md), and
[Metric Proxy](metric_proxy_zmq.md).

## Running

```bash
predictor --zmq                        # listen on a ZMQ socket
predictor -f 10 <file>                 # re-read a file at 10 Hz sampling
```

Common options: `-f` sampling rate (Hz), `-m` mode (e.g. `write_sync`), `-e`
engine (`no` to disable plots online). See `predictor -h`.

## Concurrency of predictions

By default a new prediction process is spawned for each batch of drained
messages while the loop keeps listening, so several predictions can run
concurrently. Two flags bound this:

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-predictions K` | `0` (unlimited) | Cap concurrent prediction processes. When the cap is reached, a new prediction waits for the **oldest** in-flight one, so predictions can't pile up and oversubscribe the cores. |
| `--debounce` | off | One prediction at a time (equivalent to `--max-predictions 1`), with an immediate follow-up for messages that arrived meanwhile. |

Note each fan-out prediction also spawns `--ingest-workers` sub-processes, so at
high load the single-process ingest actually sustains a *higher* prediction rate
than the fan-out (it uses fewer cores per prediction) — see the stress test in
`jit/experiments/README.md`. A sensible cap is roughly `cores / ingest-workers`.

## GLASS: GekkoFS ingestion and parallel fan-out

In GLASS the predictor is driven by GekkoFS: each flush round delivers one
message per server over a `ZMQ_PUSH` socket, in the 9-field format
(`msgpack_util.hpp`): `flush_t, hostname, pid, io_type, start_t[us], end_t[us],
req_size, total_iops, total_bytes` (`io_type` = `"w"`/`"r"`). Parsing those
messages and overlapping the intervals into the **application-level bandwidth**
is the only stage whose cost grows with the server count.

### When it matters

A single process handles ingestion within a 1 s flush up to ~512 servers
(~0.6 s of compute even at 1024). Under the multi-second flush intervals used in
practice, one process is enough. Parallel ingestion matters only at the extreme
(≈1000+ servers with a sub-second flush).

### Turning fan-out on / off

| Flag | Default | Meaning |
|------|---------|---------|
| `--ingest-workers N` | `1` | Worker processes for parse+overlap. `1` = single process (off); `>1` fans out, capped to the CPU budget. |
| `--ingest-backend MODE` | `process-resample` | Fan-out mode when workers > 1: `process-resample`, `process`, `thread`. Ignored when workers is `1`. |

```bash
# off (default, unchanged single-process behaviour)
predictor --zmq

# on — recommended fast path
predictor --zmq --ingest-workers 4

# on — keep full-resolution signal (slower)
predictor --zmq --ingest-workers 4 --ingest-backend process
```

### Why fan out at all

**Latency, not speed.** GekkoFS posts a flush round every `T` seconds (GLASS:
~5 s); ingesting one round costs `L` seconds, growing ~linearly in
`servers x events/msg`. While `L < T` the prediction for a round is ready before
the next one lands. Once `L > T` it never is: predictions pile up (or, with
`--max-predictions` / `--debounce`, the backlog moves into the ZMQ queue or
rounds get skipped), and the flush trigger starts firing for phases that are
already over. Nothing crashes — the prediction just goes **stale**.

So the question is never "how much faster", it is "**is `L` still under `T`**".

**Latency per round** (best-of-10, 4 workers,
`jit/experiments/exp_a_scaling.py`; `%` = fraction of GekkoFS's **5 s** flush
consumed — over 100% cannot keep up):

| servers | 1024 events/msg | | 4096 events/msg | |
|--------:|----------------:|---|----------------:|---|
|         | single | `process-resample` | single | `process-resample` |
| 512     | 371 ms (7%)  | 186 ms (4%)  | 1606 ms (32%) | 647 ms (13%)  |
| 1024    | 809 ms (16%) | 349 ms (7%)  | 3520 ms (70%) | 1417 ms (28%) |
| 2048    | 1840 ms (37%)| 672 ms (13%) | 8262 ms (**165%**) | 3213 ms (64%) |

Only one cell in the entire grid — 2048 servers at 4096 events/msg — fails to
keep up at a 5 s flush, and the fan-out rescues exactly that cell (165% → 64%).
That is the whole practical case for it.

**Where that puts the wall** — largest server count that still fits the flush
interval (`T / slope`; measured slope is 0.90 ms/server single, 0.33 resample, at
1024 events/msg):

| flush `T` | single (off) | `process-resample` |
|----------:|-------------:|-------------------:|
| 1 s       | ~1,100 servers | ~3,000 servers   |
| **5 s** *(GekkoFS actual)* | **~5,600 servers** | ~15,200 servers |
| 10–15 s   | ~11,000–16,700 | ~30,000–45,000   |

**At GekkoFS's real flush interval a single process handles thousands of
servers**, so the default (`--ingest-workers 1`) is the right choice for almost
any deployment. Counts above ~2048 are extrapolated from the measured slope,
which drifts up slightly with N — treat them as upper bounds.

**And the speed-up behind it:** a flat **2.0–2.8x (mean 2.5x)** across the whole
grid — constant, because both paths scale linearly. That means the fan-out
*defers* the wall by a predictable factor; it does not remove it. 2.5x of an `L`
already past `T` is still past `T`, which is why the latency table above is the
one to read first.

Practical rule: leave it off (`--ingest-workers 1`, the default) while you are
under the wall — it spends 4 cores per prediction to return a fresher answer you
did not need. Turn it on when `L` approaches `T`.

### Backends

4 workers, 1024 events/server, best-of-10:

| servers | single (off) | `thread` | `process` | `process-resample` |
|--------:|-------------:|---------:|----------:|-------------------:|
| 512     | 371 ms       | 0.4x     | 0.2x      | **2.0x (186 ms)**  |
| 1024    | 809 ms       | 0.4x     | 0.2x      | **2.3x (349 ms)**  |

- **`process-resample` (default when workers > 1):** each worker parses+overlaps
  its share and resamples the partial onto a shared grid (at the `-f` rate)
  before returning, so only a small vector crosses the process boundary — no
  big-array IPC, no GIL contention → ~2.5x across the grid. Trade-off: the
  per-round result is the resampled signal, not the full-resolution step function
  (the DFT resamples anyway, so prediction is unaffected).
- **`process`:** full resolution but pickles large arrays back → IPC-bound, ~5x
  slower than a single process.
- **`thread`:** shares memory but parse/overlap are GIL-bound → ~2.5x slower than
  a single process.

**The fan-out speeds up each prediction, not everything.** A 2.3x speed-up on 4
workers is 59% parallel efficiency, so one prediction costs 688 core-ms instead
of 403 — latency down, total CPU **up 1.7x**. With 16 cores that caps the
sustainable posting rate at 23.3 Hz, versus 39.7 Hz single-process.

In practice this rarely decides anything: the core-occupancy ceiling is ~30 Hz
against GekkoFS's 0.2 Hz, so posting rate is never the binding constraint. See
`exp_d_stress.py`.

**Rule of thumb:** leave the fan-out off. Turn it on only if `L` (the ingest
latency printed per prediction) approaches your flush interval — which, at a 5 s
flush, means several thousand servers.

The fan-out is exact (any worker count gives the same bandwidth, up to
resampling) because the application-level bandwidth is an additive sum of
per-server box functions — see `test_parallel_ingest.py`.

### GekkoFS-side settings

- **flush interval** — how often each server flushes. Ingestion must finish
  within one interval; a larger interval is the cheapest lever if a single
  process falls behind.
- **flush type** — GLASS uses the socket (ZMQ) flush type.
- FTIO matches the message `io_type` against `-m` (it compares the first
  character, so `-m write_sync` matches GekkoFS `"w"`).

### Scope

These flags are **intentionally GekkoFS/GLASS only**
(`ftio.api.gekkoFs.ftio_gekko.run`). The fan-out addresses the GLASS pattern of
*hundreds of servers pushing one message each per round*; the generic `predictor`
listens to a single aggregated stream and has no per-round fan-out to exploit, so
the flags are accepted but not used there. This is a relevance boundary, not a
missing feature.

### Reproducing

```bash
python -m ftio.api.gekkoFs.jit.experiments.exp_a_scaling --servers 512 1024 --workers 1 4
python -m ftio.api.gekkoFs.jit.experiments.profile_ingest
```

See `ftio/api/gekkoFs/jit/experiments/README.md`.
