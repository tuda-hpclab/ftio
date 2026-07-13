# FTIO Documentation

FTIO (Frequency Techniques for I/O) detects and predicts periodic I/O patterns in HPC applications using frequency analysis. It provides offline trace analysis, online prediction, and a set of companion utilities.

## Documentation Map

| Document | Contents |
|----------|----------|
| [Approach](approach.md) | Core algorithm — offline detection, online prediction, burst-width estimation |
| [CLI Reference](cli_reference.md) | Every flag for `ftio`, `predictor`, `ioplot`, and `ioparse` with type, default, and example |
| [Frequency Methods](frequency_methods.md) | DFT, STFT, ASTFT, discrete/continuous wavelets, autocorrelation, periodicity detection — when to use each |
| [Filtering](filtering.md) | Pre-analysis signal filtering: lowpass, highpass, bandpass, TFPF |
| [Phase Automaton](phase_automaton.md) | State-machine model for detecting I/O phase transitions in online prediction |
| [Prediction API](prediction_api.md) | All fields and methods of the `Prediction` class; how to consume results programmatically |
| [File Formats & Examples](file_formats.md) | Supported input formats (JSON, JSONL, MessagePack, Darshan, Recorder, custom), plus an overview of example traces |
| [Tools](tools.md) | `ioplot`, `ioparse`, `convert_trace`, `parallel_trace_analysis` |
| [API](api.md) | Python API — calling FTIO from code, GekkoFS integration, Metric Proxy |
| [ZMQ](zmq.md) | ZeroMQ-based online input: avoiding intermediate files |
| [Predictor](predictor.md) | Online prediction; GLASS/GekkoFS ingestion, keep-up vs the flush interval, parallel fan-out backends |
| [Change Point Detection](change_point_detection.md) | ADWIN, CUSUM, Page-Hinkley change-point algorithms for `predictor` |
| [Machine Learning Models](ml_models.md) | ML-based I/O classification (`-ml` flag) |
| [FTIO Server](ftio_server.md) | HTTP server wrapper around the FTIO CLI |
| [Metric Proxy ZMQ](metric_proxy_zmq.md) | FTIO ZMQ server for Metric Proxy integration |
| [Contributing](contributing.md) | Contribution guidelines for external contributors and students |

## Quick Start

### Install

```bash
git clone https://github.com/tuda-parallel/FTIO.git
cd FTIO
make install          # basic install
source .venv/bin/activate
```

### Analyse a trace offline

```bash
ftio trace.json                    # DFT, plotly output
ftio trace.msgpack -e no           # no plots, console only
ftio trace.json -tr stft           # use STFT instead of DFT
ftio trace.json -bw                # also estimate burst width / duty cycle
```

### Predict online

```bash
predictor live.jsonl -f 100 -e no
predictor live.jsonl -f 100 -w frequency_hits --phase-automaton
```

### Visualise a trace

```bash
ioplot trace.json
ioplot *.msgpack
```

### Use as a Python library

```python
from ftio.cli.ftio_core import main

predictions, args = main(["ftio", "trace.json", "-e", "no"])
p = predictions[0]
print(f"Dominant frequency: {p.get_dominant_freq():.3e} Hz")
print(f"Period:             {1 / p.get_dominant_freq():.2f} s")
print(f"Confidence:         {p.get_dominant_freq_and_conf()[1] * 100:.1f} %")
```

See [Prediction API](prediction_api.md) for the full reference.

## Repository Layout

```
ftio/
  cli/          # entry points: ftio_core.py (offline), predictor.py (online)
  freq/         # frequency analysis: _dft_workflow.py, _stft_workflow.py, …
  parse/        # input parsing: args.py, scales.py, parse_*.py
  plot/         # plotting helpers (Plotly + Matplotlib)
  prediction/   # online prediction merging, probability, shared state
  processing/   # output formatting and display
  api/          # integrations: GekkoFS, Metric Proxy, trace analysis
docs/           # this documentation
examples/       # example traces and scripts
test/           # pytest suite
```
