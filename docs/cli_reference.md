# CLI Reference

Complete flag reference for all FTIO command-line tools.

- [ftio](#ftio)
- [predictor](#predictor)
- [ioplot](#ioplot)
- [ioparse](#ioparse)

---

## ftio

Offline I/O pattern detection.  Reads a trace file and returns the dominant I/O frequency, confidence, and (optionally) burst-width statistics.

```bash
ftio [options] files [files ...]
```

### Conventions

- Every long option accepts both spellings: `--phase_automaton` and `--phase-automaton` are the same flag.
- Invalid values are rejected when the arguments are parsed, not at runtime. So are impossible combinations, such as `--filter_type bandpass` with a single `--filter_cutoff`, or `-ts` greater than or equal to `-te`.
- `--version` prints the installed FTIO version.

### Input / parsing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `files` | path(s) | — | Trace file(s) or folder(s). Supported formats: JSON, JSONL, MessagePack, Darshan, Recorder, custom TXT. See [File Formats](file_formats.md). |
| `-m`, `--mode` | str | `write_sync` | I/O mode to analyse: `write_sync`, `read_sync`, `write_async`, `read_async`. The reversed spellings (`sync_write`, …) also work, as do the shorthands `w`/`write` and `r`/`read`, which map to `write_sync` and `read_sync` (a value without `async` in it means sync). |
| `-s`, `--source` | str | `unspecified` | Force file-source detection: `tmio` or `custom`. Auto-detected by default. This is the *on-disk* format; it is unrelated to [`--zmq_format`](#online--zmq). |
| `-ts` | float | — | Restrict analysis to times ≥ this value (seconds). |
| `-te` | float | — | Restrict analysis to times ≤ this value (seconds). |
| `-cf`, `--custom_file` | path | — | Python file defining `pattern` and `translate` dicts for custom TXT parsing. See [File Formats § Custom](file_formats.md#parsing-custom-file-formats). |
| `-x`, `--dxt_mode` | str | `DXT_MPIIO` | Darshan DXT layer: `DXT_POSIX` or `DXT_MPIIO`. |
| `-l`, `--limit` | int | — | Limit the number of records read from the trace. |

### Sampling

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-f`, `--freq` | float | `10` | Sampling rate (Hz) for discretising the continuous bandwidth signal. Determines the Nyquist limit. Pass `-1` for auto mode (uses the finest time resolution in the trace). |
| `--memory_limit` | float | `0.5` | Memory ceiling (GB) used in auto-sampling mode (`-f -1`). |

### Frequency analysis

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-tr`, `--transformation` | str | `dft` | Frequency method: `dft`, `stft`, `astft`, `wave_disc`, `wave_cont`. `dwt`/`wave_dwt` alias `wave_disc`, `cwt`/`wave_cwt` alias `wave_cont`. Experimental (requires `pip install "ftio[amd-libs]"`): `efd`, `vmd`. See [Frequency Methods](frequency_methods.md). |
| `-o`, `--outlier` | str | `z-score` | Outlier detection on the spectrum: `z-score`, `dbscan`, `forest`, `lof`, `peak`. `Z-score`, `zscore`, `db-scan`, `db`, `isolation_forest` and `peaks` are accepted aliases. |
| `-t`, `--tol` | float | `0.8` | Tolerance / confidence threshold used by the outlier detector. |
| `-n`, `--n_freq` | int | `0` | Extract up to N dominant frequencies (0 = auto, finds the single dominant). |
| `-np`, `--no-psd` | flag | off | Use amplitude spectrum instead of power-spectral density. |
| `--fourier_fit` | flag | off | Fit multiple sinusoidal components (requires `-n`). |
| `-d`, `--dtw` | flag | off | Dynamic time warping on the top-3 DFT frequencies. |
| `-re`, `--reconstruction` | list | `[]` | Plot reconstruction of up to 10 signal components. |
| `-ce`, `--cepstrum` | flag | off | Enable cepstrum plot for DFT. |
| `-au`, `--autocorrelation` | flag | off | Run autocorrelation in addition to the selected method; merge results. |
| `-p`, `--periodicity_detection` | str | none | Extra periodicity check: `rpde`, `sf` (spectral flatness), `corr`, `ind`. |
| `-le`, `--level` | int | `0` (auto) | Decomposition level for the wavelet transforms. `0` or `auto` computes the maximum level automatically. |
| `--wavelet` | str | `db1` / `morl` | Wavelet family (see `pywt` docs). Defaults depend on `wave_disc` vs `wave_cont`. |
| `--stft_window` | str | `0` (auto) | Window length in samples or seconds (e.g. `20s`). For `stft`: auto uses 4× the detected period. For `astft`: auto uses the cm5 concentration measure; a non-zero value overrides it. |
| `--tfpf` | int | `0` | Time-frequency peak-filtering iterations (ASTFT). |

### Signal filtering (pre-analysis)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--filter_type` | str | none | Apply a Butterworth filter before analysis: `lowpass`, `highpass`, `bandpass`. See [Filtering](filtering.md). |
| `--filter_cutoff` | float(s) | — | Cutoff frequency (Hz). One value for low/high-pass; two values (`lo hi`) for bandpass. |
| `--filter_order` | int | `4` | Butterworth filter order. |

### Burst width & duty cycle

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-bw`, `--burst_width` | flag | off | Estimate per-period burst width and duty cycle. See [Approach § Burst Width](approach.md#burst-width-estimation). |
| `--burst_energy_fraction` | float | `0.95` | Energy fraction the burst window must contain. Must be in (0, 1]. |

### Phase automaton

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--phase-automaton` | flag | off | Enable state-machine phase tracking. See [Phase Automaton](phase_automaton.md). |
| `--pa-method` | str | `ksigma` | Statistical change-point detector: `ksigma`, `cusum`, `ph`, `adwin`, `none`. |
| `--pa-period-ratio RATIO` | float | none | Fire a transition when period changes by more than RATIO (e.g. `1.5`). |
| `--pa-no-rank-trigger` | flag | off | Disable automatic transition on rank-count change. |
| `--pa-export PATH` | path | `./phase_automaton.json` | Where to write the exported automaton JSON on exit. |

### Online / ZMQ

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--zmq` | flag | off | Input via ZeroMQ instead of a file (suppress HTML open). |
| `--zmq_address` | str | `*` | ZMQ bind address. |
| `--zmq_port` | str | `5555` | ZMQ input port. A bare port number, e.g. `5555` — not a full endpoint. |
| `--zmq_port_reply` | str | `5556` | ZMQ reply port (dominant frequency). |
| `--zmq_format` | str | `direct` | Encoding of the ZMQ payload: `direct` (raw bandwidth / start / end triples) or `tmio` (msgpack-encoded TMIO buffer). `--zmq_source` is a legacy alias. Unrelated to `--source`. |
| `--gui` | flag | off | Forward results to the `ftio-gui` dashboard (start separately). |

### Output & plotting

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-e`, `--engine` | str | `plotly` | Plot engine: `plotly` (HTML), `matplotlib`, or `no` (disable plots). `plot`/`mat` are accepted short spellings; `plotly_no_paper`/`mat_no_paper` disable the paper layout. |
| `-rp`, `--runtime_plots` | flag | off | Show each figure immediately at runtime. |
| `-r`, `--render` | str | `dynamic` | Rendering mode: `dynamic` or `static`. |
| `-v`, `--verbose` | flag | off | Verbose console output. |
| `--sum` / `--no_sum` | flag | on | Include / exclude the summed (application-level) bandwidth plot. |
| `--avr` / `--no_avr` | flag | on | Include / exclude the average bandwidth plot. |
| `--ind` / `--no_ind` | flag | on | Include / exclude per-rank individual traces. |
| `--debounce` | flag | off | Serialise predictions (prevents concurrent writes in online mode). |
| `-ml`, `--machine_learning` | flag | off | Enable ML-based analysis (API use only). |
| `-w`, `--window_adaptation` | str | none | Online window strategy (predictor mode): `frequency_hits`, `data`, `adwin`, `cusum`, `ph`. |
| `-hi`, `--hits` | int | `3` | Frequency hits required before adapting the window. |

### Examples

```bash
# Basic offline analysis
ftio trace.json

# STFT with no plots, verbose
ftio trace.msgpack -tr stft -e no -v

# Restrict to write data, time window 0–100 s
ftio trace.json -m write_sync -ts 0 -te 100

# Bandpass pre-filter, then DFT
ftio trace.json --filter_type bandpass --filter_cutoff 0.05 0.5

# Burst width with 90 % energy fraction
ftio trace.json -bw --burst_energy_fraction 0.9

# Multiple files
ftio file1.json file2.json -e no
```

---

## predictor

Online I/O prediction.  Monitors a trace file and re-runs `ftio` each time new data arrives.

```bash
predictor [options] files [files ...]
```

`predictor` accepts **all** `ftio` flags plus the following predictor-specific ones:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-w`, `--window_adaptation` | str | none | Window adaptation strategy. `frequency_hits` shifts the window after N frequency hits; `data` shifts on new data; `adwin`, `cusum`, `ph` use change-point detection (see [Change Point Detection](change_point_detection.md)). |
| `-hi`, `--hits` | float | `3` | Number of consecutive frequency hits before shifting the window. |
| `--debounce` | flag | off | Allow only one in-flight prediction at a time. |
| `--gui` | flag | off | Stream results to the `ftio-gui` live dashboard. |
| `--phase-automaton` | flag | off | Track phase transitions across predictions (see [Phase Automaton](phase_automaton.md)). |

### Examples

```bash
# Basic online prediction
predictor live.jsonl -f 100 -e no

# With frequency-hit window adaptation
predictor live.jsonl -f 100 -w frequency_hits -hi 5

# Phase transitions via change-point detection
predictor live.jsonl -f 100 --phase-automaton --pa-method ksigma

# Live dashboard
ftio-gui &
predictor live.jsonl -f 100 --gui
```

---

## ioplot

Generate interactive Plotly plots from trace files without running frequency analysis.

```bash
ioplot [options] files [files ...]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `files` | path(s) | — | Trace file(s) or folder(s). |
| `-m`, `--mode` | str | all | I/O mode to display. |
| `-s`, `--source` | str | auto | Force source detection (`tmio`, `custom`). |
| `-e`, `--engine` | str | `plotly` | Engine: `plotly`, `dash`, `matplotlib`, `no`. |
| `-z`, `--zoom` | float | — | Upper y-axis zoom limit. |
| `-nt`, `--no-threaded` | flag | off | Disable multi-threaded file loading. |
| `-r`, `--render` | str | `dynamic` | Rendering mode. |
| `--n_shown_samples` | int | `20000` | Max samples per trace (Dash only). |
| `--merge_plots` | flag | off | Merge all modes into one plot (Dash only). |
| `--no_disp` | flag | off | Write HTML without opening a browser. |
| `--sum` / `--no_sum` | flag | on | Show / hide summed trace. |
| `--avr` / `--no_avr` | flag | on | Show / hide average trace. |
| `--ind` / `--no_ind` | flag | on | Show / hide per-rank traces. |
| `-cf`, `--custom_file` | path | — | Custom parsing spec (see [File Formats](file_formats.md#parsing-custom-file-formats)). |
| `-x`, `--dxt_mode` | str | `DXT_MPIIO` | Darshan DXT layer. |
| `-l`, `--limit` | int | — | Limit number of records read. |

### Examples

```bash
ioplot trace.json
ioplot *.msgpack -e dash
ioplot . --no_disp   # write HTML files without opening browser
```

---

## ioparse

Parse and merge trace files into an [Extra-P](https://github.com/extra-p/extrap)-compatible format.

```bash
ioparse [options] files [files ...]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `files` | path(s) | — | Trace file(s) or folder(s). |
| `-m`, `--mode` | str | — | I/O mode to extract. |
| `-s`, `--source` | str | auto | Force source detection. |
| `--scale` | flag | off | Scale the Y-axis values. |
| `--sum` / `--no_sum` | flag | on | Include / exclude summed trace. |
| `--avr` / `--no_avr` | flag | on | Include / exclude average trace. |
| `--ind` / `--no_ind` | flag | on | Include / exclude per-rank traces. |
| `-cf`, `--custom_file` | path | — | Custom parsing spec. |
| `-x`, `--dxt_mode` | str | `DXT_MPIIO` | Darshan DXT layer. |
| `-l`, `--limit` | int | — | Limit number of records read. |

### Example

```bash
ioparse .               # parse all traces in current directory
ioparse *.json -m write_sync
```
