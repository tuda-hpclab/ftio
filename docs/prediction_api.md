# Prediction API

The `Prediction` class (`ftio/freq/prediction.py`) is the primary output object of FTIO.  Every call to `main()` or `core()` returns one `Prediction` per dataset.  This document covers all fields, derived properties, and helper methods.

- [Obtaining a Prediction](#obtaining-a-prediction)
- [Core fields](#core-fields)
- [Derived properties](#derived-properties)
- [Burst-width fields](#burst-width-fields)
- [Helper methods](#helper-methods)
- [Serialisation](#serialisation)
- [Full example](#full-example)

---

## Obtaining a Prediction

```python
from ftio.cli.ftio_core import main

predictions, args = main([
    "ftio",                    # program name (required first element)
    "trace.json",              # trace file
    "-e", "no",                # disable plots
])

p = predictions[0]            # one Prediction per file / I/O mode
```

The list has one entry per analysed dataset.  When multiple files or I/O modes are analysed, each produces its own `Prediction`.

---

## Core fields

All fields are exposed as properties with validation in their setters.

| Property | Type | Description |
|----------|------|-------------|
| `p.source` | `str` | Frequency method used: `"dft"`, `"stft"`, `"wave_disc"`, etc. |
| `p.dominant_freq` | `np.ndarray` | Array of detected dominant frequencies (Hz). Usually length 1; longer when `-n N` is used. |
| `p.conf` | `np.ndarray` | Confidence values in [0, 1] for each entry in `dominant_freq`. |
| `p.amp` | `np.ndarray` | DFT amplitudes for each dominant frequency. |
| `p.phi` | `np.ndarray` | DFT phase angles (radians) for each dominant frequency. |
| `p.periodicity` | `np.ndarray` | Periodicity scores (from `-p` flag); empty if not computed. |
| `p.t_start` | `float` | Start time of the analysed window (seconds). |
| `p.t_end` | `float` | End time of the analysed window (seconds). |
| `p.total_bytes` | `int` | Total bytes transferred in the trace. |
| `p.freq` | `float` | Sampling rate used for discretisation (Hz). |
| `p.ranks` | `int` | Number of I/O ranks in the trace. |
| `p.n_samples` | `int` | Number of samples in `b_sampled`. |
| `p.top_freqs` | `dict` | Dictionary of top-N frequency candidates with metadata. |
| `p.candidates` | `np.ndarray` | Autocorrelation candidate periods (set when `-au` is used). |
| `p.ranges` | `np.ndarray` | Time ranges where the dominant frequency is valid. |
| `p.b_sampled` | `np.ndarray` | Resampled bandwidth signal (set when `-au` or `-ml` is used). |

---

## Derived properties

These are computed on-the-fly from the stored arrays and never stored redundantly.

```python
f, c  = p.get_dominant_freq_and_conf()   # (float, float)
f     = p.get_dominant_freq()            # float — NaN if no detection
idx   = p.get_dominant_index()           # int — index into dominant_freq
f, a, phi = p.get_dominant_freq_amp_phi()  # (float, float, float)
period = 1.0 / p.get_dominant_freq()    # seconds
```

---

## Burst-width fields

Set when `-bw` / `--burst_width` is passed.  See [Approach § Burst Width](approach.md#burst-width-estimation).

| Property | Type | Description |
|----------|------|-------------|
| `p.burst_widths` | `np.ndarray` | Per-period burst width in seconds. Length = number of complete periods. |
| `p.burst_t_starts` | `np.ndarray` | Absolute start time of each burst window (seconds since trace start). Same length as `burst_widths`. |
| `p.burst_width_median` | `float` | Median burst width (seconds). `NaN` if not computed. |
| `p.burst_width_min` | `float` | Minimum burst width across periods. |
| `p.burst_width_max` | `float` | Maximum burst width across periods. |
| `p.duty_cycle` | `float` | `burst_width_median × dominant_freq`.  In [0, 1].  `NaN` if not computed. |

```python
predictions, _ = main(["ftio", "trace.json", "-bw", "-e", "no"])
p = predictions[0]

print(p.burst_widths)        # [0.57, 0.62, 0.55, ...]  seconds per period
print(p.burst_t_starts)      # [3.2, 14.1, 24.9, ...]   absolute start times
print(p.burst_width_median)  # 0.58
print(p.duty_cycle)          # 0.054  (5.4 %)
```

---

## Helper methods

### Reconstructing the dominant signal

```python
# Generate the cosine wave at the dominant frequency
wave = p.get_dominant_wave()       # np.ndarray, same length as b_sampled

# Or supply a custom time axis
import numpy as np
t = p.t_start + np.arange(p.n_samples) / p.freq
wave = p.get_wave(*p.get_dominant_freq_amp_phi(), t)
```

### Convenience getters

```python
p.is_empty()                        # True if no analysis was run
p.get_periodicity()                 # float — dominant freq's periodicity score
p.disp_dominant_freq_and_conf()     # Rich-formatted string for console display
p.get(key)                          # generic attribute getter
p.set(key, value)                   # generic attribute setter
p.set_from_dict({"amp": ..., ...})  # bulk set from dict
```

---

## Serialisation

```python
# Dictionary (numpy arrays preserved)
d = p.to_dict()

# JSON string (numpy arrays converted to lists)
import json
json_str = p.to_json()
print(json.loads(json_str)["dominant_freq"])

# String representation
print(repr(p))   # same as to_dict().__repr__()
```

`to_dict()` keys: `metric`, `source`, `dominant_freq`, `conf`, `periodicity`, `amp`, `phi`, `t_start`, `t_end`, `total_bytes`, `freq`, `ranks`, `n_samples`, `top_freqs`, `candidates`, `burst_widths`, `duty_cycle`.

---

## Full example

```python
from ftio.cli.ftio_core import main
import numpy as np

predictions, args = main([
    "ftio",
    "examples/tmio/ior/parallel/384.msgpack",
    "-bw",
    "-e", "no",
    "-v",
])

p = predictions[0]

# Dominant frequency
f_dom, conf = p.get_dominant_freq_and_conf()
T = 1.0 / f_dom
print(f"Frequency : {f_dom:.4e} Hz")
print(f"Period    : {T:.2f} s")
print(f"Confidence: {conf * 100:.1f} %")

# Time window
print(f"Window    : {p.t_start:.1f} s – {p.t_end:.1f} s")

# Burst width (requires -bw)
if len(p.burst_widths):
    print(f"Burst τ   : median={p.burst_width_median:.2f}s "
          f"[{p.burst_width_min:.2f}, {p.burst_width_max:.2f}]")
    print(f"Duty cycle: {p.duty_cycle * 100:.1f} %")
    print(f"Per-period widths: {np.round(p.burst_widths, 3)}")

# Serialise
import json
with open("result.json", "w") as f:
    f.write(p.to_json())
```
