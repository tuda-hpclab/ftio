# File Formats and Tools
Below, we describe the supported file formats. Aside from these formats, `ftio` and `predictor` support ZMQ as described [here](/docs/zmq.md), which avoids creating intermediate files. Furthermore, `ftio` also provides an API to different tools (e.g., [online prediction over ZMQ](/docs/api.md#online-prediction-over-zmq)) and allows easy and direct use of the Python functions as described and demonstrated [here](/docs/api.md#general).

The currently supported file Formats are:
- [File Formats and Tools](#file-formats-and-tools)
	- [JSON](#json)
		- [Custom JSON Files](#custom-json-files)
			- [JSON File With Rank-Level Metrics](#json-file-with-rank-level-metrics)
			- [JSON File With Application-Level Metrics](#json-file-with-application-level-metrics)
			- [Minimal JSON File](#minimal-json-file)
		- [TMIO JSON Files](#tmio-json-files)
	- [JSONL](#jsonl)
		- [TMIO JSONL Files](#tmio-jsonl-files)
		- [Custom JSONL Files](#custom-jsonl-files)
	- [MessagePack](#messagepack)
		- [TMIO MessagePack Files](#tmio-messagepack-files)
		- [Custom MessagePack Files](#custom-messagepack-files)
	- [Darshan](#darshan)
	- [Recorder](#recorder)
	- [Parsing Custom File Formats](#parsing-custom-file-formats)

All units in the files are by default SI units. 

## JSON
`ftio` supports JSON files as an input. These files can be either [custom-generated](/docs/file_formats.md#custom-json-files) or from [TMIO](https://github.com/tuda-parallel/TMIO). Below, we explain both and provide links to some examples. 

### Custom JSON Files

`ftio` supports custom JSON files that can be either at the rank, node, or application level. Bellow we present JSON Files with [rank-](/docs/file_formats.md#json-file-with-rank-level-metrics) or [application-](/docs/file_formats.md#json-file-with-application-level-metrics)-level metrics.

> [!note]
> `ftio` work on the application level as mentioned [here](/docs/approach.md#offline-detection). Internally the tool overlaps 
> rank-level metrics to obtain application-level metrics. If the metrics are not at the application-level (i.e., if the timestamps are not sorted in an  increasing order), the syntax for rank-level metrics should be used.

#### JSON File With Rank-Level Metrics
The JSON file provided to `ftio` should have the following structure:

```python
{
	"write_sync":{
	"total_bytes": 1.68e+09,
	"number_of_ranks": 4, 
	"bandwidth": {
		"b_rank_avr": [
			 18854940.934542, 35493987.414814, 26656768.595012, 20879002.716997,
			 28234996.909554, 21386514.858547, 26588927.552098, 18408319.012778,
			 66591429.795722, 64718227.754890, 20303409.879130, 39347431.382128,
			 21904017.847817, 26657866.416951, 38344190.339886, 67774121.679019,
			 24178995.499522, 24457899.521490, 77675298.475351, 71081178.788930,
			 69537332.852923, 72595355.489592, 20328465.984519, 22970008.962767,
			 32817028.455917, 18802080.851357, 38077100.664442, 27360825.411258,
			 39622620.596783, 35450821.093752, 68597882.969851, 34332203.448823],
		"t_rank_s": [
			 0.060111, 2.920865, 5.801426, 8.470562,
			 11.072364, 13.558426, 16.081736, 18.744577,
			 0.060017, 2.920780, 5.801340, 8.470471,
			 11.072274, 13.558340, 16.081650, 18.744492,
			 0.059990, 2.920736, 5.801290, 8.470427,
			 11.072234, 13.558294, 16.081607, 18.744447,
			 0.060026, 2.920790, 5.801344, 8.470485,
			 11.072288, 13.558349, 16.081659, 18.744501],
		"t_rank_e": [
			 2.840750, 4.397983, 7.768236, 10.981639,
			 12.929237, 16.009915, 18.053565, 21.592681,
			 0.847337, 3.730889, 8.383606, 9.802929,
			 13.465844, 15.525069, 17.448971, 19.518073,
			 2.228352, 5.064370, 6.476264, 9.208017,
			 11.826200, 14.280500, 18.660690, 21.026937,
			 1.657636, 5.709247, 7.178256, 10.386685,
			 12.395491, 15.037266, 16.845951, 20.271603]
		}
	}

}
```
The associated example is provided [here](https://github.com/tuda-parallel/FTIO/tree/main/examples/custom/JSON/custom.json), which can be simply provided to `ftio` using:
```bash
ftio custom.json 
```

The `-m`flags allows to pick the desired mode from the trace. In case the JSON file only contains a single mode (as in this example), the `-m` flag is not strictly needed:
```bash
# Explicitly specify the mode
ftio custom.json -m write_sync
```

`ftio` automatically detects the source of the JSON file. To skip this test, the source can be specified with the `-s|--source` flag, that is, for tmio `-s "tmio"` or custom `-s "custom"`. 

Several fields in the above example are self-explanatory. The metrics with `_rank_`in their names represent the rank-level metrics. As FTIO operates on the application level, these metrics are internally overlapped. 

#### JSON File With Application-Level Metrics

The application-level metrics can also be provided directly. As example 
[custom_app.json](https://github.com/tuda-parallel/FTIO/tree/main/examples/custom/JSON/custom_app.json) shows, the previous three rank-level metrics for the bandwidth (i.e., `*_rank_*`), are simply replaced by two metrics `b_overlap_avr` and `t_overlap`:
```python
{
	"write_sync":{
	"total_bytes": 1.68e+09,
	"number_of_ranks": 4, 
	"bandwidth": {
		"b_overlap_avr": [1000000000,2000000000,5000000000,2000000000,1000000000,0],
		"t_overlap": [1, 2, 4, 6, 7, 8]
		}
	}
}
```

With:
- `b_overlap_avr` representing the bandwidth at the application-level
- `t_overlap` representing the time when new values for the application-level bandwidth are attained. This means for the above example, that the bandwidth at time 1 s was 1 GB, at 2 s changed to 2 GB, and so on.

#### Minimal JSON File 

Several fields shown in the JSON Files with [rank-](/docs/file_formats.md#json-file-with-rank-level-metrics) or [application-](/docs/file_formats.md#json-file-with-application-level-metrics)-level metrics are optional. In a simpler form, a JSON file only needs the field `bandwidth`, and thus can have the following form:
```python
{
	"bandwidth": {
		"b_rank_avr": [
			 18854940.934542, 35493987.414814, 26656768.595012, 20879002.716997,
			 28234996.909554, 21386514.858547, 26588927.552098, 18408319.012778,
			 66591429.795722, 64718227.754890, 20303409.879130, 39347431.382128,
			 21904017.847817, 26657866.416951, 38344190.339886, 67774121.679019,
			 24178995.499522, 24457899.521490, 77675298.475351, 71081178.788930,
			 69537332.852923, 72595355.489592, 20328465.984519, 22970008.962767,
			 32817028.455917, 18802080.851357, 38077100.664442, 27360825.411258,
			 39622620.596783, 35450821.093752, 68597882.969851, 34332203.448823],
		"t_rank_s": [
			 0.060111, 2.920865, 5.801426, 8.470562,
			 11.072364, 13.558426, 16.081736, 18.744577,
			 0.060017, 2.920780, 5.801340, 8.470471,
			 11.072274, 13.558340, 16.081650, 18.744492,
			 0.059990, 2.920736, 5.801290, 8.470427,
			 11.072234, 13.558294, 16.081607, 18.744447,
			 0.060026, 2.920790, 5.801344, 8.470485,
			 11.072288, 13.558349, 16.081659, 18.744501],
		"t_rank_e": [
			 2.840750, 4.397983, 7.768236, 10.981639,
			 12.929237, 16.009915, 18.053565, 21.592681,
			 0.847337, 3.730889, 8.383606, 9.802929,
			 13.465844, 15.525069, 17.448971, 19.518073,
			 2.228352, 5.064370, 6.476264, 9.208017,
			 11.826200, 14.280500, 18.660690, 21.026937,
			 1.657636, 5.709247, 7.178256, 10.386685,
			 12.395491, 15.037266, 16.845951, 20.271603]
		}
}
```

In case the bandwidth filed is omitted, and only two vectors are provided, `ftio` assumes these metrics are at the *application level*. Furthermore, it searches for the first match of `b*` and `t*` to map `b_overlap_avr` and `t_overlap`, respectively. Hence, this form also works with `ftio`:
```python
{
	"b": [1000000000,2000000000,5000000000,2000000000,1000000000,0],
	"t": [1, 2, 4, 6, 7, 8]
}
```

Alternatively, the JSON files can be loaded into `ftio`manually, and the required fields can be passed through the [API](/docs/api.md#general).

### TMIO JSON Files
In the offline mode of TMIO, a JSON file is generated that contains 7 fields. Besides the standard fields: *read sync*, *write sync*, _read async_, and _write async_*, TMIO also generated fields for the required bandwidth in case of asynchronous I/O. Furthermore, the I/O time is also captured. Consequently, a JSON file has the following structure:
```python
{
	"read_sync":{
	"total_bytes": 1.34e+09,
	"max_bytes_per_rank": 1.68e+08,
	"max_bytes_per_phase": 2.10e+06,
	"max_io_phases_per_rank": 80,
	"total_io_phases": 640,
	"max_io_ops_in_phase": 1,
	"max_io_ops_per_rank": 80,
	"total_io_ops": 640,
	"number_of_ranks": 8,
	"bandwidth": {
		"weighted_harmonic_mean": 1.20e+09,
		"harmonic_mean": 1.20e+09,
		"arithmetic_mean": 1.31e+09,
		"median": 1.18e+09,
		"max": 5.28e+09,
		"min": 7.35e+08,
		"b_rank_avr": [1582905557.128160, 1338049664.395335, 1216357988.172627, 1263707703.960846, 1183417554.729751, 1149204355.590840, 1213748335.340853, 1386897971.519520, ...],
		"b_rank_sum": [1582905557.128160, 1338049664.395335, 1216357988.172627, 1263707703.960846, 1183417554.729751, 1149204355.590840, 1213748335.340853, 1386897971.519520, ...],
		"t_rank_s": [ 2.575119, 2.576482, 2.578066, 2.579800, 2.581467, 2.583249, 2.585093, 2.586831, ...],
		"t_rank_e": [2.576444, 2.578050, 2.579791, 2.581460, 2.583239, 2.585074, 2.586821, 2.588343, ...],
		⋮
		}},
	"read_async_t":{...},
	"read_async_b":{...},
	"write_async_t":{...},
	"write_async_b":{...},
	"write_sync":{...},
	"io_time":{"delta_t_agg": 1.58e+02,"delta_t_agg_io": 1.43e+02, ...}
}
```

<p align="right"><a href="#file-formats-and-tools">⬆</a></p>

## JSONL

### TMIO JSONL Files
An example with 8 ranks is located [here](https://github.com/tuda-parallel/FTIO/tree/main/examples/tmio/JSONL/8.jsonl)

### Custom JSONL Files
TBD

<p align="right"><a href="#file-formats-and-tools">⬆</a></p>

## MessagePack
### TMIO MessagePack Files
[`384.msgpack`](/examples/tmio/ior/parallel/384.msgpack) provides an example with 384 ranks and IOR. This file was generated using TMIO (online or offline).

### Custom MessagePack Files
TBD

<p align="right"><a href="#file-formats-and-tools">⬆</a></p>

## Darshan
`ftio` supports standard Darshan files. By default, `ftio` first tries to read the DXT trace. 
The DXT mode can be specified with the flag `-x DXT_MODE`, where `DXT_MODE` is either `DXT_POSIX` or `DXT_MPIIO` (default). If the file does not contain a DXT trace, `ftio` tires to read the heat map. In both cases, `ftio`uses pydarshan to read the file.

As the collected values are per rank level, `ftio` internally overlaps them to obtain the application-level bandwidth. An example trace can be downloaded from the website: https://hpcioanalysis.zdv.uni-mainz.de/. For example, [Nek5000 with 2048](https://hpcioanalysis.zdv.uni-mainz.de/trace/64ed13e0f9a07cf8244e45cc) ranks executed on the Mogon II cluster. After downloading, rename the file to `nek_2048.darshan`. `ftio` can now be called on the complete trace via:

```bash
ftio nek_2048.darshan
```

pass the `-e no` flag to avoid generating plots and just directly obtaining the result from `ftio` on the command line:

``` bash
ftio nek_2048.darshan -e no
```

To limit the time window to 56,000 s, pass the `-te 56000` flag as follows:

``` bash
ftio nek_2048.darshan  -te 56000 -e no
```

<p align="right"><a href="#file-formats-and-tools">⬆</a></p>

## Recorder
Simply specify the folder where the traces are located:
```bash
ftio folder
```

<p align="right"><a href="#file-formats-and-tools">⬆</a></p>


## Parsing Custom File Formats
`ftio` supports parsing custom file formats using regex. These values can be scaled in case they are not in SI units. 
The file must currently have the `txt`extension. 
For that, `ftio` provides two dictionaries that must be provided (pattern and translate) in a custom file similar to the [convert](/ftio/parse/custom_patterns.py) function. 

1. _**pattern** (dict[str, str])_: dictionary containing the name and a regex expression to find the custom pattern.
2. _**translate** (dict[str, tuple[str, (optional)float]])_: dictionary containing matching filed from [sample.py](/ftio/parse/sample.py) and the matching name from the pattern. The unit can be optionally specified

An example is provided in [example/txt](/examples/custom/txt/). Navigate to this folder. There, a file called [`custom_input.py`](/examples/custom/txt/custom_input.py) is located and contains the following: 

```python
pattern = {
	"avg_thruput_mib": r"avg_thruput_mib:\s+\[([\d.\d,\s]+)\]",
	"end_t_micro": r"end_t_micro:\s+\[([\d,\s]+)\]",
	"start_t_micro": r"start_t_micro:\s+\[([\d,\s]+)\]",
	"total_bytes": r"total_bytes:\s+(\d+)",
	"total_iops": r"total_iops:\s+(\d+)",
	}

# Define map according to sample.py class, along with the scale if any:
# ftio_field: ("custom_name", scale)
# ftio unit are default in bytes, b/s, ...
# scale applies ftio_field = custom_name*scale
translate = {
	"bandwidth": {
		"b_rank_avr": ("avg_thruput_mib",1.07*1e+6),
		"t_rank_e": ("end_t_micro", 1e-3),
		"t_rank_s": ("start_t_micro", 1e-3)
		},
	"total_bytes": "total_bytes",
	"max_io_ops_per_rank": "total_iops"
	}
```

The fields in translate must match the fields in [sample.py](/ftio/parse/sample.py). The field bandwidth as its three fields (`b_rank_avr`, `t_rank_e`, and `t_rank_s`) are mandatory. Though these fields are indexed with `_rank_`, they can be on any level higher than rank. Note that in case the application-level bandwidth is provided, `t_rank_e` is not needed. 
Other supported fields include (from [sample.py](/ftio/parse/sample.py)):
```python
# cutout from /ftio/parse/sample.py
⋮
def __init__(self, values, io_type, args):
        self.type # 'read_sync', 'read_async_t', 'write_async_t','write_sync'                        
        self.max_bytes_per_rank # maximum bytes transferred per rank per phase
        self.max_bytes_phase # maximum bytes transferred per rank during all phases
        self.total_bytes # total transferred bytes
        self.max_io_phases_per_rank      # maximum I/O phases
        self.total_io_phases             # total I/O phases
        self.max_io_ops_per_rank         # maximum I/O operations per rank
        self.max_io_ops_in_phase         # maximum I/O operation per phase
        self.total_io_ops                # Total I/O operations
        self.number_of_ranks             # number of ranks that did I/O
        self.bandwidth                   # Dictionary containing the fileds b_rank_avr, t_rank_e, and t_rank_s

⋮
```

Next, the fields indicated by [`custom_input.py`](/examples/custom/txt/custom_input.py) match the fields in the file [`2.txt`](/examples/custom/txt/2.txt), which contains, for example:

```
avg_thruput_mib: [0.0,0.0,1000.0,1000.0,0.0,0.0,1000.0,1000.0,0.0,0.0,1000.0,1000.0,0.0,0.0]
start_t_micro: [0500,0000,10500,10000,20500,20000,30500,30000,40500,40000,50500,50000,60500,60000]
end_t_micro: [5000,4500,15000,14500,25000,24500,35000,34500,45000,44500,55000,54500,65000,64500]
hostname: "XXX"
pid: 2063022
total_bytes: 15000
total_iops: 1024
```

Finally, `ftio` can be executed with the `-cf` flag pointing to the location of `custom_input.py`.
For this example, all files are in the current working directory [example/txt](/examples/custom/txt/).
Thus, `ftio` can be simply executed by:
```sh
ftio 2.txt -cf custom_input.py
```
<p align="right"><a href="#file-formats-and-tools">⬆</a></p>

## Examples

The [`examples/`](/examples/) directory contains ready-to-use trace files and scripts that cover all supported formats and FTIO features.

### TMIO traces

Generated by [TMIO](https://github.com/tuda-parallel/TMIO) during real HPC application runs.

| Path | Format | Description | Quick start |
|------|--------|-------------|-------------|
| `examples/tmio/ior/parallel/384.msgpack` | MessagePack | IOR benchmark, 384 MPI ranks, parallel I/O | `ftio examples/tmio/ior/parallel/384.msgpack` |
| `examples/tmio/ior/collective/1536_new.json` | JSON | IOR benchmark, 1536 ranks, collective I/O | `ftio examples/tmio/ior/collective/1536_new.json` |
| `artifacts/frontiers25/zenodo/data/hacc-io/Job47970840_9216.msgpack` | MessagePack | HACC cosmology simulation, 9216 ranks (extract from [Zenodo](https://doi.org/10.5281/zenodo.17713783) — see [frontiers25 artifact](../artifacts/frontiers25/README.md#extracting-the-data-set)) | `ftio artifacts/frontiers25/zenodo/data/hacc-io/Job47970840_9216.msgpack` |
| `examples/tmio/JSONL/8.jsonl` | JSONL | Small TMIO trace with 8 ranks | `ftio examples/tmio/JSONL/8.jsonl` |

### Custom-format traces

| Path | Format | Description | Quick start |
|------|--------|-------------|-------------|
| `examples/custom/JSON/custom.json` | JSON (rank-level) | Synthetic rank-level bandwidth signal | `ftio examples/custom/JSON/custom.json` |
| `examples/custom/JSON/custom_app.json` | JSON (app-level) | Synthetic application-level bandwidth | `ftio examples/custom/JSON/custom_app.json` |
| `examples/custom/txt/2.txt` | Custom TXT | MiB/s + microsecond timestamps, parsed via regex | `ftio examples/custom/txt/2.txt -cf examples/custom/txt/custom_input.py` |

### Feature demonstration scripts

| Path | Description | Usage |
|------|-------------|-------|
| `examples/burst_width_example.py` | Standalone matplotlib demo of the energy-concentration burst-width algorithm on a synthetic irregular signal | `python examples/burst_width_example.py` |
| `examples/jupyter_notebooks/quick_example.ipynb` | Jupyter notebook: load a trace, run `ftio`, inspect the `Prediction` object | `jupyter notebook examples/jupyter_notebooks/quick_example.ipynb` |

### Typical commands on example files

```bash
# Offline DFT on the IOR parallel trace (no plots)
ftio examples/tmio/ior/parallel/384.msgpack -e no

# STFT with burst-width estimation
ftio examples/tmio/ior/parallel/384.msgpack -tr stft -bw

# Wavelet analysis on the HACC-IO 9216-rank trace
# (extract from Zenodo first: https://doi.org/10.5281/zenodo.17713783)
ftio artifacts/frontiers25/zenodo/data/hacc-io/Job47970840_9216.msgpack -tr wave_disc -le 2 -e no

# Collective IOR trace, restrict to first 100 s
ftio examples/tmio/ior/collective/1536_new.json -te 100 -e no

# Custom TXT file with a regex parser
ftio examples/custom/txt/2.txt -cf examples/custom/txt/custom_input.py -e no

# Visualise a trace without analysis
ioplot examples/tmio/ior/parallel/384.msgpack
```

<p align="right"><a href="#file-formats-and-tools">⬆</a></p>

