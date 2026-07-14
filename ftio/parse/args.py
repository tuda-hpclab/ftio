"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import argparse

from ftio import __copyright__, __license__, __repo__, __version__

# `-e/--engine` is registered by ftio, ioplot and play, hence the constant.
# "plot"/"mat" are historical short spellings and "*_no_paper" disables the
# paper layout in ftio.plot.plot_dft; the code dispatches on substrings of the
# value, so all of them stay valid.
ENGINE_CHOICES = [
    "plotly",
    "plot",
    "plotly_no_paper",
    "matplotlib",
    "mat",
    "mat_no_paper",
    "no",
]
ENGINE_CHOICES_PLOT = ENGINE_CHOICES + ["dash"]


def _level_type(value: str) -> int:
    """Wavelet decomposition level: a non-negative int, or `auto` (== 0)."""
    if value.strip().lower() == "auto":
        return 0
    try:
        level = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer or 'auto', got '{value}'"
        ) from None
    if level < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {level}")
    return level


def parse_args(argv: list, name="") -> argparse.Namespace:
    flag = True
    if name == "":
        name = argv[0]
        name = name[name.rfind("/") + 1 :]
    else:
        # API call
        flag = False

    lname = name.lower()
    is_plot = "plot" in lname
    is_parse = "parse" in lname
    is_play = "play" in lname
    is_ftio = "ftio" in lname or "predictor" in lname

    if is_plot:
        disc = "Plots result stored in Json file to a HTML page or PDF document."
    elif "ftio" in lname:
        disc = "Captures the period of the I/O phases. Uses frequency techniques (default=discrete fourier transformation) and outlier detection methods (Z-score) on the provided file. Supported file formats are Json, Jsonlines, Msgpack, Darshan, and reorder (folder). TMIO can be used to generate the tracing file needed. There are several parameters which can be controlled by the arguments bellow."
    elif "predictor" in lname:
        disc = "Wrapper code to execute ftio online. Monitors a file for changes. Whenever the file is modified (i.e., new traces are appended) a new prediction process is executed and the result is store in a shared memory space. All parameters that can be passed to ftio are supported by predictor."
    elif is_parse:
        disc = "Parses to an extra-p format."
    else:
        disc = ""

    parser = argparse.ArgumentParser(
        prog=name,  # .capitalize(),
        description=disc,
        epilog=f"""
--------------------------------------------
Note:
Long options accept both spellings, i.e.
--phase_automaton and --phase-automaton.

Author:
Ahmad H. Tarraf

Contributors:
{__repo__}/tree/main/docs/contributors.md

Report any bugs to:
{__repo__}/issues

COPYRIGHT:
{__copyright__}

LICENSE:
{__license__}

Full documentation:
{__repo__}
--------------------------------------------
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    #! If CLI call not API, add files
    if flag:
        parser.add_argument(
            "files",
            metavar="files",
            type=str,
            nargs="+",
            help="file, file list (file 0 ... file n), folder, or folder list (folder 0.. folder n)",
        )

    #! Input and data selection (for all)
    group = parser.add_argument_group("input and data selection")
    group.add_argument(
        "-m",
        "--mode",
        dest="mode",
        type=str,
        choices=[
            "write_sync",
            "sync_write",
            "write",
            "w",
            "read_sync",
            "sync_read",
            "read",
            "r",
            "write_async",
            "async_write",
            "read_async",
            "async_read",
        ],
        metavar="{write_sync,read_sync,write_async,read_async}",
        default="write_sync" if is_ftio else ("" if is_plot else None),
        help=(
            "if the trace file contains several I/O modes, a specific mode can be "
            "selected. Supported modes are: write_sync, read_sync, write_async, "
            "read_async. The reversed spellings (sync_write, ...) are also accepted, "
            "as are the shorthands w/write and r/read, which map to write_sync and "
            "read_sync (a value without 'async' in it means sync)"
        ),
    )
    group.add_argument(
        "-s",
        "--source",
        dest="source",
        type=str,
        choices=["tmio", "custom", "unspecified"],
        default="unspecified",
        help=(
            "the source of the files: tmio, or custom. The default 'unspecified' "
            "auto-detects it. Note this is the on-disk file format and is unrelated "
            "to --zmq_format, which selects the encoding of a ZMQ payload. See "
            "https://github.com/tuda-parallel/FTIO/blob/main/docs/file_formats.md"
        ),
    )
    group.add_argument(
        "-cf",
        "--custom_file",
        "--custom-file",
        type=str,
        default="",
        help="passes a [path/filename.py] file containing the translation and pattern for a custom file format similar to:\n https://github.com/tuda-parallel/FTIO/blob/main/examples/custom/txt/custom_input.py",
    )
    group.add_argument(
        "-x",
        "--dxt_mode",
        "--dxt-mode",
        dest="dxt_mode",
        type=str,
        choices=["DXT_MPIIO", "DXT_POSIX"],
        default="DXT_MPIIO",
        help="select data to extract from Darshan traces (DXT_POSIX or DXT_MPIIO (default))",
    )
    group.add_argument(
        "-l",
        "--limit",
        type=int,
        default=-1,
        help="max ranks to consider when reading a folder",
    )

    #! PARSE Settings
    if not is_parse:
        parser.add_argument(
            "-r",
            "--render",
            dest="render",
            type=str,
            choices=["dynamic", "static"],
            default="dynamic",
            help="specifies how the plots are rendered. Either dynamic (default) or static",
        )

    #! PLAY Settings
    if is_play:
        parser.add_argument(
            "-f",
            "--freq",
            dest="freq",
            type=float,
            help="specifies the sampling rate with which the continuous signal is discretized (default=10Hz). This directly affects the highest captured frequency (Nyquist). The value is specified in Hz. In case this value is set to -1, the auto mode is launched which sets the sampling frequency automatically to the smallest change in the bandwidth detected. Note that the lowest allowed frequency in the auto mode determine by the `memory_limit`",
        )
        parser.add_argument(
            "-e",
            "--engine",
            type=str,
            choices=ENGINE_CHOICES,
            metavar="{plotly,matplotlib,no}",
            default="plotly",
            help="plot engine. Either plotly (default) or matplotlib. Specifies the engine used to display the figures. Plotly is used to generate HTML files",
        )

    #! FTIO and Predictor Settings
    if is_ftio:
        group = parser.add_argument_group("frequency analysis (ftio, predictor)")
        group.add_argument(
            "-f",
            "--freq",
            dest="freq",
            type=float,
            default=10,
            help="specifies the sampling rate with which the continuous signal is discretized (default=10Hz). This directly affects the highest captured frequency (Nyquist). The value is specified in Hz. In case this value is set to -1, the auto mode is launched which sets the sampling frequency automatically to the smallest change in the bandwidth detected. Note that the lowest allowed frequency in the auto mode determine by the `memory_limit`",
        )
        group.add_argument(
            "--memory_limit",
            "--memory-limit",
            type=float,
            default=0.5,
            help="Memory limit in GB during discretization in case `freq` is passed with -1. Default is 0.5 GB.",
        )
        group.add_argument(
            "-ts",
            "--ts",
            type=float,
            help="modifies the start time of the examined time window",
        )
        group.add_argument(
            "-te",
            "--te",
            type=float,
            help="modifies the end time of the examined time window",
        )
        group.add_argument(
            "-tr",
            "--transformation",
            dest="transformation",
            type=str,
            choices=[
                "dft",
                "stft",
                "astft",
                "wave_disc",
                "dwt",
                "wave_dwt",
                "wave_cont",
                "cwt",
                "wave_cwt",
                "efd",
                "vmd",
            ],
            metavar="{dft,stft,astft,wave_disc,wave_cont}",
            default="dft",
            help=(
                "Specifies the frequency technique to use. "
                "Supported modes: dft (default), stft, astft, wave_disc, wave_cont. "
                "'dwt' and 'wave_dwt' are aliases of wave_disc, 'cwt' and 'wave_cwt' "
                "of wave_cont. "
                "Experimental (requires pip install 'ftio[amd-libs]'): efd, vmd."
            ),
        )
        group.add_argument(
            "-le",
            "--level",
            dest="level",
            type=_level_type,
            metavar="LEVEL",
            default=0,
            help="specifies the decomposition level for the discrete wavelet transformation. If specified as auto (or 0, the default), the maximum decomposition level is automatically calculated",
        )
        group.add_argument(
            "--wavelet",
            type=str,
            help='Wavelet to use. See pywt documentation for wavelet families: pywt.wavelist(kind="continuous") or pywt.wavelist(kind="discrete") (default "morl" for continuous and "db1" for discrete)',
        )
        group.add_argument(
            "-n",
            "--n_freq",
            "--n-freq",
            dest="n_freq",
            type=int,
            default=0,
            help='number of frequencies to extract. By default FTIO finds the dominant frequency. With this flag, up to "n_freq" can be extracted from FTIO',
        )
        group.add_argument(
            "--fourier_fit",
            "--fourier-fit",
            dest="fourier_fit",
            action="store_true",
            default=False,
            help="If set, performs Fourier basis fitting on the signal by extracting multiple dominant frequencies via DFT and fitting sinusoidal components with optimized amplitudes and phases. The number of fitting sinusoidal components is set via `--n_freq",
        )
        group.add_argument(
            "-au",
            "--autocorrelation",
            dest="autocorrelation",
            action="store_true",
            default=False,
            help="if set, autocorrelation is calculated in addition to DFT. The results are merged to a single prediction at the end",
        )
        group.add_argument(
            "-d",
            "--dtw",
            action="store_true",
            default=False,
            help="performs dynamic time warping on the top 3 frequencies (highest contribution) calculated using the DFT if set (default=False)",
        )
        group.add_argument(
            "-np",
            "--no-psd",
            "--no_psd",
            dest="psd",
            action="store_false",
            default=True,
            help="if set, replace the power density spectrum (a*a/N) with the amplitude spectrum (a)",
        )
        group.add_argument(
            "--tfpf",
            type=int,
            default=0,
            help="Number of time-frequency peak filtering iterations.",
        )
        group.add_argument(
            "--stft_window",
            "--stft-window",
            dest="stft_window",
            type=str,
            default="0",
            help=(
                "Window length in samples or seconds (e.g. '20s'). Default: 0 (auto). "
                "For -tr stft: auto sets the window to 4x the dominant period found by "
                "a preliminary DFT. "
                "For -tr astft: auto determines the window via the cm5 concentration "
                "measure; a non-zero value overrides that automatic selection."
            ),
        )
        group.add_argument(
            "-bw",
            "--burst_width",
            "--burst-width",
            dest="burst_width",
            action="store_true",
            default=False,
            help=(
                "Estimate per-period burst width and duty cycle using the shortest "
                "contiguous time window that contains --burst_energy_fraction of each "
                "period's total energy (O(N) two-pointer sweep - negligible cost). "
                "Requires a valid dominant frequency. Results are stored in "
                "prediction.burst_widths and displayed in the console output. "
                "Default: off."
            ),
        )
        group.add_argument(
            "--burst_energy_fraction",
            "--burst-energy-fraction",
            dest="burst_energy_fraction",
            type=float,
            default=0.95,
            help=(
                "Energy fraction (in (0, 1]) for burst width estimation (default: 0.95). "
                "The burst window is the shortest contiguous time interval whose squared "
                "bandwidth (power) sums to at least this fraction of the period's total "
                "energy. For example, 0.95 means the detected burst captures 95%% of the "
                "period's energy. Only used with -bw / --burst_width."
            ),
        )

        group = parser.add_argument_group(
            "outlier and periodicity detection (ftio, predictor)"
        )
        group.add_argument(
            "-o",
            "--outlier",
            dest="outlier",
            type=str,
            choices=[
                "z-score",
                "Z-score",
                "zscore",
                "dbscan",
                "db-scan",
                "db",
                "forest",
                "isolation_forest",
                "lof",
                "peak",
                "peaks",
            ],
            metavar="{z-score,dbscan,forest,lof,peak}",
            default="Z-score",
            help="outlier detection method: Z-score (default), DB-Scan, Isolation_forest, LOF, find_peaks (from sci-pi)",
        )
        group.add_argument(
            "-p",
            "--periodicity_detection",
            "--periodicity-detection",
            dest="periodicity_detection",
            type=str,
            choices=["rpde", "sf", "corr", "ind"],
            default=None,
            help="periodicity detection method after outlier detection: RPDE, Spectral flatness, Correlation, Correlation for individual periods. Default: none",
        )
        group.add_argument(
            "-t",
            "--tol",
            dest="tol",
            type=float,
            default=0.8,
            help="confidence tolerance in [0, 1]. A prediction counts as periodic only if its confidence exceeds this value (default=0.8)",
        )

        group = parser.add_argument_group("filtering (ftio, predictor)")
        group.add_argument(
            "--filter_type",
            "--filter-type",
            dest="filter_type",
            type=str,
            default=None,
            choices=["lowpass", "highpass", "bandpass"],
            help="Type of filter to apply. Requires --filter_cutoff.",
        )
        group.add_argument(
            "--filter_cutoff",
            "--filter-cutoff",
            dest="filter_cutoff",
            type=float,
            nargs="+",
            help="Cutoff frequency for low/high-pass filters (one value) or low and high cutoff for bandpass (two values).",
        )
        group.add_argument(
            "--filter_order",
            "--filter-order",
            dest="filter_order",
            type=int,
            default=4,
            help="Order of Butterworth filter.",
        )

        group = parser.add_argument_group("plotting (ftio, predictor)")
        group.add_argument(
            "-e",
            "--engine",
            type=str,
            choices=ENGINE_CHOICES,
            metavar="{plotly,matplotlib,no}",
            default="plotly",
            help="specifies the engine used to display the figures. Either plotly (default) or matplotlib can be used.  Plotly is used to generate interactive plots as HTML files. Set this value to no if you do not want to generate plots",
        )
        group.add_argument(
            "-rp",
            "--runtime_plots",
            "--runtime-plots",
            dest="runtime_plots",
            action="store_true",
            default=False,
            help="if set, shows the plot at at runtime",
        )
        group.add_argument(
            "-ce",
            "--cepstrum",
            action="store_true",
            help="enable Cepstrum plotting for the DFT",
        )
        group.add_argument(
            "-re",
            "--reconstruction",
            action="store",
            nargs="*",
            default=[],
            metavar="N",
            help="plots reconstruction of top 10 signals on figure",
        )
        group.add_argument(
            "-v",
            "--verbose",
            dest="verbose",
            action="store_true",
            default=False,
            help="sets verbose on or off (default=False)",
        )

        group = parser.add_argument_group("ZMQ communication (ftio, predictor)")
        group.add_argument(
            "-w",
            "--window_adaptation",
            "--window-adaptation",
            dest="window_adaptation",
            type=str,
            choices=["frequency_hits", "data", "adwin", "cusum", "ph"],
            default=None,
            help=(
                "online window adaptation strategy. "
                "'frequency_hits': shift the time window on X frequency hits to X times the last found period. "
                "'data': move the window to X times after data has been received. "
                "'adwin': Adaptive Windowing with automatic window sizing and mathematical guarantees. "
                "'cusum': Cumulative Sum detection for rapid change detection. "
                "'ph': Page-Hinkley test for sequential change point detection. "
                "For 'adwin', 'cusum', and 'ph', the option '--gui' is supported to display detected change points."
            ),
        )
        group.add_argument(
            "-hi",
            "--hits",
            dest="hits",
            type=int,
            default=3,
            help="specifies the number of hits needed to adapt the time window. A hit occurs once a dominant frequency is found",
        )
        group.add_argument(
            "-ml",
            "--machine_learning",
            "--machine-learning",
            dest="machine_learning",
            action="store_true",
            default=False,
            help="if set, machine learning is enabled (api call only)",
        )
        group.add_argument(
            "--zmq",
            action="store_true",
            default=False,
            help="avoids opening the generated HTML file since zmq is used",
        )
        group.add_argument(
            "--gui",
            action="store_true",
            default=False,
            help="enables forwarding prediction data to the FTIO GUI dashboard. Start the GUI first with 'ftio-gui' then run predictor with this flag.",
        )
        group.add_argument(
            "--zmq_format",
            "--zmq-format",
            "--zmq_source",
            "--zmq-source",
            dest="zmq_source",
            type=str,
            choices=["direct", "tmio"],
            default="direct",
            help=(
                "encoding of the ZMQ payload: 'direct' (default) for raw bandwidth / "
                "start / end triples, or 'tmio' for a msgpack-encoded TMIO buffer. "
                "--zmq_source is a legacy alias of this flag. Unrelated to --source, "
                "which selects the on-disk file format"
            ),
        )
        group.add_argument(
            "--zmq_address",
            "--zmq-address",
            dest="zmq_address",
            type=str,
            default="*",
            help="zmq address for communication",
        )
        group.add_argument(
            "--zmq_port",
            "--zmq-port",
            dest="zmq_port",
            type=str,
            default="5555",
            help="zmq port for communication",
        )
        group.add_argument(
            "--zmq_port_reply",
            "--zmq-port-reply",
            dest="zmq_port_reply",
            type=str,
            default="5556",
            help="zmq port for communicating dominant frequency",
        )

        group = parser.add_argument_group("performance (predictor only)")
        group.add_argument(
            "--debounce",
            dest="debounce",
            action="store_true",
            default=False,
            help=(
                "Enable debounced (serial) prediction: only one prediction runs at a "
                "time.  If the monitored file changes again while a prediction is in "
                "flight, the stale stamp is detected on the next monitor call and a "
                "follow-up prediction is triggered immediately - no trigger is lost. "
                "This also prevents concurrent writes to shared state. Equivalent to "
                "--max-predictions 1. Default: off (original parallel behaviour)."
            ),
        )
        group.add_argument(
            "--max-predictions",
            "--max_predictions",
            dest="max_predictions",
            type=int,
            default=0,
            help=(
                "Cap the number of concurrent prediction processes (bounded "
                "pool). 0 (default) is unlimited - the original behaviour; a "
                "new prediction waits for the oldest in-flight one once the cap "
                "is reached, so predictions cannot pile up and oversubscribe the "
                "cores. 1 is equivalent to --debounce. Note each fan-out "
                "prediction also spawns --ingest-workers sub-processes."
            ),
        )
        group.add_argument(
            "--ingest-workers",
            "--ingest_workers",
            dest="ingest_workers",
            type=int,
            default=1,
            help=(
                "Number of worker processes used to parse and overlap the drained "
                "server messages into the application-level bandwidth. 1 (default) "
                "keeps the original single-process behaviour; >1 fans the messages "
                "out and folds the per-worker partials back (capped to the CPU "
                "budget). The result is identical regardless of the worker count."
            ),
        )
        group.add_argument(
            "--ingest-backend",
            "--ingest_backend",
            dest="ingest_backend",
            choices=["thread", "process", "process-resample"],
            default="process-resample",
            help=(
                "How --ingest-workers > 1 fans out. 'process-resample' (default) "
                "resamples each worker's partial onto a shared grid so only a small "
                "vector crosses the process boundary - the only mode that beats a "
                "single process (~2.4x), at the cost of the full-resolution signal. "
                "'thread' shares memory but is GIL-bound; 'process' keeps full "
                "resolution but is IPC-bound. Ignored when --ingest-workers is 1."
            ),
        )

        group = parser.add_argument_group("phase automaton (predictor only)")
        group.add_argument(
            "--phase-automaton",
            "--phase_automaton",
            dest="phase_automaton",
            action="store_true",
            default=False,
            help=(
                "Enable the phase automaton: models I/O behaviour as a state machine "
                "where each state is a stable frequency regime. Transitions are "
                "detected by rank changes, period-ratio threshold, and/or a "
                "statistical detector (see --pa-method)."
            ),
        )
        group.add_argument(
            "--pa-method",
            "--pa_method",
            dest="pa_method",
            choices=["cusum", "ph", "adwin", "ksigma", "none"],
            default="ksigma",
            help=(
                "Statistical change-point detector used by the phase automaton "
                "(default: ksigma). "
                "'ksigma' - state-adaptive k-sigma (recommended; robust to "
                "within-phase noise); "
                "'cusum' - adaptive-variance CUSUM; "
                "'ph'    - Page-Hinkley; "
                "'adwin' - ADWIN (needs many samples or large freq ratios); "
                "'none'  - disable statistical detection (use only rank and/or "
                "period-ratio triggers)."
            ),
        )
        group.add_argument(
            "--pa-period-ratio",
            "--pa_period_ratio",
            dest="pa_period_ratio",
            type=float,
            default=None,
            metavar="RATIO",
            help=(
                "Fire a phase transition when max(T_new/T_cur, T_cur/T_new) > RATIO. "
                "Recommended value: 1.5 (50%% period change). No warm-up needed. "
                "Can be combined with --pa-method."
            ),
        )
        group.add_argument(
            "--pa-min-cycles",
            "--pa_min_cycles",
            dest="pa_min_cycles",
            type=float,
            default=2.0,
            metavar="N",
            help=(
                "Warm-up guard: ignore a prediction whose analysis window holds "
                "fewer than N full periods (default: 2). A period cannot be "
                "measured from a single I/O phase -- before a second phase arrives "
                "the DFT reports that one burst's width as the period, at full "
                "confidence, and the automaton would learn a phase that does not "
                "exist. Set to 1 to disable."
            ),
        )
        group.add_argument(
            "--pa-no-rank-trigger",
            "--pa_no_rank_trigger",
            dest="pa_rank_trigger",
            action="store_false",
            default=True,
            help=(
                "Disable the rank-change trigger in the phase automaton. "
                "By default a prediction with a different rank count immediately "
                "opens a new state."
            ),
        )
        group.add_argument(
            "--pa-export",
            "--pa_export",
            dest="pa_export",
            type=str,
            default="./phase_automaton.json",
            metavar="PATH",
            help=(
                "Path of the JSON file written when the predictor exits "
                "(default: ./phase_automaton.json). Contains all states, "
                "transitions, and automaton configuration."
            ),
        )
        group.add_argument(
            "--pa-library",
            "--pa_library",
            dest="pa_library",
            type=str,
            default=None,
            metavar="DIR",
            help=(
                "Root directory for the phase automaton library "
                "(default: ./ftio_models when this flag is given). "
                "Each app+rank configuration is stored as "
                "<DIR>/<app_name>/ranks_<key>.json. "
                "On the first run for an app (cold start) the automaton is "
                "saved as a new reference.  On subsequent runs the distributions "
                "are updated using pooled statistics. "
                "Implies --phase-automaton."
            ),
        )
        group.add_argument(
            "--pa-app-name",
            "--pa_app_name",
            dest="pa_app_name",
            type=str,
            default=None,
            metavar="NAME",
            help=(
                "Application name used as the library subdirectory "
                "(default: stem of the monitored filename). "
                "Use this to distinguish different applications that happen to "
                "run at the same rank count."
            ),
        )
        group.add_argument(
            "--pa-match",
            "--pa_match",
            dest="pa_match",
            choices=["greedy", "dtw", "viterbi"],
            default="greedy",
            help=(
                "Matching strategy for position tracking against the reference "
                "automaton (default: greedy). "
                "greedy - nearest period at each step; "
                "dtw    - sequence alignment over an observation window; "
                "viterbi - HMM decoding with Gaussian emission on period."
            ),
        )

    #! IOPLOT Settings
    if is_plot:
        parser.add_argument(
            "-z", "--zoom", type=float, help="upper zoom limit on the y-axis"
        )
        parser.add_argument(
            "-nt",
            "--no-threaded",
            "--no_threaded",
            dest="threaded",
            action="store_false",
            default=True,
            help="turn multithreading off (default=on)",
        )
        parser.add_argument(
            "-e",
            "--engine",
            type=str,
            choices=ENGINE_CHOICES_PLOT,
            metavar="{plotly,dash,matplotlib,no}",
            default="plotly",
            help="plot engine to use. Either plotly (default), dash, matplotlib or no (disables plots)",
        )
        parser.add_argument(
            "--n_shown_samples",
            "--n-shown-samples",
            dest="n_shown_samples",
            type=int,
            default=20_000,
            help="only for dash: Number of shown samples per trace (default: 20_000). Caution: Too small numbers could lead to incorrect representations!",
        )
        parser.add_argument(
            "--merge_plots",
            "--merge-plots",
            dest="merge_plots",
            action="store_true",
            default=False,
            help="only for dash: Merges the plots to one plot for each io mode. Note: The file dropdown menu then has no functionality",
        )
        parser.add_argument(
            "--no_disp",
            "--no-disp",
            dest="no_disp",
            action="store_true",
            default=False,
            help="avoids opening the generated HTML file",
        )

    #! PARSE Settings
    if is_parse:
        parser.add_argument(
            "--scale", action="store_true", default=False, help="scales the Y-axis"
        )

    #! Data modes (for all)
    group = parser.add_argument_group("plotted bandwidth traces")
    group.add_argument(
        "--sum",
        action="store_true",
        default=True,
        help=(
            "Show the summed (application-level) bandwidth in plots (default: on). "
            "When trace data is provided at rank level (e.g. from TMIO), individual "
            "rank bandwidths are overlapped and summed to obtain the total application "
            "I/O bandwidth. Use --no_sum to hide this trace."
        ),
    )
    group.add_argument("--no_sum", "--no-sum", dest="sum", action="store_false")
    group.add_argument(
        "--avr",
        action="store_true",
        default=True,
        help=(
            "Show the average bandwidth across all ranks in plots (default: on). "
            "Relevant when trace data contains per-rank metrics (e.g. from TMIO). "
            "Use --no_avr to hide this trace."
        ),
    )
    group.add_argument("--no_avr", "--no-avr", dest="avr", action="store_false")
    group.add_argument(
        "--ind",
        action="store_true",
        default=False,
        help=(
            "Show individual per-rank bandwidth traces in plots (default: off). "
            "Useful with TMIO traces to inspect rank-level I/O patterns. "
            "Use --no_ind to hide individual traces (already the default)."
        ),
    )
    group.add_argument("--no_ind", "--no-ind", dest="ind", action="store_false")

    args = parser.parse_args(argv)

    #! Validation and derived defaults
    if is_ftio:
        if args.freq != -1 and args.freq <= 0:
            parser.error(
                f"-f/--freq must be > 0, or exactly -1 for auto mode, got {args.freq}"
            )
        if not 0.0 <= args.tol <= 1.0:
            parser.error(f"-t/--tol must be in [0, 1], got {args.tol}")
        if not 0.0 < args.burst_energy_fraction <= 1.0:
            parser.error(
                "--burst_energy_fraction must be in (0, 1], got "
                f"{args.burst_energy_fraction}"
            )
        if args.n_freq < 0:
            parser.error(f"-n/--n_freq must be >= 0, got {args.n_freq}")
        if args.hits < 1:
            parser.error(f"-hi/--hits must be >= 1, got {args.hits}")
        if args.tfpf < 0:
            parser.error(f"--tfpf must be >= 0, got {args.tfpf}")
        if args.filter_order < 1:
            parser.error(f"--filter_order must be >= 1, got {args.filter_order}")
        if args.max_predictions < 0:
            parser.error(f"--max-predictions must be >= 0, got {args.max_predictions}")
        if args.ingest_workers < 1:
            parser.error(f"--ingest-workers must be >= 1, got {args.ingest_workers}")

        for port in (args.zmq_port, args.zmq_port_reply):
            if not port.isdigit() or not 1 <= int(port) <= 65535:
                parser.error(f"zmq port must be an integer in [1, 65535], got '{port}'")

        if args.ts is not None and args.te is not None and args.ts >= args.te:
            parser.error(f"-ts ({args.ts}) must be smaller than -te ({args.te})")

        if args.filter_type and not args.filter_cutoff:
            parser.error(f"--filter_type {args.filter_type} requires --filter_cutoff")
        if args.filter_cutoff and not args.filter_type:
            parser.error("--filter_cutoff requires --filter_type")
        if args.filter_type in ("lowpass", "highpass") and len(args.filter_cutoff) != 1:
            parser.error(
                f"--filter_type {args.filter_type} expects exactly one --filter_cutoff "
                f"value, got {len(args.filter_cutoff)}"
            )
        if args.filter_type == "bandpass":
            if len(args.filter_cutoff) != 2:
                parser.error(
                    "--filter_type bandpass expects exactly two --filter_cutoff values "
                    f"(low high), got {len(args.filter_cutoff)}"
                )
            if args.filter_cutoff[0] >= args.filter_cutoff[1]:
                parser.error(
                    f"--filter_cutoff low must be < high, got {args.filter_cutoff}"
                )

        # --pa-library implies --phase-automaton
        if args.pa_library:
            args.phase_automaton = True

        is_wavelet = "wave" in args.transformation or args.transformation in (
            "dwt",
            "cwt",
        )
        if is_wavelet and not args.wavelet:
            cont = "cont" in args.transformation or "cwt" in args.transformation
            args.wavelet = "morl" if cont else "db1"

        recon = []
        if args.reconstruction:
            recon = [int(x) for val in args.reconstruction for x in val.split(",")]
        if args.n_freq and args.n_freq not in recon:
            recon.append(int(args.n_freq))
        args.reconstruction = recon

    return args
