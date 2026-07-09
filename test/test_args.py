"""
Tests for the command line argument parser.

The suite pins two things:
  1. Backwards compatibility -- every spelling that appears in the docs, the
     artifacts, the examples and the JIT settings must keep parsing to the same
     value. The consuming code dispatches on substrings of these values
     (`"mat" in args.engine`, `"dwt" in args.transformation`), so the parser
     validates them but never rewrites them.
  2. The new validation actually rejects typos and impossible combinations.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import argparse

import pytest

from ftio.parse.args import parse_args
from ftio.parse.helper import match_mode


def ftio(*flags: str) -> argparse.Namespace:
    """Parse a CLI invocation of `ftio x.json <flags>`."""
    return parse_args(["ftio", "x.json", *flags])


# ---------------------------------------------------------------------------
# Backwards compatibility: values are validated, never rewritten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", ["plotly", "plot", "matplotlib", "mat", "no", "plotly_no_paper"]
)
def test_engine_spellings_preserved(value):
    assert ftio("-e", value).engine == value


@pytest.mark.parametrize(
    "value, expected",
    [
        ("write_sync", "write_sync"),
        ("sync_write", "write_sync"),
        ("write", "write_sync"),  # ftio/api/gekkoFs/jit/jitsettings.py
        ("w", "write_sync"),
        ("read_sync", "read_sync"),
        ("sync_read", "read_sync"),
        ("read", "read_sync"),
        ("r", "read_sync"),
        ("write_async", "write_async"),
        ("read_async", "read_async"),
    ],
)
def test_mode_spellings_reach_match_mode_unchanged(value, expected):
    """`w`/`r` carry no 'async', so match_mode() maps them to the sync modes."""
    assert match_mode(ftio("-m", value).mode) == expected


@pytest.mark.parametrize(
    "value", ["dft", "stft", "astft", "wave_disc", "wave_cont", "efd", "vmd"]
)
def test_transformation_canonical(value):
    assert ftio("-tr", value).transformation == value


@pytest.mark.parametrize(
    "value, marker",
    [("dwt", "dwt"), ("wave_dwt", "dwt"), ("cwt", "cwt"), ("wave_cwt", "cwt")],
)
def test_wavelet_aliases_keep_the_substring_ftio_core_dispatches_on(value, marker):
    """ftio_core matches `"dwt" in args.transformation`, so it must survive."""
    assert marker in ftio("-tr", value).transformation


@pytest.mark.parametrize(
    "value", ["z-score", "Z-score", "zscore", "dbscan", "db", "lof", "peak"]
)
def test_outlier_spellings_preserved(value):
    """`-o Z-score` is the default and the value printed in --help."""
    assert ftio("-o", value).outlier == value


def test_outlier_default_is_an_accepted_cli_value():
    assert ftio().outlier in ftio("-o", "Z-score").outlier


@pytest.mark.parametrize("value", ["2", "0", "auto"])
def test_level_accepts_int_and_auto(value):
    expected = 0 if value == "auto" else int(value)
    assert ftio("-le", value).level == expected


def test_wavelet_defaults_follow_the_transformation():
    assert ftio("-tr", "wave_disc").wavelet == "db1"
    assert ftio("-tr", "wave_cont").wavelet == "morl"
    assert ftio("-tr", "wave_dwt").wavelet == "db1"
    assert ftio("-tr", "wave_cwt").wavelet == "morl"
    assert ftio("-tr", "wave_cont", "--wavelet", "mexh").wavelet == "mexh"


def test_zmq_ports_stay_strings():
    """They are interpolated into `tcp://{addr}:{port}` and stored as str by JIT."""
    args = ftio("--zmq_port", "5558", "--zmq_port_reply", "5559")
    assert args.zmq_port == "5558"
    assert args.zmq_port_reply == "5559"


# ---------------------------------------------------------------------------
# Hyphen / underscore twins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "underscore, hyphen",
    [
        ("--phase_automaton", "--phase-automaton"),
        ("--no_psd", "--no-psd"),
        ("--max_predictions", "--max-predictions"),
        ("--ingest_workers", "--ingest-workers"),
        ("--burst_width", "--burst-width"),
        ("--pa_no_rank_trigger", "--pa-no-rank-trigger"),
    ],
)
def test_flag_twins_agree(underscore, hyphen):
    store_value = ["2"] if "predictions" in underscore or "workers" in underscore else []
    a = ftio(underscore, *store_value)
    b = ftio(hyphen, *store_value)
    assert vars(a) == vars(b)


@pytest.mark.parametrize(
    "underscore, hyphen, value",
    [
        ("--zmq_port", "--zmq-port", "5560"),
        ("--stft_window", "--stft-window", "10"),
        ("--memory_limit", "--memory-limit", "1.0"),
        ("--pa_export", "--pa-export", "a.json"),
        ("--custom_file", "--custom-file", "c.py"),
    ],
)
def test_valued_flag_twins_agree(underscore, hyphen, value):
    assert vars(ftio(underscore, value)) == vars(ftio(hyphen, value))


def test_zmq_format_is_an_alias_of_zmq_source():
    assert ftio("--zmq_format", "tmio").zmq_source == "tmio"
    assert ftio("--zmq_source", "tmio").zmq_source == "tmio"
    assert ftio("--zmq-format", "tmio").zmq_source == "tmio"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        ("-e", "pltoly"),  # typo that used to silently disable plotting
        ("-m", "typo"),  # used to be silently coerced to read_sync
        ("-m", "async"),  # used to be silently coerced to read_async
        ("-tr", "fft"),
        ("-o", "zed"),
        ("-s", "darshan"),
        ("-x", "DXT_STDIO"),
        ("-r", "interactive"),
        ("-p", "acf"),
        ("--zmq_source", "bogus"),
        ("-le", "-1"),
        ("-le", "three"),
    ],
)
def test_invalid_values_are_rejected(flags):
    with pytest.raises(SystemExit):
        ftio(*flags)


@pytest.mark.parametrize(
    "flags",
    [
        ("-f", "0"),
        ("-f", "-5"),  # only -1 is the auto sentinel
        ("-t", "1.5"),
        ("-t", "-0.1"),
        ("-hi", "0"),
        ("-n", "-1"),
        ("--tfpf", "-1"),
        ("--filter_order", "0"),
        ("--max-predictions", "-1"),
        ("--ingest-workers", "0"),
        ("-bw", "--burst_energy_fraction", "90"),
        ("-bw", "--burst_energy_fraction", "0"),
        ("--zmq_port", "99999"),
        ("--zmq_port", "abc"),
        ("-ts", "100", "-te", "10"),
        ("-ts", "10", "-te", "10"),
    ],
)
def test_out_of_range_values_are_rejected(flags):
    with pytest.raises(SystemExit):
        ftio(*flags)


@pytest.mark.parametrize(
    "flags",
    [
        ("--filter_type", "lowpass"),  # cutoff missing
        ("--filter_cutoff", "0.5"),  # type missing
        ("--filter_type", "lowpass", "--filter_cutoff", "0.1", "0.2"),
        ("--filter_type", "bandpass", "--filter_cutoff", "0.5"),
        ("--filter_type", "bandpass", "--filter_cutoff", "0.5", "0.1"),  # low >= high
    ],
)
def test_filter_argument_combinations_are_rejected(flags):
    with pytest.raises(SystemExit):
        ftio(*flags)


@pytest.mark.parametrize(
    "flags",
    [
        ("--filter_type", "lowpass", "--filter_cutoff", "0.5"),
        ("--filter_type", "highpass", "--filter_cutoff", "0.01"),
        ("--filter_type", "bandpass", "--filter_cutoff", "0.05", "0.2"),
        (
            "--filter_type",
            "bandpass",
            "--filter_cutoff",
            "0.05",
            "0.5",
            "--filter_order",
            "8",
        ),
    ],
)
def test_valid_filter_combinations_are_accepted(flags):
    assert ftio(*flags).filter_type is not None


@pytest.mark.parametrize("freq", ["-1", "1", "10", "100", "1000"])
def test_freq_accepts_positive_and_the_auto_sentinel(freq):
    assert ftio("-f", freq).freq == float(freq)


def test_pa_library_implies_phase_automaton():
    assert ftio("--pa-library", "./models").phase_automaton is True
    assert ftio().phase_automaton is False


# ---------------------------------------------------------------------------
# Per-tool parsers
# ---------------------------------------------------------------------------


def test_predictor_accepts_every_ftio_flag():
    """ftio_core.main() re-parses the predictor's argv, so both must agree."""
    argv = ["-e", "no", "-tr", "stft", "--gui", "--debounce", "--pa-method", "cusum"]
    assert parse_args(["predictor", "x.jsonl", *argv]).transformation == "stft"
    assert parse_args(["ftio", "x.json", *argv]).transformation == "stft"


def test_ioplot_accepts_dash_but_ftio_does_not():
    assert parse_args(["ioplot", "x.json", "-e", "dash"]).engine == "dash"
    with pytest.raises(SystemExit):
        ftio("-e", "dash")


def test_tool_specific_mode_defaults():
    assert parse_args(["ftio", "x.json"]).mode == "write_sync"
    assert parse_args(["predictor", "x.jsonl"]).mode == "write_sync"
    assert parse_args(["ioplot", "x.json"]).mode == ""
    assert parse_args(["ioparse", "x.json"]).mode is None


def test_api_call_takes_no_positional_files():
    args = parse_args(["-e", "no", "-f", "10"], "ftio")
    assert args.engine == "no" and args.freq == 10
    assert not hasattr(args, "files")


def test_ioparse_has_no_render_flag():
    with pytest.raises(SystemExit):
        parse_args(["ioparse", "x.json", "-r", "static"])
