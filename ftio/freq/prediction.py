"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Mai 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import numpy as np
from rich.table import Table


class Prediction:
    """
    A class to store and manipulate prediction data related to frequency analysis.

    Attributes:
        source (str): The name or type of transformation used to generate predictions.
        dominant_freq (np.ndarray): Array of dominant frequencies identified in the data.
        conf (np.ndarray): Confidence values associated with each dominant frequency.
        periodicity (np.ndarray): Periodicity score results for each frequency.
        amp (np.ndarray): Amplitudes corresponding to each dominant frequency.
        phi (np.ndarray): Phase angles for each dominant frequency.
        t_start (float): Start time index or timestamp for the prediction interval.
        t_end (float): End time index or timestamp for the prediction interval.
        total_bytes (int): Total number of bytes processed or considered.
        freq (float): Sampling frequency (samples per unit time) used for the data.
        ranks (int): Number of ranks or parallel processes involved.
        n_samples (int): Number of samples used in the prediction.
        top_freqs (dict): Dictionary storing top frequencies and associated metadata.
        candidates (np.ndarray): Array of candidates used in autocorrelation or other analysis.
        ranges (np.ndarray): Array of ranges indicating where the dominant frequency is valid
    """

    def __init__(
        self,
        transformation: str = "",
        t_start: float = 0,
        t_end: float = 0,
        total_bytes: int = 0,
        freq: float = 0,
        ranks: int = 0,
        n_samples=0,
    ):
        self._source = transformation
        self._dominant_freq = np.array([])
        self._conf = np.array([])
        self._periodicity = np.array([])
        self._amp = np.array([])
        self._phi = np.array([])
        self._t_start = t_start
        self._t_end = t_end
        self._total_bytes = total_bytes
        self._freq = freq
        self._ranks = ranks
        self._n_samples = n_samples
        self._top_freqs = {}
        self._candidates = np.array([])
        self._ranges = np.array([])
        self._ranks_per_window = np.array([])
        self._metric = ""
        self._b_sampled = np.array([])
        self._burst_widths = np.array([])
        self._burst_t_starts = np.array([])

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        self._source = str(value)

    @property
    def dominant_freq(self):
        return self._dominant_freq

    @dominant_freq.setter
    def dominant_freq(self, value):
        # if single numeric, convert to np.array([value])
        if np.isscalar(value):
            value = np.array([value])
        elif isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "dominant_freq must be a numpy ndarray, list convertible to ndarray, or a numeric scalar"
            )
        self._dominant_freq = value

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, value):
        # same logic as dominant_freq
        if np.isscalar(value):
            value = np.array([value])
        elif isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "conf must be a numpy ndarray, list convertible to ndarray, or a numeric scalar"
            )
        self._conf = value

    @property
    def periodicity(self):
        return self._periodicity

    @periodicity.setter
    def periodicity(self, value):
        # same logic as dominant_freq
        if np.isscalar(value):
            value = np.array([value])
        elif isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "periodicity must be a numpy ndarray, list convertible to ndarray, or a numeric scalar"
            )
        self._periodicity = value

    @property
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, value):
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError("amp must be a numpy ndarray or list convertible to ndarray")
        self._amp = value

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError("phi must be a numpy ndarray or list convertible to ndarray")
        self._phi = value

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("t_start must be a number")
        self._t_start = value

    @property
    def t_end(self):
        return self._t_end

    @t_end.setter
    def t_end(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("t_end must be a number")
        self._t_end = value

    @property
    def total_bytes(self):
        return self._total_bytes

    @total_bytes.setter
    def total_bytes(self, value):
        if not isinstance(value, int):
            raise TypeError("total_bytes must be an int")
        self._total_bytes = value

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("freq must be a number")
        self._freq = value

    @property
    def ranks(self):
        return self._ranks

    @ranks.setter
    def ranks(self, value):
        if not isinstance(value, int):
            raise TypeError("ranks must be an int")
        self._ranks = value

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        if not isinstance(value, int):
            raise TypeError("n_samples must be an int")
        self._n_samples = value

    @property
    def top_freqs(self):
        return self._top_freqs

    @top_freqs.setter
    def top_freqs(self, value):
        if not isinstance(value, dict):
            raise TypeError("top_freqs must be a dict")
        self._top_freqs = value

    @property
    def candidates(self):
        return self._candidates

    @candidates.setter
    def candidates(self, value):
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "candidates must be a numpy ndarray or list convertible to ndarray"
            )
        self._candidates = value

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, value):
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "candidates must be a numpy ndarray or list convertible to ndarray"
            )
        self._ranges = value

    @property
    def ranks_per_window(self):
        """Per-STFT-window rank count, aligned index-for-index with `ranges`
        (index 0 is the whole-trace summary, indices 1..N the windows) --
        only populated when the source trace's rank count actually varies
        (see ftio_stft / Simrun.merge_fields). Empty otherwise; consumers
        should fall back to the scalar `ranks` in that case."""
        return self._ranks_per_window

    @ranks_per_window.setter
    def ranks_per_window(self, value):
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError(
                "ranks_per_window must be a numpy ndarray or list convertible to ndarray"
            )
        self._ranks_per_window = value

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = str(value)

    @property
    def b_sampled(self):
        return self._b_sampled

    @b_sampled.setter
    def b_sampled(self, value):
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError("b_sampled must be a list or numpy array")
        self._b_sampled = value

    @property
    def burst_widths(self) -> np.ndarray:
        return self._burst_widths

    @burst_widths.setter
    def burst_widths(self, value):
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError("burst_widths must be a numpy ndarray or list")
        self._burst_widths = value

    @property
    def burst_t_starts(self) -> np.ndarray:
        return self._burst_t_starts

    @burst_t_starts.setter
    def burst_t_starts(self, value):
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError("burst_t_starts must be a numpy ndarray or list")
        self._burst_t_starts = value

    @property
    def burst_width_median(self) -> float:
        return float(np.median(self._burst_widths)) if len(self._burst_widths) else np.nan

    @property
    def burst_width_min(self) -> float:
        return float(np.min(self._burst_widths)) if len(self._burst_widths) else np.nan

    @property
    def burst_width_max(self) -> float:
        return float(np.max(self._burst_widths)) if len(self._burst_widths) else np.nan

    @property
    def duty_cycle(self) -> float:
        f = self.get_dominant_freq()
        bw = self.burst_width_median
        if not np.isnan(f) and f > 0 and not np.isnan(bw):
            return float(bw * f)
        return np.nan

    def get(self, key: str):
        """
        Retrieve the value for a given attribute.

        Args:
            key (str): Attribute name.

        Returns:
            Any: Value of the attribute or None if not found.
        """
        return getattr(self, key, None)

    def set(self, key: str, value):
        """
        Set the value of a given attribute.

        Args:
            key (str): Attribute name.
            value (Any): Value to set.

        Raises:
            AttributeError: If the attribute is invalid.
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise AttributeError(f"Invalid attribute: {key}")

    def set_from_dict(self, data: dict):
        """
        Set multiple attributes from a dictionary.

        Args:
            data (dict): Keys are attribute names, values are the new values.

        Raises:
            AttributeError: If any key is not a valid attribute.
        """
        for key, value in data.items():
            self.set(key, value)

    def get_dominant_freq_and_conf(self) -> tuple[float, float]:
        """
        Return the dominant frequency and its corresponding confidence.

        Returns:
            tuple of float:
                - Dominant frequency with the highest confidence.
                - Associated confidence value.
                Returns (np.nan, np.nan) if no data is available.
        """
        out_freq = np.nan
        out_conf = np.nan
        if len(self._dominant_freq) > 0:
            dominant_index = self.get_dominant_index()
            if dominant_index is not None:
                out_freq = self._dominant_freq[dominant_index]
                out_conf = self._conf[dominant_index]
        return out_freq, out_conf

    def get_periodicity(self) -> float:
        out_periodicity = np.nan
        if len(self._periodicity) > 0:
            dominant_index = self.get_dominant_index()
            if dominant_index is not None:
                out_periodicity = self._periodicity[dominant_index]
        return out_periodicity

    def get_dominant_freq(self) -> float:
        """
        Return the dominant frequency

        Returns:
            tuple of float:
                - Dominant frequency with the highest confidence.
        """
        out_freq = np.nan
        if len(self._dominant_freq) > 0:
            dominant_index = self.get_dominant_index()
            if dominant_index is not None:
                out_freq = self._dominant_freq[dominant_index]
        return out_freq

    def get_dominant_freq_and_index(self) -> tuple[float, float]:
        """
        Return the dominant frequency and its index in the list.

        Returns:
            tuple of float:
                - Dominant frequency with the highest confidence.
                - Index of the dominant frequency.
                Returns (np.nan, np.nan) if no data is available.
        """
        out_freq = np.nan
        dominant_index = np.nan
        if len(self._dominant_freq) > 0:
            dominant_index = self.get_dominant_index()
            if dominant_index is not None:
                out_freq = self._dominant_freq[dominant_index]
        return out_freq, dominant_index

    def get_dominant_freq_amp_phi(self):
        dominant_index = self.get_dominant_index()
        if dominant_index is not None:
            return (
                self._dominant_freq[dominant_index],
                self._amp[dominant_index],
                self._phi[dominant_index],
            )
        else:
            return np.nan, np.nan, np.nan

    def get_dominant_index(self):
        # or use conf?
        if self._amp is not None and self._amp.size > 0:
            return np.argmax(self._amp)
        elif self._conf is not None and self._conf.size > 0:
            return np.argmax(self._conf)
        else:
            return None

    def is_empty(self):
        """
        Check if the prediction is empty.

        Returns:
            bool: True if source is empty, False otherwise.
        """
        return not self.source

    def get_wave(
        self, freq: float, amp: float, phi: float, t_sampled: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate a cosine wave using the given frequency, amplitude, and phase.

        Args:
            freq (float): Frequency of the cosine wave in Hz.
            amp (float): Amplitude of the wave.
            phi (float): Phase of the wave in radians.
            t_sampled (np.ndarray, optional): Array of time values. Defaults to None.

        Returns:
            np.ndarray: Array of sampled cosine wave values. Returns an empty array
            if the frequency is NaN or sampling is invalid.
        """
        if not np.isnan(freq) and self._n_samples != 0:
            if t_sampled is None:
                t_sampled = self._t_start + np.arange(0, self._n_samples) * 1 / self._freq
            if freq != 0 and freq != self._freq / 2:
                amp *= 2 / self._n_samples
            else:
                amp *= 1 / self._n_samples
            return amp * np.cos(2 * np.pi * freq * t_sampled + phi)
        else:
            return np.array([])

    def get_wave_name(self, freq: float, amp: float, phi: float) -> str:
        """
        Returns a string that describes the cosine wave

        Args:
            freq (float): Frequency of the cosine wave in Hz.
            amp (float): Amplitude of the wave.
            phi (float): Phase of the wave in radians.

        Returns:
            str: Name of the cosine wave at the specified entities
        """
        if not np.isnan(freq) and self._n_samples != 0:
            name = f"{amp:.2e}*cos(2\u03c0*{freq:.2e}*t{'+' if phi >= 0 else '-'}{abs(phi):.2e})"
        else:
            name = ""
        return name

    def get_wave_and_name(
        self, freq: float, amp: float, phi: float, t_sampled: np.ndarray = None
    ) -> tuple[np.ndarray, str]:
        """
        Generate a cosine wave using the given frequency, amplitude, and phase. Returns additionally a string which can be used to label the plots

        Args:
            freq (float): Frequency of the cosine wave in Hz.
            amp (float): Amplitude of the wave.
            phi (float): Phase of the wave in radians.
            t_sampled (np.ndarray, optional): Array of time values. Defaults to None.

        Returns:
            np.ndarray: Array of sampled cosine wave values. Returns an empty array
            str: Name of the cosine wave at the specified entities
        """
        cosine_wave = self.get_wave(freq, amp, phi, t_sampled)
        name = self.get_wave_name(freq, amp, phi)
        return cosine_wave, name

    def get_dominant_wave(self):
        if len(self._dominant_freq) > 0:
            return self.get_wave(*self.get_dominant_freq_amp_phi())
        else:
            return np.array([])

    def disp_dominant_freq_and_conf(self) -> str:
        if not self.is_empty():
            f_d, c_d = self.get_dominant_freq_and_conf()
            if not np.isnan(f_d):
                text = (
                    f"[cyan underline]Prediction results:[/]\n[cyan]Frequency:[/] {f_d:.3e} Hz"
                    f"[cyan]->[/] {np.round(1 / f_d, 4) if f_d > 0 else 0} s\n"
                    f"[cyan]Confidence:[/] {color_pred(c_d)}"
                    f"{np.round(c_d * 100, 2)}[/] %\n"
                )
                periodicity = self.get_periodicity()
                if not np.isnan(periodicity):
                    text += (
                        f"[cyan]Periodicity:[/] {color_pred(periodicity)}"
                        f"{np.round(periodicity * 100, 2)}[/] %\n"
                    )
            else:
                text = (
                    "[cyan underline]Prediction results:[/]\n"
                    "[red]No dominant frequency found[/]\n"
                )

            if len(self._burst_widths) > 0:
                dc = self.duty_cycle
                text += (
                    f"[cyan]Burst width:[/] {color_pred(dc)}"
                    f"{self.burst_width_median:.3f}[/] s"
                    f"  [min={self.burst_width_min:.3f}s, max={self.burst_width_max:.3f}s]\n"
                    f"[cyan]Duty cycle:[/] {color_pred(dc)}"
                    f"{np.round(dc * 100, 2)}[/] %\n"
                )
            return text
        else:
            return ""

    def disp_ranges(self) -> str:
        """
        Generates a formatted string representation of the valid time ranges stored
        in the 'ranges' attribute. Each valid time range is displayed with its start
        and end points rounded to two decimal places. If no valid time ranges
        exist, an empty string is returned.

        Returns:
            str: A formatted string showing all valid time ranges. Returns an empty
            string if no ranges are available.
        """
        if len(self.ranges) > 0:
            text = "[cyan]Valid time segments:\n[/]"
            for start, end in self.ranges:
                text += f"- [{start:.2f},{end:.2f}] sec\n"

            return text
        else:
            return ""

    def display_frequencies_in_ranges(self):
        if len(self.dominant_freq) > 1:
            table = Table(
                show_header=True, header_style="bold cyan", title="Identified segments"
            )
            table.add_column("Window", justify="right", style="white", no_wrap=True)
            table.add_column("Freq (Hz)", justify="right", style="white", no_wrap=True)
            table.add_column("Period (s)", justify="right", style="white", no_wrap=True)
            table.add_column("Conf. (%)", justify="right", style="white", no_wrap=True)
            table.add_column(
                "Time Range (s)", justify="right", style="white", no_wrap=True
            )

            for i in range(1, len(self.dominant_freq)):
                t_range = (
                    f"[{self.ranges[i][0]:.2f}, {self.ranges[i][1]:.2f}]"
                    if len(self.ranges) > i
                    else 0
                )

                table.add_row(
                    str(i - 1),
                    f"{self.dominant_freq[i]:.3e}",
                    f"{np.round(1 / self.dominant_freq[i], 4) if self.dominant_freq[i] > 0 else 0:.4f}",
                    f"{np.round(self.conf[i] * 100, 2):.2f}",
                    t_range,
                )
            return table
        return ""

    def to_dict(self):
        """
        Return a dictionary representation of the Prediction object.

        Returns:
            dict: A copy of all relevant instance attributes.
        """
        return {
            "metric": self._metric,
            "source": self._source,
            "dominant_freq": self._dominant_freq,
            "conf": self._conf,
            "periodicity": self._periodicity,
            "amp": self._amp,
            "phi": self._phi,
            "t_start": self.t_start,
            "t_end": self._t_end,
            "total_bytes": self._total_bytes,
            "freq": self._freq,
            "ranks": self._ranks,
            "n_samples": self._n_samples,
            "top_freqs": self._top_freqs,
            "candidates": self._candidates,
            "burst_widths": self._burst_widths,
            "duty_cycle": self.duty_cycle,
        }

    def to_json(self) -> str:
        """Convert the Prediction object to a JSON-serializable string."""
        return self.convert_json(self.to_dict())

    def convert_json(self, obj):
        """Helper method to recursively convert unsupported types."""
        if isinstance(obj, dict):
            return {k: self.convert_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_json(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        """
        Overload add operator to return a list[dict] of the predictions.

        Args:
            other (Prediction or list): Another prediction instance or list.

        Returns:
        list: Combined list of prediction dictionaries.

        Raises:
            TypeError: If an operand type is unsupported.
        """
        if isinstance(other, Prediction):
            return [self.to_dict(), other.to_dict()]
        elif isinstance(other, list):
            return other + [self.to_dict()]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: 'Prediction' and '{type(other).__name__}'"
            )

    def __bool__(self):
        """
        Boolean check for non-empty prediction.

        Returns:
            bool: True if not empty, else False.
        """
        return not self.is_empty()


def color_pred(value: float) -> str:
    """highlight color according to value

    Args:
        value (float): value between [0,1]

    Returns:
        string: color to be used with Console (from rich)
    """
    color = "[red]"
    if value >= 0.8:
        color = "[green]"
    elif value >= 0.6:
        color = "[blue]"
    elif value >= 0.3:
        color = "[yellow]"
    return color
