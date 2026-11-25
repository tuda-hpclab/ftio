"""
This file provides functions for periodicity analysis in frequency data using various methods,
including Recurrence Period Density Entropy, Spectral Flatness, Correlation, Correlation of Individual Periods, and Peak Sharpness.

Author: Anton Holderied
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Jul 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

# all
import numpy as np
from rich.panel import Panel

# find_peaks
from scipy.stats import kurtosis

# Isolation forest
# Lof
from ftio.analysis._correlation import correlation


def new_periodicity_scores(
    amp: np.ndarray,
    signal: np.ndarray,
    prediction,
    args,
    peak_window_half_width: int = 3,
) -> str:

    def compute_peak_sharpness(
        spectrum_amp: np.ndarray,
        peak_indices_in_spectrum: np.ndarray,
        window_half_width: int,
    ) -> float:
        """
        Computes the average peak sharpness (excess kurtosis) for specified peak regions.
        Higher values indicate more peaked regions.
        """
        sharpness_scores = []
        for peak_idx in peak_indices_in_spectrum:
            start_idx = max(0, peak_idx - window_half_width)
            end_idx = min(len(spectrum_amp), peak_idx + window_half_width + 1)
            peak_window_spectrum = spectrum_amp[start_idx:end_idx]
            if peak_window_spectrum.size < 4:
                sharpness_scores.append(-2.0)
                continue
            if np.std(peak_window_spectrum) < 1e-9:
                sharpness_scores.append(-2.0)
                continue
            k = kurtosis(peak_window_spectrum, fisher=True, bias=False)
            sharpness_scores.append(k)
        if not sharpness_scores:
            return -2.0

        return float(np.mean(sharpness_scores))

    def compute_rpde(freq_arr: np.ndarray) -> float:
        safe_array = np.where(freq_arr <= 0, 1e-12, freq_arr)
        sum = np.sum(safe_array * np.log(safe_array))
        max = np.log(safe_array.size)
        rpde_score = 1 - (np.divide(sum, -max))
        return rpde_score

    def compute_spectral_flatness(freq_arr: np.ndarray) -> float:
        safe_spectrum = np.where(freq_arr <= 0, 1e-12, freq_arr)
        geometric_mean = np.exp(np.mean(np.log(safe_spectrum)))
        arithmetic_mean = np.mean(safe_spectrum)
        spectral_flatness_score = 1 - float(geometric_mean / arithmetic_mean)
        return spectral_flatness_score

    def signal_correlation(
        freq: float,
        sampling_rate: float,
        signal: np.ndarray,
        phi: float,
        start_time: float,
        text: str = "",
    ) -> float:
        dt = 1 / sampling_rate
        t = start_time + dt * np.arange(len(signal))
        waveform = np.cos(2 * np.pi * freq * t + phi)

        correlation_result = correlation(waveform, signal)
        return correlation_result

    def ind_period_correlation(
        freq: float,
        sampling_rate: float,
        signal: np.ndarray,
        phi: float,
        start_time: float,
        text: str = "",
    ) -> tuple[float, str]:
        """
        Calculates a length-weighted correlation for a signal.

        The function computes a weighted average of correlations from each period.
        The weighting is based solely on how complete the period is, giving
        less weight to partial periods at the beginning or end of the signal.
        """
        dt = 1 / sampling_rate
        t = start_time + dt * np.arange(len(signal))
        waveform = np.cos(2 * np.pi * freq * t + phi)

        period = 1 / freq
        phase_offset = phi / (2 * np.pi * freq)

        correlations = []
        segment_lengths = []

        i = 0
        while True:
            center_time = (i * period) - start_time - phase_offset
            begin = round((center_time - 0.5 * period) * sampling_rate)
            end = round((center_time + 0.5 * period) * sampling_rate)

            if end < 1:
                i += 1
                continue

            begin_clamped = max(begin, 0)
            end_clamped = min(end, len(signal))

            if begin_clamped < end_clamped:
                correlation_result = correlation(
                    waveform[begin_clamped:end_clamped], signal[begin_clamped:end_clamped]
                )
                correlations.append(correlation_result)
                segment_lengths.append(end_clamped - begin_clamped)

                text += (
                    f"[green]Period {i:3d} (indices {begin_clamped:5d}:{end_clamped:5d}): "
                    f"[black]Correlation = {correlation_result:.4f}\n"
                )

            if end >= len(signal) - 1:
                break
            i += 1

        if not correlations:
            text += "[red]No full periods were found in the signal.\n"
            return 0.0, text

        full_period_samples = round(period * sampling_rate)

        # The weights are now just the ratio of the segment length to a full period length.
        length_weights = [sl / full_period_samples for sl in segment_lengths]

        for i, weight in enumerate(length_weights):
            text += (
                f"[blue]Weight for period {i:3d} (completeness): [black]{weight:.4f}\n"
            )

        # Calculate the weighted average using only the length weights.
        weighted_correlation = np.average(correlations, weights=length_weights)

        text += f"\n[bold magenta]Final Length-Weighted Correlation: {weighted_correlation:.4f}[/bold magenta]\n"

        return weighted_correlation, text

    def weighted_ind_period_correlation(
        freq: float,
        sampling_rate: float,
        signal: np.ndarray,
        phi: float,
        start_time: float,
        text: str = "",
    ) -> tuple[float, str]:
        """
        The function iterates through the signal period by period, calculating the
        correlation for each. It then computes a final weighted average of these
        correlations.

        The weighting for each period is determined by two factors:
        1.  **Length Weight**: The proportion of the period that is actually
            present in the signal (especially relevant for start/end periods).
        2.  **Recency Weight**: A linear ramp from 0.3 for the first period to
            1.0 for the last period, giving more importance to recent data.
        """
        if freq == 0.0:
            return 0.0, ""

        dt = 1 / sampling_rate
        t = start_time + dt * np.arange(len(signal))
        waveform = np.cos(2 * np.pi * freq * t + phi)

        period = 1 / freq
        phase_offset = phi / (2 * np.pi * freq)

        # --- Store individual results to be weighted later ---
        correlations = []
        segment_lengths = []

        i = 0
        while True:
            center_time = (i * period) - start_time - phase_offset
            begin = round((center_time - 0.5 * period) * sampling_rate)
            end = round((center_time + 0.5 * period) * sampling_rate)

            if end < 1:
                i += 1
                continue

            # Clamp indices to be within the signal bounds
            begin_clamped = max(begin, 0)
            end_clamped = min(end, len(signal))  # Use len(signal) for slicing

            # Calculate correlation for the valid segment
            if begin_clamped < end_clamped:
                correlation_result = correlation(
                    waveform[begin_clamped:end_clamped], signal[begin_clamped:end_clamped]
                )
                correlations.append(correlation_result)
                segment_lengths.append(end_clamped - begin_clamped)

                text += (
                    f"[green]Period {i:3d} (indices {begin_clamped:5d}:{end_clamped:5d}): "
                    f"[black]Correlation = {correlation_result:.4f}\n"
                )

            if end >= len(signal) - 1:
                break
            i += 1

        # --- Perform the weighting if any periods were found ---
        if not correlations:
            text += "[red]No full periods were found in the signal.\n"
            return 0.0, text

        num_periods = len(correlations)
        full_period_samples = round(period * sampling_rate)

        # 1. Recency weights: linear ramp from 0.3 to 1.0
        recency_weights = np.linspace(0.3, 1.0, num=num_periods)

        # 2. Combine with length weights and calculate the final weighted average
        final_weights = []
        for i in range(num_periods):
            # Length weight: actual segment length / full period length
            length_weight = segment_lengths[i] / full_period_samples
            # Final weight is the product of both factors
            combined_weight = recency_weights[i] * length_weight
            final_weights.append(combined_weight)

            text += (
                f"[blue]Weight for period {i:3d}: "
                f"[black]Recency={recency_weights[i]:.2f}, Length={length_weight:.2f} -> "
                f"Combined={combined_weight:.4f}\n"
            )

        # np.average handles the sum(value * weight) / sum(weight) calculation
        weighted_correlation = np.average(correlations, weights=final_weights)

        text += f"\n[magenta]Final Weighted Correlation: {weighted_correlation:.4f}[/]\n"

        return weighted_correlation, text

    def parallel_period_correlation(
        freq: float,
        sampling_rate: float,
        signal: np.ndarray,
        phi: float,
        start_time: float,
        workers: int,
        text: str = "",
    ) -> str:
        dt = 1 / sampling_rate
        t = start_time + dt * np.arange(len(signal))
        waveform = np.cos(2 * np.pi * freq * t + phi)

        period = 1 / freq
        phase_offset = phi / (2 * np.pi * freq)

        # Precompute all (begin, end, i) intervals
        intervals = []
        i = 0
        while True:
            center_time = (i * period) - start_time - phase_offset
            begin = round((center_time - 0.5 * period) * sampling_rate)
            end = round((center_time + 0.5 * period) * sampling_rate)
            if end < 1:
                i += 1
                continue
            begin = max(begin, 0)
            end = min(end, len(signal) - 1)
            intervals.append((begin, end, i))
            if end >= len(signal) - 1:
                break
            i += 1

        def compute_correlation(args):
            b, e, idx = args
            corr_res = correlation(waveform[b:e], signal[b:e])
            return idx, b, e, corr_res

        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(compute_correlation, interval) for interval in intervals
            ]
            for future in as_completed(futures):
                results.append(future.result())

        # Sort results by period index to keep order
        results.sort(key=lambda x: x[0])

        for idx, b, e, corr_res in results:
            text += (
                f"[green]Correlation for period {idx:3d} with indices {b:5d},{e:5d}: "
                f"[black]{corr_res:.4f} \n"
            )
        return text

    text = ""
    if args.periodicity_detection:
        if args.psd:
            amp = amp * amp / len(amp)
            # text = "[green]Spectrum[/]: Power spectrum\n"
        indices = np.arange(1, int(len(amp) / 2) + 1)
        amp_tmp = np.array(2 * amp[indices])
        # norm the data
        amp_tmp = amp_tmp / amp_tmp.sum() if amp_tmp.sum() > 0 else amp_tmp

        if "rpde" in args.periodicity_detection.lower():
            rpde_score = compute_rpde(amp_tmp)
            text += f"\n[green]RPDE score: [black]{rpde_score:.4f}[/]\n"
            prediction.periodicity = rpde_score
            if args.n_freq > 0:
                prediction.top_freqs["periodicity"] = np.full(
                    len(prediction.top_freqs["freq"]), rpde_score
                )
        if "sf" in args.periodicity_detection.lower():
            sf_score = compute_spectral_flatness(amp_tmp)
            text += f"\n[green]Spectral flatness score: [black]{sf_score:.4f}[/]\n"
            prediction.periodicity = sf_score
            if args.n_freq > 0:
                prediction.top_freqs["periodicity"] = np.full(
                    len(prediction.top_freqs["freq"]), sf_score
                )
        if (
            "corr" in args.periodicity_detection.lower()
            and len(prediction.dominant_freq) != 0
        ):
            sampling_freq = prediction.freq
            start_time = prediction.t_start
            if args.n_freq > 0:
                for i, (dominant_freq, phi) in enumerate(
                    zip(
                        prediction.top_freqs["freq"],
                        prediction.top_freqs["phi"],
                        strict=False,
                    )
                ):
                    score = signal_correlation(
                        dominant_freq, sampling_freq, signal, phi, start_time
                    )
                    prediction.top_freqs["periodicity"][i] = score
                    text += (
                        f"[green]Overall correlation score for frequency {dominant_freq:.4f}Hz: "
                        f"[black]{score:.4f}[/]\n"
                    )
                prediction.periodicity = prediction.top_freqs["periodicity"][0]
            else:
                freq, phi = prediction.dominant_freq[0], prediction.phi[0]
                score = signal_correlation(freq, sampling_freq, signal, phi, start_time)
                prediction.periodicity = score
                text += (
                    f"[green]Overall correlation score for frequency {freq:.4f}Hz: "
                    f"[black]{score:.4f}[/]\n"
                )
        if (
            "ind" in args.periodicity_detection.lower()
            and len(prediction.dominant_freq) != 0
        ):
            sampling_freq = prediction.freq
            start_time = prediction.t_start
            if args.n_freq > 0:
                for i, (dominant_freq, phi) in enumerate(
                    zip(
                        prediction.top_freqs["freq"],
                        prediction.top_freqs["phi"],
                        strict=False,
                    )
                ):
                    print(dominant_freq)
                    score, new_text = weighted_ind_period_correlation(
                        dominant_freq, sampling_freq, signal, phi, start_time
                    )
                    prediction.top_freqs["periodicity"][i] = score
                    text += f"[green]Individual correlation score for frequency {dominant_freq:.4f}Hz:[/]\n"
                    text += new_text
                prediction.periodicity = prediction.top_freqs["periodicity"][0]
            else:
                freq, phi = prediction.dominant_freq[0], prediction.phi[0]
                score, new_text = weighted_ind_period_correlation(
                    freq, sampling_freq, signal, phi, start_time
                )
                prediction.periodicity = score
                text += f"[green]Individual correlation score for frequency {freq:.4f}Hz:[/]\n"
                text += new_text

        title = "Periodicity Analysis"

        text = Panel.fit(
            text[:-1],
            style="white",
            border_style="green",
            title=title,
            title_align="left",
        )
    return text
