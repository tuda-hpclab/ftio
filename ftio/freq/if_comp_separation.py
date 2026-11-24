"""
This module provides utilities for extracting signal components from
time-frequency representations (STFT or ASTFT) using image-processing
techniques. It implements methods for creating binary images from
frequency-domain peaks, identifying significant components via
connected-component analysis, and refining them using z-score and
harmonic-removal techniques.

Author: josefinez
Editor: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Oct 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_prominences

from ftio.analysis.anomaly_detection import remove_harmonics
from ftio.analysis.anomaly_detection import z_score as z_score

# TODO:
# binary image:
# - denoise signal (preprocessing?)
# - adjust prominence threshold
# component_linking:
# - set area threshold depending on sampling frequency and minimum win size
#   (shortest relevant I/O phase period?)
# - 10-connected neighbouring set

# Rankine, L., Mesbah, M., & Boashash, B. (2007).
# IF estimation for multicomponent signals using image processing
# techniques in the time-frequency domain.
# Signal Processing, 87(6), 1234-1250.


def binary_image(Zxx):
    """
    Creates binary image with 255 at frequency domain peaks.

    Args:
        Zxx (np.ndarray): array of FFTs for each point

    Returns:
        bin_im (np.ndarray): binary image
    """
    bin_im = np.zeros_like(Zxx, dtype="uint8")
    rows = np.shape(Zxx)[0]

    for i in range(0, rows):
        freqs = np.abs(Zxx[i])
        peaks = find_peaks(freqs)

        prom = peak_prominences(freqs, peaks[0])[0]

        if prom.size > 0:
            for ind in range(0, len(prom)):
                if prom[ind] > 0.01:
                    _ind = peaks[0][ind]
                    bin_im[i][_ind] = 255

    return bin_im


def binary_image_nprom(Zxx, n=3):
    bin_im = np.zeros_like(Zxx, dtype="uint8")
    rows = np.shape(Zxx)[0]

    for i in range(0, rows):
        freqs = np.abs(Zxx[i])
        peaks = find_peaks(freqs)
        prom = peak_prominences(freqs, peaks[0])[0]

        prom_filtered = []
        if len(peaks[0]) > n:
            prom_filtered = np.argpartition(prom, -3)[-3:]
        else:
            prom_filtered = prom

        if len(prom_filtered) > 0:
            for ind in range(0, len(prom_filtered)):
                if prom_filtered[ind] > 0.01:
                    _ind = peaks[0][ind]
                    print(_ind)
                    bin_im[i][_ind] = 255

    return bin_im


def binary_image_zscore(Zxx, freq, args):
    bin_im = np.zeros_like(Zxx, dtype="uint8")
    rows = np.shape(Zxx)[0]

    for i in range(0, rows):
        yf = np.abs(Zxx[i])
        n = len(yf)
        freq_arr = freq * np.arange(0, n) / n
        original_engine = args.engine
        args.engine = "none"
        indices = z_score(yf, freq_arr, args)[0]
        args.engine = original_engine

        for ind in indices:
            bin_im[i][ind] = 255

    return bin_im


def binary_image_zscore_extended(Zxx, freq, args):
    bin_im = np.zeros_like(Zxx, dtype="uint8")
    rows = np.shape(Zxx)[0]

    zscore_freq = []

    for i in range(0, rows):
        freqs = np.abs(Zxx[i])
        peaks_ = find_peaks(freqs)
        n = len(freqs)
        freq_arr = freq * np.arange(0, n) / n
        original_engine = args.engine
        args.engine = "none"
        indices = z_score(freqs, freq_arr, args)[0]
        args.engine = original_engine

        peaks = remove_harmonics(freq_arr, freq_arr, peaks_[0])[0]
        prom = peak_prominences(freqs, peaks)[0]

        for ind in indices:
            if ind not in zscore_freq:
                zscore_freq.append(ind)
                zscore_freq.append(ind - 1)
                zscore_freq.append(ind + 1)

        if prom.size > 0:
            for ind in range(0, len(prom)):
                if prom[ind] > 0.01:
                    # _ind = peaks[0][ind]
                    _ind = peaks[ind]
                    if _ind in zscore_freq:
                        bin_im[i][_ind] = 255

    return bin_im


# https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/
def component_linking(image, fs, win_len):
    """
    Extracts individual signal components by connecting frequency peaks of binary image.

    Args:
        image (np.ndarray): binary image

    Returns:
        tuple:
            tuple:
                start (int): start of component as index of original signal
                end (int): start of component as index of original signal
            comp (np.ndarray): frequency index of component at each time
    """

    frame = np.array(image, dtype="uint8")

    analysis = cv2.connectedComponentsWithStats(frame, 8, cv2.CV_32S)
    totalLabels, label_ids, values, centroid = analysis

    output = np.zeros(image.shape, dtype="uint8")

    components = []

    # Loop through each component
    for i in range(1, totalLabels):
        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]

        if area > 80:
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)

            components.append(i)

    result = []
    for i in components:
        indices, freqs = np.where(label_ids == i)

        start = indices[0]
        end = indices[-1]
        time = start, end

        freqs_mean = np.mean(freqs)
        freqs_median = np.median(freqs)

        freq_mean = freqs_mean * fs / win_len
        freq_median = freqs_median * fs / win_len

        comp = time, freq_mean, freq_median
        result.append(comp)

    filename = "cv2_image.jpg"
    cv2.imwrite(filename, frame)
    filename = "cv2_filtered.jpg"
    cv2.imwrite(filename, output)

    # cv2.imshow("Image", frame)
    # cv2.imshow("Filtered Components", output)
    # cv2.waitKey(15000)

    return result
