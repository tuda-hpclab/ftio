"""
This file provides functions for detecting anomalies in frequency data using various methods,
including Z-Score, DBSCAN, Isolation Forest, Local Outlier Factor, and peak detection. It also
includes utilities for removing harmonics and visualizing outliers.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

# all
import numpy as np
from kneed import KneeLocator
from rich.panel import Panel

# find_peaks
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# Isolation forest
from sklearn.ensemble import IsolationForest

# Lof
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from ftio.plot.anomaly_plot import plot_decision_boundaries, plot_outliers
from ftio.plot.cepstrum_plot import plot_cepstrum


def outlier_detection(
    amp: np.ndarray, freq_arr: np.ndarray, args
) -> tuple[list[float], np.ndarray, Panel]:
    """Find the outliers in the samples

    Args:
        A (list[float]): Amplitudes array
        freq_arr (list[float]): frequency array
        args (object, optional): arguments containing: outlier detection method (Z-Score, DB-Scan).
        Defaults to 'Z-score'.

    Returns:
        dominant_index (list[float]): indecies of dominant frequencies
        conf (list[float]): confidence in the predictions
        Panel: text to display using rich
    """
    methode = args.outlier
    text = ""
    if methode.lower() in ["z-score", "zscore"]:
        dominant_index, conf, text = z_score(amp, freq_arr, args)
        title = "Z-score"
    elif methode.lower() in ["dbscan", "db-scan", "db"]:
        dominant_index, conf, text = db_scan(amp, freq_arr, args)
        title = "DB-Scan"
    elif methode.lower() in ["isolation_forest", "forest"]:
        dominant_index, conf, text = isolation_forest(amp, freq_arr, args)
        title = "Isolation Forest"
    elif methode.lower() in ["local outlier factor", "lof"]:
        dominant_index, conf, text = lof(amp, freq_arr, args)
        title = "Local Outlier Factor"
    elif methode.lower() in ["find peaks", "peaks", "peak"]:
        dominant_index, conf, text = peaks(amp, freq_arr, args)
        title = "Find Peaks"
    else:
        dominant_index, conf = [], np.array([])
        raise NotImplementedError("Unsupported method selected")
    # sort dominant index according to confidence
    # if len(dominant_index) > 1:
    #     print(f"conf: {conf} and dominant_index {dominant_index}")
    #     tmp = conf[dominant_index]
    #     dominant_index = np.array(dominant_index)
    #     dominant_index = list(dominant_index[np.argsort(tmp)])
    text = Panel.fit(
        text[:-1],
        style="white",
        border_style="green",
        title=title,
        title_align="left",
    )

    return dominant_index, conf, text


# ?#################################
# ? Z-score
# ?#################################
def z_score(
    amp: np.ndarray, freq_arr: np.ndarray, args
) -> tuple[list[float], np.ndarray, str]:
    """calculates the outliers using zscore

    Args:
        amp (np.ndarray): amplitude or psd
        freq_arr (np.ndarray): frequencies
        args (argsparse): arguments

    Returns:
        tuple[list[float], np.ndarray, str]: [dominant frequency/ies, confidence, text]
    """
    text = "[green]Spectrum[/]: Amplitude spectrum\n"
    if args.psd:
        amp = amp * amp / len(amp)
        text = "[green]Spectrum[/]: Power spectrum\n"

    indices = np.arange(1, int(len(amp) / 2) + 1)
    amp_tmp = np.array(2 * amp[indices])
    # norm the data
    amp_tmp = amp_tmp / amp_tmp.sum() if amp_tmp.sum() > 0 else amp_tmp

    tol = args.tol
    dominant_index = []
    mean = np.mean(amp_tmp)
    std = np.std(amp_tmp)
    z_k = abs(amp_tmp - mean) / std if std > 0 else np.zeros(len(amp_tmp))
    conf = np.zeros(len(z_k))
    # find outliers
    index = (
        np.where((z_k / np.max(z_k) > tol) & (z_k > 3))
        if len(z_k) > 0 and np.max(z_k) > 0
        else (np.array([], dtype=np.int64),)
    )
    text += f"[green]mean[/]: {mean / np.sum(amp_tmp) if np.sum(amp_tmp) else 0:.3e}\n[green]std[/]: {std:.3e}\n"
    text += f"Frequencies with Z-score > 3 -> [green]{len(z_k[z_k > 3])}[/] candidates\n"
    text += (
        f"         + Z > Z_max*{tol * 100}% > 3 -> [green]{len(index[0])}[/] candidates\n"
    )
    if len(index[0]) > 3:
        for counter, i in enumerate(index[0], start=1):
            text += f"           {counter}) {1 / freq_arr[i + 1]:.3f} s: z_k = {z_k[i] / np.max(z_k):.2f}\n"

    index, removed_index, msg = remove_harmonics(freq_arr, amp_tmp, indices[index[0]])
    text += msg

    if len(index) == 0:
        text += "[red]No dominant frequency -> Signal might be not periodic[/]\n"
    else:
        removed_index = [i - 1 for i in removed_index]  # tmp starts at 1
        # conf[index] = (z_k[index]/max_z  + z_k[index]/np.sum(z_k[index]) + 1/np.sum(z_k > 3))/3
        # calculate the confidence:
        # 1) check z_k/max_zk > tol
        tmp = z_k / np.max(z_k) > tol
        tmp[removed_index] = False
        conf[tmp] += z_k[tmp] / np.sum(z_k[tmp])

        # 2) check z_k > 0
        tmp = z_k > 3
        tmp[removed_index] = False
        conf[tmp] += z_k[tmp] / np.sum(z_k[tmp])
        conf = conf / 2
        conf = np.array(conf)

        # get dominant index
        dominant_index, msg = dominant(index, freq_arr, conf)
        text += msg

    if "plotly" in args.engine:
        i = np.repeat(1, len(indices))
        if len(dominant_index) != 0:
            i[np.array(dominant_index) - 1] = -1
        plot_outliers(args, freq_arr, amp, indices, conf, i)
        if args.cepstrum:
            plot_cepstrum(args, freq_arr, amp, indices)

    return dominant_index, conf, text


# ?#################################
# ? DB-Scan
# ?#################################
def db_scan(
    amp: np.ndarray, freq_arr: np.ndarray, args
) -> tuple[list[float], np.ndarray, str]:
    """calculates the outliers using dbscan

    Args:
        amp (np.ndarray): amplitude or psd
        freq_arr (np.ndarray): frequencies
        args (argsparse): arguments

    Returns:
        tuple[list[float], np.ndarray, str]: [dominant frequency/ies, confidence]
    """
    text = "[green]Spectrum[/]: Amplitude spectrum\n"
    if args.psd:
        amp = amp * amp / len(amp)
        text = "[green]Spectrum[/]: Power spectrum\n"

    indecies = np.arange(1, int(len(amp) / 2) + 1)
    amp_tmp = np.array(2 * amp[indecies])
    freq_arr_tmp = np.array(freq_arr[indecies])
    min_pts = 2

    # norm the data
    # d = np.vstack((freq_arr_tmp / freq_arr_tmp.max(), amp_tmp / amp_tmp.max())).T
    #! norm over sum for amplitude with power spectrum
    d = np.vstack((freq_arr_tmp / freq_arr_tmp.max(), amp_tmp / amp_tmp.sum())).T

    eps_mode = "range"
    if eps_mode == "avr":
        text += "Calculating eps using average\n"
        try:
            eps = np.sqrt(pow(d[:, 1].mean(), 2) + pow(d[1, 0] - d[0, 0], 2))
        except ValueError:
            eps = 0.1
        conf = d[:, 1] / d[:, 1].max()
    elif eps_mode == "median":
        text += "Calculating eps using median\n"
        eps = np.sqrt(pow(np.median(d[:, 1]), 2) + pow(d[1, 0] - d[0, 0], 2))
        conf = d[:, 1] / d[:, 1].max()
    elif eps_mode == "range":
        text += "Calculating eps using range\n"
        eps = np.sqrt(
            pow((d[:, 1].max() - d[:, 1].min()) * (1 - args.tol), 2)
            + pow(d[1, 0] - d[0, 0], 2)
        )
        conf = d[:, 1] / d[:, 1].max()
    else:  # find distance using knee method
        text += "Calculating eps using knee method\n"
        observation = int(len(amp) / 5)
        nbrs = NearestNeighbors(n_neighbors=observation).fit(d)
        # Find the k-neighbors of a point
        neigh_dist, _ = nbrs.kneighbors(d)
        # sort the neighbor distances (lengths to points) in ascending order
        sort_neigh_dist = np.sort(neigh_dist, axis=0)
        k_dist = sort_neigh_dist[:, observation - 1]  # sort_neigh_dist[:, 4]
        # plt.plot(k_dist)
        # plt.ylabel("k-NN distance")
        # plt.xlabel("Sorted observations (%ith NN)"%observation)
        # plt.show()
        kneedle = KneeLocator(
            x=range(1, len(neigh_dist) + 1),
            y=k_dist,
            S=1.0,
            curve="concave",
            direction="increasing",
            online=True,
        )
        eps = kneedle.knee_y
        conf = k_dist

    text += f"eps = [green]{eps:.4f}[/]    Minpoints = [green]{min_pts}[/]\n"
    model = DBSCAN(eps=eps, min_samples=min_pts)
    model.fit(d)
    dominant_index = model.labels_
    # normalize like the remaing methods
    dominant_index[dominant_index == -1] = dominant_index[dominant_index == -1] - 1
    dominant_index = dominant_index + 1

    if "plotly" in args.engine:
        plot_outliers(args, freq_arr, amp, indecies, conf, dominant_index, d, eps)

    clean_index, _, msg = remove_harmonics(
        freq_arr, amp_tmp, indecies[dominant_index == -1]
    )
    dominant_index, text_d = dominant(clean_index, freq_arr, conf)

    return dominant_index, conf, text + msg + text_d


# ?#################################
# ? Isolation Forest
# ?#################################
def isolation_forest(
    amp: np.ndarray, freq_arr: np.ndarray, args
) -> tuple[list[float], np.ndarray, str]:
    """calculates the outliers using isolation forest

    Args:
        amp (np.ndarray): amplitude or psd
        freq_arr (np.ndarray): frequencies
        args (argsparse): arguments

    Returns:
        tuple[list[float], np.ndarray, str]: [dominant frequency/ies, confidence]
    """
    text = "[green]Spectrum[/]: Amplitude spectrum\n"
    if args.psd:
        amp = amp * amp / len(amp)
        text = "[green]Spectrum[/]: Power spectrum\n"

    indices = np.arange(1, int(len(amp) / 2) + 1)
    amp_tmp = np.array(2 * amp[indices])
    freq_arr_tmp = np.array(freq_arr[indices])
    # norm the data
    # d = np.vstack((freq_arr_tmp / freq_arr_tmp.max(), amp_tmp / amp_tmp.max())).T
    #! norm over sum for amplitude with power spectrum
    d = np.vstack((freq_arr_tmp / freq_arr_tmp.max(), amp_tmp / amp_tmp.sum())).T

    model = IsolationForest(contamination=0.001, warm_start=True)
    # model = IsolationForest(contamination=float(0.001),warm_start=True, n_estimators=2)
    # model = IsolationForest(warm_start=True)
    model.fit(d)
    conf = model.decision_function(d)
    conf = norm_conf(conf)
    dominant_index = model.predict(d)

    if "plotly" in args.engine:
        plot_outliers(args, freq_arr, amp, indices, conf, dominant_index, d)
        plot_decision_boundaries(model, d, conf)

    clean_index, _, msg = remove_harmonics(
        freq_arr, amp_tmp, indices[dominant_index == -1]
    )
    dominant_index, text_d = dominant(clean_index, freq_arr, conf)

    return dominant_index, abs(conf), text + msg + text_d


# ?#################################
# ? Odin
# ?#################################
def lof(
    amp: np.ndarray, freq_arr: np.ndarray, args
) -> tuple[list[float], np.ndarray, str]:
    """calculates the outliers using isolation lof

    Args:
        amp (np.ndarray): amplitude or psd
        freq_arr (np.ndarray): frequencies
        args (argsparse): arguments

    Returns:
        tuple[list[float], np.ndarray, str]: [dominant frequency/ies, confidence]
    """
    text = "[green]Spectrum[/]: Amplitude spectrum\n"
    if args.psd:
        amp = amp * amp / len(amp)
        text = "[green]Spectrum[/]: Power spectrum\n"

    indices = np.arange(1, int(len(amp) / 2) + 1)
    amp_tmp = np.array(2 * amp[indices])
    freq_arr_tmp = np.array(freq_arr[indices])

    # norm the data
    # d = np.vstack((freq_arr_tmp / freq_arr_tmp.max(), amp_tmp / amp_tmp.max())).T
    #! norm over sum for amplitude with power spectrum
    d = np.vstack((freq_arr_tmp / freq_arr_tmp.max(), amp_tmp / amp_tmp.sum())).T

    model = LocalOutlierFactor(contamination=0.001, novelty=True)
    model.fit(d)
    conf = model.decision_function(d)
    dominant_index = model.predict(d)
    conf = norm_conf(conf)

    # plot
    if "plotly" in args.engine:
        plot_outliers(args, freq_arr, amp, indices, conf, dominant_index, d)

    clean_index, _, msg = remove_harmonics(
        freq_arr, amp_tmp, indices[dominant_index == -1]
    )
    dominant_index, text_d = dominant(clean_index, freq_arr, conf)

    return dominant_index, abs(conf), text + msg + text_d


# ?#################################
# ? find_peaks
# ?#################################
def peaks(
    amp: np.ndarray, freq_arr: np.ndarray, args
) -> tuple[list[float], np.ndarray, str]:
    """calculates the outliers using isolation lof

    Args:
        amp (np.ndarray): amplitude or psd
        freq_arr (np.ndarray): frequencies
        args (argsparse): arguments

    Returns:
        tuple[list[float], np.ndarray, str]: [dominant frequency/ies, confidence]
    """
    text = "[green]Spectrum[/]: Amplitude spectrum\n"
    if args.psd:
        amp = amp * amp / len(amp)
        text = "[green]Spectrum[/]: Power spectrum\n"

    indices = np.arange(1, int(len(amp) / 2) + 1)
    amp_tmp = np.array(2 * amp[indices])
    freq_arr_tmp = np.array(freq_arr[indices])
    # Normalize the data
    # Option 1: normalize amplitude by max value (commented out)
    # d = np.vstack((freq_arr_tmp / freq_arr_tmp.max(), amp_tmp / amp_tmp.max())).T

    # Option 2: normalize amplitude by total sum (power spectrum normalization)
    d = np.vstack((freq_arr_tmp / freq_arr_tmp.max(), amp_tmp / amp_tmp.sum())).T

    # Threshold for peak detection
    # limit = args.tol * np.mean(d[:, 1])
    limit = args.tol * np.max(d[:, 1])
    # threshold = limit  # if limit > 0.2 else 0.2

    # Detect peaks in normalized amplitude
    found_peaks, _ = find_peaks(d[:, 1], height=limit)
    # print(f"Found peaks: {found_peaks}")

    # Create binary confidence mask (1 for peaks, 0 elsewhere)
    conf = np.zeros(len(d[:, 1]), dtype=int)
    conf[found_peaks] = 1

    # Mark peak indices with -1 in dominant_index array
    dominant_index = np.repeat(1, len(d[:, 1]))
    dominant_index[found_peaks] = -1

    # plot
    if "plotly" in args.engine:
        plot_outliers(args, freq_arr, amp, indices, conf, dominant_index, d)

    clean_index, _, msg = remove_harmonics(
        freq_arr, amp_tmp, indices[dominant_index == -1]
    )
    dominant_index, text_d = dominant(clean_index, freq_arr, conf)

    return dominant_index, abs(conf), text + msg + text_d


def dominant(
    dominant_index: np.ndarray, freq_arr: np.ndarray, conf: np.ndarray
) -> tuple[list[float], str]:
    """_summary_

    Args:
        dominant_index (array): found indecies of dominant frequencies
        freq_arr (array): array of frequencies
        conf (float): value between -1 (strong outlier) and 1 (no outlier)

    Returns:
        dominant_index: set to 0 if more than three were found. Also remove harmonics
        text: text information
    """
    text = ""
    out = []
    if len(dominant_index) > 0:
        for i in dominant_index:
            if any(freq_arr[i] % freq_arr[out] < 0.00001):
                text += f"[yellow]Ignoring harmonic at: {freq_arr[i]:.3e} Hz (T = {1 / freq_arr[i] if freq_arr[i] > 0 else 0:.3f} s, k = {i}) -> confidence: {abs(conf[i - 1]) * 100:.3f}%[/]\n"
            else:
                text += f"Dominant frequency at: [green]{freq_arr[i]:.3e} Hz (T = {1 / freq_arr[i] if freq_arr[i] > 0 else 0:.3f} s, k = {i}) -> confidence: {abs(conf[i - 1]) * 100:.3f}%[/]\n"
                out.append(i)
            if len(out) > 2:
                text = "[red]Too many dominant frequencies -> Signal might be not periodic[/]\n"
                out = []
                break
    else:
        text = "[red]No dominant frequencies found -> Signal might be not periodic[/]\n"

    return out, text


def remove_harmonics(freq_arr, amp_tmp, index_list) -> tuple[np.ndarray, list, str]:
    """Removes harmonics

    Args:
        freq_arr (_type_): frequency array
        amp_tmp (_type_): amplitude or power array
        index_list (_type_): list of indecies starting at 1

    Returns:
        np.ndarray: indecies without harmonics
        list: removed harmonics
        str: text to print
    """
    seen = []
    removed = []
    msg = ""
    flag = True
    # sort according to amplitude in descending order
    # index_list = index_list[np.argsort(-amp_tmp[index_list])]
    for ind in index_list:
        if seen:
            flag = True
            for value in seen:
                if freq_arr[ind] % freq_arr[value] < 0.00001 and ind != value:
                    msg += (
                        f"[yellow]Ignoring harmonic at: {freq_arr[ind]:.3e} Hz "
                        f"(T = {1 / freq_arr[ind] if freq_arr[ind] > 0 else 0:.3f} s, k = {ind})[/]\n"
                    )
                    removed.append(ind)
                    flag = False
                    break
            if flag and ind not in seen:
                seen.append(ind)
        else:
            seen.append(ind)
    return np.array(seen), removed, msg


def norm_conf(conf: np.ndarray):
    # Normalize so that 1.0 = strongest outlier, 0.0 = strongest inlier
    conf_min = conf.min()
    conf_max = conf.max()

    if conf_min == conf_max:
        # All values are the same; treat all as neutral (e.g., 0.0 outlier score)
        conf = np.zeros_like(conf)
    elif conf_min == 0:
        # Avoid division by zero when conf_min is 0
        conf = 1.0 - (conf / conf_max)
    else:
        conf = (conf_max - conf) / (conf_max - conf_min)

    return conf
