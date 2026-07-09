"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import numpy as np
from numba import jit


class Bandwidth:
    """this class contains the bandwidth + time metrics at the
        different levels:
        1) request level: subscripted ind (for individual requests)
    _ovr
        2) Rank level metrics: avrerage (avr) or sum. Represent either the
        sum/avr of the ind level metrics during a phase.

        3) Application level metrics:  subscripted overlap. Calculated by
        overlapping the previous metrics. The most accurate one is ind, however,
        it has the highest overhead.
    """

    def __init__(self, b, io_type, args):  # b = values["bandwidth"]
        # App level
        self.b_overlap_sum = []
        self.b_overlap_avr = []
        self.t_overlap = []
        self.b_overlap_ind = []
        self.t_overlap_ind = []
        # Rank level
        self.b_rank_sum = []
        self.b_rank_avr = []
        self.t_rank_s = []
        self.t_rank_e = []
        # Ind level
        self.b_ind = []
        self.t_ind_s = []
        self.t_ind_e = []

        # Application level metrics
        # if "n_overlap" in b:
        # 	self.n_overlap.extend(b["n_overlap"])
        if "b_overlap_sum" in b:
            self.b_overlap_sum.extend(b["b_overlap_sum"])
        if "b_overlap_avr" in b:
            self.b_overlap_avr.extend(b["b_overlap_avr"])
        if "t_overlap" in b:
            self.t_overlap.extend(b["t_overlap"])
        else:
            if "b_overlap_sum" in b:
                self.t_overlap.extend(np.zeros(len(b["b_overlap_sum"])))
            elif "b_overlap_avr" in b:
                self.t_overlap.extend(np.zeros(len(b["b_overlap_avr"])))

        # Rank level metrics
        # remove later
        # ---------------------------------
        if "b" in b and "async" in io_type:
            print("Compatibility mode")
            self.b_rank_sum.extend(b["b"])
            self.b_rank_avr.extend(np.zeros(len(b["b"])))
            self.t_rank_s.extend(np.zeros(len(b["b"])))
            self.t_rank_e.extend(np.zeros(len(b["b"])))
        elif "b" in b and "sync" in io_type:
            print("Compatibility mode")
            self.b_rank_avr.extend(b["b"])
            self.b_rank_sum.extend(np.zeros(len(b["b"])))
            self.t_rank_s.extend(np.zeros(len(b["b"])))
            self.t_rank_e.extend(np.zeros(len(b["b"])))
        # ---------------------------------

        if args.avr or args.sum:
            # 1) assign rank level metric
            if args.sum and "b_rank_sum" in b:
                self.b_rank_sum.extend(b["b_rank_sum"])

            if args.avr and "b_rank_avr" in b:
                self.b_rank_avr.extend(b["b_rank_avr"])

            if "t_rank_s" in b:
                self.t_rank_s.extend(b["t_rank_s"])
                if "t_rank_e" in b and b["t_rank_e"]:
                    self.t_rank_e.extend(b["t_rank_e"])
                else:
                    if b["t_rank_s"]:
                        self.t_rank_e.extend(b["t_rank_s"][1:])
                        self.t_rank_e.append(b["t_rank_s"][-1])
                        b["t_rank_e"] = self.t_rank_e
                    else:
                        pass
            else:
                if "b_rank_sum" in b:
                    self.t_rank_s.extend(np.zeros(len(b["b_rank_sum"])))
                    self.t_rank_e.extend(np.zeros(len(b["b_rank_sum"])))
                elif "b_rank_avr" in b:
                    self.t_rank_s.extend(np.zeros(len(b["b_rank_avr"])))
                    self.t_rank_e.extend(np.zeros(len(b["b_rank_avr"])))

            # 2) Calculate bandwidth overlapping at rank level
            if (
                "t_rank_s" in b
                and "b_rank_avr" in b
                and "b_overlap_avr" not in b
                and args.avr
            ):
                self.b_overlap_avr, self.t_overlap = overlap(
                    b["b_rank_avr"], b["t_rank_s"], b["t_rank_e"]
                )
            if (
                "t_rank_s" in b
                and "b_rank_sum" in b
                and "b_overlap_sum" not in b
                and args.sum
            ):
                self.b_overlap_sum, self.t_overlap = overlap(
                    b["b_rank_sum"], b["t_rank_s"], b["t_rank_e"]
                )

        # overlapping thread level
        if args.ind:
            # Thread level metrics
            if "b_ind" in b:
                self.b_ind.extend(b["b_ind"])
                #! overlap ind
                self.b_overlap_ind, self.t_overlap_ind = overlap(
                    b["b_ind"], b["t_ind_s"], b["t_ind_e"]
                )

            if "t_ind_s" in b:
                self.t_ind_s.extend(b["t_ind_s"])

            if "t_ind_e" in b:
                self.t_ind_e.extend(b["t_ind_e"])

        self.app_ind = max(self.b_overlap_ind) if self.b_overlap_ind else -1
        self.app_avr = max(self.b_overlap_avr) if self.b_overlap_avr else -1
        self.app_sum = max(self.b_overlap_sum) if self.b_overlap_sum else -1

        # statistics:
        self.weighted_harmonic_mean = self.assign(b, "weighted_harmonic_mean")
        self.harmonic_mean = self.assign(b, "harmonic_mean")
        self.arithmetic_mean = self.assign(b, "arithmetic_mean")
        self.median = self.assign(b, "median")
        self.max = self.assign(b, "max")
        self.min = self.assign(b, "min")

        self.weighted_avr_harmonic_mean = self.assign(b, "weighted_avr_harmonic_mean")
        self.harmonic_avr_mean = self.assign(b, "harmonic_avr_mean")
        self.arithmetic_avr_mean = self.assign(b, "arithmetic_avr_mean")
        self.median_avr = self.assign(b, "median_avr")
        self.max_avr = self.assign(b, "max_avr")
        self.min_avr = self.assign(b, "min_avr")

        self.weighted_sum_harmonic_mean = self.assign(b, "weighted_sum_harmonic_mean")
        self.harmonic_sum_mean = self.assign(b, "harmonic_sum_mean")
        self.arithmetic_sum_mean = self.assign(b, "arithmetic_sum_mean")
        self.median_sum = self.assign(b, "median_sum")
        self.max_sum = self.assign(b, "max_sum")
        self.min_sum = self.assign(b, "min_sum")

    def assign(self, b, name):
        if name in b:
            return b[name] if not np.isnan(b[name]) else -1
        else:
            return -1


#! ----------------------- I/O analysis ------------------------------
# **********************************************************************
# *                       1. Overlap
# **********************************************************************


def overlap(b, t_s, t_e):
    t_s = np.array(t_s)
    t_e = np.array(t_e)
    b = np.array(b)
    id_s = np.argsort(t_s)
    id_e = np.argsort(t_e)
    try:
        b_overlap, t_overlap = overlap_core(b, t_s, t_e, id_s, id_e)
    except Exception:
        b_overlap, t_overlap = overlap_core_safe(b, t_s, t_e, id_s, id_e)

    return list(b_overlap), list(t_overlap)


@jit(nopython=True, cache=True, nogil=True)
def overlap_core(b, t_s, t_e, id_s, id_e):
    agg_phases = len(t_s)

    b_out = np.zeros(2 * agg_phases)
    t_out = np.zeros(2 * agg_phases)

    b_tmp = 0
    k_s = 0
    k_e = 0
    counter = 0

    while k_s < agg_phases or k_e < agg_phases:
        if k_s == agg_phases or t_e[id_e[k_e]] < t_s[id_s[k_s]]:
            b_tmp = b_tmp - b[id_e[k_e]]
            # t_out.append(t_e[id_e[k_e]])
            t_out[counter] = t_e[id_e[k_e]]
            k_e += 1
        else:
            b_tmp = b_tmp + b[id_s[k_s]]
            # t_out.append(t_s[id_s[k_s]])
            t_out[counter] = t_s[id_s[k_s]]
            k_s += 1
        # b_out.append(b_tmp)
        b_out[counter] = b_tmp
        counter += 1

    return b_out, t_out


def overlap_core_safe(b, t_s, t_e, id_s, id_e):
    agg_phases = len(t_s)

    b_out = []
    t_out = []
    b_tmp = 0
    k_s = 0
    k_e = 0

    while k_s < agg_phases or k_e < agg_phases:
        if k_s == agg_phases or t_e[id_e[k_e]] < t_s[id_s[k_s]]:
            b_tmp = b_tmp - b[id_e[k_e]]
            t_out.append(t_e[id_e[k_e]])
            k_e += 1
        else:
            b_tmp = b_tmp + b[id_s[k_s]]
            t_out.append(t_s[id_s[k_s]])
            k_s += 1
        b_out.append(b_tmp)

    return b_out, t_out


# **********************************************************************
# *                       2. Series Overlap
# **********************************************************************
def overlap_two_series(b1, t1, b2, t2):
    try:
        return overlap_two_series_jit_impl(
            np.array(b1), np.array(t1), np.array(b2), np.array(t2)
        )
    except Exception:
        return overlap_two_series_safe(b1, t1, b2, t2)


def overlap_two_series_safe(b1, t1, b2, t2):
    n1 = len(b1)
    n2 = len(b2)
    b_out = []
    t_out = []

    i1 = 0
    i2 = 0
    curr_b1 = 0
    curr_b2 = 0

    while i1 < n1 or i2 < n2:
        if i1 == n1:
            curr_b2 = b2[i2]
            t_out.append(t2[i2])
            i2 += 1
        elif i2 == n2 or t1[i1] < t2[i2]:
            curr_b1 = b1[i1]
            t_out.append(t1[i1])
            i1 += 1
        elif t2[i2] < t1[i1]:
            curr_b2 = b2[i2]
            t_out.append(t2[i2])
            i2 += 1
        else:
            curr_b1 = b1[i1]
            curr_b2 = b2[i2]
            t_out.append(t1[i1])
            i1 += 1
            i2 += 1
        b_out.append(curr_b1 + curr_b2)

    return np.array(b_out), np.array(t_out)


@jit(nopython=True, cache=True, nogil=True)
def overlap_two_series_jit_impl(b1, t1, b2, t2):
    n1 = len(b1)
    n2 = len(b2)
    max_len = n1 + n2
    b_out = np.zeros(max_len)
    t_out = np.zeros(max_len)

    i1 = 0
    i2 = 0
    counter = 0
    curr_b1 = 0
    curr_b2 = 0

    while i1 < n1 or i2 < n2:
        if i1 == n1:
            curr_b2 = b2[i2]
            t_out[counter] = t2[i2]
            i2 += 1
        elif i2 == n2 or t1[i1] < t2[i2]:
            curr_b1 = b1[i1]
            t_out[counter] = t1[i1]
            i1 += 1
        elif t2[i2] < t1[i1]:
            curr_b2 = b2[i2]
            t_out[counter] = t2[i2]
            i2 += 1
        else:
            curr_b1 = b1[i1]
            curr_b2 = b2[i2]
            t_out[counter] = t1[i1]
            i1 += 1
            i2 += 1
        b_out[counter] = curr_b1 + curr_b2
        counter += 1

    return b_out[:counter], t_out[:counter]


def merge_overlaps(b, t):
    """
    Wrapper: tries JIT version, falls back to safe version if error occurs.
    """
    try:
        return merge_overlaps_jit(np.array(b), np.array(t))
    except Exception:
        return merge_overlaps_safe(b, t)


# Safe Python version
def merge_overlaps_safe(b, t):
    """Merge overlapping timestamps by summing b values (safe Python version)."""
    if len(b) == 0:
        return np.array([]), np.array([])

    b = np.array(b)
    t = np.array(t)

    # Sort by timestamps
    order = np.argsort(t)
    b = b[order]
    t = t[order]

    merged_b = []
    unique_t = []

    for val, time in zip(b, t, strict=False):
        if unique_t and time == unique_t[-1]:
            merged_b[-1] += val
        else:
            unique_t.append(time)
            merged_b.append(val)

    return np.array(merged_b), np.array(unique_t)


# Numba JIT version
@jit(nopython=True, cache=True)
def merge_overlaps_jit(b, t):
    """Merge overlapping timestamps by summing b values (numba-accelerated)."""
    n = len(b)
    if n == 0:
        return np.zeros(0), np.zeros(0)

    # Sort by t
    idx = np.argsort(t)
    b_sorted = b[idx]
    t_sorted = t[idx]

    merged_b = np.zeros(n)
    unique_t = np.zeros(n)
    counter = 0

    for i in range(n):
        if counter == 0 or t_sorted[i] != unique_t[counter - 1]:
            unique_t[counter] = t_sorted[i]
            merged_b[counter] = b_sorted[i]
            counter += 1
        else:
            merged_b[counter - 1] += b_sorted[i]

    return merged_b[:counter], unique_t[:counter]
