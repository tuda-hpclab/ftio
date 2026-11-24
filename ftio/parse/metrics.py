"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np
import pandas as pd
from scipy import stats


class Metrics:
    def __init__(self, args):
        self.t_avr = []
        self.t_sum = []
        self.t_ind = []
        self.b_avr = []
        self.b_sum = []
        self.b_ind = []
        self.args = args

    def add(self, ranks, run, T1, T2, B1=None, B2=None):
        if B2 is None:
            B2 = []
        if B1 is None:
            B1 = []
        if self.args.avr:
            self.assign_metric("t_avr", T1["b_overlap_avr"], ranks, run, T1["t_overlap"])
        if self.args.sum:
            self.assign_metric("t_sum", T1["b_overlap_sum"], ranks, run, T1["t_overlap"])
        if self.args.ind:
            self.assign_metric(
                "t_ind", T2["b_overlap_ind"], ranks, run, T2["t_overlap_ind"]
            )
        if not isinstance(B1, list):
            if self.args.avr:
                self.assign_metric(
                    "b_avr", B1["b_overlap_avr"], ranks, run, B1["t_overlap"]
                )
            if self.args.sum:
                self.assign_metric(
                    "b_sum", B1["b_overlap_sum"], ranks, run, B1["t_overlap"]
                )
            if self.args.ind:
                self.assign_metric(
                    "b_ind",
                    B2["b_overlap_ind"],
                    ranks,
                    run,
                    B2["t_overlap_ind"],
                )

    def get_data(self):
        name = [
            "number_of_ranks",
            "run",
            "max",
            "min",
            "median",
            "hmean",
            "amean",
            "whmean",
        ]
        self.t_avr = pd.DataFrame(self.t_avr, columns=name)
        self.t_sum = pd.DataFrame(self.t_sum, columns=name)
        self.t_ind = pd.DataFrame(self.t_ind, columns=name)
        self.b_avr = pd.DataFrame(self.b_avr, columns=name)
        self.b_sum = pd.DataFrame(self.b_sum, columns=name)
        self.b_ind = pd.DataFrame(self.b_ind, columns=name)

    def assign_metric(self, name, b, ranks, run, weights=None):
        tmp = add_metric(b, ranks, run, weights)
        if tmp:
            if name == "t_avr":
                self.t_avr.append(tmp)
            elif name == "t_sum":
                self.t_sum.append(tmp)
            elif name == "t_ind":
                self.t_ind.append(tmp)
            elif name == "b_avr":
                self.b_avr.append(tmp)
            elif name == "b_sum":
                self.b_sum.append(tmp)
            elif name == "b_ind":
                self.b_ind.append(tmp)
            else:
                pass

    def get(self, name, metric):
        if "t_a" in name:
            return self.t_avr[metric]
        elif "t_s" in name:
            return self.t_sum[metric]
        elif "t_i" in name:
            return self.t_ind[metric]
        elif "b_a" in name:
            return self.b_avr[metric]
        elif "b_s" in name:
            return self.b_sum[metric]
        elif "b_i" in name:
            return self.b_ind[metric]
        else:
            pass


def add_metric(b, ranks, run, weights=None):  # b is a pandas dataframe
    b_nonzero = b[b > 0]
    if not b_nonzero.empty:
        ranks = ranks
        run = run
        max = b_nonzero.max()
        min = b_nonzero.min()
        median = b_nonzero.median()
        amean = b_nonzero.mean()
        hmean = stats.hmean(b_nonzero, axis=0)
        b = b.reset_index(drop=True)
        b[b < 0] = 0
        weights = (np.asarray([*weights, 0]) - np.asarray([0, *weights]))[1:]
        weights[-1] = 0
        weights = weights[b > 0]
        b = b[b > 0]
        whmean = stats.hmean(b, axis=0, weights=weights / sum(weights))
        return [ranks, run, max, min, median, hmean, amean, whmean]
    else:
        return []
