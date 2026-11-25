"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from threading import Thread

import numpy as np


class overlap_thread(Thread):
    # constructor
    def __init__(self, ts, te, b_rank_sum, b_rank_avr=None):
        # execute the base constructor
        if b_rank_avr is None:
            b_rank_avr = []
        Thread.__init__(self)
        # set out values
        self.n_overlap = []
        self.b_overlap_sum = []
        self.b_overlap_avr = []
        self.t_overlap = []

        # set in values
        self.ts = ts
        self.te = te
        self.b_rank_sum = b_rank_sum
        self.b_rank_avr = b_rank_avr

    # function executed in a new thread
    def run(self):
        id_s = np.argsort(self.ts)
        id_e = np.argsort(self.te)
        agg_phases = len(self.ts)
        b_ovr_sum = []
        b_ovr_avg = []
        t = []
        n = []
        n_tmp = 0
        b_tmp = 0
        b2_tmp = 0
        k_s = 0
        k_e = 0
        while k_s < agg_phases or k_e < agg_phases:
            if k_s == agg_phases or self.te[id_e[k_e]] < self.ts[id_s[k_s]]:
                t.append(self.te[id_e[k_e]])
                if self.b_rank_sum:
                    b_tmp = b_tmp - self.b_rank_sum[id_e[k_e]]
                if self.b_rank_avr:
                    b2_tmp = b2_tmp - self.b_rank_avr[id_e[k_e]]
                n_tmp -= 1
                k_e += 1
            else:
                t.append(self.ts[id_s[k_s]])
                if self.b_rank_sum:
                    b_tmp = b_tmp + self.b_rank_sum[id_s[k_s]]
                if self.b_rank_avr:
                    b2_tmp = b2_tmp + self.b_rank_avr[id_s[k_s]]
                n_tmp += 1
                k_s += 1
            b_ovr_sum.append(b_tmp)
            b_ovr_avg.append(b2_tmp)
            n.append(n_tmp)
        if self.b_rank_sum:
            self.b_overlap_sum.extend(b_ovr_sum)
            # self.b_overlap_sum.extend(np.zeros(len(b_ovr_avg)))
        if self.b_rank_avr:
            self.b_overlap_avr.extend(b_ovr_avg)
        self.t_overlap.extend(t)
        self.n_overlap.extend(n)
