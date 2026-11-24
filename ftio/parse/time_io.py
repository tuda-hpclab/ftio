"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""


class Time:
    def __init__(self, data, rank, args):
        # total
        self.delta_t_agg = self.assign(data, "delta_t_agg")
        self.delta_t_agg_io = self.assign(data, "delta_t_agg_io")
        self.delta_t_com = self.delta_t_agg - self.delta_t_agg_io

        # read
        self.delta_t_sr = self.assign(data, "delta_t_sr")
        self.delta_t_ara = self.assign(data, "delta_t_ara")
        self.delta_t_arr = self.assign(data, "delta_t_arr")
        self.delta_t_ar_lost = self.assign(data, "delta_t_ar_lost")

        # write
        self.delta_t_sw = self.assign(data, "delta_t_sw")
        self.delta_t_awa = self.assign(data, "delta_t_awa")
        self.delta_t_awr = self.assign(data, "delta_t_awr")
        self.delta_t_aw_lost = self.assign(data, "delta_t_aw_lost")
        self.file_index = args.file_index

        # lib overhead
        self.delta_t_overhead = self.assign(data, "delta_t_overhead")
        if "delta_t_overhead_post_runtime" in data:
            self.delta_t_overhead_post_runtime = self.assign(
                data, "delta_t_overhead_post_runtime"
            )
        if "delta_t_overhead_peri_runtime" in data:
            self.delta_t_overhead_peri_runtime = self.assign(
                data, "delta_t_overhead_peri_runtime"
            )

        self.delta_t_total = self.delta_t_overhead + self.delta_t_agg
        self.rank = rank

        self.delta_t_rank0 = self.assign(data, "delta_t_rank0")
        self.delta_t_rank0_app = self.assign(data, "delta_t_rank0_app")
        self.delta_t_rank0_overhead_post_runtime = self.assign(
            data, "delta_t_rank0_overhead_post_runtime"
        )
        self.delta_t_rank0_overhead_peri_runtime = self.assign(
            data, name="delta_t_rank0_overhead_peri_runtime"
        )
        # TODO: compatibility mode with earlier versions
        if "delta_t_rank0_overhead_peri_runtime" not in data:
            self.delta_t_rank0_overhead_peri_runtime = self.assign(
                data, name="delta_t_rank0_overhead_runtime"
            )

    def get_data(self):
        #! append list is much faster than append data frame
        name0 = []
        data0 = []
        name0.append("number_of_ranks")
        data0.append(self.rank)
        for attr, value in self.__dict__.items():
            name0.append(attr)
            data0.append(value)

        return name0, data0

    def assign(self, data, name):
        if name in data:
            return data[name]
        else:
            return float("NaN")
