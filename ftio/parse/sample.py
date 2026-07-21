"""
class to parse the data

Returns:
Sample: contains metrics for a single I/O type (read/write + async/sync)

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from ftio.parse.bandwidth import Bandwidth


class Sample:
    """contains metrics for a single I/O type (read/write + async/sync)"""

    def __init__(self, values: dict, io_type: str, args) -> None:
        self.type = io_type
        self.max_bytes_per_rank = self.assign(values, "max_bytes_per_rank")
        # self.max_offset_over_ranks       = self.assign(values,"max_offset_over_ranks")
        self.total_bytes = self.assign(values, "total_bytes")
        self.max_bytes_per_phase = self.assign(values, "max_bytes_per_phase")
        self.max_io_phases_per_rank = self.assign(values, "max_io_phases_per_rank")
        self.total_io_phases = self.assign(values, "total_io_phases")
        self.max_io_ops_per_rank = self.assign(values, "max_io_ops_per_rank")
        self.max_io_ops_in_phase = self.assign(values, "max_io_ops_in_phase")
        self.total_io_ops = self.assign(values, "total_io_ops")
        self.number_of_ranks, self.number_of_ranks_sequence = self._split_ranks(
            self.assign(values, "number_of_ranks")
        )
        self.total_number_of_ranks, self.total_number_of_ranks_sequence = (
            self._split_ranks(self.assign(values, "total_number_of_ranks"))
        )
        self.bandwidth = self.assign_bandwidth(values, io_type, args)
        self.file_index = args.file_index

    @staticmethod
    def _split_ranks(value):
        """Merged rank counts may be a burst-aligned list when malleability
        changed the rank count mid-run (see Simrun.merge_fields). Keep a
        scalar summary (the per-Sample value everything else expects) plus
        the full sequence (only used by find_data(), when its length lines
        up with the specific burst array being built)."""
        if isinstance(value, list):
            return (max(value) if value else float("NaN")), value
        return value, None

    def get_data(self):
        #! append list is much faster than append data frame
        name0 = []
        data0 = []
        common = ["number_of_ranks", "file_index"]
        name_rank_ovr = ["b_overlap_sum", "b_overlap_avr", "t_overlap"]
        name_rank = ["b_rank_sum", "b_rank_avr", "t_rank_s", "t_rank_e"]
        name_ind_ovr = ["b_overlap_ind", "t_overlap_ind"]
        name_ind = ["b_ind", "t_ind_s", "t_ind_e"]

        # add phase statistics. file_index IS kept here (unlike bandwidth) so
        # df0 can be matched back to the right df1 rows per source file --
        # needed once a single file's own rank count can vary (malleability).
        excluded = (
            "bandwidth",
            "number_of_ranks_sequence",
            "total_number_of_ranks_sequence",
        )
        for attr, value in self.__dict__.items():
            if attr not in excluded:
                name0.append(attr)
                data0.append(value)

        # statistics:
        name_tmp = common + name_rank_ovr + name_rank + name_ind_ovr + name_ind
        for attr, value in self.bandwidth.__dict__.items():
            if attr not in name_tmp:
                name0.append(attr)
                data0.append(value)

        name_rank_ovr, data1 = self.find_data(name_rank_ovr, common)
        name_rank, data2 = self.find_data(name_rank, common)
        name_ind_ovr, data3 = self.find_data(name_ind_ovr, common)
        name_ind, data4 = self.find_data(name_ind, common)

        return (
            name0,
            [data0],
            name_rank_ovr,
            data1,
            name_rank,
            data2,
            name_ind_ovr,
            data3,
            name_ind,
            data4,
        )

    def assign(self, values, name):
        if name in values:
            return values[name]
        else:
            return float("NaN")

    def assign_bandwidth(self, values, name, args):
        if "bandwidth" in values:
            return Bandwidth(values["bandwidth"], name, args)
        else:
            return Bandwidth({}, name, args)

    def find_data(self, name, common):
        # remove empty:
        for i in name:
            if not getattr(self.bandwidth, i):
                name.remove(i)

        # add phase info
        data = [getattr(self.bandwidth, s) for s in name]
        length = len(data[-1])
        name = name + common
        if self.number_of_ranks_sequence is not None and (
            len(self.number_of_ranks_sequence) == length
        ):
            ranks_col = list(self.number_of_ranks_sequence)
        else:
            ranks_col = [self.number_of_ranks] * length
        data.extend([ranks_col, [self.file_index] * length])
        return name, data
