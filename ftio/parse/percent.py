"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""


class Percent:
    def __init__(self, io_time):
        # XXXX
        # |||'-> Bandwidth / Throughput / Delta
        # ||'--> Write / Read
        # |'---> Async / Sync / Overhead /  IO / Compute
        # |----> Total / IO / Compute

        time = io_time.delta_t_agg
        io = io_time.delta_t_agg_io
        compute = io_time.delta_t_com
        # [ ] fix this
        # TODO: add flag to control granualaierty of sampling. Individual I/O operation can be discarded if focus is on phase (remove vectors)
        # HACK: this is a todo

        # Async Write
        self.TAWB = self.percent(time, io_time.delta_t_awr)
        self.TAWT = self.percent(time, io_time.delta_t_awa)
        self.TAWD = self.percent(time, io_time.delta_t_aw_lost)
        self.IAWB = self.percent(io, io_time.delta_t_awr)
        self.IAWT = self.percent(io, io_time.delta_t_awa)
        self.IAWD = self.percent(io, io_time.delta_t_aw_lost)
        self.CAWB = self.percent(compute, io_time.delta_t_awr)
        self.CAWT = self.percent(compute, io_time.delta_t_awa)
        self.CAWD = self.percent(compute, io_time.delta_t_aw_lost)

        # Async Read
        self.TARB = self.percent(time, io_time.delta_t_arr)
        self.TART = self.percent(time, io_time.delta_t_ara)
        self.TARD = self.percent(time, io_time.delta_t_ar_lost)
        self.IARB = self.percent(io, io_time.delta_t_arr)
        self.IART = self.percent(io, io_time.delta_t_ara)
        self.IARD = self.percent(io, io_time.delta_t_ar_lost)
        self.CARB = self.percent(compute, io_time.delta_t_arr)
        self.CART = self.percent(compute, io_time.delta_t_ara)
        self.CARD = self.percent(compute, io_time.delta_t_ar_lost)

        # Sync Write
        self.TSW = self.percent(time, io_time.delta_t_sw)
        self.ISW = self.percent(io, io_time.delta_t_sw)
        self.CSW = self.percent(compute, io_time.delta_t_sw)

        # Sync Read
        self.TSR = self.percent(time, io_time.delta_t_sr)
        self.ISR = self.percent(io, io_time.delta_t_sr)
        self.CSR = self.percent(compute, io_time.delta_t_sr)

        # Lib overhead
        self.TO = self.percent(time, io_time.delta_t_overhead)
        self.IO = self.percent(io, io_time.delta_t_overhead)
        self.CO = self.percent(compute, io_time.delta_t_overhead)

        self.TI = self.percent(time, io)
        self.TC = self.percent(time, compute)
        self.CI = self.percent(compute, io)

    def percent(self, a, b):
        return b / a * 100 if a != 0 else 0
