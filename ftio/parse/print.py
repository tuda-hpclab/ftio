"""
Converts results from TMIO / Darshan files into Extra-P supported format

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np

from ftio.parse.helper import scale_metric
from ftio.parse.scales import Scales


class Print:
    def __init__(self, args):
        self.data = Scales(args)
        self.data.assign_data()
        self.args = self.data.args

    def print_txt(self):
        with open("./scale.txt", "w") as self.file:
            self.file.write("pARAMETER x\n\n")
            self.print_points()
            self.print_regions_txt()
            self.file.write("\n")
        print("\033[1;32m------------------- done -------------------\n\033[1;0m")

    def print_json_lines(self):
        with open("./scale.jsonl", "w") as self.file:
            self.print_regions_jsonl()
        print("\033[1;32m------------------- done -------------------\n\033[1;0m")

    def print_regions_jsonl(self):
        self.print_io_read_sync("jsonl")
        self.print_io_write_sync("jsonl")
        self.print_io_read_async_t("jsonl")
        self.print_io_read_async_b("jsonl")
        self.print_io_write_async_t("jsonl")
        self.print_io_write_async_b("jsonl")
        self.print_io_time("jsonl")
        self.print_io_percent("jsonl")

    def print_regions_txt(self):
        self.print_io_read_sync()
        self.print_io_write_sync()
        self.print_io_read_async_t()
        self.print_io_read_async_b()
        self.print_io_write_async_t()
        self.print_io_write_async_b()
        self.print_io_time()
        self.print_io_percent()

    def print_io_read_sync(self, type="txt"):
        self.print_io_mode("read_sync", type)

    def print_io_write_sync(self, type="txt"):
        self.print_io_mode("write_sync", type)

    def print_io_read_async_t(self, type="txt"):
        self.print_io_mode("read_async_t", type)

    def print_io_read_async_b(self, type="txt"):
        self.print_io_mode("read_async_b", type)

    def print_io_write_async_t(self, type="txt"):
        self.print_io_mode("write_async_t", type)

    def print_io_write_async_b(self, type="txt"):
        self.print_io_mode("write_async_b", type)

    def print_io_mode(self, mode, type):
        self.print_data(mode, "total_bytes", f"{mode}->total_bytes", "Size (B)", type)
        self.print_data(
            mode,
            "max_bytes_per_rank",
            f"{mode}->max_bytes_per_rank",
            "Size (B)",
            type,
        )
        self.print_data(
            mode,
            "max_bytes_per_phase",
            f"{mode}->max_bytes_per_phase",
            "Size (B)",
            type,
        )
        self.print_data(
            mode,
            "max_io_phases_per_rank",
            f"{mode}->max_io_phases_per_rank",
            "Hits",
            type,
        )
        self.print_data(
            mode,
            "total_io_phases",
            f"{mode}->total_io_phases",
            "Hits",
            type,
        )
        self.print_data(
            mode,
            "max_io_ops_in_phase",
            f"{mode}->max_io_ops_in_phase",
            "Hits",
            type,
        )
        self.print_data(
            mode,
            "max_io_ops_per_rank",
            f"{mode}->max_io_ops_per_rank",
            "Hits",
            type,
        )
        self.print_data(mode, "total_io_ops", f"{mode}->total_io_ops", "Hits", type)
        self.print_data(mode, "number_of_ranks", f"{mode}->number_of_ranks", "Hits", type)
        self.print_data(mode, "bandwidth.app", f"{mode}->app", print_type=type)
        self.print_data(mode, "bandwidth.appH", f"{mode}->appH", print_type=type)
        # self.print_data(mode, 'bandwidth.b',                      f"{mode}->per_rank->b"                   , print_type = type)
        # self.print_data(mode, 'bandwidth.b_overlap_avr',          f"{mode}->per_rank->b_overlap_avr"         , print_type = type)
        # self.print_data(mode, 'bandwidth.b_overlap_sum',          f"{mode}->per_rank->b_overlap_sum"         , print_type = type)
        if "b" in mode:
            self.print_data(mode, "bandwidth.app_ind", f"{mode}->B_E", print_type=type)
            self.print_data(mode, "bandwidth.app_avr", f"{mode}->B_A", print_type=type)
            self.print_data(mode, "bandwidth.app_sum", f"{mode}->B_S", print_type=type)
        else:
            self.print_data(mode, "bandwidth.app_ind", f"{mode}->T_E", print_type=type)
            self.print_data(mode, "bandwidth.app_avr", f"{mode}->T_A", print_type=type)
            self.print_data(mode, "bandwidth.app_sum", f"{mode}->T_S", print_type=type)
        self.print_data(mode, "bandwidth.min", f"{mode}->per_rank->min", print_type=type)
        self.print_data(mode, "bandwidth.max", f"{mode}->per_rank->max", print_type=type)
        self.print_data(
            mode,
            "bandwidth.median",
            f"{mode}->per_rank->median",
            print_type=type,
        )
        self.print_data(
            mode,
            "bandwidth.weighted_harmonic_mean",
            f"{mode}->per_rank->weighted_harmonic_mean",
            print_type=type,
        )
        self.print_data(
            mode,
            "bandwidth.harmonic_mean",
            f"{mode}->per_rank->harmonic_mean",
            print_type=type,
        )
        self.print_data(
            mode,
            "bandwidth.arithmetic_mean",
            f"{mode}->per_rank->arithmetic_mean",
            print_type=type,
        )

    def print_io_time(self, type="txt"):
        try:
            self.print_data(
                "io_time",
                "delta_t_total",
                "io_time->total_time",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_overhead",
                "io_time->total_time->lib_overhead_time",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_overhead_post_runtime",
                "io_time->total_time->lib_overhead_time->delta_t_overhead_post_runtime",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_overhead_peri_runtime",
                "io_time->total_time->lib_overhead_time->delta_t_overhead_peri_runtime",
                "Time (s)",
                print_type=type,
            )

            self.print_data(
                "io_time",
                "delta_t_agg",
                "io_time->total_time->total_app_time",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_com",
                "io_time->total_time->total_app_time->total_com_time",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_agg_io",
                "io_time->total_time->total_app_time->total_io_time",
                "Time (s)",
                print_type=type,
            )

            self.print_data(
                "io_time",
                "delta_t_sr",
                "io_time->total_time->total_app_time->total_io_time->read_sync",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_ar_lost",
                "io_time->total_time->total_app_time->total_io_time->delta_t_ar_lost",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_sw",
                "io_time->total_time->total_app_time->total_io_time->write_sync",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_aw_lost",
                "io_time->total_time->total_app_time->total_io_time->delta_t_aw_lost",
                "Time (s)",
                print_type=type,
            )
            # self.print_data('io_time', 'delta_t_sr',      'io_time->read_sync',       'time', print_type = type)
            self.print_data(
                "io_time",
                "delta_t_ara",
                "io_time->read_async_t",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_arr",
                "io_time->read_async_b",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_ar_lost",
                "io_time->delta_t_ar_lost",
                "Time (s)",
                print_type=type,
            )

            # self.print_data('io_time', 'delta_t_sw',      'io_time->write_sync',      'time', print_type = type)
            self.print_data(
                "io_time",
                "delta_t_awa",
                "io_time->write_async_t",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_awr",
                "io_time->write_async_b",
                "Time (s)",
                print_type=type,
            )
            self.print_data(
                "io_time",
                "delta_t_aw_lost",
                "io_time->delta_t_aw_lost",
                "Time (s)",
                print_type=type,
            )

        except Exception:
            pass

    def print_io_percent(self, type="txt"):
        try:
            self.print_data(
                "io_percent",
                "TI",
                "io_percent-> I/O to Total",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "TC",
                "io_percent-> Compute to Total",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CI",
                "io_percent-> I/O to Compute",
                "Ratio",
                print_type=type,
            )

            self.print_data(
                "io_percent",
                "TAWB",
                "io_percent->AW->AW to Total (B)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "TAWT",
                "io_percent->AW->AW to Total (T)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "TAWD",
                "io_percent->AW->AW to Total (D)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "IAWB",
                "io_percent->AW->AW to IO (B)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "IAWT",
                "io_percent->AW->AW to IO (T)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "IAWD",
                "io_percent->AW->AW to IO (D)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CAWB",
                "io_percent->AW->AW to Comput (B)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CAWT",
                "io_percent->AW->AW to Comput (T)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CAWD",
                "io_percent->AW->AW to Comput (D)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "TARB",
                "io_percent->AR->AR to Total (B)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "TART",
                "io_percent->AR->AR to Total (T)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "TARD",
                "io_percent->AR->AR to Total (D)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "IARB",
                "io_percent->AR->AR to IO (B)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "IART",
                "io_percent->AR->AR to IO (T)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "IARD",
                "io_percent->AR->AR to IO (D)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CARB",
                "io_percent->AR->AR to Compute (B)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CART",
                "io_percent->AR->AR to Compute (T)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CARD",
                "io_percent->AR->AR to Compute (D)",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "TSW",
                "io_percent->SW->SW to Total",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "ISW",
                "io_percent->SW->SW to IO",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CSW",
                "io_percent->SW->SW to Compute",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "TSR",
                "io_percent->SR-> SR to Total",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "ISR",
                "io_percent->SR-> SR to IO",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CSR",
                "io_percent->SR-> SR to Compute",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "TO",
                "io_percent->overhead-> lib overhead to Total",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "IO",
                "io_percent->overhead-> lib overhead to IO",
                "Ratio",
                print_type=type,
            )
            self.print_data(
                "io_percent",
                "CO",
                "io_percent->overhead-> lib overhead to Compute",
                "Ratio",
                print_type=type,
            )

        except Exception:
            pass

    def print_data(
        self,
        io_mode,
        var="bandwidth.app",
        call_path="",
        metric="Bandwidth (B/s)",
        print_type="txt",
    ):
        if self.check_non_empty(io_mode, var):
            if not call_path:
                call_path = io_mode
            if "txt" in print_type:  # txt file
                self.file.write(f"\nREGION {call_path}\nMETRIC {metric}\n")
                for i in range(0, self.data.n):
                    value = getattr(self.data.s[i], io_mode)
                    if "bandwidth" in var:
                        art = getattr(value.bandwidth, var[10:])
                    else:
                        art = getattr(value, var)
                    if isinstance(art, list):
                        self.file.write("DATA ")
                        for i, _ in enumerate(art):
                            self.file.write(f"{art[i]:e}")
                        self.file.write("\n")
                    else:
                        self.file.write(f"DATA {art:e} \n")
                self.file.write("\n")
            elif "jsonl" in print_type:
                order = 1
                for i in range(0, self.data.n):
                    value = getattr(self.data.s[i], io_mode)
                    if "bandwidth" in var:
                        art = getattr(value.bandwidth, var[10:])
                    else:
                        art = getattr(value, var)
                        art = -1 if np.isnan(art) else art
                    if isinstance(art, list):
                        for j, _ in enumerate(art):
                            if j == 0 and self.args.scale:
                                metric, order = scale_metric(metric, art[j])
                            self.file.write(
                                f'{{"params":{{"Processes":{self.data.s[i].ranks}}},"callpath":"{call_path}","metric":"{metric}","value":{art[j]*order:e} }}\n'
                            )
                    else:
                        if self.args.scale:
                            metric, order = scale_metric(metric, art)
                        self.file.write(
                            f'{{"params":{{"Processes":{self.data.s[i].ranks}}},"callpath":"{call_path}","metric":"{metric}","value":{art*order:e} }}\n'
                        )
            else:
                pass

    def check_non_empty(self, io_mode, var):
        if self.data.n < 1:
            return False
        else:
            value = getattr(self.data.s[0], io_mode)
            if "bandwidth" in var:
                if hasattr(value.bandwidth, var[10:]):
                    art = getattr(value.bandwidth, var[10:])
                else:
                    return False
            else:
                try:
                    art = getattr(value, var)
                except AttributeError:
                    return False

            return not (isinstance(art, list) and not art)

    def print_points(self):
        for run in self.data.s:
            run.print_rank(self.file)
