"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import json

from ftio.freq.helper import MyConsole
from ftio.parse.helper import detect_source, match_mode
from ftio.parse.simrun import Simrun


class ParseJson:
    def __init__(self, path):
        self.path = path
        if self.path[-1] == "/":
            self.path = self.path[:-1]

    def to_simrun(self, args, index=0):
        """Convert to Simrun class
        Args:
            ars (argparse): command line arguments
            index: file index in case several files are passed
        Returns:
            Simrun: Simrun object
        """
        file = self.path
        with open(file) as current_file:
            data = json.load(current_file)

        source = detect_source(data, args)
        if "tmio" in source:
            pass
        else:
            data, args = self.adjust(data, args)

        return Simrun(data, "json", file, args, index)

    def adjust(self, data: dict, args):
        console = MyConsole()
        # check for mode
        fields = list(data.keys())
        if any(
            x in fields for x in ["read_sync", "read_async", "write_sync", "write_async"]
        ):
            args.source = "custom"
            if args.mode and match_mode(args.mode) in data:
                self.add_missing_fields(data, match_mode(args.mode))
            else:
                console.info(
                    f"[yellow]Warning: [/] Mode [yellow]{args.mode}[/] does not exist in trace"
                )
                for x in [
                    "read_sync",
                    "read_async",
                    "read_async",
                    "write_async",
                ]:
                    if x in fields:
                        args.mode = x
                        console.info(
                            f"[yellow]Warning: [/] Adjusting mode to [yellow]{args.mode}[/]"
                        )
                        self.add_missing_fields(data, match_mode(args.mode))
                        break
        else:
            args.mode = "read_sync"
            args.source = "custom"

            if "bandwidth" in data:
                data = {
                    "read_sync": {
                        "total_bytes": 0,
                        "number_of_ranks": 0,
                        "bandwidth": data["bandwidth"],
                    }
                }

            elif "b_rank" in data:
                data = {
                    "read_sync": {
                        "total_bytes": 0,
                        "number_of_ranks": 0,
                        "bandwidth": {
                            "b_rank_avr": data["b_rank_avr"],
                            "t_rank_s": data["t_rank_s"],
                            "t_rank_e": data["t_rank_e"],
                        },
                    }
                }
            elif "b_overlap" in data:
                data = {
                    "read_sync": {
                        "total_bytes": 0,
                        "number_of_ranks": 0,
                        "bandwidth": {
                            "b_overlap_avr": data["b_overlap"],
                            "t_overlap": data["t_overlap"],
                        },
                    }
                }
            else:
                # try to get the data out
                t_fields = [k for k in data if k.startswith("t")]
                b_fields = [k for k in data if k.startswith("b")]
                if len(b_fields) == 1:
                    if len(t_fields) == 1:
                        bandwidth = {
                            "b_overlap_avr": data[b_fields[0]],
                            "t_overlap": data[t_fields[0]],
                        }
                        console.info(
                            "[yellow]Treating Json data as application-level metrics[/]"
                        )

                    elif len(t_fields) == 2:
                        t_s = [k for k in t_fields if k.endswith("s")]
                        t_e = [k for k in t_fields if k.endswith("e")]
                        bandwidth = {
                            "b_rank_avr": data[b_fields[0]],
                            "t_rank_s": data[t_s[0]],
                            "t_rank_e": data[t_e[0]],
                        }
                        console.info(
                            "[yellow]Treating Json data as rank-level metrics[/]"
                        )

                    data = {
                        "read_sync": {
                            "total_bytes": 0,
                            "number_of_ranks": 0,
                            "bandwidth": bandwidth,
                        }
                    }
                else:
                    raise ValueError("Unable to parse JSON data")

        return data, args

    def add_missing_fields(self, data, mode):
        fields = ["total_bytes", "number_of_ranks"]
        for field in fields:
            if field not in data[mode]:
                data[mode][field] = 0
