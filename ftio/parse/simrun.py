"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import re
import statistics as st

from ftio.parse.helper import match_mode
from ftio.parse.percent import Percent
from ftio.parse.sample import Sample
from ftio.parse.time_io import Time


class Simrun:
    """Stores the result from a single simulation including the:
    1. read sync
    2. read async
    3. write sync
    4. write async
    5. I/O time (if available)
    6. I/O percent (if available)
    """

    def __init__(self, data, ext: str, name: str, args, file_index: int = 0):
        """create Simrun object

        Args:
            data (array): contains data from simulation
            ext (str): 'json', 'jsonl', 'recorder', 'darshan', 'msgpack', or 'txt'
            name (str): name of simulation (e.g., 192.json)
            args (argparse): command line arguments
            file_index (int, optional): file index, in case several files have the same name
        """
        self.name = name
        self.ranks = self.get_rank(ext)
        mode = match_mode(args.mode) if args.mode else ""
        args.file_index = file_index
        supported_modes = [
            "read_sync",
            "write_sync",
            "read_async",
            "write_async",
        ]

        #! list = JSONL
        if isinstance(data, list):
            #! JSONL need to be converted to a dict
            # if 'Time' in data[-1]:
            if "jsonl" in ext or "msgpack" in ext:
                if mode and mode in supported_modes:
                    self.reset(args)
                    self.assign(data, args, mode, "jsonl")
                else:
                    self.read_sync = self.merge_parts(data, "read_sync", args)
                    self.write_sync = self.merge_parts(data, "write_sync", args)
                    self.read_async_t = self.merge_parts(data, "read_async_t", args)
                    self.read_async_b = self.merge_parts(data, "read_async_b", args)
                    self.write_async_t = self.merge_parts(data, "write_async_t", args)
                    self.write_async_b = self.merge_parts(data, "write_async_b", args)
                    if any("io_time" in d for d in data):
                        self.io_time = self.merge_parts(data, "io_time", args)
                    else:
                        self.io_time = Time({}, self.ranks, args)
                    self.io_percent = Percent(self.io_time)
            else:
                raise ValueError("Data format empty or not supported")

        #! Darshan files or recorder files
        elif "darshan" in ext or "recorder" in ext:
            self.reset(args)
            self.read_sync = Sample(data["read_sync"], "read_sync", args)
            self.write_sync = Sample(data["write_sync"], "write_sync", args)
            self.io_time = Time(data["io_time"], self.ranks, args)

        elif "txt" in ext:
            self.reset(args)
            setattr(self, mode, Sample(data[mode], mode, args))
            self.io_time = Time(data["io_time"], self.ranks, args)

        #! json files
        else:
            #! for dft to make it faster
            if mode and mode in supported_modes:
                self.reset(args)
                self.assign(data, args, mode)
                return
            #! standard json files
            self.read_sync = Sample(data["read_sync"], "read_sync", args)
            self.write_sync = Sample(data["write_sync"], "write_sync", args)
            self.read_async_t = Sample(data["read_async_t"], "read_async_t", args)
            self.read_async_b = Sample(data["read_async_b"], "read_async_b", args)
            self.write_async_t = Sample(data["write_async_t"], "write_async_t", args)
            self.write_async_b = Sample(data["write_async_b"], "write_async_b", args)
            if "io_time" in data:
                self.io_time = Time(data["io_time"], self.ranks, args)
                self.io_percent = Percent(self.io_time)

    # ---------------------------------------------------------------------------------------------------
    # ?=======================
    # ? helper functions
    # ?=======================
    def assign(self, data, args, mode, file_format="json"):
        """assigns a single mode instead of parsing all data

        Args:
            data (array): sim data
            args (argparse): command line arguments
            mode (str): mode to assign
            format (str, optional): json or jsonl. Defaults to 'json'.
        """
        if "jsonl" in file_format:
            if mode == "read_sync":
                self.read_sync = self.merge_parts(data, "read_sync", args)
            elif mode == "read_async":
                self.read_async_t = self.merge_parts(data, "read_async_t", args)
                self.read_async_b = self.merge_parts(data, "read_async_b", args)
            elif mode == "write_sync":
                self.write_sync = self.merge_parts(data, "write_sync", args)
            elif mode == "write_async":
                self.write_async_t = self.merge_parts(data, "write_async_t", args)
                self.write_async_b = self.merge_parts(data, "write_async_b", args)
            else:
                pass

            if any("io_time" in d for d in data):
                self.io_time = self.merge_parts(data, "io_time", args)
                self.io_percent = Percent(self.io_time)

        else:
            if mode == "read_sync":
                self.read_sync = Sample(data["read_sync"], "read_sync", args)
            elif mode == "read_async":
                self.read_async_t = Sample(data["read_async_t"], "read_async_t", args)
                if "read_async_b" in data:
                    self.read_async_b = Sample(data["read_async_b"], "read_async_b", args)
            elif mode == "write_async":
                self.write_async_t = Sample(data["write_async_t"], "write_async_t", args)
                if "write_async_b" in data:
                    self.write_async_b = Sample(
                        data["write_async_b"], "write_async_b", args
                    )
            elif mode == "write_sync":
                self.write_sync = Sample(data["write_sync"], "write_sync", args)

    def reset(self, args):
        """sets all fields to empty. This is usually followed by a assign call

        Args:
            args (argparse): command line arguments
        """
        self.read_sync = Sample({"bandwidth": []}, "read_sync", args)
        self.read_async_t = Sample({"bandwidth": []}, "read_async_t", args)
        self.read_async_b = Sample({"bandwidth": []}, "read_async_b", args)
        self.write_async_t = Sample({"bandwidth": []}, "write_async_t", args)
        self.write_async_b = Sample({"bandwidth": []}, "write_async_b", args)
        self.write_sync = Sample({"bandwidth": []}, "write_sync", args)
        self.io_time = Time({}, self.ranks, args)

    def get_rank(self, ext: str) -> int:
        """Extracts the rank number from a filename.

        Args:
            ext (str): The expected file extension.

        Returns:
            int: The rank number extracted from the filename, or -1 if not found.
        """
        rank = -1

        if isinstance(self.name, int):
            rank = self.name
        elif isinstance(self.name, str) and not any(
            sep in self.name for sep in [".", "/"]
        ):
            rank = int(self.name)
        else:
            try:
                # Extract filename from path
                filename = self.name.replace("\\", "/").split("/")[
                    -1
                ]  # Normalize for Windows

                # Remove the extension
                if filename.endswith(f".{ext}"):
                    filename = filename[: -len(f".{ext}")]

                # Extract the last number in the filename
                numbers = re.findall(r"\d+", filename)
                if numbers:
                    rank = int(numbers[-1])  # Last number is the rank
            except ValueError:
                pass

        return rank

    def print_rank(self, file):
        """print rank mapped to POINTS. This is used for
        Extra-P with the text file format.

        Args:
            file (fileptr): file pointer
        """
        file.write(f"POINTS ( {self.ranks:i} )\n")

    def merge_parts(self, data, mode, args):
        """merge jsonl parts to a single simulation

        Args:
            data (2d array): array fo data elements
            mode (str): mode to merge
            args (argparse): command line arguments

        Returns:
            io_sample object: iosample of the whole simulation
        """
        data_array = [item[mode] for item in data if mode in item]
        out = []
        if len(data_array) <= 1:
            if "time" in mode:
                out = (
                    Time(data_array[0], self.ranks, args)
                    if len(data_array) > 0
                    else Time({}, self.ranks, args)
                )
            else:
                out = (
                    Sample(data_array[0], mode, args)
                    if len(data_array) > 0
                    else Sample({}, mode, args)
                )
        else:
            data_array = self.merge_fields(data_array)
            if "time" in mode:
                out = Time(data_array, self.ranks, args)
            else:
                out = Sample(data_array, mode, args)
        return out

    def merge_fields(self, data_array, keys: list[str] = None) -> dict:
        """Merges the metrics field from different files. For example,
        JsonlLines file constantly append new data. To merge the previous
        metrics with the new one, this function iterates over the fields
        and merges them.

        Args:
            data_array (dataframe): Metrics
            keys (list[str], optional): _description_. Defaults to list[str].

        Returns:
            dict: merge fields in stored in a dict variable
        """
        if keys is None:
            keys = []
        if not keys:
            keys = list(data_array[0].keys())

        my_dict = {}
        for field in keys:
            # print(f)
            if isinstance(data_array[0][field], dict):
                data_array2 = [x[field] for x in data_array if field in x]
                my_dict[field] = self.merge_fields(
                    data_array2, list(data_array2[0].keys())
                )
            else:
                if isinstance(data_array[0][field], list):
                    my_dict[field] = []
                    for i, _ in enumerate(data_array):
                        my_dict[field].extend(data_array[i][field])
                else:
                    if field in ("number_of_ranks", "total_number_of_ranks"):
                        values = [x[field] for x in data_array]
                        if len(set(values)) == 1:
                            my_dict[field] = values[0]
                        else:
                            # Ranks changed across merged parts (malleability) --
                            # collapsing to one number (e.g. max()) would hide
                            # that a rank change happened at all. Keep a
                            # burst-aligned sequence instead, so each burst can
                            # still be tagged with whichever rank count was
                            # active when it happened.
                            my_dict[field] = []
                            for x in data_array:
                                n = self._burst_count(x.get("bandwidth", {}))
                                my_dict[field].extend([x[field]] * n)
                    elif any(x in field for x in ["total", "_t_"]):
                        my_dict[field] = sum(x[field] for x in data_array)
                    elif any(x in field for x in ["max", "number"]):
                        my_dict[field] = max(x[field] for x in data_array)
                    elif "min" in field:
                        my_dict[field] = min(x[field] for x in data_array)
                    elif "arithmetic_mean" in field:
                        my_dict[field] = st.mean(x[field] for x in data_array)

        return my_dict

    @staticmethod
    def _burst_count(bandwidth: dict) -> int:
        """Number of bursts one merged-part's bandwidth dict contributes.

        Used to replicate a per-part scalar (e.g. number_of_ranks) into a
        burst-aligned sequence with the same length as whichever burst array
        that part actually reports.
        """
        for key in ("t_overlap", "t_rank_s", "t_rank_e", "b_overlap_avr", "b_rank_avr"):
            value = bandwidth.get(key)
            if isinstance(value, list):
                return len(value)
        return 1
