"""
Handles the importing of modules

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import datetime
import os

import numpy as np
import pandas as pd
from rich.console import Console

from ftio.parse.args import parse_args
from ftio.parse.helper import match_mode, print_info
from ftio.parse.parse_custom import ParseCustom
from ftio.parse.parse_darshan import ParseDarshan
from ftio.parse.parse_json import ParseJson
from ftio.parse.parse_jsonl import ParseJsonl
from ftio.parse.parse_msgpack import ParseMsgpack
from ftio.parse.parse_recorder import ParseRecorder
from ftio.parse.parse_txt import ParseTxt
from ftio.parse.parse_zmq import ParseZmq


class Scales:
    """load the data. Supports single files (json, jsonl, darshan) or folders (+ recorder)"""

    def __init__(self, argv, msg=None):
        self.prog_name = argv[0][argv[0].rfind("/") + 1 :].capitalize()
        print_info(self.prog_name)
        self.render = ""
        self.plot_mode = ""
        self.mode = ""
        self.ts = -1
        self.threaded = ""
        self.zoom = -1
        self.same_path = False
        self.names = []
        self.msg = msg
        self.s = []

        # save call
        self.save_call(argv)

        # Parse arguments
        self.args = parse_args(argv)

        if "zmq" in self.args and self.args.zmq:
            self.s.append(ParseZmq(self.msg).to_simrun(self.args, 0))
            self.n = 1
        else:
            self.load_setup()

    def load_setup(self) -> None:
        if isinstance(self.args.files, list):
            if len(self.args.files) <= 1:
                self.paths = ["."]
            else:
                self.paths = []
                for i in range(1, len(self.args.files)):
                    self.paths.append(str(self.args.files[i]))
        else:
            self.paths = [self.args.files]

        self.check_same_path()
        console = Console()

        # Mapping from directory to run_index
        dir_to_index = {}

        for path in self.paths:
            #! load folders
            # Recorder folder
            if "_text" in path[-5:]:
                console.print(
                    f"\n[cyan]Loading Recorder folder({self.paths.index(path) + 1},{len(self.paths)}):[/] {path}"
                )
                if path not in dir_to_index:
                    dir_to_index[path] = len(dir_to_index)
                    self.names.append(os.path.basename(path))

                run = ParseRecorder(path).to_simrun(self.args, dir_to_index[path])
                self.s.append(run)

            # Folder
            elif os.path.isdir(path):
                console.print(
                    f"\n[cyan]Loading folder({self.paths.index(path) + 1},{len(self.paths)}):[/] {path}"
                )
                if path[-1] == "/":
                    path = path[:-1]

                for root, _, files in os.walk(path):
                    # remove unneeded folders
                    if "io_results" in root or "exported_images" in root:
                        console.print(f"[yellow]Skipping folder:  {root}[/]")
                        continue

                    # remove unneeded files
                    skip_files = ["scale.jsonl", ".call.txt"]
                    for unwanted in skip_files:
                        if unwanted in files:
                            console.print(f"[yellow]Skipping file: {root}/{unwanted}[/]")
                            files.remove(unwanted)

                    # sort the files
                    sorted_files = sorted(files, key=len)

                    for file in sorted_files:
                        if any(
                            ext in file for ext in ["json", "darshan", "msgpack", "txt"]
                        ):
                            file_path = os.path.join(root, file)
                            # Limit the number of ranks to consider if self.limit is defined
                            try:
                                if (
                                    self.args.limit > 0
                                    and get_rank(file) >= self.args.limit
                                ):
                                    console.print(f"[yellow]Skipping file: {file}[/]")
                                    continue
                            except Exception as error:
                                console.print(
                                    f"[red]Something went wrong with the limit. Error is {error}[/]"
                                )

                            if root not in dir_to_index:
                                dir_to_index[root] = len(dir_to_index)
                                self.names.append(os.path.basename(root))

                            console.print(f"[cyan]Current file:[/] {file}")
                            self.load_file(file_path, dir_to_index[root])

            # Compare Several files
            elif (
                not self.same_path
                and ".json" in path[-6:]
                and "ftio" not in self.prog_name.lower()
            ):
                parent_dir = os.path.dirname(path) or "."
                if parent_dir not in dir_to_index:
                    dir_to_index[parent_dir] = len(dir_to_index)
                    self.names.append(path)

                console.print(f"[cyan]Current file:[/] {path}")
                self.load_file(path, dir_to_index[parent_dir])

            # Single file
            else:
                parent_dir = os.path.dirname(path) or "."
                if parent_dir not in dir_to_index:
                    dir_to_index[parent_dir] = len(dir_to_index)
                    self.names.append(get_filename(path))

                if "predictor" not in self.prog_name.lower():
                    console.print(f"[cyan]Current file:[/] {path}\n")
                self.load_file(path, dir_to_index[parent_dir])

        # print('--------------------------------------------\n')
        self.n = len(self.s)

    def load_file(self, file_path: str, file_index=0) -> None:
        """Load file content into an Simrun object

        Args:
            file_path (str): filename + absolute path
        """
        check_open(file_path, self.prog_name)
        if self.args.custom_file:
            run = ParseCustom(file_path).to_simrun(self.args, file_index)
        elif ".json" in file_path[-5:]:
            run = ParseJson(file_path).to_simrun(self.args, file_index)
        elif ".jsonl" in file_path[-6:]:
            run = ParseJsonl(file_path).to_simrun(self.args, file_index)
        elif "darshan" in file_path[-10:]:
            run = ParseDarshan(file_path).to_simrun(self.args, file_index)
        elif "msgpack" in file_path[-10:]:
            run = ParseMsgpack(file_path).to_simrun(self.args, file_index)
        elif "txt" in file_path[-10:]:
            run = ParseTxt(file_path).to_simrun(self.args, file_index)
        else:
            raise TypeError("")
        self.s.append(run)

    def save_call(self, argv):
        """save the call as a hidden file"""
        self.call = ""
        for i in argv:
            self.call = self.call + " " + i
        with open(f"{os.getcwd()}/.call.txt", "a") as f:
            f.write(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                + " :"
                + self.call
                + "\n\n"
            )

    def check_same_path(self):
        if len(self.paths) > 1:
            self.same_path = True
            same_path = ""
            for i in self.paths:
                if self.paths.index(i) == 0:
                    same_path = i[: i.rfind("/")]
                else:
                    if same_path != i[: i.rfind("/")]:
                        self.same_path = False
                        break
        else:
            self.same_path = True

    #! ----------------------- Pandas dataset functions ------------------------------

    # **********************************************************************
    # *                       1. get_mode
    # **********************************************************************
    def get_io_mode(self, mode: str):
        """
        Determines the appropriate data I/O method based on the provided mode.

        Args:
            mode (str): A string indicating the mode of operation. It should contain
                        keywords such as "read" or "write" and "async" or "sync".

        Returns:
            function: The corresponding data I/O method based on the mode.
        """
        mode = match_mode(mode)
        if "read" in mode:
            if "async" in mode:
                return self.assign_data_io("read_async_t")
            elif "sync" in mode:
                return self.assign_data_io("read_sync")
        if "write" in mode:
            if "async" in mode:
                return self.assign_data_io("write_async_t")
            elif "sync" in mode:
                return self.assign_data_io("write_sync")

        raise ValueError("Specified mode not found.")

    # **********************************************************************
    # *                       2. assign_mode
    # **********************************************************************
    def assign_mode(self, mode: str):
        """
        Assigns the appropriate data I/O method based on the provided mode.
        Parameters:
        mode (str): A string indicating the mode of operation. It can include
            'read|write_async|sync' to specify the type
        """
        mode = match_mode(mode)
        if "read" in mode:
            if "async" in mode:
                self.df_rat = self.assign_data_io("read_async_t")
            elif "sync" in mode:
                self.df_rst = self.assign_data_io("read_sync")
        if "write" in mode:
            if "async" in mode:
                self.df_wat = self.assign_data_io("write_async_t")
            elif "sync" in mode:
                self.df_wst = self.assign_data_io("write_sync")

    # **********************************************************************
    # *                       3. assign_data
    # **********************************************************************
    def assign_data(self):
        self.df_rst = self.assign_data_io("read_sync")
        self.df_wst = self.assign_data_io("write_sync")
        self.df_rat = self.assign_data_io("read_async_t")
        self.df_rab = self.assign_data_io("read_async_b")
        self.df_wat = self.assign_data_io("write_async_t")
        self.df_wab = self.assign_data_io("write_async_b")
        self.df_time = self.assign_data_time("io_time")

        # df = pd.concat([df,self.get_data_core('write_async_b')])

    # **********************************************************************
    # *                       4. assign_data_io
    # **********************************************************************
    def assign_data_io(self, io_mode="read_sync"):
        """Extract data from the file(s) and gathers in dataframes.
        The fields name are store in 'name_[level]' and their values are stored
        in 'data_[level]'. There are 4 levels provided:
        (1) Application level (overlap or rank metrics): [..]_rank_ovr
        (2) rank level (sum/average of I/O requests): [..]_rank
        (3) high precision rank level (overlap of I/O requests): [..]_ind_ovr
        (4) I/O request level (lowest level): [..]_rank_over

        if the 'ind' flag is not provided, level (3) and (4) are skipped as they are
        expensive to calculate.


        Args:
            io_mode (str, optional): Can be read or write and
            sync or async (required [b] or actual [t]). Supported modes are:
            'read_sync', 'write_sync', 'read_async_t', 'read_async_b', 'write_async_t',
            and 'write_async_b'.Defaults to 'read_sync'.

        Returns:
            tuple[pd.DataFrame,pd.DataFrame,
            pd.DataFrame, pd.DataFrame, pd.DataFrame,]: Five dataframes are
            returned. The first one contains metrics like total bytes transferred, number
            of phases, etc.. The next 4 dataframes contain the I/O data at the 4
            levels explained above
        """
        name = ""
        data_metrics = np.array([])
        for i in range(0, self.n):
            value = getattr(self.s[i], io_mode)
            data = value.get_data()
            if i == 0:
                name = data[0]
                name_rank_ovr = data[2]
                name_rank = data[4]
                name_ind_ovr = data[6]
                name_ind = data[8]
                data_metrics = np.array(data[1])
                data_rank_ovr = np.array(data[3])
                data_rank = np.array(data[5])
                data_ind_ovr = np.array(data[7])
                data_ind = np.array(data[9])

            else:
                data_metrics = np.vstack((data_metrics, data[1]))
                data_rank_ovr = np.concatenate((data_rank_ovr, data[3]), axis=1)
                data_rank = np.concatenate((data_rank, data[5]), axis=1)
                data_ind_ovr = np.concatenate((data_ind_ovr, data[7]), axis=1)
                data_ind = np.concatenate((data_ind, data[9]), axis=1)
            # data.append(d0)
        # check if there is data
        if data_metrics.size == 0:
            raise RuntimeError(
                f"The mode {self.args.mode} contains no values\nChange the mode by using the -m argument."
            )

        df0 = pd.DataFrame(data_metrics, columns=name)
        df0 = df0.sort_values(by=["number_of_ranks"])
        df1 = pd.DataFrame(data_rank_ovr.transpose(), columns=name_rank_ovr)
        df1 = df1.astype({"number_of_ranks": "int"})
        df2 = pd.DataFrame(data_rank.transpose(), columns=name_rank)
        df3 = pd.DataFrame(data_ind_ovr.transpose(), columns=name_ind_ovr)
        df4 = pd.DataFrame(data_ind.transpose(), columns=name_ind)

        return df0, df1, df2, df3, df4

    # **********************************************************************
    # *                       5. assign_data_time
    # **********************************************************************
    def assign_data_time(self, io_mode="io_time"):
        data = []
        name = ""
        for i in range(0, self.n):
            value = getattr(self.s[i], io_mode)
            if i == 0:
                name, d0 = value.get_data()  # Time function
            else:
                _, d0 = value.get_data()

            data.append(d0)
        df0 = pd.DataFrame(data, columns=name)
        df0 = df0.sort_values(by=["number_of_ranks"])
        return df0


def check_open(file: str, name: str = "") -> None:
    """Checks that the file is accessible

    Args:
        file (str): filename
    """
    if os.path.isfile(file):
        pass
    else:
        console = Console()
        if "predictor" in name.lower():
            console.print(
                f"[yellow]Waiting for [b]{file}[/b] to appear in [b]{os.getcwd()}[/b]. \n\n[/]"
            )
        else:
            console.print(
                f"[red]--- Error --- [/]\n"
                f"[red]-> Could not open file [b]{file}[/b]. \n[/]"
                f"[yellow]Make sure [b]{file}[/b] exists in [b]{os.getcwd()}[/b]. \n\n[/]"
            )
        exit()


def get_rank(name: str) -> int:
    """Get the number of ranks from the name of the file

    Args:
        name (str): name of file

    Returns:
        int: number of ranks
    """
    if isinstance(name, int):
        return name
    else:
        start = name.rfind("/")
        end = max(name.rfind(".json"), name.rfind(".darshan"), name.rfind(".msgpack"))
        rank = name[start + 1 : end]
        strs = ["_", "-", " "]
        if any(x in rank for x in strs):
            for x in strs:
                if x in rank:
                    end = rank.rfind(x)
                    rank = rank[:end]
        return int(rank)


def get_filename(path: str) -> str:
    """Returns filename from absolute path

    Args:
        path (str): absolute path to file (including name)

    Returns:
        str: filename
    """
    tmp = path.rfind("/")
    out = path[tmp + 1 :] if tmp > 0 else path
    return out
