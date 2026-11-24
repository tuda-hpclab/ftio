"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import pandas as pd

import ftio.plot.dash_files.constants.io_mode as io_mode


class FileData:
    def __init__(
        self,
        run: int,
        rank: int,
        name: str,
        data_actual: list[pd.DataFrame],
        data_required: list[pd.DataFrame],
    ):
        self._run = run
        self._rank = rank
        self._name = name
        self._data_actual = data_actual
        self._data_required = data_required
        self._init_masks()

    def _init_masks(self):
        self._mask = self._data_actual[1]["number_of_ranks"].isin([self.rank])
        self._mask_ind = self._data_actual[3]["number_of_ranks"].isin([self.rank])
        self._mask2 = self._data_actual[1]["file_index"][self._mask].isin([self.run])
        self._mask2_ind = self._data_actual[3]["file_index"][self._mask_ind].isin(
            [self.run]
        )

    @property
    def run(self) -> int:
        return self._run

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def name(self) -> str:
        return self._name

    @property
    def data_actual_is_not_empty(self) -> bool:
        return len(self._data_actual) != 0

    @property
    def data_required_is_not_empty(self) -> bool:
        return len(self._data_required) != 0

    @property
    def actual_time_overlap(self):
        return self._data_actual[1]["t_overlap"][self._mask][self._mask2]

    @property
    def required_time_overlap(self):
        return self._data_required[1]["t_overlap"][self._mask][self._mask2]

    @property
    def actual_time_overlap_individual(self):
        return self._data_actual[3]["t_overlap_ind"][self._mask_ind][self._mask2_ind]

    @property
    def required_time_overlap_individual(self):
        return self._data_required[3]["t_overlap_ind"][self._mask_ind][self._mask2_ind]

    @property
    def actual_bandwidth_overlap_average(self):
        return self._data_actual[1]["b_overlap_avr"][self._mask][self._mask2]

    @property
    def required_bandwidth_overlap_average(self):
        return self._data_required[1]["b_overlap_avr"][self._mask][self._mask2]

    @property
    def actual_bandwidth_overlap_sum(self):
        return self._data_actual[1]["b_overlap_sum"][self._mask][self._mask2]

    @property
    def required_bandwidth_overlap_sum(self):
        return self._data_required[1]["b_overlap_sum"][self._mask][self._mask2]

    @property
    def actual_bandwidth_overlap_individual(self):
        return self._data_actual[3]["b_overlap_ind"][self._mask_ind][self._mask2_ind]

    @property
    def required_bandwidth_overlap_individual(self):
        return self._data_required[3]["b_overlap_ind"][self._mask_ind][self._mask2_ind]


class DataSource:
    def __init__(self, plot_core, io_mode: str):
        self._io_mode = io_mode
        self._plot_core = plot_core

        self._data = plot_core.data

        self._init_data_actual_and_required()
        self._init_names_and_ranks()
        self._init_file_data_dictionary()

    def _init_data_actual_and_required(self):
        match self.io_mode:
            case io_mode.ASYNC_READ:
                self._data_actual: list[pd.DataFrame] = (
                    self._data.df_rat
                    if not (self._data.df_rat is None or self._data.df_rat[1].empty)
                    else []
                )
                self._data_required: list[pd.DataFrame] = (
                    self._data.df_rab
                    if not (self._data.df_rab is None or self._data.df_rab[1].empty)
                    else []
                )
            case io_mode.ASYNC_WRITE:
                self._data_actual: list[pd.DataFrame] = (
                    self._data.df_wat
                    if not (self._data.df_wat is None or self._data.df_wat[1].empty)
                    else []
                )
                self._data_required: list[pd.DataFrame] = (
                    self._data.df_wab
                    if not (self._data.df_wab is None or self._data.df_wab[1].empty)
                    else []
                )
            case io_mode.SYNC_READ:
                self._data_actual: list[pd.DataFrame] = (
                    self._data.df_rst
                    if not (self._data.df_rst is None or self._data.df_rst[1].empty)
                    else []
                )
                self._data_required: list[pd.DataFrame] = []
            case io_mode.SYNC_WRITE:
                self._data_actual: list[pd.DataFrame] = (
                    self._data.df_wst
                    if not (self._data.df_wst is None or self._data.df_wst[1].empty)
                    else []
                )
                self._data_required: list[pd.DataFrame] = []
            case _:
                raise Exception("invalid mode")

    def _init_names_and_ranks(self) -> None:
        self._ranks: pd.Series = pd.Series()
        if len(self._data_actual) != 0:
            self._ranks = self._data_actual[0]["number_of_ranks"].astype(int)
        elif len(self._data_required) != 0:
            self._ranks = self._data_required[0]["number_of_ranks"].astype(int)
        names_and_ranks_df = pd.concat(
            [pd.Series(self._plot_core.names, name="filenames"), self._ranks],
            axis=1,
        )
        self._filenames = names_and_ranks_df["filenames"].to_list()
        self._ranks = names_and_ranks_df["number_of_ranks"].to_list()
        self._ranks_unique = pd.unique(self.ranks)
        self._ranks_unique.sort()

    def _init_file_data_dictionary(self):
        self._file_data_by_filename: dict[str, FileData] = {}
        for idx, filename in enumerate(self._filenames):
            self._file_data_by_filename[filename] = FileData(
                idx,
                self._ranks[idx],
                self._filenames[idx],
                self._data_actual,
                self._data_required,
            )

    @property
    def io_mode(self) -> str:
        return self._io_mode

    @property
    def filenames(self) -> list[str]:
        return self._filenames

    @property
    def ranks(self) -> list[int]:
        return self._ranks

    @property
    def ranks_unique(self) -> list[int]:
        return self._ranks_unique

    @property
    def fontfamily(self) -> str:
        return "Courier New, monospace"

    @property
    def fontsize(self) -> int:
        return 17

    @property
    def width_figure(self) -> int:
        return self._plot_core.width

    @property
    def height_figure(self) -> int:
        return self._plot_core.height

    @property
    def file_data_by_filename(self) -> dict[str, FileData]:
        return self._file_data_by_filename

    @property
    def individual_is_selected(self) -> bool:
        return self._data.args.ind

    @property
    def n_shown_samples(self) -> int:
        return self._data.args.n_shown_samples

    @property
    def merge_plots_is_selected(self) -> bool:
        return self._data.args.merge_plots


def get_data_source(plot_core, io_mode_: str) -> DataSource:
    """Creates and returns a suitable DataSource-class depending on the specified io_mode.

    Args:
        plot_core (plot_core): plot_core-class
        mode (str): input-output-mode
            current valid modes:
                io_mode.ASYNC_READ,
                io_mode.ASYNC_WRITE,
                io_mode.SYNC_READ,
                io_mode.SYNC_WRITE

    Raises:
        Exception: raises if mode can't be handled

    Returns:
        DataSource: Suitable DataSource-class for the mode
    """
    match io_mode_:
        case io_mode.ASYNC_READ:
            return DataSource(plot_core, io_mode.ASYNC_READ)
        case io_mode.ASYNC_WRITE:
            return DataSource(plot_core, io_mode.ASYNC_WRITE)
        case io_mode.SYNC_READ:
            return DataSource(plot_core, io_mode.SYNC_READ)
        case io_mode.SYNC_WRITE:
            return DataSource(plot_core, io_mode.SYNC_WRITE)
        case _:
            raise Exception("invalid mode")
