"""
This module provides the `JitTime` class for managing and recording timing information
for various stages of a process. It includes functionality to display timing data,
convert it to a dictionary, and save it in JSON format.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Dec 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import json
import os
from datetime import datetime

from rich.console import Console
from rich.table import Table

from ftio.api.gekkoFs.jit.jitsettings import JitSettings


class JitTime:
    """
    A class to manage and record timing information for application, stage-in, and stage-out processes.
    """

    def __init__(self) -> None:
        """
        Initialize the JitTime instance with default timing values set to 0.
        """
        self._app: float = 0
        self._stage_in: float = 0
        self._stage_out: float = 0

    @property
    def app(self) -> float:
        """
        Get the application time.

        Returns:
            float: The application time.
        """
        return self._app

    @app.setter
    def app(self, app: float) -> None:
        """
        Set the application time.

        Args:
            app (float): The application time to set.
        """
        self._app = app

    @app.deleter
    def app(self) -> None:
        """
        Delete the application time.
        """
        del self._app

    @property
    def stage_out(self) -> float:
        """
        Get the stage-out time.

        Returns:
            float: The stage-out time.
        """
        return self._stage_out

    @stage_out.setter
    def stage_out(self, stage_out: float) -> None:
        """
        Set the stage-out time.

        Args:
            stage_out (float): The stage-out time to set.
        """
        self._stage_out = stage_out

    @stage_out.deleter
    def stage_out(self) -> None:
        """
        Delete the stage-out time.
        """
        del self._stage_out

    @property
    def stage_in(self) -> float:
        """
        Get the stage-in time.

        Returns:
            float: The stage-in time.
        """
        return self._stage_in

    @stage_in.setter
    def stage_in(self, stage_in: float) -> None:
        """
        Set the stage-in time.

        Args:
            stage_in (float): The stage-in time to set.
        """
        self._stage_in = stage_in

    @stage_in.deleter
    def stage_in(self) -> None:
        """
        Delete the stage-in time.
        """
        del self._stage_in

    def print_time(self) -> str:
        """
        Print the timing information in a formatted table and return it as a string.

        Returns:
            str: The formatted timing information.
        """
        console = Console()
        text = (
            f"App time      : {self._app:.6f}s\n"
            f"Stage out time: {self._stage_out:.6f}s\n"
            f"Stage in time : {self._stage_in:.6f}s\n"
            "--------------------------------\n"
            f"Total time : {self._app + self._stage_out + self._stage_in:.6f}s\n"
        )

        # This block of code is creating a table using the `rich` library in Python to display the time data
        # in a structured format. Here's a breakdown of what each part does:
        # This block of code is creating a table using the `rich` library in Python to display the time data
        # related to different stages. Here's a breakdown of what each part does:
        # Create a Table object
        table = Table(show_header=True, header_style="bold magenta")

        # Add columns to the table
        table.add_column("Task", justify="left")
        table.add_column("Time (s)", justify="left")

        # Add rows with the time data
        table.add_row("App", f"{self._app:.8f}")
        table.add_row("Stage out", f"{self.stage_out:.8f}")
        table.add_row("Stage in", f"{self.stage_in:.8f}")
        table.add_row("-" * 10, "")
        table.add_row(
            "Total",
            f"{self._app + self._stage_out + self._stage_in:.8f}",
            style="bold green",
        )

        # Print the table to the console
        console.print(table)

        return text

    def to_dict(self) -> dict:
        """
        Convert the timing information to a dictionary.

        Returns:
            dict: A dictionary containing the timing information.
        """
        return {
            "app": self._app,
            "stage_in": self._stage_in,
            "stage_out": self._stage_out,
        }

    def dump_json(self, settings: JitSettings, add_timestamp: bool = True) -> None:
        """
        Save the timing information and settings to a JSON file, with an option to add timestamps to new entries.

        Args:
            settings (JitSettings): The settings object containing additional data.
            add_timestamp (bool): Whether to add a timestamp to new entries (default is True).
        """
        # Create the base data dictionary from the instance and settings
        # data = {**self.to_dict(), **settings.to_dict()}\
        if add_timestamp:
            data = {
                **self.to_dict(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **settings.to_dict(),
            }
        else:
            data = {**self.to_dict(), **settings.to_dict()}

        # result.json lives at the per-app level (logs/<app>), one file per app so
        # a multi-app sweep does not overwrite entries that share nodes+mode.
        parent = getattr(settings, "result_dir", "") or os.path.dirname(settings.log_dir)
        json_path = os.path.join(parent, "result.json")

        try:
            with open(json_path, "r+") as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []

                # Check if there is an existing entry with the same number of nodes and mode
                for entry in existing_data:
                    if entry.get("nodes") == data["nodes"]:
                        if add_timestamp:
                            entry["data"].append(data)
                            break
                        else:
                            for i, data_entry in enumerate(entry["data"]):
                                if data_entry["mode"] == data["mode"]:
                                    # Update existing entry (no timestamp added here)
                                    entry["data"][i] = data
                                    break
                            else:
                                # If no mode match, add the new entry (with timestamp if enabled)
                                entry["data"].append(data)
                            break
                else:
                    # If no entry with the same nodes exists, add a new one (with timestamp if enabled)
                    existing_data.append({"nodes": data["nodes"], "data": [data]})

                # Write the updated data back to the file
                file.seek(0)
                json.dump(existing_data, file, indent=4)
                file.truncate()

        except FileNotFoundError:
            # If the file does not exist, create it with the new data (with timestamp if enabled)
            with open(json_path, "w") as file:
                json.dump([{"nodes": data["nodes"], "data": [data]}], file, indent=4)

    def print_and_save_time(self, settings: JitSettings) -> None:
        """
        Print the timing information and save it to a log file and JSON file.

        Args:
            settings (JitSettings): The settings object containing the log directory.
        """
        # get the time
        text = self.print_time()
        # write it out to the file
        time_log_file = os.path.join(settings.log_dir, "time.log")
        with open(time_log_file, "a") as file:
            file.write(text)

        self.dump_json(settings)
