"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from ftio.parse.recorder_reader import extract
from ftio.parse.simrun import Simrun


class ParseRecorder:

    def __init__(self, path):
        self.path = path
        if self.path[-1] == "/":
            self.path = self.path[:-1]

    def to_simrun(self, args, index: int = 0):
        """Convert to Simrun class
        Args:
            ts (double): [optional] desired start time
            index (int): file index
        Returns:
            Simrun: Simrun object
        """
        data, ranks = extract(self.path, args)

        return Simrun(data, "recorder", str(ranks), args, index)
