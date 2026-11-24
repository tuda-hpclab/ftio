"""
Parse text file contaiting three fields:
the bandwidth, the start time, and the end time.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from ftio.parse.simrun import Simrun
from ftio.parse.txt_reader import extract


class ParseCustom:
    """class to parse txt file"""

    def __init__(self, path):
        self.path = path
        if self.path[-1] == "/":
            self.path = self.path[:-1]

    def to_simrun(self, args, index: int = 0):
        """Convert to Simrun class
        Args:
            ars (argparse): command line arguments
            index: file index in case several files are passed
        Returns:
            Simrun: Simrun object
        """
        dataframe, ranks = extract(self.path, args, True)

        return Simrun(dataframe, "txt", str(ranks), args, index)
