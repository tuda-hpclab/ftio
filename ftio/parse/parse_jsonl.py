"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import jsonlines

from ftio.parse.simrun import Simrun


class ParseJsonl:
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
        with jsonlines.open(file, "r") as jsonl_f:
            data = list(jsonl_f)

        return Simrun(data, "jsonl", file, args, index)
