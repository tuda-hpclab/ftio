"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""


class FreqData:
    """contains the data from the frequency analysis including the
    1. bandwidth over time spaced with constant steps (1/freq)
    2. settings
    3. original data before sampling
    """

    def __init__(self, df1, df2, df3):
        self.data_df = df1
        self.settings_df = df2
        self.original_df = df3
