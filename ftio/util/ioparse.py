"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import sys

from ftio.parse.print import Print


def main(args=sys.argv):
    out = Print(args)
    out.print_json_lines()


if __name__ == "__main__":
    main(sys.argv)
