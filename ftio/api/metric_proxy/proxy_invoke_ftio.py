"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import sys

from ftio.api.metric_proxy.helper import data_to_json
from ftio.api.metric_proxy.parse_proxy import load_proxy_trace_stdin
from ftio.prediction.tasks import ftio_metric_task_save


def main():
    argv = sys.argv[1:]
    metrics = load_proxy_trace_stdin(deriv_and_not_deriv=False)
    ranks = 32

    # Make sure we do no plots
    argv.extend(["-e", "no"])

    data = []

    for metric, arrays in metrics.items():
        ftio_metric_task_save(data, metric, arrays, argv, ranks, False)
    data_to_json(data)


if __name__ == "__main__":
    main()
