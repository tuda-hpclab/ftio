"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Jul 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import plotly.express as px


def quick_plot(x, y, x_label="time", y_label="Bandwidth"):
    fig = px.scatter(x=x, y=y, labels={"x": x_label, "y": y_label})
    fig.show()
