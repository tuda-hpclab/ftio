"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go


class StackPlot:
    def __init__(self, args):
        self.engine = "plotly"
        if args.engine:
            self.engine = args.engine
            if "plotly" in args.engine:
                self.fig = go.Figure()

            if "mat" in args.engine:
                self.b = []
                self.t = []

    def add(self, b, t, name="", group=0):
        if "plotly" in self.engine:
            txt = {}
            if group != 0:
                txt = {"legendgroup": group, "legendgrouptitle_text": group}
            self.fig.add_trace(
                go.Scatter(
                    x=t,
                    y=b,
                    name=name,
                    **txt,
                    line={"shape": "hv", "width": 0.5},
                    fill="tozeroy",
                    hoverinfo="x+y",
                    stackgroup="one",
                )
            )
        elif "mat" in self.engine:
            if group != 0:
                label = f"ranks {group} -> {name}"
            else:
                label = name
            plt.plot([], [], label=label)
            if not self.b:
                self.t = t
            self.b.append(b)

    def finilize(self):
        if "plotly" in self.engine:
            # template = "plotly_dark"
            template = "plotly"
            width = 1100
            height = 600
            self.fig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Bandwidth (B/s)",
                font={"family": "Courier New, monospace", "size": 24},
                width=width,
                height=height / 1.1,
                template=template,
            )
            self.fig.update_layout(legend={"groupclick": "toggleitem"})
            self.fig.show()
        elif "mat" in self.engine:
            plt.stackplot(self.t, self.b, baseline="zero")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Bandwidth (B/s)")
            plt.show()
