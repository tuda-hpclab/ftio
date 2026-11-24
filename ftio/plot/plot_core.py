"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import importlib.util
import os
import socket
from multiprocessing import Process
from threading import Thread

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ftio.freq.helper import MyConsole
from ftio.parse.metrics import Metrics
from ftio.parse.scales import Scales
from ftio.plot.helper import (
    add_fig_col,
    add_fig_row,
    format_plot,
    legend_fix,
    save_fig,
)
from ftio.plot.plot_error import plot_error_bar, plot_time_error_bar
from ftio.plot.print_html import PrintHtml
from ftio.plot.units import find_unit

CONSOLE = MyConsole()


def _find_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


class PlotCore:
    def __init__(self, args):
        self.data = Scales(args)
        self.data.assign_data()
        self.names = self.data.names
        self.barprecision = "%{y:.1f}"
        # ? IEEE
        self.width = 900
        self.height = 400

        self.barprecision = "%{y:.0f}"

    def plot_io(self):
        if "dash" in self.data.args.engine.lower():
            self.plot_dash()
        else:
            self.plot_plotly()

    def plot_dash(self):
        """Starts a dash server to display the figures dynamically."""
        if importlib.util.find_spec("dash") is not None:
            import dash.dependencies

            if not hasattr(dash.dependencies, "_Wildcard"):
                dash.dependencies._Wildcard = object
            from ftio.plot.dash_files.dash_app import IOAnalysisApp

            app = IOAnalysisApp(self)
            app.run(port=_find_free_port(), debug=False)
        else:
            from rich.markup import escape

            CONSOLE.info(
                "[red]"
                + escape(
                    "Please install plot-libs: pip install '.[external-libs,development-libs,plot-libs]'"
                )
            )

    def plot_plotly(self):
        self.f_aw = []
        self.f_ar = []
        self.f_sw = []
        self.f_sr = []
        self.f_t = []
        print("\n\033[1;34mGenerating plots\033[1;0m")
        args = self.data.args
        path = self.data.paths

        if args.render:
            if "stat" in args.render:  # only generate a single image
                args.threaded = False
                if not args.mode:
                    raise Exception("To use render=static, --mode=... must be also set")
        else:
            args.render = "dynamic"

        (
            print("\033[1;32mMultithreaded is on \033[1;0m")
            if args.threaded
            else print("\033[1;31mMultithreaded is off \033[1;0m")
        )

        out = PrintHtml(path, self.names, args)
        out.generate_html_start()
        if args.threaded:
            t = []
            t.append(
                Thread(
                    target=self.plot_and_generate_html,
                    args=(
                        "time",
                        out,
                    ),
                )
            )
            t.append(
                Thread(
                    target=self.plot_and_generate_html,
                    args=(
                        "async write",
                        out,
                    ),
                )
            )  # needs comma to be treated as a tuple
            t.append(
                Thread(
                    target=self.plot_and_generate_html,
                    args=(
                        "async read",
                        out,
                    ),
                )
            )
            t.append(
                Thread(
                    target=self.plot_and_generate_html,
                    args=(
                        "sync write",
                        out,
                    ),
                )
            )
            t.append(
                Thread(
                    target=self.plot_and_generate_html,
                    args=(
                        "sync read",
                        out,
                    ),
                )
            )

            for thread in t:
                thread.start()

            for thread in t:
                thread.join()
        else:
            if "stat" in args.render:
                if "async" in args.mode:
                    if "write" in args.mode:
                        self.plot_io_mode(
                            "async write", self.data.df_wat, self.data.df_wab
                        )
                    else:
                        self.plot_io_mode(
                            "async read", self.data.df_rat, self.data.df_rab
                        )
                elif "sync" in args.mode:
                    if "write" in args.mode:
                        self.plot_io_mode("sync write", self.data.df_wst)
                    else:
                        self.plot_io_mode("sync read", self.data.df_rst)
                else:
                    pass
                exit(0)
            else:
                self.plot_time()
                out.generate_html_core("time.html", self.get_figure("time.html"))
                self.plot_io_mode("async write", self.data.df_wat, self.data.df_wab)
                out.generate_html_core(
                    "write_async.html", self.get_figure("write_async.html")
                )
                self.plot_io_mode("async read", self.data.df_rat, self.data.df_rab)
                out.generate_html_core(
                    "read_async.html", self.get_figure("read_async.html")
                )
                self.plot_io_mode("sync write", self.data.df_wst)
                out.generate_html_core(
                    "write_sync.html", self.get_figure("write_sync.html")
                )
                self.plot_io_mode("sync read", self.data.df_rst)
                out.generate_html_core(
                    "read_sync.html", self.get_figure("read_sync.html")
                )

        out.generate_html_end()
        print("\033[1;32m------------------- done -------------------\n\033[1;0m")

        return [*self.f_aw, *self.f_ar, *self.f_sr, *self.f_sr, *self.f_t]

    # **********************************************************************
    # *                       2. plot Multithreaded
    # **********************************************************************
    def plot_and_generate_html(self, mode, out):
        if "async" in mode:
            if "write" in mode:
                self.plot_io_mode("async write", self.data.df_wat, self.data.df_wab)
                out.generate_html_core(
                    "write_async.html", self.get_figure("write_async.html")
                )
            else:
                self.plot_io_mode("async read", self.data.df_rat, self.data.df_rab)
                out.generate_html_core(
                    "read_async.html", self.get_figure("read_async.html")
                )
        elif "sync" in mode:
            if "write" in mode:
                self.plot_io_mode("sync write", self.data.df_wst)
                out.generate_html_core(
                    "write_sync.html", self.get_figure("write_sync.html")
                )
            else:
                self.plot_io_mode("sync read", self.data.df_rst)
                out.generate_html_core(
                    "read_sync.html", self.get_figure("read_sync.html")
                )
        else:
            self.plot_time()
            out.generate_html_core("time.html", self.get_figure("time.html"))

    # **********************************************************************
    # *                       2. plot
    # **********************************************************************
    def plot_io_mode(self, mode, df_t, df_b=None):
        """Plots I/O

        Args:
            mode (string): _description_
            df_t (list): list containing throughput
            df_b (list, optional): list containing bandwidth (async). Defaults to [].
        """
        if df_b is None:
            df_b = []
        args = self.data.args
        CONSOLE.print(f"│   ├── [green]Creating plot {mode}[/]")
        f = []
        if isinstance(df_t[1], pd.DataFrame) and df_t[1].empty:
            # Empty throughput by B might exist
            df_t = []
            # empty dataset
            if not df_b or isinstance(df_b[1], pd.DataFrame) and df_b[1].empty:
                self.set_figure(f, mode)
                return

        f.append(go.Figure())
        if df_t:
            f[-1].add_trace(
                go.Box(
                    x=df_t[2]["number_of_ranks"],
                    y=df_t[2]["b_rank_avr"],
                    name="$T_{i,j}$",
                )
            )
        if df_b:
            f[-1].add_trace(
                go.Box(
                    x=df_b[2]["number_of_ranks"],
                    y=df_b[2]["b_rank_sum"],
                    name="$B_{i,j}$",
                )
            )

        if df_t:
            # f.append(go.Figure())
            f[-1].add_trace(
                go.Scatter(
                    x=df_t[0]["number_of_ranks"],
                    y=df_t[0]["harmonic_mean"],
                    mode="lines+markers",
                    name="$hmean(T_{i,j})$",
                    marker_symbol=0,
                    line_dash="dash",
                )
            )
            f[-1].add_trace(
                go.Scatter(
                    x=df_t[0]["number_of_ranks"],
                    y=df_t[0]["arithmetic_mean"],
                    mode="lines+markers",
                    name="$amean(T_{i,j})$",
                    marker_symbol=1,
                    line_dash="dash",
                )
            )
            f[-1].add_trace(
                go.Scatter(
                    x=df_t[0]["number_of_ranks"],
                    y=df_t[0]["median"],
                    mode="lines+markers",
                    name="$median(T_{i,j})$",
                    marker_symbol=2,
                    line_dash="dash",
                )
            )
        if df_b:
            f[-1].add_trace(
                go.Scatter(
                    x=df_b[0]["number_of_ranks"],
                    y=df_b[0]["harmonic_mean"],
                    mode="lines+markers",
                    name="$hmean(B_{i,j})$",
                    marker_symbol=100,
                )
            )
            f[-1].add_trace(
                go.Scatter(
                    x=df_b[0]["number_of_ranks"],
                    y=df_b[0]["arithmetic_mean"],
                    mode="lines+markers",
                    name="$amean(B_{i,j})$",
                    marker_symbol=101,
                )
            )
            f[-1].add_trace(
                go.Scatter(
                    x=df_b[0]["number_of_ranks"],
                    y=df_b[0]["median"],
                    mode="lines+markers",
                    name="$median(B_{i,j})$",
                    marker_symbol=102,
                )
            )
        f[-1].update_layout(
            xaxis_title="Ranks",
            yaxis_title="Transfer Rate (B/s)",
            width=self.width,
            height=self.height,
        )
        f[-1] = format_plot(f[-1], 17)

        f.append(go.Figure())
        if df_t:
            f[-1].add_trace(
                go.Box(
                    x=df_t[2]["number_of_ranks"],
                    y=df_t[2]["b_rank_avr"],
                    name="$T_{i,j}$",
                    boxpoints="all",
                    jitter=0.3,
                )
            )
        if df_b:
            f[-1].add_trace(
                go.Box(
                    x=df_b[2]["number_of_ranks"],
                    y=df_b[2]["b_rank_sum"],
                    name="$B_{i,j}$",
                    boxpoints="all",
                    jitter=0.3,
                )
            )
        f[-1].update_layout(
            xaxis_title="Ranks",
            yaxis_title="Transfer Rate (B/s)",
            width=self.width,
            height=self.height,
        )  # ,legend_title_text='Metrics' )
        f[-1] = format_plot(f[-1], 17)

        f.append(go.Figure())
        if df_t:
            f[-1].add_trace(
                go.Scatter(
                    x=df_t[0]["number_of_ranks"],
                    y=df_t[0]["harmonic_mean"],
                    mode="lines+markers",
                    name="$hmean(T_{i,j})$",
                    marker_symbol=0,
                    line_dash="dash",
                )
            )
            f[-1].add_trace(
                go.Scatter(
                    x=df_t[0]["number_of_ranks"],
                    y=df_t[0]["arithmetic_mean"],
                    mode="lines+markers",
                    name="$amean(T_{i,j})$",
                    marker_symbol=1,
                    line_dash="dash",
                )
            )
            f[-1].add_trace(
                go.Scatter(
                    x=df_t[0]["number_of_ranks"],
                    y=df_t[0]["median"],
                    mode="lines+markers",
                    name="$median(T_{i,j})$",
                    marker_symbol=2,
                    line_dash="dash",
                )
            )
        if df_b:
            f[-1].add_trace(
                go.Scatter(
                    x=df_b[0]["number_of_ranks"],
                    y=df_b[0]["harmonic_mean"],
                    mode="lines+markers",
                    name="$hmean(B_{i,j})$",
                    marker_symbol=100,
                )
            )
            f[-1].add_trace(
                go.Scatter(
                    x=df_b[0]["number_of_ranks"],
                    y=df_b[0]["arithmetic_mean"],
                    mode="lines+markers",
                    name="$amean(B_{i,j})$",
                    marker_symbol=101,
                )
            )
            f[-1].add_trace(
                go.Scatter(
                    x=df_b[0]["number_of_ranks"],
                    y=df_b[0]["median"],
                    mode="lines+markers",
                    name="$median(B_{i,j})$",
                    marker_symbol=102,
                )
            )
        f[-1].update_layout(
            xaxis_title="Ranks",
            yaxis_title="Transfer Rate (B/s)",
            width=self.width,
            height=self.height,
        )
        f[-1] = format_plot(f[-1], 17)

        _ = go.Figure(
            layout={"template": "plotly"}
        )  #!fixes plotly error for default values
        if df_t:
            fig_tmp = px.histogram(
                df_t[1],
                x="b_overlap_avr",
                color="number_of_ranks",
                marginal="box",
                labels={
                    "b_overlap_avr": "Throughput  (B/s)",
                    "number_of_ranks": "Ranks",
                },
            )  # "number_of_ranks": "idth (cm)","number_of_ranks": "Rank"
            fig_tmp.update_layout(
                width=self.width,
                height=self.height,
                title="Distribution",
                xaxis_title="Throughput (B/s)",
                yaxis_title="Count",
            )
            fig_tmp = format_plot(fig_tmp, 17)
            f.append(fig_tmp)

        if df_b:
            fig_tmp = px.histogram(
                df_b[1],
                x="b_overlap_sum",
                color="number_of_ranks",
                marginal="box",
                labels={
                    "b_overlap_sum": "Bandwidth  (B/s)",
                    "number_of_ranks": "Ranks",
                },
            )  # "number_of_ranks": "idth (cm)","number_of_ranks": "Rank"
            fig_tmp.update_layout(
                width=self.width,
                height=self.height,
                title="Distribution",
                xaxis_title="Bandwidth (B/s)",
                yaxis_title="Count",
            )
            fig_tmp = format_plot(fig_tmp, 17)
            f.append(fig_tmp)

        io_stats = Metrics(args)
        if df_t:
            ranks = df_t[0]["number_of_ranks"].sort_index()
        elif df_b:
            ranks = df_b[0]["number_of_ranks"].sort_index()
        for i in pd.unique(ranks):
            try:
                i = int(i)
                index = df_t[1]["number_of_ranks"].isin([i])
                index_ind = df_t[3]["number_of_ranks"].isin([i])
            except Exception:
                index = df_t[1]["number_of_ranks"].astype(str).isin([i])
                index_ind = df_t[3]["number_of_ranks"].astype(str).isin([i])
            if (
                len(df_t[1]["file_index"][index]) != 0
                or len(df_t[3]["file_index"][index_ind]) != 0
            ):
                for j in pd.unique(df_t[1]["file_index"][index]):
                    j = int(j)
                    f.append(go.Figure())
                    index2 = df_t[1]["file_index"][index].isin([j])
                    index2_ind = df_t[3]["file_index"][index_ind].isin([j])
                    # find the order and unit of the plots on y-axis
                    unit, order = find_unit(
                        df_t, index, index2, index_ind, index2_ind, args
                    )
                    # # compatibility mode with earlier version
                    # CONSOLE.info(f"[yellow]WARNING:[/] Compatibility Mode")
                    # unit, order = find_unit(df_t,index,index2,index_ind,index2_ind,args,pow(10,6))
                    # order = order*pow(10,6)

                    # if no async, show only T instead of T_A

                    T_name = "$T_A$"
                    if not df_b:
                        T_name = "$T$"

                    # ? Avr plot
                    if args.avr:
                        if df_t:
                            f[-1].add_trace(
                                go.Scatter(
                                    x=df_t[1]["t_overlap"][index][index2],
                                    y=df_t[1]["b_overlap_avr"][index][index2] * order,
                                    mode="lines",
                                    name=T_name,  # name="$T$", # For BW limit paper
                                    line={"shape": "hv"},
                                    fill="tozeroy",
                                )
                            )

                        if df_b:
                            f[-1].add_trace(
                                go.Scatter(
                                    x=df_b[1]["t_overlap"][index][index2],
                                    y=df_b[1]["b_overlap_avr"][index][index2] * order,
                                    mode="lines",
                                    name="$B_A$",  # name="$B_L$", # For BW limit paper
                                    line={"shape": "hv"},
                                    fill="tozeroy",
                                )
                            )

                    # ? Sum plot
                    if args.sum:
                        if df_t and df_b:  # shows this only in async mode
                            f[-1].add_trace(
                                go.Scatter(
                                    x=df_t[1]["t_overlap"][index][index2],
                                    y=df_t[1]["b_overlap_sum"][index][index2] * order,
                                    mode="lines",
                                    name="$T_S$",
                                    line={"shape": "hv"},
                                    fill="tozeroy",
                                )
                            )

                        if df_b:
                            f[-1].add_trace(
                                go.Scatter(
                                    x=df_b[1]["t_overlap"][index][index2],
                                    y=df_b[1]["b_overlap_sum"][index][index2] * order,
                                    mode="lines",
                                    name="$B_S$",  # name="$B$", # For BW limit paper
                                    line={"shape": "hv"},
                                    fill="tozeroy",
                                )
                            )

                    # ? ind plot
                    if args.ind:
                        if df_t:
                            f[-1].add_trace(
                                go.Scatter(
                                    x=df_t[3]["t_overlap_ind"][index_ind][index2_ind],
                                    y=df_t[3]["b_overlap_ind"][index_ind][index2_ind]
                                    * order,
                                    mode="lines",
                                    name="$T_E$",
                                    line={"shape": "hv"},
                                    fill="tozeroy",
                                )
                            )

                        if df_b:
                            f[-1].add_trace(
                                go.Scatter(
                                    x=df_b[3]["t_overlap_ind"][index_ind][index2_ind],
                                    y=df_b[3]["b_overlap_ind"][index_ind][index2_ind]
                                    * order,
                                    mode="lines",
                                    name="$B_E$",
                                    line={"shape": "hv"},
                                    fill="tozeroy",
                                )
                            )

                    f[-1].update_layout(
                        barmode="stack",
                        xaxis_title="Time (s)",
                        yaxis_title=f"Transfer Rate ({unit})",
                        width=self.width,
                        height=self.height,
                        showlegend=True,
                        title=f"{i} Ranks (Run {j})",
                    )
                    f[-1] = format_plot(f[-1], 17)

                    if self.names and (args.avr or args.sum or args.ind):
                        f[-1].update_layout(
                            title=f"{i} Ranks (Run {j}: {self.names[j]})",
                        )

                    if args.avr or args.sum or args.ind:
                        if args.zoom:
                            f[-1].update_layout(yaxis_range=[0, args.zoom])
                        else:
                            f[-1].update_layout(yaxis_rangemode="nonnegative")
                        f[-1].update_xaxes(
                            ticks="outside",
                            tickcolor="black",
                            ticklen=6,
                            minor={
                                "ticklen": 3,
                                "tickcolor": "black",
                                "tickmode": "auto",
                                "nticks": 10,
                                "showgrid": True,
                            },
                        )
                        f[-1].update_yaxes(
                            ticks="outside",
                            tickcolor="black",
                            ticklen=6,
                            minor={
                                "ticklen": 3,
                                "tickcolor": "black",
                                "tickmode": "auto",
                                "nticks": 10,
                                "showgrid": True,
                            },
                        )

                    #! compute statistics
                    if df_b:
                        io_stats.add(
                            i,
                            j,
                            df_t[1][index][index2],
                            df_t[3][index_ind][index2_ind],
                            df_b[1][index][index2],
                            df_b[3][index_ind][index2_ind],
                        )
                    else:
                        if df_t:
                            io_stats.add(
                                i,
                                j,
                                df_t[1][index][index2],
                                df_t[3][index_ind][index2_ind],
                            )

            if "stat" in args.render:
                path = os.path.join(os.getcwd(), f"exported_images_{i}")
                if not os.path.exists(path):
                    os.mkdir(path)
                if not self.names:
                    self.names.append(str(i))
                self.names = [x.replace(".", "_") for x in self.names]
                procs = []
                for fig in f:
                    procs.append(
                        Process(
                            target=save_fig,
                            args=(fig, f, path, self.names[-1]),
                        )
                    )
                    procs[-1].start()

                while procs:
                    for p in procs:
                        if p.is_alive():
                            pass
                        else:
                            p.join()
                            procs.remove(p)

                print(f"Images  saved to {path}\nExciting")
                exit(0)

        #! plot overlap statistics
        if self.nRun >= 1:
            io_stats.get_data()
            labels = ["A", "S", "E"]
            for count, type in enumerate(["average", "sum", "ind"]):
                f.append(go.Figure())
                f[-1].update_layout(width=self.width, height=self.height)
                f[-1].update_xaxes(
                    minor={"ticklen": 6, "tickcolor": "black", "showgrid": True}
                )
                for i in ["max", "min", "median", "hmean", "amean"]:
                    if df_t:
                        f[-1].add_trace(
                            go.Scatter(
                                x=io_stats.get(f"t_{type[0:3]}", "number_of_ranks"),
                                y=io_stats.get(f"t_{type[0:3]}", i),
                                name=f"$\\text{{{i}}}(T_{{{labels[count]}}})$",
                                hovertemplate="<b>Throughput</b><br>Ranks: %{x:i}<br>"
                                + i
                                + ": %{y:.2f}<br>",
                            )
                        )
                    if df_b:
                        f[-1].add_trace(
                            go.Scatter(
                                x=io_stats.get(f"b_{type[0:3]}", "number_of_ranks"),
                                y=io_stats.get(f"b_{type[0:3]}", i),
                                name=f"$\\text{{{i}}}(B_{{{labels[count]}}})$",
                                hovertemplate="<b>Bandwidth</b><br>Ranks: %{x:i}<br>"
                                + i
                                + ": %{y:.2f}<br>",
                            )
                        )
                f[-1].update_layout(
                    barmode="stack",
                    xaxis_title="Ranks",
                    yaxis_title="Transfer Rate (B/s)",
                    title=f"Overlap Statistics: {type.capitalize()}",
                )
                f[-1] = format_plot(f[-1], 17)

            f_1 = plot_error_bar(io_stats.t_avr, "T_{A}")
            f_2 = plot_error_bar(io_stats.t_sum, "T_{S}")
            f_3 = plot_error_bar(io_stats.t_ind, "T_{E}")
            if df_b:
                f_1 = plot_error_bar(io_stats.b_avr, "B_{A}", f_1)
                f_2 = plot_error_bar(io_stats.b_sum, "B_{S}", f_2)
                f_3 = plot_error_bar(io_stats.b_ind, "B_{E}", f_3)
            f.extend([f_1, f_2, f_3])

        self.set_figure(f, mode)

    # **********************************************************************
    # *                       3. plot_Time
    # **********************************************************************
    def plot_time(self):
        # ? Init
        CONSOLE.print("│   ├── [green]Creating plot I/O time [/]")
        self.nRun = len(pd.unique(self.data.df_time["file_index"]))
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
        symbols = [
            "square",
            "circle",
            "cross",
            "star-triangle-down",
            "hexagon",
            "x",
            "diamond",
            "star-triangle-up",
        ]
        markeredgecolor = "DarkSlateGrey"
        self.width_increase = 0
        self.height = 1.0 * self.height
        if self.nRun == 1:
            x = self.data.df_time["number_of_ranks"].astype(str)

        else:
            self.data.df_time = self.data.df_time.sort_values(
                by=["number_of_ranks", "file_index"]
            )
            x = [
                self.data.df_time["number_of_ranks"].astype(str),
                self.data.df_time["file_index"].astype(str),
            ]
            # x = [self.data.df_time.sort_values(by=["number_of_ranks","file_index"])["file_index"].astype(str), self.data.df_time.sort_values(by=["number_of_ranks","file_index"])["file_index"].astype(str)]

            self.width = self.width
            self.width_increase = (self.nRun - 1) * self.width / 5
            # self.width_increase = -self.width + self.nRun * len(self.data.df_time['number_of_ranks'])*self.width/24

        # ? Generate Figures
        self.f_t.append(add_fig_col(self.nRun))
        for i in range(0, self.nRun):
            index = self.data.df_time["file_index"].isin([i])
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_total"][index],
                    mode="lines+markers",
                    fill="tozeroy",
                    name="Total",
                    legendgroup="Total",
                    marker={
                        "symbol": symbols[0],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[0],
                    showlegend=legend_fix(self, i),
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_agg"][index],
                    mode="lines+markers",
                    fill="tozeroy",
                    name="App",
                    legendgroup="App",
                    marker={
                        "symbol": symbols[1],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[1],
                    showlegend=legend_fix(self, i),
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_overhead"][index],
                    mode="lines+markers",
                    fill="tozeroy",
                    name="Overhead",
                    legendgroup="Overhead",
                    marker={
                        "symbol": symbols[2],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[2],
                    showlegend=legend_fix(self, i),
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].update_xaxes(title_text="Ranks", row=1, col=i + 1)
        self.f_t[-1].update_layout(
            xaxis_title="Ranks",
            yaxis_title="Time (s)",
            width=self.width + self.width_increase,
            height=self.height,
            title="Time Distribution",
        )
        self.f_t[-1].update_layout(hovermode="x", legend_tracegroupgap=1)
        self.f_t[-1].update_xaxes(showspikes=True, spikethickness=1, spikecolor="black")
        self.f_t[-1].update_yaxes(showspikes=True, spikethickness=1, spikecolor="black")
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        modes = ["delta_t_total", "delta_t_agg", "delta_t_overhead"]
        names = ["Total", "App", "Overhead"]
        self.f_t.append(
            plot_time_error_bar(
                self.data.df_time,
                modes,
                names,
                colors,
                symbols,
                markeredgecolor,
            )
        )

        self.f_t.append(go.Figure())
        if self.nRun != 1:
            self.f_t[-1].update_layout(barmode="relative")
            self.f_t[-1].update_layout(
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                }
            )
        self.f_t[-1].add_trace(
            go.Bar(
                x=x,
                y=self.data.df_time["delta_t_agg"]
                / self.data.df_time["delta_t_total"]
                * 100,
                text=self.data.df_time["delta_t_agg"]
                / self.data.df_time["delta_t_total"]
                * 100,
                name="App",
                textposition="inside",
                textangle=0,
                texttemplate=self.barprecision,
                textfont={"color": "white"},
            )
        )
        self.f_t[-1].add_trace(
            go.Bar(
                x=x,
                y=self.data.df_time["delta_t_overhead"]
                / self.data.df_time["delta_t_total"]
                * 100,
                text=self.data.df_time["delta_t_overhead"]
                / self.data.df_time["delta_t_total"]
                * 100,
                name="Overhead",
                textposition="inside",
                textangle=0,
                texttemplate=self.barprecision,
                textfont={"color": "white"},
            )
        )
        self.f_t[-1].update_layout(
            xaxis_title_text="Ranks",
            yaxis_title_text="Percentage (%)",
            height=self.height,
            title="Total time",
            barmode="stack",
        )  # ,uniformtext_minsize=12, uniformtext_mode='show')
        self.f_t[-1].update_layout(barmode="relative")
        self.f_t[-1].update_yaxes(range=[0, 100])
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        self.f_t.append(go.Figure())
        delmiter = "<br>"
        if self.nRun != 1:
            delmiter = " "
            self.f_t[-1].update_layout(barmode="relative")
            self.f_t[-1].update_layout(
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                }
            )
        self.f_t[-1].add_trace(
            go.Bar(
                x=x,
                y=self.data.df_time["delta_t_com"]
                / self.data.df_time["delta_t_total"]
                * 100,
                text=self.data.df_time["delta_t_com"]
                / self.data.df_time["delta_t_total"]
                * 100,
                name="Compute",
                textposition="inside",
                textangle=0,
                marker_color=colors[0],
                texttemplate=self.barprecision,
                textfont={"color": "white"},
            )
        )
        self.f_t[-1].add_trace(
            go.Bar(
                x=x,
                y=self.data.df_time["delta_t_agg_io"]
                / self.data.df_time["delta_t_total"]
                * 100,
                text=self.data.df_time["delta_t_agg_io"]
                / self.data.df_time["delta_t_total"]
                * 100,
                name="Visible I/O",
                textposition="inside",
                textangle=0,
                marker_color=colors[1],
                texttemplate=self.barprecision,
                textfont={"color": "white"},
            )
        )
        if (
            "delta_t_overhead_post_runtime" in self.data.df_time.head()
            and "delta_t_overhead_peri_runtime" in self.data.df_time.head()
        ):
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_overhead_peri_runtime"]
                    / self.data.df_time["delta_t_total"]
                    * 100,
                    text=self.data.df_time["delta_t_overhead_peri_runtime"]
                    / self.data.df_time["delta_t_total"]
                    * 100,
                    name=f"Overhead{delmiter}peri-run",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[7],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_overhead_post_runtime"]
                    / self.data.df_time["delta_t_total"]
                    * 100,
                    text=self.data.df_time["delta_t_overhead_post_runtime"]
                    / self.data.df_time["delta_t_total"]
                    * 100,
                    name=f"Overhead{delmiter}post-run",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[8],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
        else:
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_overhead"]
                    / self.data.df_time["delta_t_total"]
                    * 100,
                    text=self.data.df_time["delta_t_overhead"]
                    / self.data.df_time["delta_t_total"]
                    * 100,
                    name="Overhead",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[9],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
        self.f_t[-1].update_layout(
            xaxis_title_text="Ranks",
            yaxis_title_text="Percentage (%)",
            width=self.width + self.width_increase,
            height=self.height,
            title="Total Time",
            barmode="stack",
        )  # ,uniformtext_minsize=12, uniformtext_mode='show')
        self.f_t[-1].update_yaxes(range=[0, 100])
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        self.f_t.append(add_fig_col(self.nRun))
        for i in range(0, self.nRun):
            index = self.data.df_time["file_index"].isin([i])
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_agg"][index],
                    mode="lines+markers",
                    fill="tozeroy",
                    name="app",
                    marker={
                        "symbol": symbols[1],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[0],
                    legendgroup="app",
                    showlegend=legend_fix(self, i),
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_com"][index],
                    mode="lines+markers",
                    fill="tozeroy",
                    name="Compute",
                    marker={
                        "symbol": symbols[3],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[1],
                    legendgroup="Compute",
                    showlegend=legend_fix(self, i),
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_agg_io"][index],
                    mode="lines+markers",
                    fill="tozeroy",
                    name="Visible I/O",
                    marker={
                        "symbol": symbols[4],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[2],
                    legendgroup="I/O",
                    showlegend=legend_fix(self, i),
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].update_xaxes(title_text="Ranks", row=1, col=i + 1)
        self.f_t[-1].update_layout(
            legend_tracegroupgap=1,
            xaxis_title="Ranks",
            yaxis_title="Time (s)",
            width=self.width + self.width_increase,
            height=self.height,
            title="Application Time",
        )
        self.f_t[-1].update_layout(hovermode="x", legend_tracegroupgap=1)
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        modes = ["delta_t_agg", "delta_t_com", "delta_t_agg_io"]
        names = ["App", "Compute", "I/O"]
        self.f_t.append(
            plot_time_error_bar(
                self.data.df_time,
                modes,
                names,
                colors,
                symbols,
                markeredgecolor,
            )
        )
        self.f_t[-1] = format_plot(self.f_t[-1], 17)
        self.f_t[-1].update_yaxes(type="log", row=1, col=1, showspikes=True)

        self.f_t.append(go.Figure())
        if self.nRun != 1:
            self.f_t[-1].update_layout(barmode="relative")
            self.f_t[-1].update_layout(
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                }
            )
        self.f_t[-1].add_trace(
            go.Bar(
                x=x,
                y=self.data.df_time["delta_t_com"]
                / self.data.df_time["delta_t_agg"]
                * 100,
                text=self.data.df_time["delta_t_com"]
                / self.data.df_time["delta_t_agg"]
                * 100,
                name="Compute",
                textposition="inside",
                textangle=0,
                texttemplate=self.barprecision,
                textfont={"color": "white"},
            )
        )
        self.f_t[-1].add_trace(
            go.Bar(
                x=x,
                y=self.data.df_time["delta_t_agg_io"]
                / self.data.df_time["delta_t_agg"]
                * 100,
                text=self.data.df_time["delta_t_agg_io"]
                / self.data.df_time["delta_t_agg"]
                * 100,
                name="I/O",
                textposition="inside",
                textangle=0,
                texttemplate=self.barprecision,
                textfont={"color": "white"},
            )
        )

        self.f_t[-1].update_layout(
            xaxis_title_text="Ranks",
            yaxis_title_text="Percentage (%)",
            width=self.width + self.width_increase,
            height=self.height,
            title="Application Time",
            barmode="stack",
        )  # ,uniformtext_minsize=12, uniformtext_mode='show')
        self.f_t[-1].update_yaxes(range=[0, 100])
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        # ? I/O to compute ration
        self.f_t.append(add_fig_col(self.nRun))
        for i in range(0, self.nRun):
            index = self.data.df_time["file_index"].isin([i])
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["delta_t_com"][index],
                    y=self.data.df_time["delta_t_agg_io"][index],
                    mode="lines+markers",
                    name="Compute to I/O time",
                    legendgroup="Compute",
                    marker={
                        "symbol": symbols[3],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[0],
                    showlegend=legend_fix(self, i),
                    text=self.data.df_time["number_of_ranks"][index],
                    hovertemplate="<b>Ranks: %{text}</b><br>"
                    + "Comp. time : %{x:.2f}<br>"
                    + "I/O time: %{y:.2}"
                    + "<extra></extra>",
                ),
                row=1,
                col=i + 1,
            )  # , secondary_y=False)
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["delta_t_com"][index],
                    y=self.data.df_time["delta_t_com"][index],
                    mode="lines+markers",
                    name="Ref",
                    legendgroup="Ref",
                    marker={
                        "symbol": symbols[4],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    line_dash="dash",
                    marker_color=colors[1],
                    showlegend=legend_fix(self, i),
                ),
                row=1,
                col=i + 1,
            )  # , secondary_y=False)
            # self.f_t[-1].add_trace(go.Scatter(x=self.data.df_time['delta_t_com'][index],     y=self.data.df_time['delta_t_agg_io'][index], mode = 'lines+markers', name = 'Compute to I/O time', marker_symbol=0, marker_color=colors[0], showlegend = False), row=1, col=i+1), secondary_y=True)
            self.f_t[-1].update_xaxes(title_text="Compute time (s)", row=1, col=i + 1)
        self.f_t[-1].update_layout(
            legend_tracegroupgap=1,
            xaxis_title="Compute time (s)",
            yaxis_title="I/O time (s)",
            width=self.width + self.width_increase,
            height=self.height,
            title="Compute to I/O time",
        )
        self.f_t[-1].update_yaxes(
            title_text="Ranks",
            secondary_y=True,
            ticktext=self.data.df_time["number_of_ranks"][index],
            tickvals=self.data.df_time["delta_t_agg_io"][index],
            showspikes=True,
        )
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        #! I/O instensity for the malleability manager
        self.f_t.append(add_fig_col(self.nRun))
        for i in range(0, self.nRun):
            index = self.data.df_time["file_index"].isin([i])
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_agg_io"][index]
                    / self.data.df_time["delta_t_com"][index],
                    mode="lines+markers",
                    name="I/O time to compute",
                    legendgroup="IO1",
                    marker={
                        "symbol": symbols[3],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[0],
                    showlegend=legend_fix(self, i),
                    text=self.data.df_time["number_of_ranks"][index],
                    hovertemplate="<b>Ranks: %{text}</b><br>"
                    + "Ratio: %{y:.2}"
                    + "<extra></extra>",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_agg_io"][index]
                    / self.data.df_time["delta_t_agg"][index],
                    mode="lines+markers",
                    name="I/O time to total",
                    legendgroup="IO2",
                    marker={
                        "symbol": symbols[5],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[2],
                    showlegend=legend_fix(self, i),
                    text=self.data.df_time["number_of_ranks"][index],
                    hovertemplate="<b>Ranks: %{text}</b><br>"
                    + "Ratio: %{y:.2}"
                    + "<extra></extra>",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=np.repeat(0.5, len(self.data.df_time["number_of_ranks"][index])),
                    mode="lines+markers",
                    name="Ref",
                    legendgroup="Ref",
                    marker={
                        "symbol": symbols[4],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    line_dash="dash",
                    marker_color=colors[1],
                    showlegend=legend_fix(self, i),
                ),
                row=1,
                col=i + 1,
            )  # , secondary_y=False)
            # self.f_t[-1].add_trace(go.Scatter(x=self.data.df_time['delta_t_com'][index],     y=self.data.df_time['delta_t_agg_io'][index], mode = 'lines+markers', name = 'Compute to I/O time', marker_symbol=0, marker_color=colors[0], showlegend = False), row=1, col=i+1), secondary_y=True)
            self.f_t[-1].update_xaxes(title_text="Ranks", row=1, col=i + 1)
        self.f_t[-1].update_layout(
            legend_tracegroupgap=1,
            xaxis_title="Ranks",
            yaxis_title="I/O intensity",
            width=self.width + self.width_increase,
            height=self.height,
            title="I/O intensity",
        )
        self.f_t[-1] = format_plot(self.f_t[-1], 17)
        self.f_t[-1].update_yaxes(
            title_text="Ranks",
            secondary_y=True,
            ticktext=self.data.df_time["number_of_ranks"][index],
            tickvals=self.data.df_time["delta_t_agg_io"][index],
            showspikes=True,
        )
        self.f_t[-1].update_xaxes(showspikes=True)
        self.f_t[-1].update_yaxes(range=[0, 1])

        # ? I/O instensity error bar
        modes = [
            "delta_t_agg_io / delta_t_com",
            "delta_t_agg_io / delta_t_agg",
        ]
        names = ["I/O to compute", "I/O to total"]
        self.f_t.append(
            plot_time_error_bar(
                self.data.df_time,
                modes,
                names,
                colors,
                symbols,
                markeredgecolor,
            )
        )

        self.f_t.append(go.Figure())
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3
        y = (
            self.data.df_time["delta_t_com"]
            - (self.data.df_time["delta_t_ara"] - self.data.df_time["delta_t_ar_lost"])
            - (self.data.df_time["delta_t_awa"] - self.data.df_time["delta_t_aw_lost"])
        )
        if self.nRun != 1:
            self.f_t[-1].update_layout(barmode="relative")
            self.f_t[-1].update_layout(
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                }
            )
        if not y.empty and not all(y == 0):
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=y / self.data.df_time["delta_t_agg"] * 100,
                    text=y / self.data.df_time["delta_t_agg"] * 100,
                    name="Compute (I/O free)",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[0],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
        if not self.data.df_time["delta_t_awa"].empty and not all(
            self.data.df_time["delta_t_awa"] == 0
        ):
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=(
                        self.data.df_time["delta_t_awa"]
                        - self.data.df_time["delta_t_aw_lost"]
                    )
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    text=(
                        self.data.df_time["delta_t_awa"]
                        - self.data.df_time["delta_t_aw_lost"]
                    )
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    name="Async write exploit",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[1],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
        if not self.data.df_time["delta_t_ara"].empty and not all(
            self.data.df_time["delta_t_ara"] == 0
        ):
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=(
                        self.data.df_time["delta_t_ara"]
                        - self.data.df_time["delta_t_ar_lost"]
                    )
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    text=(
                        self.data.df_time["delta_t_ara"]
                        - self.data.df_time["delta_t_ar_lost"]
                    )
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    name="Async read exploit",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[2],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
        if not self.data.df_time["delta_t_awa"].empty and not all(
            self.data.df_time["delta_t_awa"] == 0
        ):
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_aw_lost"]
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    text=self.data.df_time["delta_t_aw_lost"]
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    name="Async write lost",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[3],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
        if not self.data.df_time["delta_t_ara"].empty and not all(
            self.data.df_time["delta_t_ara"] == 0
        ):
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_ar_lost"]
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    text=self.data.df_time["delta_t_ar_lost"]
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    name="Async read lost",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[4],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
        if not self.data.df_time["delta_t_sw"].empty and not all(
            self.data.df_time["delta_t_sw"] == 0
        ):
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_sw"]
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    text=self.data.df_time["delta_t_sw"]
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    name="Sync write",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[5],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
        if not self.data.df_time["delta_t_sr"].empty and not all(
            self.data.df_time["delta_t_sr"] == 0
        ):
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_sr"]
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    text=self.data.df_time["delta_t_sr"]
                    / self.data.df_time["delta_t_agg"]
                    * 100,
                    name="Sync read",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[6],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
        self.f_t[-1].update_layout(
            xaxis_title="Ranks",
            yaxis_title="Percentage (%)",
            width=self.width + self.width_increase,
            height=1.3 * self.height,
            title="Detailed Application Time",
            barmode="stack",
            margin={"t": 200},
        )
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        self.f_t.append(add_fig_col(self.nRun))
        for i in range(0, self.nRun):
            index = self.data.df_time["file_index"].isin([i])
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_sr"][index],
                    mode="lines+markers",
                    name="Sync read",
                    marker={
                        "symbol": symbols[0],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[0],
                    showlegend=legend_fix(self, i),
                    legendgroup="delta_t_sr",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_ara"][index],
                    mode="lines+markers",
                    name="Async read act.",
                    marker={
                        "symbol": symbols[1],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[1],
                    showlegend=legend_fix(self, i),
                    legendgroup="delta_t_ara",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_arr"][index],
                    mode="lines+markers",
                    name="Async read req.",
                    marker={
                        "symbol": symbols[2],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[2],
                    showlegend=legend_fix(self, i),
                    legendgroup="delta_t_arr",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_ar_lost"][index],
                    mode="lines+markers",
                    name="Async read lost",
                    marker={
                        "symbol": symbols[3],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[3],
                    showlegend=legend_fix(self, i),
                    legendgroup="delta_t_ar_lost",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_sw"][index],
                    mode="lines+markers",
                    name="Sync write",
                    marker={
                        "symbol": symbols[4],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[4],
                    showlegend=legend_fix(self, i),
                    legendgroup="delta_t_sw",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_awa"][index],
                    mode="lines+markers",
                    name="Async write act.",
                    marker={
                        "symbol": symbols[5],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[5],
                    showlegend=legend_fix(self, i),
                    legendgroup="delta_t_awa",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_awr"][index],
                    mode="lines+markers",
                    name="Async write req.",
                    marker={
                        "symbol": symbols[6],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[6],
                    showlegend=legend_fix(self, i),
                    legendgroup="delta_t_awr",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_aw_lost"][index],
                    mode="lines+markers",
                    name="Async write lost",
                    marker={
                        "symbol": symbols[7],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[7],
                    showlegend=legend_fix(self, i),
                    legendgroup="delta_t_aw_lost",
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].update_xaxes(title_text="Ranks", row=1, col=i + 1)
        self.f_t[-1].update_layout(
            legend_tracegroupgap=1,
            xaxis_title="Ranks",
            yaxis_title="Time (s)",
            width=self.width + self.width_increase,
            height=self.height,
            title="I/O time",
        )
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        self.f_t.append(add_fig_row(self.nRun, onefig=True))
        for i in range(0, self.nRun):
            index = self.data.df_time["file_index"].isin([i])
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_sr"][index],
                    mode="lines+markers",
                    legendgroup="delta_t_sr",
                    legendgrouptitle_text="Sync read",
                    name=f"run {i}",
                    marker={
                        "symbol": symbols[0],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                ),
                row=1,
                col=1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_ara"][index],
                    mode="lines+markers",
                    legendgroup="delta_t_ara",
                    legendgrouptitle_text="Async read act.",
                    name=f"run {i}",
                    marker={
                        "symbol": symbols[1],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                ),
                row=1,
                col=1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_arr"][index],
                    mode="lines+markers",
                    legendgroup="delta_t_arr",
                    legendgrouptitle_text="Async read req.",
                    name=f"run {i}",
                    marker={
                        "symbol": symbols[2],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                ),
                row=1,
                col=1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_ar_lost"][index],
                    mode="lines+markers",
                    legendgroup="delta_t_ar_lost",
                    legendgrouptitle_text="Async read lost.",
                    name=f"run {i}",
                    marker={
                        "symbol": symbols[3],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                ),
                row=1,
                col=1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_sw"][index],
                    mode="lines+markers",
                    legendgroup="delta_t_sw",
                    legendgrouptitle_text="Sync write ",
                    name=f"run {i}",
                    marker={
                        "symbol": symbols[4],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                ),
                row=1,
                col=1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_awa"][index],
                    mode="lines+markers",
                    legendgroup="delta_t_awa",
                    legendgrouptitle_text="Async write act.",
                    name=f"run {i}",
                    marker={
                        "symbol": symbols[5],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                ),
                row=1,
                col=1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_awr"][index],
                    mode="lines+markers",
                    legendgroup="delta_t_awr",
                    legendgrouptitle_text="Async write req.",
                    name=f"run {i}",
                    marker={
                        "symbol": symbols[6],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                ),
                row=1,
                col=1,
            )
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_aw_lost"][index],
                    mode="lines+markers",
                    legendgroup="delta_t_aw_lost",
                    legendgrouptitle_text="Async lost",
                    name=f"run {i}",
                    marker={
                        "symbol": symbols[7],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                ),
                row=1,
                col=1,
            )
        self.f_t[-1].update_layout(
            legend_tracegroupgap=1,
            xaxis_title="Ranks",
            yaxis_title="Time (s)",
            width=self.width + self.width_increase,
            height=self.height,
            title="I/O time",
        )
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        #! Error bar
        # modes = ["delta_t_sr", "delta_t_sw",  "delta_t_ar_lost", "delta_t_aw_lost",  "delta_t_ara",     "delta_t_awa",     "delta_t_arr",      "delta_t_awr"]
        # names = ["Sync read",  "Sync write ", "Async read lost", "Async write lost", "Async read act.", "Async write act.", "Async read req.",  "Async write req.", ]
        modes = [
            "delta_t_sr",
            "delta_t_ara",
            "delta_t_arr",
            "delta_t_ar_lost",
            "delta_t_sw",
            "delta_t_awa",
            "delta_t_awr",
            "delta_t_aw_lost",
        ]
        names = [
            "Sync read",
            "Async read act.",
            "Async read req.",
            "Async read lost",
            "Sync write ",
            "Async write act.",
            "Async write req.",
            "Async write lost",
        ]
        self.f_t.append(
            plot_time_error_bar(
                self.data.df_time,
                modes,
                names,
                colors,
                symbols,
                markeredgecolor,
            )
        )

        self.f_t.append(add_fig_col(self.nRun))
        for i in range(0, self.nRun):
            index = self.data.df_time["file_index"].isin([i])
            self.f_t[-1].add_trace(
                go.Scatter(
                    x=self.data.df_time["number_of_ranks"][index],
                    y=self.data.df_time["delta_t_overhead"][index],
                    mode="lines+markers",
                    name="Overhead",
                    legendgroup="delta_t_overhead",
                    marker={
                        "symbol": symbols[0],
                        "line": {"width": 1, "color": markeredgecolor},
                    },
                    marker_color=colors[0],
                    showlegend=legend_fix(self, i),
                ),
                row=1,
                col=i + 1,
            )
            self.f_t[-1].update_xaxes(title_text="Ranks", row=1, col=i + 1)
            if "delta_t_overhead_post_runtime" in self.data.df_time.head():
                self.f_t[-1].add_trace(
                    go.Scatter(
                        x=self.data.df_time["number_of_ranks"][index],
                        y=self.data.df_time["delta_t_overhead_post_runtime"][index],
                        mode="lines+markers",
                        name="Post runtime",
                        legendgroup="delta_t_overhead_post_runtime",
                        marker={
                            "symbol": symbols[1],
                            "line": {"width": 1, "color": markeredgecolor},
                        },
                        marker_color=colors[1],
                        showlegend=legend_fix(self, i),
                    ),
                    row=1,
                    col=i + 1,
                )
            if "delta_t_overhead_peri_runtime" in self.data.df_time.head():
                self.f_t[-1].add_trace(
                    go.Scatter(
                        x=self.data.df_time["number_of_ranks"][index],
                        y=self.data.df_time["delta_t_overhead_peri_runtime"][index],
                        mode="lines+markers",
                        name="During runtime",
                        legendgroup="delta_t_overhead_peri_runtime",
                        marker={
                            "symbol": symbols[2],
                            "line": {"width": 1, "color": markeredgecolor},
                        },
                        marker_color=colors[2],
                        showlegend=legend_fix(self, i),
                    ),
                    row=1,
                    col=i + 1,
                )
        self.f_t[-1].update_layout(
            legend_tracegroupgap=1,
            xaxis_title="Ranks",
            yaxis_title="Time (s)",
            width=self.width + self.width_increase,
            height=self.height,
            title="Overhead",
        )
        self.f_t[-1] = format_plot(self.f_t[-1], 17)

        if (
            "delta_t_overhead_post_runtime" in self.data.df_time.head()
            and "delta_t_overhead_peri_runtime" in self.data.df_time.head()
        ):
            self.f_t.append(go.Figure())
            if self.nRun != 1:
                self.f_t[-1].update_layout(barmode="relative")
                self.f_t[-1].update_layout(
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.02,
                        "xanchor": "right",
                        "x": 1,
                    }
                )
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_overhead_peri_runtime"]
                    / self.data.df_time["delta_t_overhead"]
                    * 100,
                    text=self.data.df_time["delta_t_overhead_peri_runtime"]
                    / self.data.df_time["delta_t_overhead"]
                    * 100,
                    name="During runtime",
                    textposition="inside",
                    textangle=0,
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_overhead_post_runtime"]
                    / self.data.df_time["delta_t_overhead"]
                    * 100,
                    text=self.data.df_time["delta_t_overhead_post_runtime"]
                    / self.data.df_time["delta_t_overhead"]
                    * 100,
                    name="Post runtime",
                    textposition="inside",
                    textangle=0,
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
            self.f_t[-1].update_layout(
                xaxis_title_text="Ranks",
                yaxis_title_text="Percentage (%)",
                width=self.width + self.width_increase,
                height=self.height,
                title="Overhead",
                barmode="stack",
            )  # ,uniformtext_minsize=12, uniformtext_mode='show')
            self.f_t[-1].update_yaxes(range=[0, 100])
            self.f_t[-1] = format_plot(self.f_t[-1], 17)

        if "delta_t_rank0" in self.data.df_time.head():
            self.f_t.append(go.Figure())
            delmiter = "<br>"
            if self.nRun != 1:
                delmiter = " "
                self.f_t[-1].update_layout(barmode="relative")
                self.f_t[-1].update_layout(
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.02,
                        "xanchor": "right",
                        "x": 1,
                    }
                )
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_rank0_app"]
                    / self.data.df_time["delta_t_rank0"]
                    * 100,
                    text=self.data.df_time["delta_t_rank0_app"]
                    / self.data.df_time["delta_t_rank0"]
                    * 100,
                    name="App",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[1],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_rank0_overhead_peri_runtime"]
                    / self.data.df_time["delta_t_rank0"]
                    * 100,
                    text=self.data.df_time["delta_t_rank0_overhead_peri_runtime"]
                    / self.data.df_time["delta_t_rank0"]
                    * 100,
                    name=f"Overhead{delmiter}peri-run",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[7],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
            self.f_t[-1].add_trace(
                go.Bar(
                    x=x,
                    y=self.data.df_time["delta_t_rank0_overhead_post_runtime"]
                    / self.data.df_time["delta_t_rank0"]
                    * 100,
                    text=self.data.df_time["delta_t_rank0_overhead_post_runtime"]
                    / self.data.df_time["delta_t_rank0"]
                    * 100,
                    name=f"Overhead{delmiter}post-run",
                    textposition="inside",
                    textangle=0,
                    marker_color=colors[8],
                    texttemplate=self.barprecision,
                    textfont={"color": "white"},
                )
            )
            self.f_t[-1].update_layout(
                xaxis_title_text="Ranks",
                yaxis_title_text="Percentage (%)",
                width=self.width + self.width_increase,
                height=self.height,
                title="Overhead Rank 0",
                barmode="stack",
            )
            self.f_t[-1].update_yaxes(range=[0, 100])
            self.f_t[-1] = format_plot(self.f_t[-1], 17)

            self.f_t.append(add_fig_col(self.nRun))
            for i in range(0, self.nRun):
                index = self.data.df_time["file_index"].isin([i])
                self.f_t[-1].add_trace(
                    go.Scatter(
                        x=self.data.df_time["number_of_ranks"][index],
                        y=self.data.df_time["delta_t_rank0"][index],
                        mode="lines+markers",
                        fill="tozeroy",
                        name="Total",
                        legendgroup="Total",
                        marker={
                            "symbol": symbols[0],
                            "line": {"width": 1, "color": markeredgecolor},
                        },
                        marker_color=colors[0],
                        showlegend=legend_fix(self, i),
                    ),
                    row=1,
                    col=i + 1,
                )
                self.f_t[-1].add_trace(
                    go.Scatter(
                        x=self.data.df_time["number_of_ranks"][index],
                        y=self.data.df_time["delta_t_rank0_app"][index],
                        mode="lines+markers",
                        fill="tozeroy",
                        name="App",
                        legendgroup="App",
                        marker={
                            "symbol": symbols[1],
                            "line": {"width": 1, "color": markeredgecolor},
                        },
                        marker_color=colors[1],
                        showlegend=legend_fix(self, i),
                    ),
                    row=1,
                    col=i + 1,
                )
                self.f_t[-1].add_trace(
                    go.Scatter(
                        x=self.data.df_time["number_of_ranks"][index],
                        y=self.data.df_time["delta_t_rank0_overhead_post_runtime"][index]
                        + self.data.df_time["delta_t_rank0_overhead_peri_runtime"][index],
                        mode="lines+markers",
                        fill="tozeroy",
                        name="Overhead",
                        legendgroup="Overhead",
                        marker={
                            "symbol": symbols[2],
                            "line": {"width": 1, "color": markeredgecolor},
                        },
                        marker_color=colors[2],
                        showlegend=legend_fix(self, i),
                    ),
                    row=1,
                    col=i + 1,
                )
                self.f_t[-1].update_xaxes(title_text="Ranks", row=1, col=i + 1)
            self.f_t[-1].update_layout(
                xaxis_title="Ranks",
                yaxis_title="Time (s)",
                width=self.width + self.width_increase,
                height=self.height,
                title="Time Distribution Rank 0",
            )
            self.f_t[-1].update_layout(hovermode="x", legend_tracegroupgap=1)
            self.f_t[-1].update_xaxes(
                showspikes=True, spikethickness=1, spikecolor="black"
            )
            self.f_t[-1].update_yaxes(
                showspikes=True, spikethickness=1, spikecolor="black"
            )
            self.f_t[-1] = format_plot(self.f_t[-1], 17)

            modes = [
                "delta_t_rank0",
                "delta_t_rank0_app",
                "delta_t_rank0_overhead_post_runtime + delta_t_rank0_overhead_peri_runtime",
            ]
            names = ["Total", "App", "Overhead"]
            self.f_t.append(
                plot_time_error_bar(
                    self.data.df_time,
                    modes,
                    names,
                    colors,
                    symbols,
                    markeredgecolor,
                )
            )

        # print('%i figures'%len(self.f))

    # **********************************************************************
    # *                       6. Set and Get Functions
    # **********************************************************************

    def get_figure(self, mode):
        if "write_async" in mode:
            return self.f_aw
        if "read_async" in mode:
            return self.f_ar
        if "write_sync" in mode:
            return self.f_sw
        if "read_sync" in mode:
            return self.f_sr
        if "time" in mode:
            return self.f_t

    def set_figure(self, f, mode):
        # assign
        if "async" in mode:
            if "write" in mode:
                self.f_aw = f
            else:  # read
                self.f_ar = f
        else:  # sync
            if "write" in mode:
                self.f_sw = f
            else:  # read
                self.f_sr = f
