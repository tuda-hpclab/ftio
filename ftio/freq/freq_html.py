"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import os
from sys import platform
from threading import Thread

import plotly.io
from plotly.offline import get_plotlyjs
from rich.console import Console


def create_html(figs: list, render: str, configuration: dict, name: str = "freq") -> None:
    console = Console()
    if platform == "linux" or platform == "linux2":
        os.system(f"rm -f ./{name}.html || true")
        os.system("rm -rf io_anomality_freq_images || true ")

    if render == "dynamic":
        template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8"/>
        <script>{plotly}</script>
        </head>
        <body>
        {plots}
        </body>
        </html>
        """
        s = ""
        for fig in figs:
            # s = s + fig.to_html(include_plotlyjs=False) + "\n"
            s = s + fig.to_html(config=configuration, include_plotlyjs=False) + "\n"

        template = template.format(plotly=get_plotlyjs(), plots=s)
        with open(f"{name}.html", "w") as file:
            file.write(template.strip())

        os.system(f'open "./{name}.html"')
        console.print(f"[cyan]{name}.html created[/cyan]")
    else:
        os.mkdir("io_predicition_anomality_images")
        # extension='jpg'
        extension = "svg"
        # extension = "png"
        console.print(f"-> Generating {extension.upper()} figures")
        threads = []
        length = len(figs)
        plotly.io.json.config.default_engine = "orjson"
        for fig in figs:
            index = figs.index(fig)
            threads.append(
                Thread(
                    target=create_static_figure,
                    args=(fig, index, length, extension),
                )
            )

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        console.print(f"[cyan]{name}.html created[/cyan]")
        os.system("open . &")


def create_static_figure(fig, index: int, length: int, extension: str) -> None:
    """Creates static HTML figures

    Args:
        fig (plotly figure): _description_
        index (int): index of the figure
        length (int): total figures
        extension (str): desired extension
    """
    console = Console()
    scale = 1
    console.print(f"working on [cyan]figure ({index + 1}/{length}) [/]")
    if fig.layout.title.text:
        file_name = f"io_predicition_freq_images/{fig.layout.title.text.replace(' ', '_').replace('(', '_').replace(')', '_')}.{extension}"
    else:
        file_name = f"io_predicition_freq_images/{index}.{extension}"

    if "svg" not in extension and "pdf" not in extension:
        scale = 5

    fig.write_image(file_name, scale=scale)
    if index == 0:
        os.system(f"open {file_name} || true ")

    console.print(f"[cyan]figure ({index + 1}/{length}) [/]created")
