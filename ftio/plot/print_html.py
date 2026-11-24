"""
Generates HTML page out of plotly images
Used by plot_core.py

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Feb 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import os
import platform as plat
from sys import platform
from threading import Lock

import plotly.offline
from plotly.graph_objects import Figure

from ftio.freq.helper import MyConsole

CONSOLE = MyConsole()
CONSOLE.set(True)


class PrintHtml:
    """Class to Create HTML files with sub-pages"""

    def __init__(
        self, path, names, args=None, filename="main.html", outdir="io_results"
    ) -> None:
        """generates HTML report
        Args:
            filename (str): main html file name
            args (list): contains render (static or dynamic)
                        and show (True or False) flag
            path (str): output location
            outdir (str): output dir name
        """
        self.path = path
        self.names = names
        self.render = args.render if args else "dynamic"
        self.show = not args.no_disp if args else True
        self.filename = filename
        self.outdir = outdir
        self.lock = Lock()

    #! ----------------------- Plot to HTML ------------------------------
    # **********************************************************************
    # *                       1. generate_html
    # **********************************************************************
    def generate_html_start(self) -> None:
        """Creates main.html which is an entry point for the HTML File. Moreover,
        a folder is created that hosts all sub-pages
        """
        # only execute if dynmaic plots are needed
        if "stat" not in self.render:  # only generate a single image

            # ? 1. set current working directory
            pwd = os.getcwd()
            if len(self.path) <= 1 and not any(
                ext in self.path[0] for ext in ["json", "darshan", "msgpack", "txt"]
            ):
                pwd = os.path.join(pwd, os.path.relpath(self.path[0], pwd))

            self.path = os.path.join(pwd, self.outdir)
            CONSOLE.print(
                f"[green]\nGenerating HTML report [/]\n├── Directory is: {self.path}"
            )

            # ? 2. Create directory if needed
            if not os.path.exists(self.path):
                os.mkdir(self.path)
                CONSOLE.print(f"├── Directory {self.outdir} created in {pwd}")
            else:
                CONSOLE.print(f"└── Directory {self.outdir}  exists  in {pwd}")

            # ? 3. create main HTML file
            # html_files = ["write_async.html","read_async.html","write_sync.html","read_sync.html","time.html"]
            CONSOLE.print(
                f"├── [cyan]To see intermediate result call: \n[/]│[cyan]   open {self.path}/main.html [/]\n│"
            )
            self.filename = os.path.join(self.path, self.filename)
            CONSOLE.print("├── Generated main.html ")
            with open(self.filename, "w") as file:
                file.write("<html><head></head><body>\n")
                file.write(
                    '<center><h1 style="color:black;"> Results </h1> </center><hr style="height:2px;border-width:0;color:gray;background-color:gray">\n'
                )

            CONSOLE.print("└── [green]Creating plots (sub-pages):[/]")

    def generate_html_core(self, html_file: str, f: list[Figure]) -> None:
        """Generates sub HTML pages by writing into self.path/self.outdir

        Args:
            html_file (str): name of HTML file
            f (list[go.Figure]): list of plotly figures (go or px)
        """
        CONSOLE.print(f"    ├── [green]Started generating {html_file}    [/]")

        # Remove invalid characters
        invalid = ["/", " "]
        if any(x in html_file for x in invalid):
            for i in invalid:
                if i in html_file:
                    CONSOLE.print(
                        f"[yellow]Warning: removing character {i} in {html_file}[/]"
                    )
                    html_file = html_file.replace(i, "")

        # convert all figures to  html
        figures_to_html(f, os.path.join(self.path, html_file), self.names)

        # Mark entry in main.html
        with open(self.filename, "a") as file:
            self.lock.acquire()
            tmp = html_file.replace("_", " ").replace(".html", " ").capitalize()
            file.write(
                f'<h3 ><a href="file://{self.path}/{html_file}" style="color:black;">{tmp}: {len(f)} figures </a>\n'
            )
            self.lock.release()

        CONSOLE.print(f"    ├──  [green]Finished generating {html_file}    [/]")

    def generate_html_end(self) -> None:
        """closes the File and displays the location of the HTML files"""

        # close the file
        with open(self.filename, "a") as file:
            file.write(
                '<hr style="height:2px;border-width:0;color:gray;background-color:gray">\n'
            )
            if self.names:
                # self.names = list(set(self.names))
                file.write("<br><br> Folders map to:<ul>\n")
                for index, name in enumerate(self.names):
                    file.write(f"<li> Run {index}: {name} </li>\n")
                file.write("</ul> \n")
            file.write("</body></html> \n")
        CONSOLE.print("    └──  [green]done    [/]")

        CONSOLE.print(
            f"[cyan]\nTo see the result call \nopen {self.path}/main.html \n[/]"
        )
        if self.show:
            if platform == "linux" or platform == "linux2":
                if "WSL" in plat.uname().release:
                    os.system(f"powershell.exe start ./{self.outdir}/main.html ")
                else:
                    os.system(f"open {self.path}/main.html \n")

            if "windows" in platform:
                try:
                    os.system(f"powershell.exe start {self.path}/main.html &\n")
                except Exception:
                    os.system(f"powershell.exe start./{self.path}/main.html")


# **********************************************************************
# *                       1. figures_to_html
# **********************************************************************
def figures_to_html(
    figs: list, filename: str = "write_async.html", names: list = None
) -> None:
    """Convert list of figures to a HTML file

    Args:
        figs (list): list of figures from plotly (go or px)
        filename (str, optional): Name of the HTML file. Defaults to "write_async.html".
        names (list, optional): folders to map the runs if needed. Defaults to [].
    """
    # conf = {  "toImageButtonOptions": {     "format": "svg", "scale":1  }}
    if names is None:
        names = []
    conf = {"toImageButtonOptions": {"format": "png", "scale": 2}}
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8"/>
    <script>{plotly}</script>
    </head>
    <body>
    {names}
    {plots}
    </body>
    </html>
    """
    html_parts = ""
    current_file = os.path.basename(filename)
    for fig in figs:
        if figs.index(fig) + 1 == len(figs):
            CONSOLE.print(
                f"    │    └──  {current_file} total figures {len(figs)}      ",
                end="\n",
            )
        else:
            CONSOLE.print(
                f"    │    ├──   {current_file} figure  ({figs.index(fig)+1}/{len(figs)})\r",
                end="\r",
            )
        html_parts = (
            html_parts
            + fig.to_html(config=conf, include_plotlyjs=False, include_mathjax="cdn")
            + "\n"
        )

    run_names = ""
    if names:
        run_names = "<br><br> Folders map to:<ul>\n"
        for index, name in enumerate(names):
            run_names = run_names + (f"<li> Run {index}: {name} </li>\n")
        run_names = run_names + "</ul> \n"

    template = template.format(
        plotly=plotly.offline.get_plotlyjs(), plots=html_parts, names=run_names
    )
    with open(filename, "w") as file:
        file.write(template)
