"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import argparse
import shlex
import threading
import time

from flask import Flask, jsonify, request
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from ftio.cli.ftio_core import main

app = Flask(__name__)
console = Console()


@app.route("/ftio", methods=["POST"])
def ftio():
    # Assume raw args are sent as plain text in the body
    raw_args = request.get_data(as_text=True)  # gets raw POST body as string
    cmd_input = ["ftio"] + shlex.split(raw_args)
    prediction_list, _ = main(cmd_input)
    prediction = prediction_list[0]
    out = prediction.to_json()
    print(out)
    if not isinstance(out, dict):
        return "Unsupported prediction type", 500
    else:
        return jsonify(out), 200


def run_flask(host, port):
    app.run(port=port, host=host)


def main_cli():
    parser = argparse.ArgumentParser(description="FTIO HTTP Server")
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host address to bind to"
    )
    args = parser.parse_args()

    console.print(f"[green]Starting FTIO server on {args.host}:{args.port}...[/]")

    # Start Flask in background thread
    flask_thread = threading.Thread(
        target=run_flask, args=(args.host, args.port), daemon=True
    )
    flask_thread.start()

    console.print(
        "[green]FTIO is running. Send commands with:[/]\n"
        f'[green]curl -X POST http://{args.host}:{args.port}/ftio --data [/]"--file data.txt --mode auto"\n'
    )

    spinner = Spinner("dots", text="[cyan]FTIO is ready[/]")
    with Live(spinner, console=console, refresh_per_second=10):
        try:
            while flask_thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            console.print("\n[green]Exiting...[/]")

    console.print("[green]FTIO server stopped.[/]")


if __name__ == "__main__":
    main_cli()
