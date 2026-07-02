"""
Author: Amine Aherbil
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Feb 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import json
import socket
import time

from rich.console import Console
from rich.text import Text

_socket_logger = None
_gui_enabled: bool = False


class SocketLogger:
    """TCP socket client for sending prediction and change point data to the GUI dashboard.

    Establishes a connection to the dashboard server and sends JSON-formatted messages
    containing prediction results, change point detections, and other log data for
    real-time visualization.

    Attributes:
        host: The hostname of the GUI server (default: "localhost").
        port: The port number of the GUI server (default: 9999).
        socket: The TCP socket connection.
        connected: Boolean indicating if currently connected to the server.
    """

    def __init__(self, host: str = "localhost", port: int = 9999):
        """Initialize the socket logger and attempt connection to the GUI server.

        Args:
            host: The hostname of the GUI server.
            port: The port number of the GUI server.
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected: bool = False
        self._connect()

    def _connect(self):
        """Attempt to establish a TCP connection to the GUI dashboard server.

        Creates a socket with a 1-second timeout and attempts to connect to the
        SocketListener running in the GUI dashboard process. If connection fails
        (e.g., GUI not running), sets connected=False and continues without GUI
        logging - predictions still work, just without real-time visualization.

        The connection is optional: the predictor works fine without the GUI.
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(1.0)  # 1 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"[INFO] Connected to GUI server at {self.host}:{self.port}")
        except (TimeoutError, OSError, ConnectionRefusedError) as e:
            self.connected = False
            if self.socket:
                self.socket.close()
                self.socket = None
            print(
                f"[WARNING] Failed to connect to GUI server at {self.host}:{self.port}: {e}"
            )
            print("[WARNING] GUI logging disabled - messages will only appear in console")

    def send_log(self, log_type: str, message: str, data=None):
        """Send a log message to the GUI dashboard for visualization.

        Constructs a JSON message with timestamp, type, message, and optional data,
        then sends it over the TCP socket. If sending fails, marks the connection
        as closed and stops further send attempts.

        Args:
            log_type: Category of the message. Common types:
                - "prediction": New FTIO prediction result
                - "change_point": Change point detection event
                - "info": General information message
            message: Human-readable description of the event.
            data: Dictionary containing structured data for the GUI to display.
                For predictions, includes frequency, confidence, time window, etc.
                For change points, includes old/new frequency and detection time.
        """
        if not self.connected:
            return

        try:
            log_data = {
                "timestamp": time.time(),
                "type": log_type,
                "message": message,
                "data": data or {},
            }

            json_data = json.dumps(log_data) + "\n"
            self.socket.send(json_data.encode("utf-8"))

        except (OSError, BrokenPipeError, ConnectionResetError) as e:
            print(f"[WARNING]  Failed to send to GUI: {e}")
            self.connected = False
            if self.socket:
                self.socket.close()
                self.socket = None

    def close(self):
        """Close the socket connection to the GUI server.

        Safe to call multiple times. After closing, no more messages can be sent.
        """
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False


def init_socket_logger(gui_enabled: bool = False):
    """Initialize the socket logger based on --gui flag"""
    global _socket_logger, _gui_enabled
    _gui_enabled = gui_enabled
    if gui_enabled:
        _socket_logger = SocketLogger()
    else:
        _socket_logger = None


def get_socket_logger():
    """Retrieve the global socket logger if GUI logging is enabled.

    Lazily creates the logger if needed.
    """
    global _socket_logger, _gui_enabled
    if not _gui_enabled:
        return None
    if _socket_logger is None:
        _socket_logger = SocketLogger()
    return _socket_logger


def log_to_gui_and_console(
    gui_enabled: bool,
    console: Console,
    message: str,
    log_type: str = "info",
    data: dict | None = None,
):
    """
    Logs a message to both the Rich console (with formatting) and a GUI/socket logger (plain text).

    Args:
        gui_enabled (bool): Whether to log to the GUI socket logger.
        console (Console): Rich console instance to print formatted text.
        message (str): Message containing optional Rich markup.
        log_type (str): Logging type/severity (default: 'info').
        data (dict | None): Optional additional data to send with the log.
    """
    console.print(message)

    if gui_enabled:
        logger = get_socket_logger()
        if logger is not None:
            logger.send_log(log_type, Text.from_markup(message).plain, data)
