"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Okt 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import importlib.util
import logging
import os
import sys

COLORLOG_AVAILABLE = importlib.util.find_spec("colorlog") is not None
if COLORLOG_AVAILABLE:
    from colorlog import ColoredFormatter

# ANSI codes fallback if colorlog is unavailable
ANSI_COLORS = {
    "ftio": "\033[36m",  # cyan
    "proxy": "\033[35m",  # purple
    "daemon": "\033[95m",  # bright magenta
    "fuse": "\033[34m",  # blue
    "dlio": "\033[33m",  # yellow
    "lammp": "\033[33m",  # yellow
    "jit": "\033[32m",  # green
    "trigger": "\033[97m",  # bright white
    "error": "\033[31m",  # red
    "reset": "\033[0m",
}

# colorlog color names matching each prefix (used for INFO message color)
_COLORLOG_MSG_COLORS = {
    "ftio": "cyan",
    "proxy": "purple",
    "daemon": "bold_purple",
    "fuse": "blue",
    "dlio": "yellow",
    "lammp": "yellow",
    "jit": "green",
    "trigger": "bold_white",
    "error": "red",
}


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)  # emit the log
        self.flush()  # flush immediately


class Logger:
    def __init__(self, level=None, prefix="", stream=None):
        self.prefix = (prefix or "GENERAL").lower()  # normalize prefix
        self._stream = stream or sys.stderr
        self.logger = logging.getLogger(prefix)
        self.logger.setLevel(self._resolve_level(level))

        # Only add a handler if the logger has no handlers yet
        if not self.logger.handlers:
            handler = FlushStreamHandler(self._stream)
            handler.setFormatter(self._resolve_formatter())
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _resolve_level(self, cli_level):
        env_level = os.environ.get("LOG_LEVEL")
        level_name = env_level or cli_level or "DEBUG"
        return getattr(logging, level_name.upper(), logging.INFO)

    def _resolve_formatter(self):
        is_tty = hasattr(self._stream, "isatty") and self._stream.isatty()
        prefix_upper = self.prefix.upper()
        prefix_color = ANSI_COLORS.get(self.prefix, ANSI_COLORS["reset"])
        msg_colorlog = _COLORLOG_MSG_COLORS.get(self.prefix, "green")

        if is_tty and COLORLOG_AVAILABLE:
            base_format = (
                f"[%(asctime)s|{prefix_color}{prefix_upper}{ANSI_COLORS['reset']}"
                f"|%(levelname)-5s]: %(log_color)s%(message)s%(reset)s"
            )
            return ColoredFormatter(
                fmt=base_format,
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": msg_colorlog,
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
                style="%",
            )
        elif is_tty:
            # ANSI fallback when colorlog is unavailable
            reset = ANSI_COLORS["reset"]
            fmt = (
                f"[%(asctime)s| {prefix_color}{prefix_upper}{reset} | %(levelname)-8s]: "
                f"{prefix_color}%(message)s{reset}"
            )
            return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            # Plain text for non-TTY output (piped to files, log aggregators)
            fmt = "[%(asctime)s|%(name)s|%(levelname)-5s]: %(message)s"
            return logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")

    def get(self):
        return self.logger

    def set_prefix(self, new_prefix: str):
        self.prefix = new_prefix.lower()
        # Re-set formatter for all handlers
        for handler in self.logger.handlers:
            handler.setFormatter(self._resolve_formatter())

    def set_level(self, level):
        lvl = self._resolve_level(level)
        self.logger.setLevel(lvl)

    def get_level(self):
        return self.logger.level
