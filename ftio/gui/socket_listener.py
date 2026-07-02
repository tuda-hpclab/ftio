"""
Socket listener for receiving FTIO prediction data via direct JSON transmission.

This module provides a TCP socket server that receives structured prediction
data from FTIO's online predictor via direct JSON transmission.

Author: Amine Aherbil
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: January 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
from collections.abc import Callable

from ftio.gui.data_models import (
    ChangePoint,
    FrequencyCandidate,
    PredictionData,
)


class SocketListener:
    """TCP server that receives FTIO prediction data from the online predictor.

    This class is the receiving end of the predictor-to-GUI communication channel.
    The FTIO online predictor (predictor.py) runs as a separate process and sends
    prediction results via TCP socket. This SocketListener runs inside the GUI
    dashboard process and receives those messages.

    Architecture:
        [FTIO Predictor Process] --TCP/JSON--> [SocketListener] --> [GUI Dashboard]

    The socket connection allows the predictor and GUI to run on different machines
    if needed (e.g., predictor on HPC cluster, GUI on local workstation).

    Each received message contains:
        - Prediction ID and timestamp
        - Detected dominant frequency and confidence
        - Time window that was analyzed
        - Whether a change point was detected
        - Change point details (old/new frequency, when it occurred)

    Messages are JSON-formatted and parsed into PredictionData objects, which are
    then passed to the dashboard via the data_callback function for visualization.

    Attributes:
        host: The hostname to bind the server to (default: "localhost").
        port: The port number to listen on (default: 9999).
        data_callback: Function called when new prediction data arrives.
        running: Boolean indicating if the server is currently running.
        server_socket: The main TCP server socket.
        client_connections: List of active client connections.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9999,
        data_callback: Callable | None = None,
    ):
        """Initialize the socket listener with connection parameters.

        Args:
            host: The hostname to bind the server to. Use "localhost" for local
                connections only, or "0.0.0.0" to accept connections from any host.
            port: The port number to listen on. Must match the port used by the
                FTIO predictor's SocketLogger (default: 9999).
            data_callback: Function to call when prediction data is received.
                The callback receives a dict with 'type' and 'data' keys.
        """
        self.host = host
        self.port = port
        self.data_callback = data_callback
        self.running = False
        self.server_socket = None
        self.client_connections = []

    def start_server(self):
        """Start the TCP server and begin listening for predictor connections.

        This method blocks and runs the main server loop. It binds to the configured
        host:port, listens for incoming connections, and spawns a new thread for
        each connected client (predictor process).

        The server continues running until stop_server() is called or an error occurs.
        Each client connection is handled in a separate daemon thread, allowing
        multiple predictors to connect simultaneously.

        Raises:
            OSError: If the port is already in use (errno 98) or other socket errors.
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            print(f"Attempting to bind to {self.host}:{self.port}")
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True

            print(f" Socket server successfully listening on {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f" Client connected from {address}")

                    client_thread = threading.Thread(
                        target=self._handle_client, args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except OSError as e:
                    if self.running:
                        print(f"Error accepting client connection: {e}")
                    break
                except KeyboardInterrupt:
                    print(" Socket server interrupted")
                    break

        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(
                    f"Port {self.port} is already in use! Please use a different port or kill the process using it."
                )
            else:
                print(f"OS Error starting socket server: {e}")
            self.running = False
        except Exception as e:
            print(f"Unexpected error starting socket server: {e}")
            import traceback

            traceback.print_exc()
            self.running = False
        finally:
            self.stop_server()

    def _handle_client(self, client_socket, address):
        """Handle an individual client connection in a dedicated thread.

        Continuously receives JSON messages from the connected predictor process,
        parses them into PredictionData objects, and forwards them to the dashboard
        via the data_callback.

        Message format expected:
            {
                "type": "prediction",
                "data": {
                    "prediction_id": int,
                    "timestamp": float,
                    "dominant_freq": float,
                    "confidence": float,
                    "is_change_point": bool,
                    "change_point": {...} or null,
                    ...
                }
            }

        Args:
            client_socket: The socket connection to the client (predictor).
            address: Tuple of (host, port) identifying the client.
        """
        try:
            while self.running:
                try:
                    data = client_socket.recv(4096).decode("utf-8")
                    if not data:
                        break

                    try:
                        message_data = json.loads(data)

                        if (
                            message_data.get("type") == "prediction"
                            and "data" in message_data
                        ):
                            print(
                                f"[DEBUG] Direct prediction data received: #{message_data['data']['prediction_id']}"
                            )

                            pred_data = message_data["data"]

                            candidates = []
                            for cand in pred_data.get("candidates", []):
                                candidates.append(
                                    FrequencyCandidate(
                                        frequency=cand["frequency"],
                                        confidence=cand["confidence"],
                                    )
                                )

                            change_point = None
                            if pred_data.get("is_change_point") and pred_data.get(
                                "change_point"
                            ):
                                cp_data = pred_data["change_point"]
                                change_point = ChangePoint(
                                    prediction_id=cp_data["prediction_id"],
                                    timestamp=cp_data["timestamp"],
                                    old_frequency=cp_data["old_frequency"],
                                    new_frequency=cp_data["new_frequency"],
                                    frequency_change_percent=cp_data[
                                        "frequency_change_percent"
                                    ],
                                    sample_number=cp_data["sample_number"],
                                    cut_position=cp_data["cut_position"],
                                    total_samples=cp_data["total_samples"],
                                )

                            prediction_data = PredictionData(
                                prediction_id=pred_data["prediction_id"],
                                timestamp=pred_data["timestamp"],
                                dominant_freq=pred_data["dominant_freq"],
                                dominant_period=pred_data["dominant_period"],
                                confidence=pred_data["confidence"],
                                candidates=candidates,
                                time_window=tuple(pred_data["time_window"]),
                                total_bytes=pred_data["total_bytes"],
                                bytes_transferred=pred_data["bytes_transferred"],
                                current_hits=pred_data["current_hits"],
                                periodic_probability=pred_data["periodic_probability"],
                                frequency_range=tuple(pred_data["frequency_range"]),
                                period_range=tuple(pred_data["period_range"]),
                                is_change_point=pred_data["is_change_point"],
                                change_point=change_point,
                                sample_number=pred_data.get("sample_number"),
                            )

                            if self.data_callback:
                                self.data_callback(
                                    {"type": "prediction", "data": prediction_data}
                                )

                    except json.JSONDecodeError:
                        pass

                except OSError:
                    break

        except Exception as e:
            logging.error(f"Error handling client {address}: {e}")
        finally:
            try:
                client_socket.close()
                print(f"Client {address} disconnected")
            except Exception:
                pass

    def stop_server(self):
        """Stop the server and close all connections.

        Sets running to False to signal all client handler threads to stop,
        closes the main server socket, and closes all active client connections.
        This method is safe to call multiple times.
        """
        self.running = False
        if self.server_socket:
            with contextlib.suppress(BaseException):
                self.server_socket.close()

        for client_socket in self.client_connections:
            with contextlib.suppress(BaseException):
                client_socket.close()
        self.client_connections.clear()
        print("Socket server stopped")

    def start_in_thread(self):
        """Start the server in a background daemon thread.

        This is the recommended way to start the server when running alongside
        the GUI dashboard. The daemon thread allows the program to exit cleanly
        even if the server is still running.

        Returns:
            threading.Thread: The thread object running the server. Can be used
                to check if the server is still alive via thread.is_alive().
        """
        server_thread = threading.Thread(target=self.start_server)
        server_thread.daemon = True
        server_thread.start()
        return server_thread
