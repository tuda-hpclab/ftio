"""
Main Dash application for FTIO prediction visualization.

This module provides a real-time dashboard for visualizing FTIO predictions
and change point detection results. It includes auto-updating visualizations,
statistics display, and prediction selection controls.

Author: Amine Aherbil
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: January 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import argparse
import time

import dash
import numpy as np
from dash import Input, Output, callback_context, dcc, html

from ftio.gui.data_models import PredictionDataStore
from ftio.gui.socket_listener import SocketListener
from ftio.gui.visualizations import CosineWaveViz


class FTIODashApp:
    """Main Dash application for FTIO prediction visualization"""

    def __init__(self, host="localhost", port=8050, socket_port=9999):
        self.app = dash.Dash(__name__)
        self.host = host
        self.port = port
        self.socket_port = socket_port

        self.data_store = PredictionDataStore()
        self.selected_prediction_id = None
        self.last_update = time.time()

        self.socket_listener = SocketListener(
            port=socket_port, data_callback=self._on_data_received
        )

        self._setup_layout()
        self._setup_callbacks()

        self.socket_thread = self.socket_listener.start_in_thread()

        print(f"FTIO Dashboard starting on http://{host}:{port}")
        print(f"Socket listener on port {socket_port}")

    def _setup_layout(self):
        """Setup the Dash app layout"""

        self.app.layout = html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "FTIO Prediction Visualizer",
                            style={
                                "textAlign": "center",
                                "color": "#2c3e50",
                                "marginBottom": "20px",
                            },
                        ),
                        html.Div(
                            [
                                html.P(
                                    f"Socket listening on port {self.socket_port}",
                                    style={
                                        "textAlign": "center",
                                        "color": "#7f8c8d",
                                        "margin": "0",
                                    },
                                ),
                                html.P(
                                    id="connection-status",
                                    children="Waiting for predictions...",
                                    style={
                                        "textAlign": "center",
                                        "color": "#e74c3c",
                                        "margin": "0",
                                    },
                                ),
                            ]
                        ),
                    ],
                    style={"marginBottom": "30px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("View Mode:"),
                                dcc.Dropdown(
                                    id="view-mode",
                                    options=[
                                        {
                                            "label": "Dashboard (Merged Cosine Wave)",
                                            "value": "dashboard",
                                        },
                                        {
                                            "label": "Individual Prediction (Single Wave)",
                                            "value": "cosine",
                                        },
                                    ],
                                    value="dashboard",
                                    style={"width": "250px"},
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "20px"},
                        ),
                        html.Div(
                            [
                                html.Label("Select Prediction:"),
                                dcc.Dropdown(
                                    id="prediction-selector",
                                    options=[],
                                    value=None,
                                    placeholder="Select prediction for cosine view",
                                    style={"width": "250px"},
                                ),
                            ],
                            style={"display": "inline-block", "marginRight": "20px"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Clear Data",
                                    id="clear-button",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#e74c3c",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "8px 16px",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                            style={"display": "inline-block"},
                        ),
                    ],
                    style={
                        "textAlign": "center",
                        "marginBottom": "20px",
                        "padding": "20px",
                        "backgroundColor": "#ecf0f1",
                        "borderRadius": "5px",
                    },
                ),
                html.Div(id="stats-bar", style={"marginBottom": "20px"}),
                html.Div(id="main-viz", style={"height": "600px"}),
                html.Div(
                    [
                        html.Hr(),
                        html.H3(
                            "All Predictions",
                            style={"color": "#2c3e50", "marginTop": "30px"},
                        ),
                        html.Div(
                            id="recent-predictions-table",
                            style={
                                "maxHeight": "400px",
                                "overflowY": "auto",
                                "border": "1px solid #ddd",
                                "borderRadius": "8px",
                                "padding": "10px",
                                "backgroundColor": "#f9f9f9",
                            },
                        ),
                    ],
                    style={"marginTop": "20px"},
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=2000,  # Update every 2 seconds
                    n_intervals=0,
                ),
                dcc.Store(id="data-store-trigger"),
            ]
        )

    def _setup_callbacks(self):
        """Setup Dash callbacks"""

        @self.app.callback(
            [
                Output("main-viz", "children"),
                Output("prediction-selector", "options"),
                Output("prediction-selector", "value"),
                Output("connection-status", "children"),
                Output("connection-status", "style"),
                Output("stats-bar", "children"),
            ],
            [
                Input("interval-component", "n_intervals"),
                Input("view-mode", "value"),
                Input("prediction-selector", "value"),
                Input("clear-button", "n_clicks"),
            ],
        )
        def update_visualization(n_intervals, view_mode, selected_pred_id, clear_clicks):

            ctx = callback_context
            if (
                ctx.triggered
                and ctx.triggered[0]["prop_id"] == "clear-button.n_clicks"
                and clear_clicks > 0
            ):
                self.data_store.clear_data()
                self.selected_prediction_id = None

            pred_options = []
            pred_value = selected_pred_id

            if self.data_store.predictions:
                pred_options = [
                    {
                        "label": f"Prediction #{p.prediction_id} ({p.dominant_freq:.2f} Hz)",
                        "value": p.prediction_id,
                    }
                    for p in self.data_store.predictions[-50:]  # Last 50 predictions
                ]

                if pred_value is None and self.data_store.predictions:
                    pred_value = self.data_store.predictions[-1].prediction_id

            if self.data_store.predictions:
                status_text = (
                    f"Connected - {len(self.data_store.predictions)} predictions received"
                )
                status_style = {"textAlign": "center", "color": "#27ae60", "margin": "0"}
            else:
                status_text = "Waiting for predictions..."
                status_style = {"textAlign": "center", "color": "#e74c3c", "margin": "0"}

            stats_bar = self._create_stats_bar()

            if view_mode == "cosine" and pred_value is not None:
                fig = CosineWaveViz.create_cosine_plot(self.data_store, pred_value)
                viz_component = dcc.Graph(figure=fig, style={"height": "600px"})

            elif view_mode == "dashboard":
                fig = self._create_cosine_timeline_plot(self.data_store)
                viz_component = dcc.Graph(figure=fig, style={"height": "600px"})

            else:
                viz_component = html.Div(
                    [
                        html.H3(
                            "Select a view mode and prediction to visualize",
                            style={
                                "textAlign": "center",
                                "color": "#7f8c8d",
                                "marginTop": "200px",
                            },
                        )
                    ]
                )

            return (
                viz_component,
                pred_options,
                pred_value,
                status_text,
                status_style,
                stats_bar,
            )

        @self.app.callback(
            Output("recent-predictions-table", "children"),
            [Input("interval-component", "n_intervals")],
        )
        def update_recent_predictions_table(n_intervals):
            """Update the recent predictions table"""

            if not self.data_store.predictions:
                return html.P(
                    "No predictions yet",
                    style={"textAlign": "center", "color": "#7f8c8d"},
                )

            recent_preds = self.data_store.predictions

            seen_ids = set()
            unique_preds = []
            for pred in reversed(recent_preds):  # Newest first
                if pred.prediction_id not in seen_ids:
                    seen_ids.add(pred.prediction_id)
                    unique_preds.append(pred)

            rows = []
            for i, pred in enumerate(unique_preds):
                row_style = {
                    "backgroundColor": "#ffffff" if i % 2 == 0 else "#f8f9fa",
                    "padding": "8px",
                    "borderBottom": "1px solid #dee2e6",
                }

                if pred.dominant_freq == 0 or pred.dominant_freq is None:
                    row = html.Tr(
                        [
                            html.Td(
                                f"#{pred.prediction_id}",
                                style={"fontWeight": "bold", "color": "#999"},
                            ),
                            html.Td(
                                "—",
                                style={
                                    "color": "#999",
                                    "textAlign": "center",
                                    "fontStyle": "italic",
                                },
                            ),
                            html.Td(
                                "No pattern detected",
                                style={"color": "#999", "fontStyle": "italic"},
                            ),
                        ],
                        style=row_style,
                    )
                else:
                    change_point_text = ""
                    if pred.is_change_point and pred.change_point:
                        cp = pred.change_point
                        change_point_text = (
                            f"🔴 {cp.old_frequency:.2f} → {cp.new_frequency:.2f} Hz"
                        )

                    row = html.Tr(
                        [
                            html.Td(
                                f"#{pred.prediction_id}",
                                style={"fontWeight": "bold", "color": "#495057"},
                            ),
                            html.Td(
                                f"{pred.dominant_freq:.2f} Hz", style={"color": "#007bff"}
                            ),
                            html.Td(
                                change_point_text,
                                style={
                                    "color": "red" if pred.is_change_point else "black"
                                },
                            ),
                        ],
                        style=row_style,
                    )

                rows.append(row)

            table = html.Table(
                [
                    html.Thead(
                        [
                            html.Tr(
                                [
                                    html.Th(
                                        "ID",
                                        style={
                                            "backgroundColor": "#6c757d",
                                            "color": "white",
                                            "padding": "12px",
                                        },
                                    ),
                                    html.Th(
                                        "Frequency",
                                        style={
                                            "backgroundColor": "#6c757d",
                                            "color": "white",
                                            "padding": "12px",
                                        },
                                    ),
                                    html.Th(
                                        "Change Point",
                                        style={
                                            "backgroundColor": "#6c757d",
                                            "color": "white",
                                            "padding": "12px",
                                        },
                                    ),
                                ]
                            )
                        ]
                    ),
                    html.Tbody(rows),
                ],
                style={
                    "width": "100%",
                    "borderCollapse": "collapse",
                    "marginTop": "10px",
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    "borderRadius": "8px",
                    "overflow": "hidden",
                },
            )

            return table

    def _create_stats_bar(self):
        """Create statistics bar component"""

        if not self.data_store.predictions:
            return html.Div()

        total_preds = len(self.data_store.predictions)
        total_changes = len(self.data_store.change_points)
        latest_pred = self.data_store.predictions[-1]

        stats_items = [
            html.Div(
                [
                    html.H4(str(total_preds), style={"margin": "0", "color": "#2c3e50"}),
                    html.P(
                        "Total Predictions",
                        style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d"},
                    ),
                ],
                style={"textAlign": "center", "flex": "1"},
            ),
            html.Div(
                [
                    html.H4(
                        str(total_changes), style={"margin": "0", "color": "#e74c3c"}
                    ),
                    html.P(
                        "Change Points",
                        style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d"},
                    ),
                ],
                style={"textAlign": "center", "flex": "1"},
            ),
            html.Div(
                [
                    html.H4(
                        f"{latest_pred.dominant_freq:.2f} Hz",
                        style={"margin": "0", "color": "#27ae60"},
                    ),
                    html.P(
                        "Latest Frequency",
                        style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d"},
                    ),
                ],
                style={"textAlign": "center", "flex": "1"},
            ),
            html.Div(
                [
                    html.H4(
                        f"{latest_pred.confidence:.1f}%",
                        style={"margin": "0", "color": "#3498db"},
                    ),
                    html.P(
                        "Latest Confidence",
                        style={"margin": "0", "fontSize": "12px", "color": "#7f8c8d"},
                    ),
                ],
                style={"textAlign": "center", "flex": "1"},
            ),
        ]

        return html.Div(
            stats_items,
            style={
                "display": "flex",
                "justifyContent": "space-around",
                "backgroundColor": "#f8f9fa",
                "padding": "15px",
                "borderRadius": "5px",
                "border": "1px solid #dee2e6",
            },
        )

    def _on_data_received(self, data):
        """Callback when new data is received from socket"""
        print(f"[DEBUG] Dashboard received data: {data}")

        if data["type"] == "prediction":
            prediction_data = data["data"]
            self.data_store.add_prediction(prediction_data)

            print(
                f"[DEBUG] Added prediction #{prediction_data.prediction_id}: "
                f"{prediction_data.dominant_freq:.2f} Hz "
                f"({'CHANGE POINT' if prediction_data.is_change_point else 'normal'})"
            )

            self.last_update = time.time()
        else:
            print(f"[DEBUG] Received non-prediction data: type={data.get('type')}")

    def _create_cosine_timeline_plot(self, data_store):
        """Create single continuous cosine wave showing I/O pattern evolution"""
        import plotly.graph_objs as go

        if not data_store.predictions:
            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="Waiting for predictions...",
                showarrow=False,
                font={"size": 16, "color": "gray"},
            )
            fig.update_layout(
                xaxis={"visible": False},
                yaxis={"visible": False},
                title="I/O Pattern Timeline (Continuous Cosine Wave)",
            )
            return fig

        last_3_predictions = data_store.get_latest_predictions(3)
        print(f"[DEBUG] Merged view using {len(last_3_predictions)} predictions")

        sorted_predictions = sorted(last_3_predictions, key=lambda p: p.time_window[0])

        global_time = []
        global_cosine = []
        cumulative_time = 0.0
        segment_info = []  # For change point markers

        # Normalized display: each prediction gets equal width showing 5 cycles
        display_cycles = 5  # Show 5 complete cycles per prediction

        for pred in sorted_predictions:
            freq = pred.dominant_freq

            if freq == 0 or freq is None:
                # No frequency - show flat gap
                display_duration = 1.0  # Fixed width for gaps
                num_points = 100
                t_local = np.linspace(0, display_duration, num_points)
                t_global = cumulative_time + t_local

                global_time.extend(t_global.tolist())
                global_cosine.extend([None] * num_points)
            else:
                # Normalized: show 5 cycles regardless of actual duration
                display_duration = display_cycles / freq  # Time for 5 cycles
                num_points = 1000

                t_local = np.linspace(0, display_duration, num_points)
                cosine_segment = np.cos(2 * np.pi * freq * t_local)

                t_global = cumulative_time + t_local

                global_time.extend(t_global.tolist())
                global_cosine.extend(cosine_segment.tolist())

            segment_start = cumulative_time
            segment_end = cumulative_time + display_duration
            segment_info.append((segment_start, segment_end, pred))

            cumulative_time += display_duration

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=global_time,
                y=global_cosine,
                mode="lines",
                name="I/O Pattern Evolution",
                line={"color": "#1f77b4", "width": 2},
                connectgaps=False,  # DON'T connect across None values - creates visible gaps
                hovertemplate="<b>I/O Pattern</b><br>"
                + "Time: %{x:.3f} s<br>"
                + "Amplitude: %{y:.3f}<extra></extra>",
            )
        )

        for seg_start, seg_end, pred in segment_info:
            if pred.dominant_freq == 0 or pred.dominant_freq is None:
                fig.add_vrect(
                    x0=seg_start,
                    x1=seg_end,
                    fillcolor="gray",
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                    annotation_text="No pattern",
                    annotation_position="top",
                )

        for seg_start, _seg_end, pred in segment_info:
            if pred.is_change_point and pred.change_point:
                marker_time = seg_start  # Mark at the START of the changed segment

                fig.add_vline(
                    x=marker_time,
                    line_dash="solid",
                    line_color="red",
                    line_width=4,
                    opacity=0.8,
                )

                fig.add_annotation(
                    x=marker_time,
                    y=1.1,
                    text=f"🔴 CHANGE<br>{pred.change_point.old_frequency:.2f}→{pred.change_point.new_frequency:.2f} Hz",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=0,
                    ay=-40,
                    font={"size": 12, "color": "red", "family": "Arial Black"},
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="red",
                    borderwidth=2,
                )

        fig.update_layout(
            title="I/O Pattern Timeline (Continuous Evolution)",
            xaxis_title="Time (s) - Concatenated Segments",
            yaxis_title="I/O Pattern Amplitude",
            showlegend=True,
            height=600,
            hovermode="x unified",
            yaxis={"range": [-1.2, 1.2]},
            uirevision="constant",  # Prevents full page refresh - keeps zoom/pan state
        )

        return fig

    def run(self, debug=False):
        """Run the Dash application"""
        try:
            self.app.run(host=self.host, port=self.port, debug=debug)
        except KeyboardInterrupt:
            print("\nShutting down FTIO Dashboard...")
            self.socket_listener.stop_server()
        except Exception as e:
            print(f"Error running dashboard: {e}")
            self.socket_listener.stop_server()


def main():
    """Entry point for ftio-gui command"""
    parser = argparse.ArgumentParser(description="FTIO Prediction GUI Dashboard")
    parser.add_argument(
        "--host", default="localhost", help="Dashboard host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Dashboard port (default: 8050)"
    )
    parser.add_argument(
        "--socket-port",
        type=int,
        default=9999,
        help="Socket listener port (default: 9999)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    print("=" * 60)
    print("FTIO Prediction GUI Dashboard")
    print("=" * 60)
    print(f"Dashboard URL: http://{args.host}:{args.port}")
    print(f"Socket listener: {args.socket_port}")
    print("")
    print("Instructions:")
    print("1. Start this dashboard")
    print("2. Run your FTIO predictor with socket logging enabled")
    print("3. Watch real-time predictions and change points in the browser")
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        dashboard = FTIODashApp(
            host=args.host, port=args.port, socket_port=args.socket_port
        )
        dashboard.run(debug=args.debug)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()
