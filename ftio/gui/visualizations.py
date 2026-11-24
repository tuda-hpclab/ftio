"""
Plotly/Dash visualization components for FTIO prediction data.

This module provides visualization components for the FTIO dashboard including
frequency timeline plots, continuous cosine wave displays, change point markers,
and combined dashboard views.

Author: Amine Aherbil
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: January 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ftio.gui.data_models import PredictionDataStore


class FrequencyTimelineViz:
    """Creates frequency timeline visualization"""

    @staticmethod
    def create_timeline_plot(
        data_store: PredictionDataStore, title="FTIO Frequency Timeline"
    ):
        """Create main frequency timeline plot"""

        pred_ids, frequencies, confidences = data_store.get_frequency_timeline()

        if not pred_ids:
            fig = go.Figure()
            fig.add_annotation(
                text="No prediction data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 16, "color": "gray"},
            )
            fig.update_layout(
                title=title,
                xaxis_title="Prediction Index",
                yaxis_title="Frequency (Hz)",
                height=500,
            )
            return fig

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=pred_ids,
                y=frequencies,
                mode="lines+markers",
                name="Dominant Frequency",
                line={"color": "blue", "width": 2},
                marker={
                    "size": 8,
                    "opacity": [conf / 100.0 for conf in confidences],
                    "color": "blue",
                    "line": {"width": 1, "color": "darkblue"},
                },
                hovertemplate="<b>Prediction #%{x}</b><br>"
                + "Frequency: %{y:.2f} Hz<br>"
                + "Confidence: %{customdata:.1f}%<extra></extra>",
                customdata=confidences,
            )
        )

        candidates_dict = data_store.get_candidate_frequencies()
        for pred_id, candidates in candidates_dict.items():
            for candidate in candidates:
                if (
                    candidate.frequency
                    != data_store.get_prediction_by_id(pred_id).dominant_freq
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=[pred_id],
                            y=[candidate.frequency],
                            mode="markers",
                            name=f"Candidate (conf: {candidate.confidence:.2f})",
                            marker={
                                "size": 6,
                                "opacity": candidate.confidence,
                                "color": "orange",
                                "symbol": "diamond",
                            },
                            showlegend=False,
                            hovertemplate="<b>Candidate Frequency</b><br>"
                            + f"Frequency: {candidate.frequency:.2f} Hz<br>"
                            + f"Confidence: {candidate.confidence:.2f}<extra></extra>",
                        )
                    )

        cp_pred_ids, cp_frequencies, cp_labels = (
            data_store.get_change_points_for_timeline()
        )

        if cp_pred_ids:
            fig.add_trace(
                go.Scatter(
                    x=cp_pred_ids,
                    y=cp_frequencies,
                    mode="markers",
                    name="Change Points",
                    marker={
                        "size": 12,
                        "color": "red",
                        "symbol": "diamond",
                        "line": {"width": 2, "color": "darkred"},
                    },
                    hovertemplate="<b>Change Point</b><br>"
                    + "Prediction #%{x}<br>"
                    + "%{customdata}<extra></extra>",
                    customdata=cp_labels,
                )
            )

            for pred_id, _freq, label in zip(
                cp_pred_ids, cp_frequencies, cp_labels, strict=False
            ):
                fig.add_vline(
                    x=pred_id,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.7,
                    annotation_text=label,
                    annotation_position="top",
                )

        fig.update_layout(
            title={"text": title, "font": {"size": 18, "color": "darkblue"}},
            xaxis={
                "title": "Prediction Index",
                "showgrid": True,
                "gridcolor": "lightgray",
                "tickmode": "linear",
            },
            yaxis={"title": "Frequency (Hz)", "showgrid": True, "gridcolor": "lightgray"},
            hovermode="closest",
            height=500,
            margin={"l": 60, "r": 60, "t": 80, "b": 60},
            plot_bgcolor="white",
            showlegend=True,
            legend={
                "x": 0.02,
                "y": 0.98,
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "gray",
                "borderwidth": 1,
            },
        )

        return fig


class CosineWaveViz:
    """Creates cosine wave visualization for individual predictions"""

    @staticmethod
    def create_cosine_plot(
        data_store: PredictionDataStore, prediction_id: int, title=None, num_points=1000
    ):
        """Create cosine wave plot for a specific prediction"""

        prediction = data_store.get_prediction_by_id(prediction_id)
        if not prediction:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Prediction #{prediction_id} not found",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 16, "color": "gray"},
            )
            fig.update_layout(
                title=f"Cosine Wave - Prediction #{prediction_id}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=400,
            )
            return fig

        t, primary_wave, candidate_waves = data_store.generate_cosine_wave(
            prediction_id, num_points
        )

        if title is None:
            title = (
                f"Cosine Wave - Prediction #{prediction_id} "
                f"(f = {prediction.dominant_freq:.2f} Hz)"
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=t,
                y=primary_wave,
                mode="lines",
                name=f"I/O Pattern: {prediction.dominant_freq:.2f} Hz",
                line={"color": "#1f77b4", "width": 3},
                hovertemplate="<b>I/O Pattern</b><br>"
                + "Time: %{x:.3f} s<br>"
                + "Amplitude: %{y:.3f}<br>"
                + f"Frequency: {prediction.dominant_freq:.2f} Hz<extra></extra>",
            )
        )

        if prediction.is_change_point and prediction.change_point:
            cp_time = prediction.change_point.timestamp
            start_time, end_time = prediction.time_window
            if start_time <= cp_time <= end_time:
                cp_relative = cp_time - start_time
                fig.add_vline(
                    x=cp_relative,
                    line_dash="dash",
                    line_color="red",
                    line_width=3,
                    opacity=0.8,
                    annotation_text=(
                        f"Change Point<br>"
                        f"{prediction.change_point.old_frequency:.2f} → "
                        f"{prediction.change_point.new_frequency:.2f} Hz"
                    ),
                    annotation_position="top",
                )

        start_time, end_time = prediction.time_window
        duration = end_time - start_time
        fig.update_layout(
            title={"text": title, "font": {"size": 16, "color": "darkblue"}},
            xaxis={
                "title": f"Time (s) - Duration: {duration:.2f}s",
                "range": [0, duration],
                "showgrid": True,
                "gridcolor": "lightgray",
            },
            yaxis={
                "title": "Amplitude",
                "range": [-1.2, 1.2],
                "showgrid": True,
                "gridcolor": "lightgray",
            },
            height=400,
            margin={"l": 60, "r": 60, "t": 60, "b": 60},
            plot_bgcolor="white",
            showlegend=True,
            legend={
                "x": 0.02,
                "y": 0.98,
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "gray",
                "borderwidth": 1,
            },
        )

        return fig


class DashboardViz:
    """Creates comprehensive dashboard visualization"""

    @staticmethod
    def create_dashboard(data_store: PredictionDataStore, selected_prediction_id=None):
        """Create comprehensive dashboard with multiple views"""

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Frequency Timeline",
                "Latest Predictions",
                "Cosine Wave View",
                "Statistics",
            ),
            specs=[[{"colspan": 2}, None], [{}, {}]],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.1,
        )

        timeline_fig = FrequencyTimelineViz.create_timeline_plot(data_store)
        for trace in timeline_fig.data:
            fig.add_trace(trace, row=1, col=1)

        if selected_prediction_id is not None:
            cosine_fig = CosineWaveViz.create_cosine_plot(
                data_store, selected_prediction_id
            )
            for trace in cosine_fig.data:
                fig.add_trace(trace, row=2, col=1)

        stats = DashboardViz._calculate_stats(data_store)
        fig.add_trace(
            go.Bar(
                x=list(stats.keys()),
                y=list(stats.values()),
                name="Statistics",
                marker_color="lightblue",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800, title_text="FTIO Prediction Dashboard", showlegend=True
        )

        fig.update_xaxes(title_text="Prediction Index", row=1, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_xaxes(title_text="Metric", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)

        return fig

    @staticmethod
    def _calculate_stats(data_store: PredictionDataStore) -> dict[str, float]:
        """Calculate basic statistics from prediction data"""
        if not data_store.predictions:
            return {}

        frequencies = [p.dominant_freq for p in data_store.predictions]
        confidences = [p.confidence for p in data_store.predictions]

        stats = {
            "Total Predictions": len(data_store.predictions),
            "Change Points": len(data_store.change_points),
            "Avg Frequency": np.mean(frequencies),
            "Avg Confidence": np.mean(confidences),
            "Freq Std Dev": np.std(frequencies),
        }

        return stats
