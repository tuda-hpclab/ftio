"""
Burst-width plot for FTIO.

Produces a dedicated Plotly figure showing the resampled bandwidth signal
with period boundaries and estimated burst regions shaded — analogous to
the example in examples/burst_width_example.py but using FTIO's real data.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Jun 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ftio.plot.units import set_unit


def plot_burst_width(
    prediction, b_sampled: np.ndarray, energy_fraction: float = 0.95
) -> go.Figure:
    """Return a Plotly figure of the bandwidth signal with burst-width shading.

    Shows:
      - The resampled bandwidth signal (blue)
      - Reconstructed dominant-frequency signal (orange, as a trace)
      - Dashed vertical lines at every period boundary (grey)
      - Shaded salmon region per burst window (all toggle together in legend)
      - Hover text per burst with energy fraction
      - Per-burst energy % annotation (when <= 12 periods)
      - Median tau bar with min/max whiskers

    Args:
        prediction:       Prediction object with burst_widths, burst_t_starts,
                          freq, t_start, t_end, and dominant_freq set.
        b_sampled:        Uniformly resampled bandwidth signal.
        energy_fraction:  Energy fraction used during burst estimation (default 0.95).

    Returns:
        go.Figure
    """
    bws = prediction.burst_widths
    t_starts = prediction.burst_t_starts

    if len(bws) == 0 or len(b_sampled) == 0:
        return go.Figure()

    fs = prediction.freq
    t0 = prediction.t_start
    t_end = prediction.t_end
    f_dom = prediction.get_dominant_freq()

    unit, order = set_unit(b_sampled)
    b_plot = b_sampled * order
    t_sampled = t0 + np.arange(len(b_sampled)) / fs
    y_max = float(np.max(b_plot)) if len(b_plot) else 1.0
    y_top = y_max * 1.30

    total_energy = float(np.sum(b_sampled**2))

    fig = go.Figure()

    # ── Bandwidth signal ────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=t_sampled,
            y=b_plot,
            mode="lines",
            name="Bandwidth",
            line={"color": "rgba(0,120,210,0.9)", "shape": "hv", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(0,120,210,0.15)",
        )
    )

    # ── Dominant frequency reconstructed signal ─────────────────────────────
    f_d, amp_d, phi_d = prediction.get_dominant_freq_amp_phi()
    if not np.isnan(f_d):
        dom_wave = prediction.get_wave(f_d, amp_d, phi_d, t_sampled)
        if len(dom_wave) > 0:
            dom_wave_plot = dom_wave * order
            fig.add_trace(
                go.Scatter(
                    x=t_sampled,
                    y=dom_wave_plot,
                    mode="lines",
                    name=f"Dominant freq ({f_d:.3e} Hz)",
                    line={"color": "rgba(255,140,0,0.85)", "width": 1.5, "dash": "dash"},
                )
            )

    # ── Burst regions — one trace per period, all in the same legendgroup ───
    # Using filled Scatter so all toggle together when clicking the legend.
    burst_legend_name = f"Burst region ({int(energy_fraction * 100)}% energy)"
    for k, (t_s, bw) in enumerate(zip(t_starts, bws, strict=False)):
        left_idx = max(0, int((t_s - t0) * fs))
        right_idx = min(len(b_sampled), left_idx + int(bw * fs))
        burst_energy = float(np.sum(b_sampled[left_idx:right_idx] ** 2))
        energy_pct = 100.0 * burst_energy / total_energy if total_energy > 0 else 0.0

        x_poly = [t_s, t_s + bw, t_s + bw, t_s, t_s]
        y_poly = [0, 0, y_top, y_top, 0]

        fig.add_trace(
            go.Scatter(
                x=x_poly,
                y=y_poly,
                fill="toself",
                fillcolor="rgba(250,128,114,0.22)",
                mode="none",
                name=burst_legend_name,
                legendgroup="burst",
                showlegend=(k == 0),
                hovertemplate=(
                    f"<b>Burst {k + 1}</b><br>"
                    f"Start: {t_s:.3f} s<br>"
                    f"Width: {bw:.3f} s<br>"
                    f"Energy: {energy_pct:.1f} %"
                    "<extra></extra>"
                ),
            )
        )

        if len(bws) <= 12:
            fig.add_annotation(
                x=t_s + bw / 2,
                y=y_max * 1.08,
                text=f"{energy_pct:.0f}%",
                showarrow=False,
                font={"size": 10, "color": "rgba(200,80,60,0.9)"},
            )

    # ── Period boundaries ───────────────────────────────────────────────────
    if not np.isnan(f_dom) and f_dom > 0:
        T = 1.0 / f_dom
        n_periods = int((t_end - t0) / T) + 2
        for k in range(n_periods):
            tb = t0 + k * T
            if t0 <= tb <= t_end:
                fig.add_vline(
                    x=tb,
                    line_dash="dash",
                    line_color="grey",
                    line_width=1,
                    opacity=0.4,
                )

    # ── Median tau bar with min/max whiskers ────────────────────────────────
    median_bw = prediction.burst_width_median
    min_bw = prediction.burst_width_min
    max_bw = prediction.burst_width_max

    if not np.isnan(median_bw) and not np.isnan(f_dom) and f_dom > 0:
        T = 1.0 / f_dom
        ref_center = t0 + 0.5 * T
        bar_y = y_max * 1.18

        fig.add_shape(
            type="line",
            x0=ref_center - median_bw / 2,
            x1=ref_center + median_bw / 2,
            y0=bar_y,
            y1=bar_y,
            line={"color": "rgba(200,80,60,1)", "width": 3},
        )
        fig.add_shape(
            type="line",
            x0=ref_center - max_bw / 2,
            x1=ref_center - min_bw / 2,
            y0=bar_y,
            y1=bar_y,
            line={"color": "rgba(200,80,60,0.5)", "width": 2, "dash": "dot"},
        )
        fig.add_shape(
            type="line",
            x0=ref_center + min_bw / 2,
            x1=ref_center + max_bw / 2,
            y0=bar_y,
            y1=bar_y,
            line={"color": "rgba(200,80,60,0.5)", "width": 2, "dash": "dot"},
        )
        fig.add_annotation(
            x=ref_center,
            y=bar_y * 1.04,
            text=f"τ median = {median_bw:.2f} s  [{min_bw:.2f}, {max_bw:.2f}]",
            showarrow=False,
            font={"size": 12, "color": "rgba(200,80,60,1)"},
            yanchor="bottom",
        )

    # ── Layout ──────────────────────────────────────────────────────────────
    n_complete = len(bws)
    dc = prediction.duty_cycle
    T_str = f"{1.0 / f_dom:.3f} s" if (not np.isnan(f_dom) and f_dom > 0) else "N/A"
    dc_str = f"{dc * 100:.1f} %" if not np.isnan(dc) else "N/A"
    f_str = f"{f_dom:.3e} Hz" if (not np.isnan(f_dom) and f_dom > 0) else "N/A"

    subtitle = (
        f"f = {f_str}  |  T = {T_str}  |  duty cycle = {dc_str}"
        "  |  "
        f"median = {median_bw:.3f} s  |  min = {min_bw:.3f} s  |  max = {max_bw:.3f} s"
        f"  |  {n_complete} complete period(s)"
        if not np.isnan(median_bw)
        else ""
    )

    fig.update_layout(
        title={
            "text": f"Burst width estimation<br><sup>{subtitle}</sup>",
            "x": 0.05,
        },
        xaxis_title="Time (s)",
        yaxis_title=f"Bandwidth ({unit})",
        xaxis={"range": [t0, t_end]},
        yaxis={"range": [0, y_top * 1.05]},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        template="plotly",
        width=1100,
        height=500,
        font={"family": "Courier New, monospace", "size": 16, "color": "black"},
    )

    return fig
