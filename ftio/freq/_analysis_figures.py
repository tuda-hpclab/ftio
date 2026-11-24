"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from argparse import Namespace

import matplotlib.figure  # to check type
import matplotlib.pyplot as plt
import numpy as np

from ftio.freq.freq_html import create_html


class AnalysisFigures:
    def __init__(
        self,
        args: Namespace = None,
        b=None,
        t=None,
        b_sampled=None,
        t_sampled=None,
        freqs=None,
        amp=None,
        phi=None,
        conf=None,
        ranks=None,
        scales=None,
        coefficients=None,
    ):
        self.set_bulk(
            args,
            b,
            t,
            b_sampled,
            t_sampled,
            freqs,
            amp,
            phi,
            conf,
            ranks,
            scales,
            coefficients,
        )
        self.figures = []
        self.figure_titles = []

    def set_bulk(
        self,
        args: Namespace = None,
        b=None,
        t=None,
        b_sampled=None,
        t_sampled=None,
        freqs=None,
        amp=None,
        phi=None,
        conf=None,
        ranks=None,
        scales=None,
        coefficients=None,
    ):
        self.args = args
        self.b = b
        self.t = t
        self.b_sampled = b_sampled
        self.t_sampled = t_sampled
        self.freqs = freqs
        self.amp = amp
        self.phi = phi
        self.conf = conf
        self.ranks = ranks
        self.scales = scales
        self.coefficients = coefficients

    def sort(self):
        """
        Return the indices that would sort the data by conf (if available),
        using amp as a tiebreaker. If conf is not available, sorts by amp only.
        Does not modify the instance.

        Returns:
            np.ndarray: Indices for sorting.
        """
        if self.amp is None:
            return None

        if self.conf is not None and len(self.conf) > 0:
            # Sort by conf, then amp
            idx = np.lexsort((self.amp, self.conf))
        else:
            idx = np.argsort(self.amp)

        return idx

    def set(self, field_name, value):
        """
        Set an individual attribute by name.

        Args:
            field_name (str): Name of the attribute to set.
            value (array-like): The value to set (converted to np.array).
        """
        if not hasattr(self, field_name):
            raise AttributeError(
                f"'AnalysisFigures' object has no attribute '{field_name}'"
            )
        setattr(self, field_name, np.array(value))

    def get(self, field_name):
        """
        Get an individual attribute by name.

        Args:
            field_name (str): Name of the attribute to get.

        Returns:
            np.ndarray: The attribute value.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if not hasattr(self, field_name):
            raise AttributeError(
                f"'AnalysisFigures' object has no attribute '{field_name}'"
            )
        return getattr(self, field_name)

    def is_empty(self) -> bool:
        return len(self.figures) == 0

    def __bool__(self):
        return self.is_empty() or any(x in self.args.engine for x in ["mat", "plot"])

    def __len__(self):
        return len(self.figures)

    def add_figure(self, fig_list=None, source: str = ""):
        if fig_list is not None:
            if not isinstance(fig_list, list):
                fig_list = [fig_list]
            self.figures.append(fig_list)
            self.figure_titles.append(source)

    def add_figure_and_show(self, fig_list: list, source: str = ""):
        self.add_figure(fig_list, source)
        if self.args.runtime_plots:
            self.show_figs(fig_list, source)

    def show_figs(self, fig_list, name, condition=None):
        if condition is None:
            condition = any(x in self.args.engine for x in ["mat", "plot"])
        if condition and "mat" in self.args.engine:
            figs_to_show = []
            for fig in fig_list:
                # Matplotlib clears the figure after show()
                # If it has no axes, it's already been shown and cleared
                if (
                    isinstance(fig, matplotlib.figure.Figure)
                    and fig.get_axes()
                    and plt.fignum_exists(fig.number)
                ):
                    plt.figure(fig.number)
                    figs_to_show.append(fig)
            if figs_to_show:
                plt.show()
        elif condition:
            conf = {"toImageButtonOptions": {"format": "png", "scale": 4}}
            create_html(fig_list, self.args.render, conf, name)

    def show(self):
        if self.args is not None:
            # Track shown figures by list id to avoid duplicates
            shown_ids = set()
            for i, fig_list in enumerate(self.figures):
                if not fig_list:
                    continue
                list_id = id(fig_list)
                if list_id not in shown_ids:
                    self.show_figs(fig_list, self.figure_titles[i])
                    shown_ids.add(list_id)

    def __str__(self):
        attrs = [
            "b",
            "t",
            "b_sampled",
            "t_sampled",
            "freqs",
            "amp",
            "phi",
            "conf",
            "scales",
            "coefficients",
            "ranks",
        ]
        lines = [f"{attr}: {getattr(self, attr)}" for attr in attrs]
        return "AnalysisFigures with data:\n" + "\n".join(lines)

    def __add__(self, other):
        """
        Add two AnalysisFigures instances.
        - extends the list of figures and figure titles.
        - Overwrites all other attributes with those from `other`.

        Args:
            other (AnalysisFigures): Another instance to combine with.

        Returns:
            AnalysisFigures: A new combined instance.
        """
        if not isinstance(other, AnalysisFigures):
            return NotImplemented

        result = AnalysisFigures()

        # Initialize with copies of self's lists
        result.figures = self.figures
        result.figure_titles = self.figure_titles

        # Append lists
        result.figures.extend(other.figures)
        result.figure_titles.extend(other.figure_titles)

        for attr, value in other.__dict__.items():
            if attr in ["figures", "figure_titles"]:
                continue

            if attr in [
                "b",
                "t",
                "b_sampled",
                "t_sampled",
                "freqs",
                "amp",
                "phi",
                "conf",
                "scales",
                "coefficients",
            ]:
                self_val = getattr(self, attr, None)
                other_val = value
                if self_val is None and other_val is not None:
                    setattr(result, attr, other_val)
                elif self_val is not None and other_val is None:
                    setattr(result, attr, self_val)
                else:
                    # both None or both exist: prefer other
                    setattr(result, attr, other_val)
            else:
                # Overwrite all other attributes from other
                setattr(result, attr, value)

        return result

    def __iadd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        s = f"AnalysisFigures Class containing the following elements:{self.figure_titles}\n"
        attrs = [
            "b",
            "t",
            "b_sampled",
            "t_sampled",
            "freqs",
            "amp",
            "phi",
            "conf",
            "scales",
            "coefficients",
            "ranks",
        ]
        lines = [f"{attr}: {type(getattr(self, attr)).__name__}" for attr in attrs]
        return s + "AnalysisFigures with fields and types:\n" + "\n".join(lines)
