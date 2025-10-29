"""Plotting utilities for group-sequential operating characteristics.

This module provides a small plotting class that accepts the output of
`compute_oc_curve` (a sequence of OCPointResult) and draws curves
and stopping-distribution markers using the visual style from
`earlysign.stats.design.gst.common.visualization`.

The implementation is intentionally lightweight so it can be used in
reporting code paths without pulling in the heavier scenario result
classes used by the design-level plotter.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    OCPointResult,
)


class OCCurvePlotter:
    """Plotter for compute_oc_curve() results (List[OCPointResult]).

    Usage:
        plotter = OCCurvePlotter()
        plotter.plot_oc_curve(results, target_effect=..., null_value=..., save_path=...)

    The plot shows Expected Sample Size (ESS) vs effect size. For each
    effect size we overlay scatter markers that represent the stopping
    distribution across analyses (transparency proportional to probability).

    Examples
    --------
    A small doctest that creates two synthetic `OCPointResult` objects
    and plots them. The test switches Matplotlib to the non-interactive
    'Agg' backend so it can run in CI without a display.

    >>> import matplotlib
    >>> matplotlib.use('Agg')
    >>> from matplotlib.axes import Axes
    >>> from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    ...     OCPointResult,
    ... )
    >>> # Create two minimal OCPointResult instances
    >>> r1 = OCPointResult(
    ...     effect_size=0.0,
    ...     expected_sample_size=100.0,
    ...     power=0.05,
    ...     max_sample_size=200,
    ...     stop_distribution={1: 100},
    ...     metadata={"sample_sizes": [100, 200], "n_per_analysis": 50, "n_looks": 2},
    ... )
    >>> r2 = OCPointResult(
    ...     effect_size=0.2,
    ...     expected_sample_size=80.0,
    ...     power=0.8,
    ...     max_sample_size=200,
    ...     stop_distribution={2: 200},
    ...     metadata={"sample_sizes": [100, 200], "n_per_analysis": 50, "n_looks": 2},
    ... )
    >>> plotter = OCCurvePlotter(figsize=(6, 4))
    >>> ax = plotter.plot_oc_curve([r1, r2], target_effect=0.2, null_value=0.0)
    >>> isinstance(ax, Axes)
    True
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (16, 6),
        colors: Optional[List[str]] = None,
        linestyles: Optional[List[str]] = None,
    ) -> None:
        self.figsize = figsize
        self.colors = colors or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        self.linestyles = linestyles or ["-", "--", "-.", ":"]

    def plot_oc_curve(
        self,
        results: List[OCPointResult],
        target_effect: Optional[float] = None,
        null_value: float = 0.0,
        effect_label: str = "Effect Size",
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot ESS vs effect size and overlay stopping distribution.

        Args:
            results: List of OCPointResult returned from compute_oc_curve
            target_effect: Optional target effect (draws a vertical line)
            null_value: Value to add to effect sizes to convert to absolute scale
            effect_label: Label for x-axis
            ax: Optional matplotlib Axes to draw onto; if None a new figure/axes
                will be created using this plotter's `figsize`.

        Returns:
            The matplotlib Axes that was used for plotting.
        """
        if not results:
            raise ValueError(
                "`results` must be a non-empty list of OCPointResult"
            )

        # Sort by effect_size to ensure monotonic curve
        results_sorted = sorted(results, key=lambda r: float(r.effect_size))

        effect_sizes = np.array([r.effect_size for r in results_sorted], dtype=float)
        ess_values = np.array(
            [r.expected_sample_size for r in results_sorted], dtype=float
        )
        np.array([r.power for r in results_sorted], dtype=float)

        created_fig = None
        if ax is None:
            created_fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        ax.set_title(
            "Operating Characteristics: ESS vs Effect Size",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(effect_label, fontsize=12)
        ax.set_ylabel("Expected Sample Size (ESS)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot ESS curve
        ax.plot(
            effect_sizes + null_value,
            ess_values,
            color=self.colors[0],
            linestyle=self.linestyles[0],
            linewidth=2.5,
            label="ESS",
            alpha=0.9,
        )

        # Mark each result as a scatter point and overlay stopping distribution
        for idx, r in enumerate(results_sorted):
            eff = float(r.effect_size) + null_value
            # marker color scales with power (darker for higher power)
            c = self.colors[min(idx + 1, len(self.colors) - 1)]

            ax.scatter(
                [eff],
                [r.expected_sample_size],
                color=c,
                s=80,
                zorder=5,
                edgecolors="black",
            )

            # Overlay stopping distribution if present
            stop_dist = r.stop_distribution or {}
            if stop_dist:
                # Determine number of looks from maximum analysis index seen
                max_look = max(stop_dist.keys())

                # Determine sample sizes at each analysis. Prefer metadata if available.
                sample_sizes = self._resolve_sample_sizes_from_result(r, max_look)

                total_sims = (
                    float(sum(stop_dist.values()))
                    if sum(stop_dist.values()) > 0
                    else 1.0
                )

                for analysis_idx, count in stop_dist.items():
                    # sample_sizes is 1-based index list
                    try:
                        n_total = sample_sizes[analysis_idx - 1]
                    except Exception:
                        # fallback: use proportional allocation of max_sample_size
                        n_total = int(r.max_sample_size * (analysis_idx / max_look))

                    prob = float(count) / total_sims
                    alpha_val = min(max(prob * 0.5, 0.02), 0.6)

                    ax.scatter(
                        [eff],
                        [n_total],
                        color=c,
                        s=150,
                        alpha=alpha_val,
                        edgecolors="none",
                        zorder=3,
                    )

        # Optionally mark target effect
        if target_effect is not None:
            ax.axvline(
                x=target_effect + null_value,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label=f"Target: {target_effect + null_value:.4f}",
            )

        ax.legend(loc="best", fontsize=10)

        # If we created the figure here, run tight_layout to avoid clipping.
        if created_fig is not None:
            created_fig.tight_layout()

        # Return the axes to the caller so they can show/save/compose as needed.
        return ax

    def _resolve_sample_sizes_from_result(
        self, r: OCPointResult, max_look: int
    ) -> List[int]:
        """Try to obtain sample sizes at each analysis from result.metadata.

        Accepts a few common metadata keys that simulator implementations may
        provide (for example 'n_per_analysis', 'n_looks', or explicit
        'sample_sizes'). If none are present we fall back to equally spaced
        fractions of r.max_sample_size.
        """
        meta = r.metadata or {}

        # If explicit sample_sizes provided
        sample_sizes = meta.get("sample_sizes")
        if (
            sample_sizes is not None
            and isinstance(sample_sizes, (list, tuple))
            and len(sample_sizes) >= max_look
        ):
            return [int(x) for x in sample_sizes][:max_look]

        # If n_per_analysis and n_looks provided
        n_per_analysis = meta.get("n_per_analysis")
        n_looks = meta.get("n_looks")
        if n_per_analysis and n_looks:
            try:
                n_per = int(n_per_analysis)
                n_looks = int(n_looks)
                return [n_per * (i + 1) * 2 for i in range(min(n_looks, max_look))]
            except Exception:
                pass

        # Fallback: equally spaced fractions of max_sample_size
        if r.max_sample_size and max_look > 0:
            return [
                int(r.max_sample_size * ((i + 1) / max_look)) for i in range(max_look)
            ]

        # Last resort: use ESS as a proxy (same value repeated)
        return [int(r.expected_sample_size) for _ in range(max_look)]
