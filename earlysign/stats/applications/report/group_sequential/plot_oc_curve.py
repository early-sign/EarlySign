"""Plotting utilities for group-sequential operating characteristics.

This module provides a small plotting class that accepts the output of
`compute_oc_curve` (a sequence of OCPointResult) and draws curves
and stopping-distribution markers using the visual style from
`earlysign.stats.design.gst.common.visualization`.

The implementation is intentionally lightweight so it can be used in
reporting code paths without pulling in the heavier scenario result
classes used by the design-level plotter.

Stop distribution data format (spec)
-------------------------------
The plotting utilities accept an `OCPointResult` with a `stop_distribution`
payload. To avoid ambiguity the simulator/reporting code SHOULD prefer the
following canonical mapping when producing `stop_distribution`:

- Keys: integers representing the actual total sample size at which trials
    stopped (i.e. the cumulative sample size across arms at the analysis
    time). Example: {1279: 100, 2558: 200, 3837: 700}.
- Values: integer counts of simulated trials that stopped at that sample size.

Legacy/compatibility: older code may instead use analysis indices (1-based)
or ordinal labels (e.g. 3,5,8). The plotter will accept those but gives
preference to actual sample-size keys when they match the computed
`sample_sizes` for the result. If the keys do not match sample sizes, the
plotter maps ordinal labels to analyses by sorted order (smallest key → first
look) as a best-effort fallback. Prefer emitting sample-size keyed maps to
avoid ambiguity.
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
        plot_options: Optional[dict] = None,
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
            raise ValueError("`results` must be a non-empty list of OCPointResult")

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

        # Plot ESS curve (GST) using main color
        gst_color = self.colors[0]
        fsd_color = self.colors[1] if len(self.colors) > 1 else self.colors[0]

        ax.plot(
            effect_sizes + null_value,
            ess_values,
            color=gst_color,
            linestyle=self.linestyles[0],
            linewidth=2.5,
            label="GST: ESS",
            alpha=0.95,
        )

        # For each effect point, draw the ESS marker and stopping-distribution
        for r in results_sorted:
            eff = float(r.effect_size) + null_value
            # ESS scatter marker
            ax.scatter([eff], [r.expected_sample_size], color=gst_color, s=64, zorder=5, edgecolors="black")

            # Stop-distribution: expect keys to be actual total sample sizes
            stop_dist = r.stop_distribution or {}
            if isinstance(stop_dist, dict) and stop_dist:
                total = float(sum(stop_dist.values())) if sum(stop_dist.values()) > 0 else 1.0
                # plot each stop bin at its actual sample-size y coordinate
                for raw_key, count in stop_dist.items():
                    try:
                        n_total = int(raw_key)
                    except Exception:
                        # try legacy: 1..n_looks mapping
                        md = r.metadata or {}
                        n_looks_val = md.get("n_looks")
                        if n_looks_val is not None:
                            try:
                                n_looks = int(n_looks_val)
                            except Exception:
                                n_looks = len(stop_dist)
                        else:
                            n_looks = len(stop_dist)
                        k_int = int(raw_key)
                        if 1 <= k_int <= n_looks:
                            sample_sizes = self._resolve_sample_sizes_from_result(r, n_looks)
                            n_total = int(sample_sizes[k_int - 1])
                        else:
                            # skip non-integer/unknown keys
                            continue
                    prob = float(count) / total
                    size = max(30, min(300, int(prob * 600)))
                    alpha_val = min(max(prob * 0.8, 0.05), 0.9)
                    ax.scatter([eff], [n_total], color=gst_color, s=size, alpha=alpha_val, edgecolors="black", zorder=6)

        # Draw a star marker at the planned fixed-sample design point when
        # metadata provides a planned_max_n. We choose the effect that has
        # the largest ESS as a proxy for the fixed-sample point if no
        # explicit target_effect was supplied. This restores the previous
        # visual cue for the FSD point even when target_effect is omitted.
        try:
            if plot_options is None or plot_options.get("show_fsd_star", True):
                # Find any planned_max_n in metadata and draw one star at the
                # effect with maximum ESS as a representative location.
                planned_vals = [
                    (i, (r.metadata or {}).get("planned_max_n"))
                    for i, r in enumerate(results_sorted)
                ]
                # pick the first non-None planned_max_n if present
                planned = next((v for _, v in planned_vals if v is not None), None)
                if planned is not None:
                    idx_max_ess = int(np.argmax(ess_values))
                    ax.scatter(
                        [float(effect_sizes[idx_max_ess]) + null_value],
                        [int(planned)],
                        marker="*",
                        color=fsd_color,
                        s=350,
                        edgecolors="black",
                        zorder=8,
                    )
                    ax.axhline(
                        y=int(planned),
                        color=fsd_color,
                        linestyle="--",
                        alpha=0.7,
                        linewidth=1.5,
                    )
        except Exception:
            # Non-fatal: we prefer the plot to render even if drawing the
            # FSD star fails for unexpected metadata shapes.
            pass

        # Optionally mark target effect and draw FSD marker/horizontal line
        if target_effect is not None:
            ax.axvline(
                x=target_effect + null_value,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label=f"Target: {target_effect + null_value:.4f}",
            )

            # Find the OCPointResult closest to the target effect
            eff_arr = np.array(
                [float(r.effect_size) for r in results_sorted], dtype=float
            )
            idx_closest = int(np.argmin(np.abs(eff_arr - float(target_effect))))
            r_closest = results_sorted[idx_closest]

            # If the simulator provided planned_max_n (or fsd info) in metadata,
            # draw a star marker at (target_effect, planned_max_n) and a horizontal line.
            md = r_closest.metadata or {}
            planned_max_n = md.get("planned_max_n") or md.get("fsd_total")
            if planned_max_n is not None:
                ax.scatter(
                    [target_effect + null_value],
                    [int(planned_max_n)],
                    marker="*",
                    color=fsd_color,
                    s=300,
                    edgecolors="black",
                    zorder=8,
                )
                ax.axhline(
                    y=int(planned_max_n),
                    color=fsd_color,
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                )

        # Apply any plotting options provided by the caller
        # plot_options can include keys like 'ylim', 'y_bottom', 'xlim'
        if plot_options:
            # Explicit ylim tuple takes precedence. Allow None for upper bound
            ylim = plot_options.get("ylim")
            if (
                ylim is not None
                and isinstance(ylim, (list, tuple))
                and len(ylim) == 2
            ):
                lo, hi = ylim
                if lo is None and hi is None:
                    pass
                elif lo is None:
                    ax.set_ylim(top=float(hi))
                elif hi is None:
                    ax.set_ylim(bottom=float(lo))
                else:
                    ax.set_ylim(float(lo), float(hi))
            else:
                y_bottom = plot_options.get("y_bottom")
                if y_bottom is not None:
                    # preserve current top if present
                    try:
                        cur_top = ax.get_ylim()[1]
                        ax.set_ylim(float(y_bottom), float(cur_top))
                    except Exception:
                        ax.set_ylim(bottom=float(y_bottom))

            xlim = plot_options.get("xlim")
            if xlim is not None and isinstance(xlim, (list, tuple)) and len(xlim) == 2:
                ax.set_xlim(float(xlim[0]), float(xlim[1]))

            # Disable autoscale so the requested limits remain in effect
            try:
                ax.set_autoscale_on(False)
            except Exception:
                pass

        # Re-enforce a requested y_bottom after all plotting is complete.
        # Some backends or additional Matplotlib layout/legend operations can
        # nudge axis limits; re-setting the bottom here makes the intent
        # explicit and reduces flakiness in notebooks.
        try:
            if plot_options and (plot_options.get("y_bottom") is not None):
                _tmp_yb = plot_options.get("y_bottom")
                if _tmp_yb is None:
                    # defensive: skip if unexpectedly None
                    _tmp_yb = None
                else:
                    _tmp_yb = float(_tmp_yb)
                if _tmp_yb is not None:
                    yb = float(_tmp_yb)
                else:
                    # nothing to enforce
                    yb = None
                try:
                    top = float(ax.get_ylim()[1])
                    if yb is not None:
                        ax.set_ylim(yb, top)
                except Exception:
                    if yb is not None:
                        ax.set_ylim(bottom=yb)
                # request a draw so interactive/nbagg backends honor the change
                try:
                    fig = ax.figure
                    if fig is not None and hasattr(fig, "canvas"):
                        fig.canvas.draw_idle()
                except Exception:
                    pass
        except Exception:
            # non-fatal; continue rendering
            pass

        # Ensure axis label reflects absolute proportion when null_value is non-zero
        if effect_label is None or effect_label == "Effect Size":
            if null_value != 0.0:
                ax.set_xlabel("Treated proportion", fontsize=12)
            else:
                ax.set_xlabel(effect_label, fontsize=12)

        ax.legend(loc="best", fontsize=10)

        # Simple enforcement: if the caller requested a specific lower bound,
        # set it now after legend/layout which can shift axis limits.
        try:
            if plot_options and (plot_options.get("y_bottom") is not None):
                try:
                    _yb_val = plot_options.get("y_bottom")
                    if _yb_val is not None:
                        ax.set_ylim(bottom=float(_yb_val))
                    # request a synchronous draw so notebooks show the update.
                    fig = ax.figure
                    if fig is not None and hasattr(fig, "canvas"):
                        try:
                            fig.canvas.draw()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

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
