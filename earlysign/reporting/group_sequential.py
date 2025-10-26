"""
Reporting utilities for Group Sequential designs.

This module provides plotting and visualization functions for group sequential
testing designs and results.
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from earlysign.stats.essentials.methods.group_sequential.boundary import (
    BoundaryCalculator,
)


def plot_design_boundaries(
    design: Dict[str, Any],
    n_points: int = 50,
) -> Figure:
    """
    Plot group sequential design boundaries.

    Parameters
    ----------
    design : Dict[str, Any]
        Design configuration dictionary containing alpha, tails, scale,
        efficacy, and futility specifications.
    n_points : int, optional
        Number of information time points to evaluate boundaries at, by default 50.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.

    Examples
    --------
    >>> design = {
    ...     "alpha": 0.05, "tails": 2, "scale": "z",
    ...     "efficacy": {"style": "alpha_spending", "family": "obf"},
    ...     "futility": {"mode": "symmetric"}
    ... }
    >>> fig = plot_design_boundaries(design)  # doctest: +SKIP
    """
    # Generate information time points
    info_times = np.linspace(0.01, 1.0, n_points)
    upper_bounds = []
    lower_bounds = []

    calc = BoundaryCalculator(spec=design, process=None)

    # Calculate boundaries at each information time
    for t in info_times:
        upper, lower, scale = calc.compute_boundary(info_time=float(t))
        upper_bounds.append(upper)
        lower_bounds.append(lower)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot efficacy (upper) boundary
    ax.plot(
        info_times,
        upper_bounds,
        "r-",
        linewidth=2,
        label="Efficacy Boundary",
        marker="o",
        markersize=3,
        markevery=max(1, n_points // 10),
    )

    # Plot futility (lower) boundary
    ax.plot(
        info_times,
        lower_bounds,
        "b-",
        linewidth=2,
        label="Futility Boundary",
        marker="s",
        markersize=3,
        markevery=max(1, n_points // 10),
    )

    # Add horizontal line at zero
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Labels and title
    ax.set_xlabel("Information Fraction", fontsize=12)
    scale = design.get("scale", "z")
    if scale == "z":
        ax.set_ylabel("Z-statistic", fontsize=12)
    elif scale == "bm":
        ax.set_ylabel("Brownian Motion B(t)", fontsize=12)
    else:
        ax.set_ylabel(f"Test Statistic ({scale})", fontsize=12)

    # Title with design details
    design.get("efficacy", {}).get("style", "unknown")
    efficacy_family = design.get("efficacy", {}).get("family", "")
    alpha = design.get("alpha", 0.05)
    title = f"Group Sequential Design (α={alpha}"
    if efficacy_family:
        title += f", {efficacy_family.upper()}"
    title += ")"
    ax.set_title(title, fontsize=14)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close(fig)  # Prevent duplicate display in notebooks
    return fig
