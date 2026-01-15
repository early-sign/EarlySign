"""Plotting helpers for group-sequential boundary specifications."""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pydantic import ValidationError

from earlysign.v0.methods.group_sequential.boundary import BoundaryCalculator
from earlysign.v0.methods.group_sequential.design.records.design import DesignPayloadModel


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
        Number of information time points to evaluate boundaries at.
    """
    try:
        design_model = DesignPayloadModel.model_validate(design)
        boundary_spec = design_model.boundary_spec()
    except ValidationError:
        boundary_spec = design

    info_times = np.linspace(0.01, 1.0, n_points)
    upper_bounds = []
    lower_bounds = []

    calc = BoundaryCalculator(spec=boundary_spec, process=None)

    for t in info_times:
        upper, lower, _ = calc.compute_boundary(info_time=float(t))
        upper_bounds.append(upper)
        lower_bounds.append(lower)

    fig, ax = plt.subplots(figsize=(10, 6))

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

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Information Fraction", fontsize=12)
    scale = boundary_spec.get("scale", "z")
    if scale == "z":
        ax.set_ylabel("Z-statistic", fontsize=12)
    elif scale == "bm":
        ax.set_ylabel("Brownian Motion B(t)", fontsize=12)
    else:
        ax.set_ylabel(f"Test Statistic ({scale})", fontsize=12)

    boundary_spec.get("efficacy", {}).get("style", "unknown")
    efficacy_family = boundary_spec.get("efficacy", {}).get("family", "")
    alpha = boundary_spec.get("alpha", 0.05)
    title = f"Group Sequential Design (α={alpha}"
    if efficacy_family:
        title += f", {efficacy_family.upper()}"
    title += ")"
    ax.set_title(title, fontsize=14)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close(fig)
    return fig
