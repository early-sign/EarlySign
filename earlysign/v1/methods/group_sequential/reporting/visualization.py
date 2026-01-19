"""
Visualization utilities for Group Sequential Testing.
"""

from typing import Any, List, Tuple

import ibis
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from earlysign.schema.ES3.GST.Log import LookResult


def reconstruct_z_history(table: ibis.Expr) -> Tuple[List[int], List[float]]:
    """
    Generic helper to reconstruct Z-statistic history from LookResult records in the ledger.
    """
    results_df = table.filter(table.payload_type == "LookResult").execute()

    if results_df.empty:
        return [], []

    history_n = []
    history_z = []

    for _, row in results_df.iterrows():
        payload = row["payload"]
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        lr = LookResult.model_validate(payload)
        history_n.append(lr.sample_n)
        history_z.append(lr.z_stat)

    return history_n, history_z


def plot_gst_summary(
    protocol: Any,
    history_n: List[int],
    history_z: List[float],
    title: str = "GST Monitoring",
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """
    Generates a generic summary plot for Group Sequential Test results.
    """
    from earlysign.v1.methods.group_sequential.execution.binomial import (
        BinomialGSTEngine,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Theoretical Boundaries
    # Use BinomialGSTEngine logic to reconstruct boundaries for visualization
    # (Even if the test isn't binomial, the GST structure for boundaries is shared)
    try:
        engine = BinomialGSTEngine(protocol)

        schedule = protocol.method.efficacy.schedule
        look_ns = schedule.interim_points or []
        n_max = max(look_ns) if look_ns else 1

        eff_boundaries = []
        fut_boundaries = []

        for i, n in enumerate(look_ns):
            frac = n / n_max
            # Efficacy
            eb = engine._get_boundary(
                protocol.method.efficacy, engine.efficacy_calc, "efficacy", i, frac
            )
            eff_boundaries.append(eb)

            # Futility
            fb = None
            if protocol.method.futility:
                fb = engine._get_boundary(
                    protocol.method.futility, engine.futility_calc, "futility", i, frac
                )
            fut_boundaries.append(fb)

        if look_ns:
            if any(b is not None for b in eff_boundaries):
                eff_plot = np.array(
                    [b if b is not None else np.nan for b in eff_boundaries]
                )
                ax.plot(look_ns, eff_plot, "r--", label="Efficacy Boundary")

            if any(b is not None for b in fut_boundaries):
                # Filter out -inf for plotting
                fut_plot = np.array(
                    [
                        b if (b is not None and b > -10) else np.nan
                        for b in fut_boundaries
                    ]
                )
                ax.plot(look_ns, fut_plot, "k--", label="Futility Boundary")

    except Exception:
        # Fallback if engine fails or protocol is missing rules
        n_max = max(history_n) if history_n else 1
        pass

    # 2. Realized Trajectory
    plot_ns = [0] + history_n
    plot_zs = [0.0] + history_z
    ax.plot(plot_ns, plot_zs, "b.-", label="Z-Statistic")

    if history_n:
        ax.scatter(history_n, history_z, color="blue", zorder=5)

    # Styling
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Z-Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linestyle=":", alpha=0.3)

    # Info Time Axis (Secondary)
    def n_to_info(x: float) -> float:
        return x / n_max if n_max > 0 else 0.0

    def info_to_n(x: float) -> float:
        return x * n_max

    secax = ax.secondary_xaxis("top", functions=(n_to_info, info_to_n))  # type: ignore[arg-type]
    secax.set_xlabel("Information Time")

    return fig
