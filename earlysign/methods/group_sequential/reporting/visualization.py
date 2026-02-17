"""
Visualization utilities for Group Sequential Testing.
"""

from typing import Any, List, Optional, Tuple

import ibis
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from earlysign.schema.ES3.GST.Log import LookResult


def reconstruct_z_history(table: ibis.Expr) -> Tuple[List[int], List[float]]:
    """
    Generic helper to reconstruct Z-statistic history from LookResult records in the ledger.
    """
    history = reconstruct_full_history(table)
    return [lr.sample_n for lr in history], [lr.z_stat for lr in history]


def reconstruct_full_history(table: ibis.Expr) -> List[LookResult]:
    """
    Reconstructs full LookResult history from the ledger.
    """
    results_df = table.filter(table.type == "LookResult").execute()

    if results_df.empty:
        return []

    history = []
    for _, row in results_df.iterrows():
        payload = row["payload"]
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        lr = LookResult.model_validate(payload)
        history.append(lr)

    return history


def plot_gst_summary(
    protocol: Any,
    history_n: Optional[List[int]] = None,
    history_z: Optional[List[float]] = None,
    title: str = "GST Monitoring",
    adaptation_logs: List[Any] | None = None,
    full_history: Optional[List[Any]] = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """
    Generates a generic summary plot for Group Sequential Test results.
    Supports optional overlay of Adaptive Design events.
    """
    from earlysign.methods.group_sequential.execution.binomial import (
        BinomialGSTEngine,
    )

    # Resolve history from full_history if provided as trajectory data
    if full_history and isinstance(full_history, list) and len(full_history) > 0:
        if isinstance(full_history[0], tuple):
            # It's a list of (timestamp, LookResult)
            raw_history = [state for _, state in full_history]
        else:
            raw_history = full_history

        if history_n is None:
            history_n = [res.sample_n for res in raw_history]
        if history_z is None:
            history_z = [res.z_stat for res in raw_history]
    else:
        raw_history = []

    history_n = history_n or []
    history_z = history_z or []

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Planned Boundaries (Faint)
    try:
        engine = BinomialGSTEngine(protocol)
        timer = protocol.method.stopping_policy.timer
        n_max = 0
        if hasattr(timer, "max_sample_size"):
            n_max = timer.max_sample_size
        elif hasattr(timer, "max_events"):
            n_max = timer.max_events

        if n_max > 0:
            # Create a dense grid for faint planned boundaries
            dense_t = np.linspace(0.05, 1.0, 100)
            dense_n = dense_t * n_max
            eff_planned = []
            fut_planned = []

            for t in dense_t:
                eff_planned.append(engine.get_boundary_at_look(0, t, "efficacy"))
                fut_planned.append(engine.get_boundary_at_look(0, t, "futility"))

            if any(b is not None for b in eff_planned):
                eff_filt = [b if b is not None else np.nan for b in eff_planned]
                ax.plot(
                    dense_n,
                    eff_filt,
                    color="red",
                    linestyle="--",
                    alpha=0.2,
                    label="Planned Efficacy",
                )
            if any(b is not None for b in fut_planned):
                fut_filt = [
                    b if (b is not None and b > -10) else np.nan for b in fut_planned
                ]
                ax.plot(
                    dense_n,
                    fut_filt,
                    color="black",
                    linestyle="--",
                    alpha=0.2,
                    label="Planned Futility",
                )

    except Exception:
        n_max = max(history_n) if history_n else 1

    # 2. Realized Boundaries (Solid Segments)
    dx = n_max * 0.05  # Segment length: 5% of axis

    for res in raw_history:
        n = res.sample_n
        # Efficacy Segment
        if res.efficacy_boundary is not None:
            ax.hlines(
                y=res.efficacy_boundary,
                xmin=n - dx / 2,
                xmax=n + dx / 2,
                color="red",
                linewidth=2,
                label="Realized Boundary" if res == raw_history[0] else "",
            )
        # Futility Segment
        if res.futility_boundary is not None and res.futility_boundary > -10:
            ax.hlines(
                y=res.futility_boundary,
                xmin=n - dx / 2,
                xmax=n + dx / 2,
                color="black",
                linewidth=2,
            )

    # 2. Realized Trajectory
    plot_ns = [0] + history_n
    plot_zs = [0.0] + history_z
    ax.plot(plot_ns, plot_zs, "b.-", label="Z-Statistic")

    if history_n:
        ax.scatter(history_n, history_z, color="blue", zorder=5)

    # 3. Adaptation Overlay
    if adaptation_logs:
        for log in adaptation_logs:
            # Assuming history_n has length >= look.
            if log.look <= len(history_n):
                n_val = history_n[log.look - 1]
                z_val = history_z[log.look - 1]

                if (
                    hasattr(log, "promising_zone_status")
                    and log.promising_zone_status == "promising"
                ):
                    ax.annotate(
                        f"Promising Zone\nCP={log.conditional_power:.2f}",
                        xy=(n_val, z_val),
                        xytext=(n_val, z_val + 0.5),
                        arrowprops=dict(facecolor="orange", shrink=0.05),
                        fontsize=9,
                        color="orange",
                    )
                if (
                    hasattr(log, "recommended_sample_size")
                    and log.recommended_sample_size
                ):
                    if log.recommended_sample_size != log.original_sample_size:
                        ax.axvline(
                            x=log.recommended_sample_size,
                            color="green",
                            linestyle=":",
                            label="New N_max",
                        )

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
