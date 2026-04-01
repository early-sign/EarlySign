"""
Visualization utilities for Group Sequential Testing.
"""

import json
from typing import Any, List, Optional, Tuple, Union

import ibis
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import earlysign.schema.ES3.GST as GST
from earlysign.builtin.group_sequential.design.operating_characteristics.binomial import (
    BinomialOperatingCharacteristicsEvaluator,
)
from earlysign.builtin.group_sequential.design.operating_characteristics.continuous import (
    ContinuousOperatingCharacteristicsEvaluator,
)
from earlysign.builtin.group_sequential.design.operating_characteristics.engines import (
    SimulationCurve,
)
from earlysign.builtin.group_sequential.engine.engine import GroupSequentialEngine
from earlysign.parts.visualization.visual import VisualizationResult
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
) -> VisualizationResult:
    """
    Generates a generic summary plot for Group Sequential Test results.
    Supports optional overlay of Adaptive Design events.
    """

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

    fig = Figure(figsize=(10, 6), layout="constrained")
    ax = fig.subplots()

    # 1. Planned Boundaries (Faint)
    try:
        engine = GroupSequentialEngine(protocol)
        timer = protocol.method.stopping_policy.timer
        n_max = 0
        if hasattr(timer, "max_sample_size"):
            n_max_raw = timer.max_sample_size
            n_max = (
                sum(n_max_raw.values()) if isinstance(n_max_raw, dict) else n_max_raw
            )
        elif hasattr(timer, "max_events"):
            n_max_raw = timer.max_events
            n_max = (
                sum(n_max_raw.values()) if isinstance(n_max_raw, dict) else n_max_raw
            )

        if n_max > 0:
            # Create a dense grid for faint planned boundaries
            dense_t = np.linspace(0.05, 1.0, 100)
            dense_n = list(dense_t * n_max)
            eff_planned = []
            fut_planned = []

            for t in dense_t:
                eff_planned.append(
                    engine.get_boundary_at_look(
                        0, t, "efficacy", method="numerical_integration"
                    )
                )
                fut_planned.append(
                    engine.get_boundary_at_look(
                        0, t, "futility", method="numerical_integration"
                    )
                )

            if history_n and max(history_n) > n_max:
                dense_n.append(max(history_n))
                if eff_planned:
                    eff_planned.append(eff_planned[-1])
                if fut_planned:
                    fut_planned.append(fut_planned[-1])

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

            # Cap the Y-axis so early infinite Z-scores don't compress the plot
            ax.set_ylim(bottom=-5, top=6)

    except Exception:
        n_max = max(history_n) if history_n else 1

    # 2. Realized Boundaries (Solid Segments)
    plot_width = max(n_max, max(history_n) if history_n else 1)
    dx = plot_width * 0.05  # Segment length: 5% of axis

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
                label="Realized Boundary" if res == raw_history[0] else "",
            )

    # 2. Realized Trajectory
    plot_ns = history_n
    plot_zs = history_z
    if plot_ns:
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
    ax.set_xlim(left=0)

    # Info Time Axis (Secondary)
    def n_to_info(x: float) -> float:
        return x / n_max if n_max > 0 else 0.0

    def info_to_n(x: float) -> float:
        return x * n_max

    secax = ax.secondary_xaxis("top", functions=(n_to_info, info_to_n))  # type: ignore[arg-type]
    secax.set_xlabel("Information Time")

    return VisualizationResult(figure=fig)


def generate_operating_characteristics_table(results: SimulationCurve) -> pd.DataFrame:
    """Generates a summary DataFrame from a SimulationCurve with enhanced diagnostics."""
    rows = []
    n_max_total = results.n_max_total or 0.0
    n_fixed_total = results.n_fixed_total or 0.0

    for i, r in enumerate(results.results):
        x_val = results.x_values[i]

        # Use total N for summary
        ess_total = (
            (
                sum(r.expected_n_per_arm.values())
                if r.expected_n_per_arm
                else r.asn * n_max_total
            )
            if n_max_total > 0
            else (r.asn if r.asn is not None else 0.0)
        )

        # 1. Basic Stats & Parameter
        row = {
            "Effect Size (%)": x_val,
            "Power": r.power,
        }

        row["Expected N"] = ess_total

        # 3. Sequential vs Fixed Comparisons
        if n_fixed_total > 0:
            row["Fixed N (Ref)"] = n_fixed_total
            row["ESS / Fixed (%)"] = ess_total / n_fixed_total

        # 4. Capacity / Max N
        if n_max_total > 0:
            row["ESS / Max (%)"] = ess_total / n_max_total
            row["Max N (Total)"] = n_max_total

        # 5. Risk metrics
        if n_fixed_total > 0 and r.n_per_arm_schedule and r.prob_stop_total is not None:
            total_n_schedule = np.zeros_like(r.prob_stop_total)
            for arm_schedule in r.n_per_arm_schedule.values():
                total_n_schedule += arm_schedule
            exceed_mask = total_n_schedule > (n_fixed_total + 1e-6)
            prob_exceed = np.sum(r.prob_stop_total[exceed_mask])
            row["Prob > Fixed N"] = prob_exceed

        # 6. Stopping Breakdown (Pushed to end)
        if r.prob_stop_total is not None and len(r.prob_stop_total) > 0:
            for j, prob in enumerate(r.prob_stop_total):
                row[f"Prob Stop (Look {j+1})"] = prob

        rows.append(row)

    return pd.DataFrame(rows)


def visualize_protocol_design(
    protocol: GST.Protocol,
    effect_sizes: List[float],
    return_fig: bool = True,
) -> Union[pd.DataFrame, VisualizationResult]:
    """
    Visualizes the Operating Characteristics (OC) of a protocol design.

    Args:
        protocol: The design to evaluate.
        effect_sizes: List of effect sizes to evaluate (interpreted as %).

    Returns:
        VisualizationResult or pd.DataFrame:
            If return_fig=True, returns a VisualizationResult with "summary" (Styler) and "figure".
            If return_fig=False, returns only the summary (DataFrame).
    """
    # 1. Select Evaluator
    evaluator: Any
    metric_type: str
    if protocol.task.response_type == GST.ResponseType.BINARY:
        evaluator = BinomialOperatingCharacteristicsEvaluator(protocol, n_sims=10000)
        metric_type = "relative_lift_pct"
    else:
        evaluator = ContinuousOperatingCharacteristicsEvaluator(protocol, n_sims=10000)
        metric_type = "absolute_diff_pct"

    # 2. Evaluate Curve
    # We treat effect_sizes as percentage change relative to baseline for binomial
    curve = evaluator.evaluate_metric_at(effect_sizes, metric_type=metric_type)

    # 3. Build DataFrame
    df = generate_operating_characteristics_table(curve)

    format_dict = {
        "Power": "{:.1%}",
        "Expected N": "{:.1f}",
        "ESS / Fixed (%)": "{:.1%}",
        "ESS / Max (%)": "{:.1%}",
        "Prob > Fixed N": "{:.1%}",
        "Fixed N (Ref)": "{:.1f}",
        "Max N (Total)": "{:.1f}",
    }
    for col in df.columns:
        if col.startswith("Prob Stop (Look"):
            format_dict[col] = "{:.1%}"

    summary = df.style.format(format_dict)

    if not return_fig:
        return df

    # 4. Plot
    fig = Figure(figsize=(10, 6), layout="constrained")
    ax = fig.subplots()

    # ASN Bubble Curve
    ax.set_title(
        "Operating Characteristics: ESS vs Effect Size", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel(
        (
            "Relative Effect Size (%)"
            if metric_type == "relative_lift_pct"
            else "Effect Size"
        ),
        fontsize=12,
    )
    ax.set_ylabel("Expected Sample Size", fontsize=12)

    ax.plot(
        df["Effect Size (%)"],
        df["Expected N"],
        color="#1f77b4",
        linewidth=2.5,
        label="ESS (Total)",
        alpha=0.9,
    )

    for i, res in enumerate(curve.results):
        x_val = curve.x_values[i]

        ax.scatter(
            [x_val],
            [df["Expected N"].iloc[i]],
            color="#1f77b4",
            s=80,
            zorder=5,
            edgecolors="black",
        )

        if res.prob_stop_total is not None and len(res.prob_stop_total) > 0:
            sample_sizes = []
            if getattr(res, "n_per_arm_schedule", None):
                total_n_schedule = np.zeros_like(res.prob_stop_total)
                for arm_schedule in res.n_per_arm_schedule.values():
                    total_n_schedule += arm_schedule
                sample_sizes = total_n_schedule.tolist()
            else:
                max_look = len(res.prob_stop_total)
                n_total = 0
                if evaluator.n_max_per_arm:
                    n_total = sum(evaluator.n_max_per_arm.values())
                else:
                    n_total = 1000  # Fallback
                for k in range(1, max_look + 1):
                    sample_sizes.append(int(n_total * (k / max_look)))

            for analysis_idx, prob in enumerate(res.prob_stop_total):
                if prob == 0:
                    continue
                alpha_val = min(max(prob * 0.5, 0.02), 0.6)

                try:
                    n_total = sample_sizes[analysis_idx]
                except Exception:
                    n_total = sample_sizes[-1]

                ax.scatter(
                    [x_val],
                    [n_total],
                    color="#1f77b4",
                    s=max(20, 500 * prob),  # Scale size by probability
                    alpha=alpha_val,
                    edgecolors="none",
                    zorder=3,
                )

    # Mark max N and Fixed N Star
    max_n = sum(evaluator.n_max_per_arm.values())
    ax.axhline(max_n, color="#1f77b4", linestyle="--", alpha=0.3, label="Max N (Total)")

    fixed_n = getattr(curve, "n_fixed_total", None)
    if fixed_n and fixed_n > 0 and protocol.task.futility:
        # Approximate the effect size targeted by the design using the nearest power
        target_power = protocol.task.futility.power
        nearest_idx = (df["Power"] - target_power).abs().idxmin()
        target_eff = df.loc[nearest_idx, "Effect Size (%)"]

        ax.scatter(
            [target_eff],
            [fixed_n],
            color="#1f77b4",
            marker="*",
            s=400,
            label="Fixed Design (Total)",
            edgecolors="black",
            zorder=4,
        )

    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=10,
    )

    return VisualizationResult(
        summary=summary,
        figure=fig,
    )
