from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
import pandas as pd

# lazy import for plotting:
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.plan.operating_characteristics.binomial import (
    BinomialABOperatingCharacteristicsEvaluator,
)
from earlysign.v1.methods.group_sequential.plan.operating_characteristics.engines import (
    EvaluationResult,
    OperatingCharacteristicsEvaluator,
    SimulationCurve,
)

# Internal rendering defaults
_DEFAULT_GRID_POINTS = 100


class OCCurvePlotter:
    """Plotter for compute_oc_curve() results (List[EvaluationResult])."""

    def __init__(
        self,
        figsize: tuple[int, int] = (10, 6),
        colors: Optional[List[str]] = None,
        linestyles: Optional[List[str]] = None,
    ) -> None:
        self.figsize = figsize
        self.colors = colors or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        self.linestyles = linestyles or ["-", "--", "-.", ":"]

    def plot_oc_curve(
        self,
        results: List[EvaluationResult],
        x_values: List[float],
        target_effect: Optional[float] = None,
        null_value: float = 0.0,
        effect_label: str = "Effect Size",
        plot_options: Optional[Dict[str, Any]] = None,
        arms_to_plot: Optional[List[str]] = None,  # List of arm names or "Total"
        n_max_per_arm: Optional[Dict[str, int]] = None,
        n_fixed_per_arm: Optional[Dict[str, float]] = None,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Plot ESS vs effect size and overlay stopping distribution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting.")

        if not results:
            raise ValueError("`results` must be a non-empty list of EvaluationResult")

        # Sort by x_value
        paired = sorted(zip(x_values, results), key=lambda p: float(p[0]))
        sorted_x = np.array([p[0] for p in paired], dtype=float)
        sorted_res = [p[1] for p in paired]

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Determine which targets to plot
        available_arms = []
        if results and results[0].expected_n_per_arm:
            available_arms = list(results[0].expected_n_per_arm.keys())

        if arms_to_plot is None:
            # Default: Everything available
            targets = ["Total"] + available_arms
        else:
            targets = arms_to_plot

        for i, target in enumerate(targets):
            color = self.colors[i % len(self.colors)]
            linestyle = (
                self.linestyles[0]
                if target == "Total"
                else self.linestyles[(i + 1) % len(self.linestyles)]
            )

            if target == "Total":

                def get_ess(r: "EvaluationResult") -> float:
                    return (
                        sum(r.expected_n_per_arm.values())
                        if r.expected_n_per_arm
                        else (
                            r.asn * sum(n_max_per_arm.values())
                            if n_max_per_arm
                            else (r.expected_sample_size or r.asn)
                        )
                    )

                def get_schedule(r: "EvaluationResult") -> NDArray[np.float64]:
                    times = (
                        r.info_times if r.info_times is not None else np.array([1.0])
                    )
                    if r.n_per_arm_schedule:
                        return cast(
                            NDArray[np.float64],
                            np.array(list(r.n_per_arm_schedule.values())).sum(axis=0),
                        )
                    return (
                        times * sum(n_max_per_arm.values()) if n_max_per_arm else times
                    )

                ref_max_n = sum(n_max_per_arm.values()) if n_max_per_arm else None
                ref_fixed_n = sum(n_fixed_per_arm.values()) if n_fixed_per_arm else None
            else:
                arm_name = target

                def get_ess(r: "EvaluationResult") -> float:
                    return (
                        r.expected_n_per_arm[arm_name]
                        if r.expected_n_per_arm and arm_name in r.expected_n_per_arm
                        else (
                            r.asn * n_max_per_arm[arm_name]
                            if n_max_per_arm and arm_name in n_max_per_arm
                            else r.asn
                        )
                    )

                def get_schedule(r: "EvaluationResult") -> NDArray[np.float64]:
                    times = (
                        r.info_times if r.info_times is not None else np.array([1.0])
                    )
                    if r.n_per_arm_schedule and arm_name in r.n_per_arm_schedule:
                        return r.n_per_arm_schedule[arm_name]
                    return (
                        times * n_max_per_arm[arm_name]
                        if (n_max_per_arm and arm_name in n_max_per_arm)
                        else times
                    )

                ref_max_n = (
                    n_max_per_arm[arm_name]
                    if (n_max_per_arm and arm_name in n_max_per_arm)
                    else None
                )
                ref_fixed_n = (
                    n_fixed_per_arm[arm_name]
                    if (n_fixed_per_arm and arm_name in n_fixed_per_arm)
                    else None
                )

            # 1. Plot ESS curve
            ess = [get_ess(r) for r in sorted_res]
            ax.plot(
                sorted_x + null_value,
                ess,
                label=f"ESS ({target})",
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                zorder=5,
            )

            # 2. Add Bubbles (Stopping Distribution) - only for "Total" or if single target to avoid clutter
            if target == "Total" or len(targets) == 1:
                bubble_color = color
                for j, r in enumerate(sorted_res):
                    eff = sorted_x[j] + null_value
                    ax.scatter(
                        [eff],
                        [ess[j]],
                        color=bubble_color,
                        s=80,
                        edgecolors="black",
                        zorder=7,
                    )

                    ns = get_schedule(r)
                    probs = r.prob_stop_total
                    for n, p in zip(ns, probs):
                        if p > 0.005:
                            ax.scatter(
                                [eff],
                                [n],
                                s=p * 1500,
                                color=bubble_color,
                                alpha=0.15,
                                edgecolors=bubble_color,
                                zorder=4,
                            )

            # 3. Draw Reference Markers
            if ref_max_n is not None:
                ax.axhline(
                    y=ref_max_n,
                    color=color,
                    linestyle="--",
                    alpha=0.3,
                    linewidth=1,
                    label=f"Max N ({target})",
                )

            if target_effect is not None and ref_fixed_n is not None:
                ax.scatter(
                    [target_effect + null_value],
                    [ref_fixed_n],
                    marker="*",
                    color=color,
                    s=350,
                    edgecolors="black",
                    zorder=8,
                    label=f"Fixed Design ({target})",
                )
                ax.axhline(
                    y=ref_fixed_n,
                    color=color,
                    linestyle=":",
                    alpha=0.4,
                    linewidth=1.5,
                )

        ax.set_title(
            "Operating Characteristics: ESS vs Effect Size",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(effect_label, fontsize=12)
        ax.set_ylabel("Expected Sample Size", fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
        ax.set_ylim(bottom=0)

        return ax


def get_evaluator_for_task(
    protocol: GST.Protocol,
    method: Literal["simulation", "numerical_integration"] = "simulation",
    n_sims: int = 5000,
    seed: int = 42,
) -> OperatingCharacteristicsEvaluator:
    """
    Factory to get the appropriate OC evaluator based on the protocol task.
    """

    task = protocol.task
    if task.response_type == GST.ResponseType.BINARY:
        return BinomialABOperatingCharacteristicsEvaluator(
            protocol, method=method, n_sims=n_sims, seed=seed
        )

    raise NotImplementedError(
        f"Operating Characteristics evaluation not implemented for response type: {task.response_type}"
    )


def plot_design_characteristics(
    protocol: GST.Protocol,
    num_points: int = 50,
    arms_to_plot: Optional[List[str]] = None,
    n_sims: int = 5000,
    seed: int = 42,
) -> Axes:
    """Plots the ASN and sampling distribution for a Group Sequential Design."""
    evaluator = get_evaluator_for_task(
        protocol, method="simulation", n_sims=n_sims, seed=seed
    )

    if not isinstance(evaluator, BinomialABOperatingCharacteristicsEvaluator):
        raise NotImplementedError(
            "plot_design_characteristics only supports Binomial tasks currently."
        )

    curve = evaluator.evaluate_lift_curve(
        range_min=-0.5,
        range_max=1.0,
        n_points=num_points,
        metric_type="relative_lift_pct",
    )

    plotter = OCCurvePlotter()
    ax = plotter.plot_oc_curve(
        curve.results,
        x_values=list(curve.x_values),
        target_effect=curve.target_x_value,
        null_value=curve.null_x_value if curve.null_x_value else 0.0,
        effect_label="Relative Lift (%)",
        arms_to_plot=arms_to_plot,
        n_max_per_arm=curve.n_max_per_arm,
        n_fixed_per_arm=curve.n_fixed_per_arm,
    )

    return ax


def plot_operating_characteristics(
    result: SimulationCurve,
    ax: Optional[Axes] = None,
    figsize: tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Expected Sample Size (ESS)",
    colors: Optional[List[str]] = None,
    show_bubbles: bool = True,
) -> Axes:
    """
    Plots the Operating Characteristics (ASN) curve and stopping probability bubbles.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = colors or ["#1f77b4", "#ff7f0e"]
    gst_color = colors[0]

    # 1. Plot ASN Curve
    # SimulationCurve has 'results' list and 'x_values' array.
    asn_values = [
        r.expected_sample_size if r.expected_sample_size is not None else r.asn
        for r in result.results
    ]
    x_axis = result.x_values

    ax.plot(
        x_axis, asn_values, label="GST: ESS", linewidth=2, color=gst_color, zorder=3
    )

    # 2. Bubbles
    if show_bubbles and result.n_max:
        for i, r in enumerate(result.results):
            x_val = x_axis[i]
            probs = r.prob_stop_total

            # Infer Ns at looks: Use precise sizes if available, else approximate
            if r.sample_sizes is not None:
                ns = r.sample_sizes
            else:
                k = len(probs)
                ns = np.linspace(result.n_max / k, result.n_max, k).astype(int)

            for look_idx, p in enumerate(probs):
                if p > 0.001:
                    size = 30 + p * 150
                    alpha_val = min(0.8, p * 4 + 0.15)

                    ax.scatter(
                        [x_val],
                        [ns[look_idx]],
                        s=size,
                        color=gst_color,
                        alpha=alpha_val,
                        edgecolors="none",
                        zorder=2,
                    )

    # Styling
    ax.set_ylabel(ylabel, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    elif result.metric_type == "relative_lift_pct":
        ax.set_xlabel("Relative Lift (%)", fontsize=12)
    elif result.metric_type == "absolute_diff_pct":
        ax.set_xlabel("Absolute Difference (%pt)", fontsize=12)

    cur_ylim = ax.get_ylim()
    # Handle optional n_max
    top_limit = result.n_max * 1.1 if result.n_max else max(cur_ylim[1], 100)
    ax.set_ylim(bottom=0, top=top_limit)

    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if result.null_x_value is not None:
        ax.axvline(
            result.null_x_value,
            color="gray",
            linestyle=":",
            label=f"Null ({result.null_x_value:.1f})",
        )

    if result.target_x_value is not None:
        ax.axvline(
            result.target_x_value,
            color="green",
            linestyle="--",
            label=f"Target ({result.target_x_value:.3f})",
        )

    if result.n_max:
        ax.axhline(result.n_max, color="orange", linestyle="--", label="Max N")

    ax.legend(loc="best")
    return ax


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
        # Note: If n_max_total is 0, we fallback to showing ASN (0-1) if absolute data is missing.
        # This is better than guessing 1000.

        # 1. Basic Stats & Parameter
        row = {
            "Rel. Lift (%) : (pt - pc) / pc": x_val,
            "Power / Rejection Prob": f"{r.power:.2%}",
        }

        # 2. Stopping Breakdown (Placed before ESS as requested)
        early_stop_prob = (
            np.sum(r.prob_stop_total[:-1]) if len(r.prob_stop_total) > 1 else 0.0
        )
        row["Early Stop Prob"] = f"{early_stop_prob:.1%}"
        row["Stop (Efficacy)"] = f"{np.sum(r.prob_stop_efficacy):.1%}"
        row["Stop (Futility)"] = f"{np.sum(r.prob_stop_futility):.1%}"

        # 3. Total ESS
        row["Total ESS"] = f"{ess_total:.1f}"

        # 4. Sequential vs Fixed Comparisons
        if n_fixed_total > 0:
            row["Fixed N (Ref)"] = f"{n_fixed_total:.1f}"
            row["ESS / Fixed N (%)"] = f"{ess_total / n_fixed_total:.1%}"

        # 5. Capacity / Max N (Placed right after Efficiency)
        if n_max_total > 0:
            row["ESS / Max (%)"] = f"{ess_total / n_max_total:.1%}"
            row["Max N (Total)"] = f"{n_max_total:.1f}"

        # 6. Risk metrics
        if n_fixed_total > 0 and r.n_per_arm_schedule and r.prob_stop_total is not None:
            total_n_schedule = np.zeros_like(r.prob_stop_total)
            for arm_schedule in r.n_per_arm_schedule.values():
                total_n_schedule += arm_schedule
            exceed_mask = total_n_schedule > (n_fixed_total + 1e-6)
            prob_exceed = np.sum(r.prob_stop_total[exceed_mask])
            row["Prob > Fixed N"] = f"{prob_exceed:.1%}"

        rows.append(row)

    return pd.DataFrame(rows)


def visualize_protocol_design(
    protocol: GST.Protocol,
    effect_sizes: List[float],
    plot: bool = True,
    method: Literal["simulation", "numerical_integration"] = "simulation",
    n_sims: int = 5000,
    seed: int = 42,
    arms_to_plot: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Visualizes a given protocol design (Table and Plot).

    Args:
        protocol: The GST Protocol to visualize.
        effect_sizes: Specific effect sizes (relative lift %) to evaluate.
        plot: Whether to generate a plot.
        method: Evaluation method.
        simulation_kwargs: Optional kwargs for simulation (e.g. n_sims, seed).
        plot_arm: Which arm (or "Total") to plot sample sizes for.

    Returns:
        Dictionary with 'summary' (DataFrame) and 'figure' (matplotlib Figure or None).
    """
    evaluator = get_evaluator_for_task(
        protocol, method=method, n_sims=n_sims, seed=seed
    )

    # Point Results for Table and Plot
    if isinstance(evaluator, BinomialABOperatingCharacteristicsEvaluator):
        point_results = evaluator.evaluate_lift_at(
            effect_sizes_pct=effect_sizes,
            metric_type="relative_lift_pct",
        )
    else:
        # Fallback for future evaluators
        raise NotImplementedError(
            f"Explicit point evaluation not implemented for evaluator: {type(evaluator)}"
        )

    # Use the evaluated results for the table
    df = generate_operating_characteristics_table(point_results)

    # 2. Generate Plot
    fig = None
    if plot:
        plotter = OCCurvePlotter()

        # Update effect label based on task
        effect_label = "Effect Size"
        if protocol.task.response_type == GST.ResponseType.BINARY:
            effect_label = "Rel. Lift (%) : (pt - pc) / pc"

        ax = plotter.plot_oc_curve(
            point_results.results,
            x_values=list(point_results.x_values),
            target_effect=point_results.target_x_value,
            null_value=(
                point_results.null_x_value if point_results.null_x_value else 0.0
            ),
            effect_label=effect_label,
            arms_to_plot=arms_to_plot,
            n_max_per_arm=point_results.n_max_per_arm,
            n_fixed_per_arm=point_results.n_fixed_per_arm,
        )
        fig = ax.figure
        try:
            import matplotlib.pyplot as plt

            if fig is not None:
                from matplotlib.figure import Figure

                if isinstance(fig, Figure):
                    plt.close(cast(Any, fig))
        except ImportError:
            pass

    return {"summary": df, "figure": fig}


def visualize_design_from_params(
    alpha: float = 0.05,
    power: float = 0.8,
    delta: float = 0.1,
    control_rate: float = 0.1,
    looks: int = 4,
    spending_function: str = "obrien_fleming",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Plans and visualizes a Binomial design from parameters.
    """
    from earlysign.v1.methods.group_sequential.plan.protocol_design import (
        ProtocolDesigner,
    )
    from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
        CanonicalJointModel,
        Config,
    )

    designer = ProtocolDesigner(
        model=CanonicalJointModel(Config(info_times=np.array([1.0])))
    )

    protocol_obj = designer.plan_binomial_ab(
        alpha=alpha,
        power=power,
        delta=delta,
        k=looks,
        p_control=control_rate,
        # spending_fn=..., # plan_binomial_ab handles default if None
    )

    return visualize_protocol_design(
        protocol_obj, effect_sizes=[-10, 0, 10, 20], plot=plot
    )
