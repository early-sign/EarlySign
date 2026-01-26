from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# lazy import for plotting:
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.figure import Figure
if "Figure" not in globals():
    from typing import Any

    Figure = Any  # Placeholder type hint if matplotlib missing

# For type hinting Axex
if "Axes" not in globals():
    Axes = Any

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.plan.simulation import (
    OperatingCharacteristicsResult,
    calculate_stopping_probabilities,
)
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
)


@dataclass
class OCPointResult:
    """Operating characteristics at a single effect size.

    Attributes:
        effect_size: effect size evaluated (absolute difference or as configured)
        expected_sample_size: expected sample size (ASN)
        power: estimated power (probability of rejection)
        max_sample_size: maximum sample size observed in simulations
        stop_distribution: mapping from actual total sample size (int) at stop time
            to counts.
        metadata: optional additional info
    """

    effect_size: float
    expected_sample_size: float
    power: float
    max_sample_size: float
    stop_distribution: Dict[int, int]
    metadata: Optional[Dict[str, Any]] = None


class OCCurvePlotter:
    """Plotter for compute_oc_curve() results (List[OCPointResult]).

    Ported from v0 to maintain visual consistency.
    """

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
        results: List[OCPointResult],
        target_effect: Optional[float] = None,
        null_value: float = 0.0,
        effect_label: str = "Effect Size",
        ax: Optional[Axes] = None,
        plot_options: Optional[Dict[str, Any]] = None,
    ) -> Axes:
        """Plot ESS vs effect size and overlay stopping distribution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting.")

        if not results:
            raise ValueError("`results` must be a non-empty list of OCPointResult")

        # Sort by effect_size to ensure monotonic curve
        results_sorted = sorted(results, key=lambda r: float(r.effect_size))

        effect_sizes = np.array([r.effect_size for r in results_sorted], dtype=float)
        ess_values = np.array(
            [r.expected_sample_size for r in results_sorted], dtype=float
        )

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

        # For each effect point, draw the ESS marker and stopping-distribution of sample sizes
        for r in results_sorted:
            eff = float(r.effect_size) + null_value
            # ESS scatter marker (Mean)
            ax.scatter(
                [eff],
                [r.expected_sample_size],
                color=gst_color,
                s=64,
                zorder=5,
                edgecolors="black",
            )

            # Stop-distribution: Bubbles
            stop_dist = r.stop_distribution or {}
            if isinstance(stop_dist, dict) and stop_dist:
                total = (
                    float(sum(stop_dist.values()))
                    if sum(stop_dist.values()) > 0
                    else 1.0
                )
                for raw_key, count in stop_dist.items():
                    n_total = int(raw_key)
                    prob = float(count) / total
                    # Bubble size proportional to prob
                    size = max(30, min(300, int(prob * 600)))
                    # Opacity proportional to prob
                    alpha_val = min(max(prob * 0.8, 0.05), 0.9)

                    ax.scatter(
                        [eff],
                        [n_total],
                        color=gst_color,
                        s=size,
                        alpha=alpha_val,
                        edgecolors="black",
                        zorder=6,
                    )

        # Optionally mark target effect and draw FSD marker/horizontal line
        if target_effect is not None:
            ax.axvline(
                x=target_effect + null_value,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label=f"Target: {target_effect + null_value:.4f}",
            )

            # Find result closest to target to check for FSD metadata
            eff_arr = np.array(
                [float(r.effect_size) for r in results_sorted], dtype=float
            )
            idx_closest = int(np.argmin(np.abs(eff_arr - float(target_effect))))
            r_closest = results_sorted[idx_closest]

            md = r_closest.metadata or {}
            planned_max_n = md.get("planned_max_n")
            if planned_max_n is not None:
                ax.scatter(
                    [target_effect + null_value],
                    [int(planned_max_n)],
                    marker="*",
                    color=fsd_color,
                    s=350,
                    edgecolors="black",
                    zorder=8,
                    label="Fixed Design",
                )
                ax.axhline(
                    y=int(planned_max_n),
                    color=fsd_color,
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1.5,
                )

        ax.legend(loc="best", fontsize=10)

        # Ensure Y starts at 0 unless overridden
        if plot_options and plot_options.get("ylim"):
            ax.set_ylim(plot_options.get("ylim"))
        else:
            cur_ylim = ax.get_ylim()
            ax.set_ylim(bottom=0, top=max(cur_ylim[1], 10))

        if created_fig is not None:
            created_fig.tight_layout()

        return ax


def plot_design_characteristics(
    protocol: GST.Protocol,
    effect_sizes: Optional[List[float]] = None,
    num_points: int = 50,
    control_proportion: Optional[float] = None,
) -> Figure:
    """
    Plots the ASN and sampling distribution for a Group Sequential Design.

    Args:
        protocol: The GS protocol to analyze.
        effect_sizes: Optional list of effect sizes to plot.
                      If None, defaults to [0, 1.5 * target_effect].
        num_points: Number of points to plot on X-axis.
        control_proportion: Baseline proportion.

    Returns:
        Matplotlib Figure containing the plot.
    """

    # 1. Extract Design Constants
    method = protocol.method
    task = protocol.task

    if not isinstance(task.efficacy, GST.EfficacyRequirement):
        raise ValueError("Protocol must have efficacy requirement")

    timer = method.stopping_policy.timer
    if not isinstance(timer, GST.SampleSizeTimer):
        raise ValueError("Only SampleSizeTimer supported")
    n_max = timer.max_sample_size

    schedule = method.stopping_policy.schedule
    if isinstance(schedule, GST.FixedSchedule):
        info_sim = np.array(schedule.analyses)
    elif isinstance(schedule, GST.EquidistantSchedule):
        k = schedule.n_looks
        info_sim = np.linspace(1 / k, 1.0, k)
    else:
        raise ValueError("Unsupported schedule")

    k = len(info_sim)
    ns = np.ceil(info_sim * n_max).astype(int)

    # 2. Derive Target Drift and Effect Sizes
    p_c = 0.5
    target_delta = 0.5

    if isinstance(task.hypotheses.target_effect, GST.BinaryEffectSize):
        props = task.hypotheses.target_effect.proportions
        vals = list(props.values())

        # Control Proportion
        if control_proportion is None:
            if "control" in props:
                p_c = props["control"]
            else:
                p_c = vals[0] if vals else 0.5
        else:
            p_c = control_proportion

        # Target Delta
        if len(vals) >= 2:
            target_delta = abs(vals[1] - vals[0])
            if "treatment" in props and "control" in props:
                target_delta = abs(props["treatment"] - props["control"])
        else:
            target_delta = 0.1

    sigma = np.sqrt(p_c * (1 - p_c))
    i_max = n_max / (4 * sigma**2)
    target_drift = target_delta * np.sqrt(i_max)

    # 3. Solve Boundaries with CORRECT Drift
    model = CanonicalJointModel.from_spec(protocol)
    upper_bounds, lower_bounds = model.solve_boundaries(drift=target_drift)

    if upper_bounds is None:
        upper_bounds = np.full(k, 10.0)
    if lower_bounds is None:
        lower_bounds = np.full(k, -10.0)

    # 4. Generate Plot Points
    if effect_sizes is None:
        es_linspace = np.linspace(0, 1.5 * target_delta, num_points)
    else:
        es_linspace = np.array(effect_sizes)

    oc_results = []

    for delta in es_linspace:
        drift = delta * np.sqrt(i_max)

        res = calculate_stopping_probabilities(
            info_times=info_sim, upper=upper_bounds, lower=lower_bounds, drift=drift
        )
        probs = res["probs"]

        asn = np.sum(probs * ns)

        # Stop distribution for bubbles
        stop_dist = {}
        for i, p in enumerate(probs):
            if p > 0:
                count = int(p * 1000)
                if count > 0:
                    stop_dist[ns[i]] = count

        # Approx Power = prob of stopping early + prob of stopping at final look crossing upper
        # Since we don't have separate upper/lower probs in the loop easily, we skip explicit power for now.
        power_val = 0.0

        oc_results.append(
            OCPointResult(
                effect_size=delta,
                expected_sample_size=asn,
                power=power_val,
                max_sample_size=n_max,
                stop_distribution=stop_dist,
                metadata={"planned_max_n": n_max},
            )
        )

    # 5. Plot
    plotter = OCCurvePlotter(figsize=(10, 6))
    plotter.plot_oc_curve(
        oc_results,
        target_effect=target_delta,
        null_value=0.0,
        effect_label="Effect Size",  # Changed from "Effect Size (Absolute)"
    )


def plot_operating_characteristics(
    result: OperatingCharacteristicsResult,
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

    Ported and Refined from GSD Demo Notebook logic.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = colors or ["#1f77b4", "#ff7f0e"]  # GST Blue, FSD Orange/Gray?
    gst_color = colors[0]

    # 1. Plot ASN Curve
    asn_values = [r["asn"] for r in result.results]
    x_axis = result.x_values

    ax.plot(
        x_axis, asn_values, label="GST: ESS", linewidth=2, color=gst_color, zorder=3
    )

    # 2. Plot Stopping Distribution Bubbles (Compressed Scaling)
    if show_bubbles:
        for item in result.results:
            # item has {p_t, asn, probs, ns}
            # We need to map p_t to x_value.
            # Ideally the result structure would have x_val directly,
            # but simulation returns result dicts keyed by p_t.
            # We can re-derive or rely on x_values array index if synchronized.
            # Let's re-derive X coordinate for safety or assume 1-to-1 mapping if sorted.

            p_t = item["p_t"]
            ns = item["ns"]
            probs = item["probs"]

            # Map p_t to X logic (same as simulation.py)
            p_c = result.p_control
            metric = result.metric_type

            if metric == "relative_lift_pct":
                x_val = (p_t / p_c - 1) * 100
            elif metric == "absolute_diff_pct":
                x_val = (p_t - p_c) * 100
            else:
                x_val = p_t - p_c

            for i, p in enumerate(probs):
                if p > 1e-4:
                    # Compressed Scaling: Min ~30, Max ~180
                    size = 30 + p * 150
                    alpha_val = min(0.8, p * 4 + 0.15)

                    ax.scatter(
                        [x_val],
                        [ns[i]],
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

    # Y-axis start at 0
    cur_ylim = ax.get_ylim()
    ax.set_ylim(bottom=0, top=max(cur_ylim[1], result.n_max * 1.1))

    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Plot formatting lines
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

    ax.axhline(result.n_max, color="orange", linestyle="--", label="Max N")

    ax.legend(loc="best")
    return ax


def overlay_fixed_design_reference(
    ax: Axes,
    x_value: float,
    sample_size: int,
    label: str = "Fixed Design N",
    color: str = "gray",
    marker: str = "*",
    marker_size: int = 300,
) -> None:
    """
    Overlays a Fixed Design reference point (Star) and horizontal line.
    """
    # Plot Star
    ax.scatter(
        [x_value],
        [sample_size],
        marker=marker,
        s=marker_size,
        color=color,
        label=label,
        zorder=4,
    )
    # Plot Line
    ax.axhline(sample_size, color=color, linestyle="--", alpha=0.6, zorder=1)

    # Update legend to include new item
    # (Matplotlib update often requires re-calling legend, or passing handlers)
    ax.legend(loc="best")
