"""
GST Design Tutorial - Operating Characteristics Curves

This demonstrates the operating characteristics (expected sample size vs. effect size)
for different group sequential designs.

Plot A (Scenario A): Fixed Max N
- One point: Fixed design at target (δ, power)
- Curves: k-interim GST with same max N, showing ESS across various p1 values

Plot B (Scenario B): Fixed Power at δ
- Horizontal line: Fixed design ESS across various p1 values
- Curves: k-interim GST with increased max N to maintain power at δ, showing ESS across various p1
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab
from earlysign.stats.design.gst.common.types import InformationSpacing

# Fix random seed for reproducibility
np.random.seed(42)


@dataclass
class ESSCurveResult:
    """Result of computing ESS curve across effect sizes."""

    p1_values: np.ndarray  # Treatment proportions
    ess_values: np.ndarray  # Expected sample sizes
    power_values: np.ndarray  # Statistical power values
    stop_dists: list  # List of stopping distributions (dict per p1)


def calculate_sample_size_two_proportions(p0, p1, alpha=0.05, power=0.8):
    """Calculate required sample size per group for two-proportion z-test.

    Args:
        p0: Control proportion
        p1: Treatment proportion
        alpha: Type I error rate (two-sided)
        power: Target statistical power

    Returns:
        Required sample size per group (integer)
    """
    delta = p1 - p0

    # Standard errors
    se_alt = np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))

    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Sample size formula
    n = ((z_alpha + z_beta) * se_alt / delta) ** 2

    return int(np.ceil(n))


def compute_ess_curve(spec_template, p0, p1_values, n_per_analysis) -> ESSCurveResult:
    """Compute expected sample size curve across different effect sizes.

    Args:
        spec_template: Template ProportionsDesignSpec with design parameters
        p0: Control proportion
        p1_values: Array of treatment proportions to evaluate
        n_per_analysis: Sample size per analysis (determines max N)

    Returns:
        ESSCurveResult with p1_values, ess_values, power_values, and stop_dists
    """
    ess_values = []
    power_values = []
    stop_dists = []

    for p1 in p1_values:
        spec = ProportionsDesignSpec()
        spec.test.alpha = spec_template.test.alpha
        spec.sequential.n_analyses = spec_template.sequential.n_analyses
        spec.sequential.info_times = spec_template.sequential.info_times
        spec.sequential.info_spacing = spec_template.sequential.info_spacing
        spec.sample_size.n_per_analysis = n_per_analysis
        spec.effect.p_control = p0
        spec.effect.effect_size = p1 - p0

        lab = DesignLab(spec)
        lab.compute_boundaries().run_simulations()

        if lab.simulation_results:
            ess = lab.simulation_results["expected_sample_size"]
            power = lab.simulation_results["power"]

            # Extract and save stopping distribution
            stop_dist = lab.simulation_results.get("stop_distribution", {})
            stop_dists.append(stop_dist)
        else:
            # If no simulation, use max N
            summary = lab.get_summary()
            max_n_per_group = summary["N Control"].iloc[-1]
            ess = max_n_per_group * 2
            power = 0.0
            stop_dists.append({})  # Empty stop distribution

        ess_values.append(ess)
        power_values.append(power)

    return ESSCurveResult(
        p1_values=p1_values,
        ess_values=np.array(ess_values),
        power_values=np.array(power_values),
        stop_dists=stop_dists,
    )


@pytest.mark.basic
@pytest.mark.parametrize(
    "alpha, power, p0, effect_size_target",
    [
        (0.05, 0.8, 0.2, 0.1),
    ],
)
def test_gst_operating_characteristics(alpha, power, p0, effect_size_target):
    """GST operating characteristics curves across effect sizes."""

    print("\n" + "=" * 80)
    print("GST OPERATING CHARACTERISTICS")
    print("=" * 80)

    p1_target = p0 + effect_size_target

    # 1. Fixed design at target effect size
    print("\n1. Reference Fixed Design")
    print(f"   Target: alpha={alpha}, power={power}, p0={p0}, p1={p1_target}")

    n_fixed_correct = calculate_sample_size_two_proportions(p0, p1_target, alpha, power)
    n_fixed_total_correct = n_fixed_correct * 2

    print(f"   Calculated n per group: {n_fixed_correct}")
    print(f"   Total N: {n_fixed_total_correct}")

    # Verify power at target
    spec_fixed = ProportionsDesignSpec()
    spec_fixed.test.alpha = alpha
    spec_fixed.sequential.n_analyses = 1
    spec_fixed.sample_size.n_per_analysis = n_fixed_correct
    spec_fixed.effect.p_control = p0
    spec_fixed.effect.effect_size = effect_size_target

    lab_fixed = DesignLab(spec_fixed)
    lab_fixed.compute_boundaries().run_simulations()

    fixed_power_at_target = (
        lab_fixed.simulation_results["power"] if lab_fixed.simulation_results else power
    )
    print(f"   Verified power at p1={p1_target}: {fixed_power_at_target:.3f}")

    # 2. Define range of effect sizes to evaluate
    # From slightly above null (p0) to substantial effect
    p1_min = p0 + 0.02  # Minimum detectable
    p1_max = min(p0 + 0.35, 0.95)  # Extended range to see full OC curve
    p1_values = np.linspace(
        p1_min, p1_max, 50
    )  # More points for smoother quantile curves

    print("\n2. Evaluating operating characteristics")
    print(f"   p1 range: {p1_min:.3f} to {p1_max:.3f}")

    # 3. Scenario A: Fixed max N
    print(f"\n3. Scenario A: Fixed Max N = {n_fixed_total_correct}")

    n_interim_list = [1, 2]
    gst_curves_a = {}

    # Fixed design curve (for reference, though it's constant ESS)
    result_fixed = compute_ess_curve(spec_fixed, p0, p1_values, n_fixed_correct)
    gst_curves_a["fixed"] = {
        "p1": result_fixed.p1_values,
        "ess": result_fixed.ess_values,
        "n_interim": 0,
        "n_looks": 1,
    }

    for n_interim in n_interim_list:
        n_looks = n_interim + 1
        info_times = np.linspace(0, 1, n_looks + 1)[1:].tolist()

        # Calculate n_per_analysis to achieve approximately the same max N
        # For n_fixed_correct=291:
        # - 2 looks: 291/2 = 145.5 → use 146 to get 292 per group (closer to 291)
        # - 3 looks: 291/3 = 97 → use 97 to get 291 per group (exact)
        n_per_analysis_floor = int(n_fixed_correct / n_looks)
        n_per_analysis_ceil = n_per_analysis_floor + 1

        # Choose the one that gives max N closer to fixed design
        max_n_floor = n_per_analysis_floor * n_looks
        max_n_ceil = n_per_analysis_ceil * n_looks

        # Prefer ceil if the difference is equal or ceil is closer
        if abs(max_n_ceil - n_fixed_correct) <= abs(max_n_floor - n_fixed_correct):
            n_per_analysis = n_per_analysis_ceil
        else:
            n_per_analysis = n_per_analysis_floor

        spec_template = ProportionsDesignSpec()
        spec_template.test.alpha = alpha
        spec_template.sequential.n_analyses = n_looks
        spec_template.sequential.info_times = info_times
        spec_template.sequential.info_spacing = InformationSpacing.CUSTOM

        result = compute_ess_curve(spec_template, p0, p1_values, n_per_analysis)

        # Find power and ESS at target p1
        idx_target = np.argmin(np.abs(result.p1_values - p1_target))
        power_at_target = result.power_values[idx_target]
        ess_at_target = result.ess_values[idx_target]
        max_n_total = n_per_analysis * n_looks * 2

        gst_curves_a[f"{n_interim}_interim"] = {
            "p1": result.p1_values,
            "ess": result.ess_values,
            "power": result.power_values,
            "stop_dists": result.stop_dists,
            "n_interim": n_interim,
            "n_looks": n_looks,
            "n_per_analysis": n_per_analysis,
        }

        print(f"   {n_interim} interim ({n_looks} looks):")
        print(f"     n_per_analysis={n_per_analysis}, Max N={max_n_total}")
        print(f"     Power at p1={p1_target:.2f}: {power_at_target:.3f}")
        print(f"     ESS at p1={p1_target:.2f}: {ess_at_target:.1f}")

    # 4. Scenario B: Fixed power at target δ
    print(f"\n4. Scenario B: Fixed Power ≈ {power} at p1={p1_target}")

    gst_curves_b = {}

    # Fixed design curve (same as before)
    gst_curves_b["fixed"] = gst_curves_a["fixed"]

    for n_interim in n_interim_list:
        n_looks = n_interim + 1
        info_times = np.linspace(0, 1, n_looks + 1)[1:].tolist()

        # Use heuristic for n_per_analysis to achieve target power
        n_per_analysis_b = int(n_fixed_correct / n_looks * 1.15)

        spec_template = ProportionsDesignSpec()
        spec_template.test.alpha = alpha
        spec_template.sequential.n_analyses = n_looks
        spec_template.sequential.info_times = info_times
        spec_template.sequential.info_spacing = InformationSpacing.CUSTOM

        result = compute_ess_curve(spec_template, p0, p1_values, n_per_analysis_b)

        gst_curves_b[f"{n_interim}_interim"] = {
            "p1": result.p1_values,
            "ess": result.ess_values,
            "power": result.power_values,
            "stop_dists": result.stop_dists,
            "n_interim": n_interim,
            "n_looks": n_looks,
            "n_per_analysis": n_per_analysis_b,
        }

        # Find power and ESS at target p1
        idx_target = np.argmin(np.abs(result.p1_values - p1_target))
        power_at_target = result.power_values[idx_target]
        ess_at_target = result.ess_values[idx_target]
        max_n_total = n_per_analysis_b * n_looks * 2

        print(f"   {n_interim} interim ({n_looks} looks):")
        print(
            f"     Max N={max_n_total}, Power at p1={p1_target:.2f}: {power_at_target:.3f}"
        )
        print(f"     ESS at p1={p1_target:.2f}: {ess_at_target:.1f}")
        print(f"     Fixed design ESS at p1={p1_target:.2f}: {n_fixed_total_correct}")
        print(
            f"     Difference: {ess_at_target - n_fixed_total_correct:.1f} (ESS - Fixed)"
        )

    # 5. Visualization
    print("\n5. Creating operating characteristics plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    linestyles = ["-", "--", "-.", ":"]

    # Plot A: Scenario A (Fixed Max N)
    ax1.set_title("Scenario A: Fixed Max N", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Treatment Proportion (p1)", fontsize=12)
    ax1.set_ylabel("Expected Sample Size (ESS)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot point for fixed design at target
    ax1.scatter(
        [p1_target],
        [n_fixed_total_correct],
        color=colors[0],
        s=150,
        marker="o",
        label=f"Fixed (0 interim), N={n_fixed_total_correct}",
        zorder=5,
        edgecolors="black",
        linewidths=2,
    )

    # Plot curves for GST designs
    idx = 0
    for key, data in gst_curves_a.items():
        if key == "fixed":
            continue  # Skip fixed curve (constant)

        # Find power at target p1
        idx_target = np.argmin(np.abs(data["p1"] - p1_target))
        power_at_target = data["power"][idx_target]

        # Plot ESS curve (mean)
        ax1.plot(
            data["p1"],
            data["ess"],
            color=colors[idx + 1],
            linestyle=linestyles[idx + 1],
            linewidth=2.5,
            label=f"{data['n_interim']} interim ({data['n_looks']} looks), Power@p1={p1_target:.2f}: {power_at_target:.3f}",
            alpha=0.9,
        )

        # Overlay stopping distribution as scatter points with transparency
        if "stop_dists" in data and "n_per_analysis" in data:
            n_per_analysis = data["n_per_analysis"]
            n_looks = data["n_looks"]
            # Sample sizes at each analysis (total N)
            sample_sizes_at_analysis = np.array(
                [n_per_analysis * (i + 1) * 2 for i in range(n_looks)]
            )

            # For each p1, plot points at each analysis with transparency based on stopping probability
            for p1_idx, p1_val in enumerate(data["p1"]):
                stop_dist = data["stop_dists"][p1_idx]
                if not stop_dist:
                    continue

                # Total number of simulations
                total_sims = sum(stop_dist.values())

                # Plot a point for each analysis where stopping occurred
                for analysis_idx, count in stop_dist.items():
                    # analysis_idx is 1-based
                    n_total = sample_sizes_at_analysis[analysis_idx - 1]
                    prob = count / total_sims  # Stopping probability

                    # Use probability as alpha (transparency)
                    # Scale so that sum of all alphas = 0.5 (since sum of probs = 1)
                    alpha_val = prob * 0.5

                    ax1.scatter(
                        [p1_val],
                        [n_total],
                        color=colors[idx + 1],
                        s=150,  # Same size as the fixed design point
                        alpha=alpha_val,
                        edgecolors="none",
                        zorder=3,
                    )

        idx += 1

    # Mark the target p1 with vertical line
    ax1.axvline(
        x=p1_target,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"Target p1={p1_target:.2f}",
    )

    ax1.legend(loc="best", fontsize=10)

    # Plot B: Scenario B (Fixed Power at target)
    ax2.set_title("Scenario B: Fixed Power at Target δ", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Treatment Proportion (p1)", fontsize=12)
    ax2.set_ylabel("Expected Sample Size (ESS)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Plot fixed design curve (horizontal line at max N)
    ax2.axhline(
        y=n_fixed_total_correct,
        color=colors[0],
        linestyle=linestyles[0],
        linewidth=2,
        label=f"Fixed (0 interim), N={n_fixed_total_correct}",
        alpha=0.6,
    )

    # Plot point for fixed design at target
    ax2.scatter(
        [p1_target],
        [n_fixed_total_correct],
        color=colors[0],
        s=150,
        marker="o",
        zorder=5,
        edgecolors="black",
        linewidths=2,
    )

    # Plot curves for GST designs
    idx = 0
    for key, data in gst_curves_b.items():
        if key == "fixed":
            continue

        # Plot ESS curve (mean)
        ax2.plot(
            data["p1"],
            data["ess"],
            color=colors[idx + 1],
            linestyle=linestyles[idx + 1],
            linewidth=2.5,
            label=f"{data['n_interim']} interim ({data['n_looks']} looks)",
            alpha=0.9,
        )

        # Overlay stopping distribution as scatter points with transparency
        if "stop_dists" in data and "n_per_analysis" in data:
            n_per_analysis = data["n_per_analysis"]
            n_looks = data["n_looks"]
            # Sample sizes at each analysis (total N)
            sample_sizes_at_analysis = np.array(
                [n_per_analysis * (i + 1) * 2 for i in range(n_looks)]
            )

            # For each p1, plot points at each analysis with transparency based on stopping probability
            for p1_idx, p1_val in enumerate(data["p1"]):
                stop_dist = data["stop_dists"][p1_idx]
                if not stop_dist:
                    continue

                # Total number of simulations
                total_sims = sum(stop_dist.values())

                # Plot a point for each analysis where stopping occurred
                for analysis_idx, count in stop_dist.items():
                    # analysis_idx is 1-based
                    n_total = sample_sizes_at_analysis[analysis_idx - 1]
                    prob = count / total_sims  # Stopping probability

                    # Use probability as alpha (transparency)
                    # Scale so that sum of all alphas = 0.5 (since sum of probs = 1)
                    alpha_val = prob * 0.5

                    ax2.scatter(
                        [p1_val],
                        [n_total],
                        color=colors[idx + 1],
                        s=150,  # Same size as the fixed design point
                        alpha=alpha_val,
                        edgecolors="none",
                        zorder=3,
                    )

        idx += 1

    # Mark the target p1
    ax2.axvline(
        x=p1_target,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"Target p1={p1_target:.2f}",
    )

    ax2.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(
        "/Users/teshima/2025/EarlySign/gst_operating_characteristics.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(
        "   Plot saved to: /Users/teshima/2025/EarlySign/gst_operating_characteristics.png"
    )
    plt.close()

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
