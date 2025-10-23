"""
GST Design Tutorial - Operating Characteristics Curves

This demonstrates the operating characteristics (expected sample size vs. effect size)
for different group sequential designs using the new scenario abstractions.
"""

import numpy as np
import pytest

from earlysign.stats.design.gst.common.visualization import (
    OCCurvePlotter,
    print_scenario_summary,
)
from earlysign.stats.design.gst.scenarios import (
    run_scenario_a,
    run_scenario_b,
)
from earlysign.stats.design.gst.schemes.two_proportions.calculator import (
    TwoProportionsCalculator,
)

# Fix random seed for reproducibility
np.random.seed(42)


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

    # 1. Setup calculator
    print("\n1. Setup: Two-Proportion Test")
    print(f"   Control (p0):    {p0:.2f}")
    print(f"   Treatment (p1):  {p1_target:.2f}")
    print(f"   Effect size:     {effect_size_target:.2f}")
    print(f"   Alpha:           {alpha:.3f}")
    print(f"   Power:           {power:.2f}")

    calculator = TwoProportionsCalculator(p_control=p0)

    # Calculate fixed design sample size
    n_per_group = calculator.calculate_sample_size(effect_size_target, alpha, power)
    n_total = n_per_group * 2

    print("\n   Fixed design sample size:")
    print(f"   - Per group: {n_per_group:,}")
    print(f"   - Total:     {n_total:,}")

    # 2. Define range of effect sizes to evaluate (as differences, not absolute p1)
    effect_size_min = 0.02  # Minimum detectable difference
    effect_size_max = min(0.35, 0.95 - p0)  # Extended range
    effect_sizes = np.linspace(effect_size_min, effect_size_max, 50)

    p1_min = p0 + effect_size_min
    p1_max = p0 + effect_size_max

    print("\n2. Evaluating operating characteristics")
    print(f"   Treatment p1 range:   {p1_min:.3f} to {p1_max:.3f}")
    print(f"   Effect size range:    {effect_size_min:.3f} to {effect_size_max:.3f}")
    print(f"   Number of points:     {len(effect_sizes)}")

    # 3. Run Scenario A: Fixed Max N
    print(f"\n3. Running Scenario A: Fixed Max N = {n_total}")

    n_interim_list = [1, 2]
    result_a = run_scenario_a(
        calculator=calculator,
        target_effect=effect_size_target,
        effect_sizes=effect_sizes,
        alpha=alpha,
        power=power,
        n_interim_list=n_interim_list,
    )

    print_scenario_summary(
        result=result_a,
        target_effect=effect_size_target,
        null_value=p0,
        scenario_name="A",
    )

    # 4. Run Scenario B: Fixed Power at target δ
    print(f"\n4. Running Scenario B: Fixed Power ≈ {power} at target")

    result_b = run_scenario_b(
        calculator=calculator,
        target_effect=effect_size_target,
        effect_sizes=effect_sizes,
        alpha=alpha,
        power=power,
        n_interim_list=n_interim_list,
        inflation_factor=1.15,
    )

    print_scenario_summary(
        result=result_b,
        target_effect=effect_size_target,
        null_value=p0,
        scenario_name="B",
    )

    # 5. Visualization
    print("\n5. Creating operating characteristics plots...")

    plotter = OCCurvePlotter(figsize=(16, 6))

    # Combined plot
    plotter.plot_comparison(
        result_a=result_a,
        result_b=result_b,
        target_effect=effect_size_target,
        null_value=p0,
        effect_label="Treatment Proportion (p1)",
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


@pytest.mark.basic
def test_gst_operating_characteristics_realistic_ctr():
    """GST operating characteristics with realistic CTR (0.5% vs 0.7%)."""

    print("\n" + "=" * 80)
    print("GST OPERATING CHARACTERISTICS - REALISTIC CTR")
    print("=" * 80)

    # Realistic click-through rate scenario
    p0 = 0.005  # 0.5% baseline
    p1 = 0.007  # 0.7% target
    effect_size_target = p1 - p0
    alpha = 0.05
    power = 0.80

    print("\n1. Setup: Realistic Click-Through Rate Test")
    print(f"   Control CTR (p0):    {p0*100:.2f}%")
    print(f"   Treatment CTR (p1):  {p1*100:.2f}%")
    print(f"   Effect size:         {effect_size_target*100:.2f} pp")
    print(f"   Relative lift:       {(p1/p0 - 1)*100:.0f}%")
    print(f"   Alpha:               {alpha:.3f}")
    print(f"   Power:               {power:.2f}")

    calculator = TwoProportionsCalculator(p_control=p0)

    # Calculate fixed design sample size
    n_per_group = calculator.calculate_sample_size(effect_size_target, alpha, power)
    n_total = n_per_group * 2

    print("\n   Fixed design sample size:")
    print(f"   - Per group: {n_per_group:,}")
    print(f"   - Total:     {n_total:,}")

    # 2. Define range of effect sizes (as differences, not absolute p1)
    effect_size_min = 0.0001  # Minimum detectable difference
    effect_size_max = min(0.005, 0.015 - p0)  # Up to 1.5% absolute or +0.5pp
    effect_sizes = np.linspace(effect_size_min, effect_size_max, 30)

    p1_min = p0 + effect_size_min
    p1_max = p0 + effect_size_max

    print("\n2. Evaluating operating characteristics")
    print(f"   CTR range:         {p1_min*100:.3f}% to {p1_max*100:.3f}%")
    print(
        f"   Effect size range: {effect_size_min*100:.3f}pp to {effect_size_max*100:.3f}pp"
    )
    print(f"   Number of points:  {len(effect_sizes)}")

    # 3. Run scenarios
    n_interim_list = [1, 2]

    print(f"\n3. Running Scenario A: Fixed Max N = {n_total:,}")
    result_a = run_scenario_a(
        calculator=calculator,
        target_effect=effect_size_target,
        effect_sizes=effect_sizes,
        alpha=alpha,
        power=power,
        n_interim_list=n_interim_list,
    )

    print(f"\n4. Running Scenario B: Fixed Power ≈ {power}")
    result_b = run_scenario_b(
        calculator=calculator,
        target_effect=effect_size_target,
        effect_sizes=effect_sizes,
        alpha=alpha,
        power=power,
        n_interim_list=n_interim_list,
        inflation_factor=1.15,
    )

    # 5. Visualization
    print("\n5. Creating plots...")

    plotter = OCCurvePlotter(figsize=(16, 6))

    # Scenario A only
    print("\n   Plotting Scenario A...")
    plotter.plot_scenario_a(
        result=result_a,
        target_effect=effect_size_target,
        null_value=p0,
        effect_label="Treatment CTR (%)",
    )

    # Scenario B only
    print("\n   Plotting Scenario B...")
    plotter.plot_scenario_b(
        result=result_b,
        target_effect=effect_size_target,
        null_value=p0,
        effect_label="Treatment CTR (%)",
    )

    # Both scenarios
    print("\n   Plotting comparison...")
    plotter.plot_comparison(
        result_a=result_a,
        result_b=result_b,
        target_effect=effect_size_target,
        null_value=p0,
        effect_label="Treatment CTR (%)",
    )

    # Print summaries
    print_scenario_summary(
        result=result_a,
        target_effect=effect_size_target,
        null_value=p0,
        scenario_name="A (Fixed Max N)",
    )

    print_scenario_summary(
        result=result_b,
        target_effect=effect_size_target,
        null_value=p0,
        scenario_name="B (Fixed Power)",
    )

    print("\n" + "=" * 80)
    print("COMPLETE - Realistic CTR Scenario")
    print("=" * 80)
    print("\nKey Insights:")
    print("  • With low baseline CTR (0.5%), large sample sizes are needed")
    print("  • GST can still provide savings through early stopping")
    print("  • Multiple interim analyses offer flexibility without much penalty")
    print("=" * 80)
