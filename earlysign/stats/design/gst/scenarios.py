"""Scenario patterns for GST design evaluation.

This module provides standardized scenario patterns for evaluating
group sequential test designs across different effect sizes.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from earlysign.stats.design.gst.essentials.operating_characteristics import (
    OCCurveResult,
    compute_oc_curve,
)
from earlysign.stats.essentials.schemes.protocols import FixedDesignEffectCalculator


@dataclass
class ScenarioAResult:
    """Results for Scenario A: Fixed Maximum N.

    Attributes:
        fixed_n: Total sample size for fixed design
        gst_results: Dictionary mapping n_interim -> OCCurveResult
    """

    fixed_n: int
    gst_results: dict[int, OCCurveResult]


@dataclass
class ScenarioBResult:
    """Results for Scenario B: Fixed Power at Target Effect.

    Attributes:
        fixed_n: Total sample size for fixed design
        target_power: Target power level
        gst_results: Dictionary mapping n_interim -> OCCurveResult
    """

    fixed_n: int
    target_power: float
    gst_results: dict[int, OCCurveResult]


def run_scenario_a(
    calculator: FixedDesignEffectCalculator,
    target_effect: float,
    effect_sizes: np.ndarray,
    alpha: float,
    power: float,
    n_interim_list: List[int],
) -> ScenarioAResult:
    """Run Scenario A: Fixed Maximum N.

    This scenario evaluates GST designs that use the same maximum sample size
    as the fixed design. Power will be lower due to the GST penalty, but
    expected sample size can be reduced if the true effect is strong.

    Args:
        calculator: Fixed-design effect size calculator
        target_effect: Target effect size for power calculation
        effect_sizes: Array of effect sizes to evaluate
        alpha: Type I error rate
        power: Target power (for fixed design)
        n_interim_list: List of interim analysis counts to try

    Returns:
        ScenarioAResult with operating characteristics
    """
    # Calculate fixed design sample size
    n_per_group = calculator.calculate_sample_size(target_effect, alpha, power)
    n_total = n_per_group * 2

    gst_results = {}

    for n_interim in n_interim_list:
        n_looks = n_interim + 1

        # Calculate n_per_analysis to achieve approximately the same max N
        n_per_analysis_floor = int(n_per_group / n_looks)
        n_per_analysis_ceil = n_per_analysis_floor + 1

        # Choose the one that gives max N closer to fixed design
        max_n_floor = n_per_analysis_floor * n_looks
        max_n_ceil = n_per_analysis_ceil * n_looks

        if abs(max_n_ceil - n_per_group) <= abs(max_n_floor - n_per_group):
            n_per_analysis = n_per_analysis_ceil
        else:
            n_per_analysis = n_per_analysis_floor

        # Compute OC curve
        oc_result = compute_oc_curve(
            calculator=calculator,
            effect_sizes=effect_sizes,
            n_looks=n_looks,
            n_per_analysis=n_per_analysis,
            alpha=alpha,
        )

        gst_results[n_interim] = oc_result

    return ScenarioAResult(fixed_n=n_total, gst_results=gst_results)


def run_scenario_b(
    calculator: FixedDesignEffectCalculator,
    target_effect: float,
    effect_sizes: np.ndarray,
    alpha: float,
    power: float,
    n_interim_list: List[int],
    inflation_factor: float = 1.15,
) -> ScenarioBResult:
    """Run Scenario B: Fixed Power at Target Effect.

    This scenario increases the maximum sample size to maintain target power
    at the target effect size. Expected sample size can still be reduced
    through early stopping.

    Args:
        calculator: Fixed-design effect size calculator
        target_effect: Target effect size for power calculation
        effect_sizes: Array of effect sizes to evaluate
        alpha: Type I error rate
        power: Target power to maintain
        n_interim_list: List of interim analysis counts to try
        inflation_factor: Factor to inflate max N (default 1.15 for ~15% increase)

    Returns:
        ScenarioBResult with operating characteristics
    """
    # Calculate fixed design sample size
    n_per_group_base = calculator.calculate_sample_size(target_effect, alpha, power)
    n_total_base = n_per_group_base * 2

    gst_results = {}

    for n_interim in n_interim_list:
        n_looks = n_interim + 1

        # Inflate sample size to maintain power
        n_per_analysis = int(np.ceil(n_per_group_base / n_looks * inflation_factor))

        # Compute OC curve
        oc_result = compute_oc_curve(
            calculator=calculator,
            effect_sizes=effect_sizes,
            n_looks=n_looks,
            n_per_analysis=n_per_analysis,
            alpha=alpha,
        )

        gst_results[n_interim] = oc_result

    return ScenarioBResult(
        fixed_n=n_total_base, target_power=power, gst_results=gst_results
    )
