"""Scenario patterns for GST design evaluation.

This module provides standardized scenario patterns for evaluating
group sequential test designs across different effect sizes.
"""

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab


class EffectSizeCalculator(Protocol):
    """Protocol for calculating sample size for a given effect."""

    def calculate_sample_size(
        self, effect_size: float, alpha: float, power: float
    ) -> int:
        """Calculate required sample size per group.

        Args:
            effect_size: Effect size in natural units
            alpha: Type I error rate (two-sided)
            power: Target statistical power

        Returns:
            Required sample size per group
        """
        ...

    def get_null_value(self) -> float:
        """Get the null hypothesis value."""
        ...


@dataclass
class TwoProportionsCalculator:
    """Sample size calculator for two-proportion z-test.

    Attributes:
        p_control: Control proportion
    """

    p_control: float

    def calculate_sample_size(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.8
    ) -> int:
        """Calculate required sample size per group."""
        from scipy import stats

        p1 = self.p_control + effect_size
        delta = effect_size

        # Standard errors
        se_alt = np.sqrt(self.p_control * (1 - self.p_control) + p1 * (1 - p1))

        # Critical values
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Sample size formula
        n = ((z_alpha + z_beta) * se_alt / delta) ** 2

        return int(np.ceil(n))

    def get_null_value(self) -> float:
        """Get the null hypothesis value (control proportion)."""
        return self.p_control


@dataclass
class OCSinglePointResult:
    """Result of computing operating characteristics at a single effect size.

    Attributes:
        effect_size: Effect size evaluated
        expected_sample_size: Expected sample size (ESS)
        power: Statistical power
        max_sample_size: Maximum sample size
        stop_distribution: Distribution of stopping times (dict: analysis_idx -> count)
    """

    effect_size: float
    expected_sample_size: float
    power: float
    max_sample_size: float
    stop_distribution: dict[int, int]


@dataclass
class OCCurveResult:
    """Result of computing operating characteristics across effect sizes.

    Attributes:
        effect_sizes: Array of effect sizes evaluated
        results: List of OCSinglePointResult for each effect size
        n_looks: Number of analyses
        n_per_analysis: Sample size per analysis (per group)
    """

    effect_sizes: np.ndarray
    results: List[OCSinglePointResult]
    n_looks: int
    n_per_analysis: int

    @property
    def ess_values(self) -> np.ndarray:
        """Expected sample sizes across effect sizes."""
        return np.array([r.expected_sample_size for r in self.results])

    @property
    def power_values(self) -> np.ndarray:
        """Power values across effect sizes."""
        return np.array([r.power for r in self.results])


def compute_oc_curve(
    calculator: EffectSizeCalculator,
    effect_sizes: np.ndarray,
    n_looks: int,
    n_per_analysis: int,
    alpha: float,
    info_times: List[float] | None = None,
) -> OCCurveResult:
    """Compute operating characteristics curve across effect sizes.

    Args:
        calculator: Effect size calculator (determines test type)
        effect_sizes: Array of effect sizes to evaluate
        n_looks: Number of analyses
        n_per_analysis: Sample size per analysis (per group)
        alpha: Type I error rate
        info_times: Information times (optional, defaults to equally spaced)

    Returns:
        OCCurveResult with operating characteristics
    """
    from earlysign.stats.design.gst.common.types import InformationSpacing

    if info_times is None:
        info_times = np.linspace(0, 1, n_looks + 1)[1:].tolist()

    results = []

    for effect_size in effect_sizes:
        # Create design specification
        spec = ProportionsDesignSpec()
        spec.test.alpha = alpha
        spec.sequential.n_analyses = n_looks
        spec.sequential.info_times = info_times
        spec.sequential.info_spacing = InformationSpacing.CUSTOM
        spec.sample_size.n_per_analysis = n_per_analysis
        spec.effect.p_control = calculator.get_null_value()
        spec.effect.effect_size = effect_size

        # Run design
        lab = DesignLab(spec)
        lab.compute_boundaries().run_simulations()

        if lab.simulation_results:
            result = OCSinglePointResult(
                effect_size=effect_size,
                expected_sample_size=lab.simulation_results["expected_sample_size"],
                power=lab.simulation_results["power"],
                max_sample_size=lab.simulation_results["max_sample_size"],
                stop_distribution=lab.simulation_results.get("stop_distribution", {}),
            )
        else:
            # Fallback if simulation fails
            summary = lab.get_summary()
            max_n = summary["N Control"].iloc[-1] * 2
            result = OCSinglePointResult(
                effect_size=effect_size,
                expected_sample_size=max_n,
                power=0.0,
                max_sample_size=max_n,
                stop_distribution={},
            )

        results.append(result)

    return OCCurveResult(
        effect_sizes=effect_sizes,
        results=results,
        n_looks=n_looks,
        n_per_analysis=n_per_analysis,
    )


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
    calculator: EffectSizeCalculator,
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
        calculator: Effect size calculator
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
    calculator: EffectSizeCalculator,
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
        calculator: Effect size calculator
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
