"""Effect size and sample size calculations for different test types."""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from earlysign.stats.design.gst.common.config import (
    DesignSpec,
    MeansDesignSpec,
    ProportionsDesignSpec,
    TimeToEventDesignSpec,
)


class EffectCalculator(ABC):
    """Abstract base class for test-specific effect size and sample size calculations.

    Different statistical tests (two proportions, two means, time-to-event)
    require different formulations for:
    1. Standardized effect size (drift parameter for Brownian motion)
    2. Sample size at each analysis (may vary by allocation ratio)
    3. Information accrual (may be based on events rather than sample size)

    This protocol defines the interface that all test-specific calculators
    must implement. Concrete implementations handle the mathematical details
    for each test type.

    Methods
    -------
    standardized_effect(spec, info_time) -> float
        Compute the standardized effect (drift) at a given information time.
        This determines the mean of the Z-statistic under the alternative
        hypothesis. For example, for two proportions:
            drift = (p_treatment - p_control) / SE(p_treatment - p_control)

    sample_sizes(spec) -> Dict[str, np.ndarray]
        Compute sample sizes at each planned analysis. Returns dictionary
        with keys like 'n_control', 'n_treatment', 'n_total', containing
        arrays of length n_analyses.

    Examples
    --------
    >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    >>> spec = ProportionsDesignSpec()
    >>>
    >>> # Concrete implementation for proportions
    >>> from earlysign.stats.design.gst.common.effects import ProportionsEffectCalculator
    >>> calc = ProportionsEffectCalculator()
    >>> effect = calc.standardized_effect(spec, info_time=1.0)
    >>> effect > 0  # Positive effect
    True
    >>>
    >>> sizes = calc.sample_sizes(spec)
    >>> 'n_total' in sizes
    True

    See Also
    --------
    ProportionsEffectCalculator : For two-proportion tests
    MeansEffectCalculator : For two-sample t-tests
    TimeToEventEffectCalculator : For survival analysis
    SimulationEngine : Uses standardized_effect for drift
    DesignLab : Orchestrates effect calculations
    """

    @abstractmethod
    def standardized_effect(self, spec: DesignSpec, info_time: float) -> float:
        """Calculate standardized effect at given information time.

        Args:
            spec: Design specification
            info_time: Information time in (0, 1]

        Returns:
            Standardized effect size
        """
        pass

    @abstractmethod
    def sample_sizes(self, spec: DesignSpec) -> Dict[str, np.ndarray]:
        """Calculate sample sizes at each analysis.

        Args:
            spec: Design specification

        Returns:
            Dictionary with sample size arrays
        """
        pass


class ProportionsEffectCalculator(EffectCalculator):
    """Effect size and sample size calculations for two-proportion tests.

    This calculator implements the standard formulas for comparing two
    binomial proportions using a Z-test with pooled variance estimate.
    It handles unequal allocation ratios and computes the standardized
    effect (drift) for use in power calculations and simulations.

    The standardized effect is:
        δ = (p_treatment - p_control) / SE_pooled

    where SE_pooled uses the pooled proportion under H0:
        p_pooled = (p_control + p_treatment) / 2
        SE_pooled = sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))

    Methods
    -------
    standardized_effect(spec, info_time) -> float
        Compute standardized difference at information time.
    sample_sizes(spec) -> Dict[str, np.ndarray]
        Compute n_control, n_treatment, n_total at each analysis.

    Examples
    --------
    >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    >>> spec = ProportionsDesignSpec()
    >>> spec.effect.p_control = 0.10
    >>> spec.effect.delta = 0.05  # 5% absolute increase
    >>>
    >>> calc = ProportionsEffectCalculator()
    >>> effect = calc.standardized_effect(spec, 1.0)
    >>> effect > 0  # Positive effect
    True
    >>>
    >>> sizes = calc.sample_sizes(spec)
    >>> 'n_total' in sizes
    True
    >>> len(sizes['n_total'])  # One value per analysis
    3

    Notes
    -----
    The pooled variance estimator is used for consistency with the null
    distribution in group sequential testing. This differs from the
    unpooled estimator sometimes used in fixed-sample tests.

    Information accrual is proportional to sample size for proportion tests,
    so information time directly corresponds to the fraction of planned
    sample accumulated.

    See Also
    --------
    ProportionsDesignSpec : Specifies proportions effect parameters
    MeansEffectCalculator : Similar calculator for continuous outcomes
    SimulationEngine : Uses standardized_effect for power simulation
    """

    def standardized_effect(self, spec: DesignSpec, info_time: float) -> float:
        """Calculate standardized effect (delta) for proportions test.

        Args:
            spec: Proportions design specification
            info_time: Information time in (0, 1]

        Returns:
            Standardized difference (delta)
        """
        assert isinstance(spec, ProportionsDesignSpec)

        p_A = spec.effect.p_control
        p_B = spec.effect.get_treatment_proportion()

        # Pooled proportion under H0
        p_pooled = (p_A + p_B) / 2.0

        # Sample sizes at this information time
        n_total_A = spec.sample_size.n_per_analysis * spec.sequential.n_analyses
        n_A = int(n_total_A * info_time)
        n_B = int(n_A * spec.allocation.alloc_ratio)

        if n_A == 0 or n_B == 0:
            return 0.0

        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_A + 1 / n_B))

        if se == 0:
            return 0.0

        # Standardized difference
        return float((p_B - p_A) / se)

    def sample_sizes(self, spec: DesignSpec) -> Dict[str, np.ndarray]:
        """Calculate sample sizes at each analysis.

        Args:
            spec: Proportions design specification

        Returns:
            Dictionary with n_control, n_treatment, n_total, info_fraction
        """
        assert isinstance(spec, ProportionsDesignSpec)

        t = spec.resolved_info_times()
        n_total_A = spec.sample_size.n_per_analysis * spec.sequential.n_analyses

        n_A = (n_total_A * t).astype(int)
        n_B = (n_A * spec.allocation.alloc_ratio).astype(int)

        return {
            "n_control": n_A,
            "n_treatment": n_B,
            "n_total": n_A + n_B,
            "info_fraction": t,
        }


class TimeToEventEffectCalculator(EffectCalculator):
    """Effect size calculator for time-to-event.

    >>> from earlysign.stats.design.gst.common.config import TimeToEventDesignSpec
    >>> spec = TimeToEventDesignSpec()
    >>> calc = TimeToEventEffectCalculator()
    >>> effect = calc.standardized_effect(spec, 1.0)
    >>> effect < 0  # HR < 1 means negative standardized effect
    True
    >>> sizes = calc.sample_sizes(spec)
    >>> 'events' in sizes
    True
    """

    def standardized_effect(self, spec: DesignSpec, info_time: float) -> float:
        """Calculate standardized effect for survival analysis.

        Args:
            spec: Time-to-event design specification
            info_time: Information time in (0, 1]

        Returns:
            Standardized log hazard ratio
        """
        assert isinstance(spec, TimeToEventDesignSpec)

        # For log-rank test, effect is based on events
        hr = spec.effect.hazard_ratio
        log_hr = np.log(hr)

        # Events at this information time
        events = int(spec.sample_size.total_events * info_time)

        if events == 0:
            return 0.0

        # Standard error of log(HR) ~ sqrt(4/events) for equal allocation
        r = spec.allocation.alloc_ratio
        se = np.sqrt((1 + r) ** 2 / (r * events))

        return float(log_hr / se)

    def sample_sizes(self, spec: DesignSpec) -> Dict[str, np.ndarray]:
        """Calculate events and sample sizes at each analysis.

        Args:
            spec: Time-to-event design specification

        Returns:
            Dictionary with events, n_control, n_treatment, n_total, info_fraction
        """
        assert isinstance(spec, TimeToEventDesignSpec)

        t = spec.resolved_info_times()

        events = (spec.sample_size.total_events * t).astype(int)
        n_total = spec.sample_size.total_sample_size
        n_A = int(n_total / (1 + spec.allocation.alloc_ratio))
        n_B = n_total - n_A

        return {
            "events": events,
            "n_control": np.full_like(events, n_A),
            "n_treatment": np.full_like(events, n_B),
            "n_total": np.full_like(events, n_total),
            "info_fraction": t,
        }


class MeansEffectCalculator(EffectCalculator):
    """Effect size calculator for two means.

    >>> from earlysign.stats.design.gst.common.config import MeansDesignSpec
    >>> spec = MeansDesignSpec()
    >>> calc = MeansEffectCalculator()
    >>> effect = calc.standardized_effect(spec, 1.0)
    >>> effect > 0  # Positive effect
    True
    >>> sizes = calc.sample_sizes(spec)
    >>> 'n_total' in sizes
    True
    """

    def standardized_effect(self, spec: DesignSpec, info_time: float) -> float:
        """Calculate standardized effect for means test.

        Args:
            spec: Means design specification
            info_time: Information time in (0, 1]

        Returns:
            Standardized mean difference
        """
        assert isinstance(spec, MeansDesignSpec)

        mu_A = spec.effect.mean_control
        mu_B = (
            spec.effect.mean_treatment
            if spec.effect.mean_treatment is not None
            else mu_A + (spec.effect.effect_size or 0.0)
        )
        sigma = spec.effect.std_dev

        # Sample sizes at this information time
        n_total_A = spec.sample_size.n_per_analysis * spec.sequential.n_analyses
        n_A = int(n_total_A * info_time)
        n_B = int(n_A * spec.allocation.alloc_ratio)

        if n_A == 0 or n_B == 0:
            return 0.0

        # Standard error
        se = sigma * np.sqrt(1 / n_A + 1 / n_B)

        if se == 0:
            return 0.0

        return float((mu_B - mu_A) / se)

    def sample_sizes(self, spec: DesignSpec) -> Dict[str, np.ndarray]:
        """Calculate sample sizes at each analysis.

        Args:
            spec: Means design specification

        Returns:
            Dictionary with n_control, n_treatment, n_total, info_fraction
        """
        assert isinstance(spec, MeansDesignSpec)

        t = spec.resolved_info_times()
        n_total_A = spec.sample_size.n_per_analysis * spec.sequential.n_analyses

        n_A = (n_total_A * t).astype(int)
        n_B = (n_A * spec.allocation.alloc_ratio).astype(int)

        return {
            "n_control": n_A,
            "n_treatment": n_B,
            "n_total": n_A + n_B,
            "info_fraction": t,
        }
