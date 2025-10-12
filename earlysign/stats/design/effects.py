"""Effect size and sample size calculations for different test types."""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from earlysign.stats.design.config import (
    DesignSpec,
    MeansDesignSpec,
    ProportionsDesignSpec,
    TimeToEventDesignSpec,
)


class EffectCalculator(ABC):
    """Abstract base class for effect size calculations."""

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
    """Effect size calculator for two proportions.

    >>> from earlysign.stats.design.config import ProportionsDesignSpec
    >>> spec = ProportionsDesignSpec()
    >>> calc = ProportionsEffectCalculator()
    >>> effect = calc.standardized_effect(spec, 1.0)
    >>> effect > 0  # Positive effect
    True
    >>> sizes = calc.sample_sizes(spec)
    >>> 'n_total' in sizes
    True
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

    >>> from earlysign.stats.design.config import TimeToEventDesignSpec
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

    >>> from earlysign.stats.design.config import MeansDesignSpec
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
