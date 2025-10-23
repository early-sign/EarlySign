"""Effect size calculator for time-to-event (survival) analysis.

This module provides a concrete implementation of effect calculations
for survival analysis, such as log-rank tests and Cox models.
"""

from typing import Dict

import numpy as np

from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.schemes.survival.config import TimeToEventDesignSpec
from earlysign.stats.schemes.base.effects import EffectCalculator


class TimeToEventEffectCalculator(EffectCalculator):
    """Effect size calculator for time-to-event (survival) analysis.

    Computes standardized effect (log hazard ratio / SE) and sample sizes/events
    at each analysis for group sequential designs with time-to-event endpoints.

    Examples
    --------
    >>> from earlysign.stats.design.gst.schemes.survival.config import TimeToEventDesignSpec
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
