"""Effect size calculator for time-to-event designs."""

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from earlysign.stats.schemes.protocols import EffectSizeCalculator


@dataclass
class TimeToEventEffect:
    """Effect specification for survival outcomes."""

    hazard_ratio: float = 0.6


@dataclass
class TimeToEventSampleSize:
    """Sample size specification for survival outcomes."""

    total_events: int = 171
    total_sample_size: int = 296
    time_unit: Literal["days", "weeks", "months", "years"] = "months"
    accrual_duration: float | None = None
    follow_up_duration: float | None = None


@dataclass
class TimeToEventEffectSizeCalculator(EffectSizeCalculator):
    """Effect size utilities for survival analysis (log-rank framework)."""

    def standardized_effect(
        self,
        effect: TimeToEventEffect,
        info_time: float,
        *,
        sample_size: TimeToEventSampleSize,
        allocation_ratio: float,
        info_times: np.ndarray,
    ) -> float:
        """Return standardized log hazard ratio at the specified information fraction."""
        if info_times.size == 0:
            return 0.0

        log_hr = float(np.log(effect.hazard_ratio))
        total_events = int(sample_size.total_events * info_time)
        if total_events == 0:
            return 0.0

        standard_error = float(
            np.sqrt((1 + allocation_ratio) ** 2 / (allocation_ratio * total_events))
        )
        return log_hr / standard_error

    def sample_sizes(
        self,
        sample_size: TimeToEventSampleSize,
        *,
        allocation_ratio: float,
        info_times: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Return planned event and sample counts for each analysis."""
        events = (sample_size.total_events * info_times).astype(int)
        total_sample = sample_size.total_sample_size
        control_size = int(total_sample / (1 + allocation_ratio))
        treatment_size = total_sample - control_size

        n_control = np.full_like(events, control_size)
        n_treatment = np.full_like(events, treatment_size)
        n_total = np.full_like(events, total_sample)

        return {
            "events": events,
            "n_control": n_control,
            "n_treatment": n_treatment,
            "n_total": n_total,
            "info_fraction": info_times,
        }
