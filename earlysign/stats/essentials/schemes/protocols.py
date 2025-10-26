"""Shared protocols for scheme effect size calculators."""

from typing import Any, Protocol

import numpy as np


class EffectSizeCalculator(Protocol):
    """Protocol for group sequential effect size calculators."""

    def standardized_effect(
        self,
        effect: Any,
        info_time: float,
        *,
        sample_size: Any,
        allocation_ratio: float,
        info_times: np.ndarray,
    ) -> float:
        """Return standardized effect at a given information fraction."""

    def sample_sizes(
        self,
        sample_size: Any,
        *,
        allocation_ratio: float,
        info_times: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return planned sample sizes keyed by descriptive labels."""


class FixedDesignEffectCalculator(Protocol):
    """Protocol for fixed-design effect size utilities."""

    def calculate_sample_size(
        self, effect_size: float, alpha: float, power: float
    ) -> int:
        """Return the per-group sample size for the targeted effect."""

    def get_null_value(self) -> float:
        """Return the baseline or null parameter value for the design."""


__all__ = ["EffectSizeCalculator", "FixedDesignEffectCalculator"]
