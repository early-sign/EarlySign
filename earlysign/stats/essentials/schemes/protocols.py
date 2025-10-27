"""Shared protocols for scheme effect size calculators."""

from typing import Any, Protocol, Sequence

import numpy as np

from earlysign.stats.essentials.primitives.group_sequential import ASNDesignSummary


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


class ASNCalculator(Protocol):
    """Protocol for expected sample size evaluators."""

    def evaluate(self, information_rates: Sequence[float]) -> ASNDesignSummary:
        """Return a design summary from the supplied information schedule."""
