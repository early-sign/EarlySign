"""Shared protocols for scheme effect size calculators."""

from typing import Any, Protocol, Sequence

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
        ...

    def sample_sizes(
        self,
        sample_size: Any,
        *,
        allocation_ratio: float,
        info_times: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return planned sample sizes keyed by descriptive labels."""
        ...


class FixedDesignEffectCalculator(Protocol):
    """Protocol for fixed-design effect size utilities."""

    def calculate_sample_size(
        self, effect_size: float, alpha: float, power: float
    ) -> int:
        """Return the per-group sample size for the targeted effect."""
        ...

    def get_null_value(self) -> float:
        """Return the baseline or null parameter value for the design."""
        ...


class ASNCalculator(Protocol):
    """Protocol for expected sample size evaluators.

    The ASN calculator evaluates a schedule of information fractions and
    returns the expected sample size (ASN) as a floating point value.
    Detailed design summaries (boundaries, per-stage metrics) are not
    part of this protocol and should be produced by other utilities.
    """

    def evaluate(self, information_rates: Sequence[float]) -> float:
        """Return the expected sample size (ASN) for the supplied schedule."""
        ...
