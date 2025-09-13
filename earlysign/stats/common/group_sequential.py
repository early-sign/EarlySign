"""
earlysign.stats.common.group_sequential
======================================

Generic group sequential testing methods.

This module provides the core mathematical algorithms for group sequential
testing that are independent of the specific statistical scheme being tested.
These methods are applied to specific schemes in the earlysign.stats.schemes
modules.

The functions here implement the theoretical foundations of sequential testing
including spending functions, boundary calculations, and information time
management.
"""

from __future__ import annotations
import math
from typing import List, Optional, Union, Dict, Any, TypedDict
from dataclasses import dataclass, field

from scipy.stats import norm


class GstBoundaryPayload(TypedDict):
    """Payload structure for GST boundary events."""

    upper: float
    lower: float
    info_time: float
    alpha_i: float


def lan_demets_spending(alpha_total: float, t: float, style: str = "obf") -> float:
    """
    Compute cumulative alpha spending at information time t.

    Args:
        alpha_total: Total Type I error rate to spend
        t: Information fraction (0 < t ≤ 1)
        style: Spending function style ("obf" for O'Brien-Fleming, "pocock" for Pocock)

    Returns:
        Cumulative alpha spent up to information time t

    Examples:
        >>> lan_demets_spending(0.05, 0.5, "obf")  # doctest: +SKIP
        0.006...
        >>> lan_demets_spending(0.05, 1.0, "obf")  # doctest: +SKIP
        0.05
    """
    if not (0 < t <= 1):
        raise ValueError(f"Information time t must be in (0, 1], got {t}")

    if style.lower() == "obf":
        # O'Brien-Fleming spending: α(t) = 2(1 - Φ(z_{α/2}/√t))
        z_alpha_2 = norm.ppf(1 - alpha_total / 2)
        return float(2 * (1 - norm.cdf(z_alpha_2 / math.sqrt(t))))
    elif style.lower() == "pocock":
        # Pocock spending: α(t) = α * ln(1 + (e-1)*t)
        return alpha_total * math.log(1 + (math.e - 1) * t)
    else:
        raise ValueError(f"Unknown spending style: {style}")


def nominal_alpha_increments(
    alpha_total: float, t_sequence: List[float], style: str = "obf"
) -> List[float]:
    """
    Compute nominal alpha increments for a sequence of information times.

    Args:
        alpha_total: Total Type I error rate
        t_sequence: Increasing sequence of information fractions
        style: Spending function style

    Returns:
        List of alpha increments for each analysis

    Examples:
        >>> nominal_alpha_increments(0.05, [0.25, 0.5, 0.75, 1.0], "obf")  # doctest: +SKIP
        [0.000..., 0.003..., 0.012..., 0.033...]
    """
    if not t_sequence or not all(0 < t <= 1 for t in t_sequence):
        raise ValueError("t_sequence must contain values in (0, 1]")

    if not all(t_sequence[i] <= t_sequence[i + 1] for i in range(len(t_sequence) - 1)):
        raise ValueError("t_sequence must be non-decreasing")

    cumulative_alphas = [lan_demets_spending(alpha_total, t, style) for t in t_sequence]

    # Convert to increments
    increments = [cumulative_alphas[0]]
    for i in range(1, len(cumulative_alphas)):
        increments.append(cumulative_alphas[i] - cumulative_alphas[i - 1])

    return increments


@dataclass
class AdaptiveInfoTime:
    """
    Generic adaptive information time re-estimation.

    This class provides adaptive group sequential testing where information
    fractions are re-estimated based on actual observed sample sizes rather
    than pre-planned targets. This is scheme-agnostic and can be applied to
    any sequential testing procedure.

    Attributes:
        initial_target: Initial planned sample size target
        looks: Number of planned analysis looks
        min_increment: Minimum increment between consecutive information fractions
        max_info_fraction: Maximum allowed information fraction (default 1.0)
    """

    initial_target: int
    looks: int
    min_increment: float = 0.01
    max_info_fraction: float = 1.0

    # Internal state
    planned_fractions: List[float] = field(init=False)
    planned_targets: List[int] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize planned information fractions and targets."""
        self.planned_fractions = [(i + 1) / self.looks for i in range(self.looks)]
        self.planned_targets = [
            int(self.initial_target * f) for f in self.planned_fractions
        ]

    def get_info_fraction(self, current_look: int, observed_n: int) -> float:
        """
        Re-estimate information fraction based on observed sample size.

        Args:
            current_look: Current analysis look (1-indexed)
            observed_n: Observed sample size (scheme-specific interpretation)

        Returns:
            Adapted information fraction ensuring monotonicity and bounds

        Algorithm:
            1. Use planned fraction for first look
            2. Scale planned fraction by observed/planned ratio
            3. Ensure monotonicity with previous fractions
            4. Bound between 0 and max_info_fraction
        """
        if current_look <= 0:
            raise ValueError("current_look must be positive")

        if current_look > len(self.planned_fractions):
            return self.max_info_fraction

        if current_look == 1:
            return self.planned_fractions[0]

        # Get planned values
        planned_n = self.planned_targets[current_look - 1]
        base_fraction = self.planned_fractions[current_look - 1]

        # Scale by observed vs planned ratio
        scaling_factor = observed_n / planned_n if planned_n > 0 else 1.0
        adapted_fraction = base_fraction * scaling_factor

        # Ensure monotonicity: must be > previous fraction
        if current_look > 1:
            prev_fraction = self.planned_fractions[current_look - 2]
            adapted_fraction = max(adapted_fraction, prev_fraction + self.min_increment)

        # Bound to [0, max_info_fraction]
        adapted_fraction = min(max(adapted_fraction, 0.0), self.max_info_fraction)

        return adapted_fraction

    def get_all_adapted_fractions(self, observed_ns: List[int]) -> List[float]:
        """
        Get adapted information fractions for all looks.

        Args:
            observed_ns: List of observed sample sizes at each look

        Returns:
            List of adapted information fractions
        """
        if len(observed_ns) > self.looks:
            raise ValueError(
                f"Too many observations: {len(observed_ns)} > {self.looks}"
            )

        adapted = []
        for i, n in enumerate(observed_ns, start=1):
            adapted.append(self.get_info_fraction(i, n))

        return adapted

    def get_planned_vs_adapted_summary(self, observed_ns: List[int]) -> Dict[str, Any]:
        """
        Compare planned vs adapted information fractions.

        Args:
            observed_ns: Observed sample sizes at each look

        Returns:
            Dictionary with planned, adapted, and comparison metrics
        """
        num_looks = len(observed_ns)
        planned = self.planned_fractions[:num_looks]
        adapted = self.get_all_adapted_fractions(observed_ns)

        return {
            "looks": list(range(1, num_looks + 1)),
            "planned_fractions": planned,
            "adapted_fractions": adapted,
            "observed_n": observed_ns,
            "planned_n": self.planned_targets[:num_looks],
            "scaling_factors": [
                a / p if p > 0 else 1.0 for a, p in zip(adapted, planned)
            ],
            "max_deviation": (
                max(abs(a - p) for a, p in zip(adapted, planned)) if adapted else 0.0
            ),
        }


@dataclass
class SampleSizeTracker:
    """
    Generic utility for tracking cumulative sample sizes across analysis looks.

    This helper class maintains running totals and can be used with any
    sequential testing procedure for accurate information time estimation.
    The interpretation of samples is scheme-specific.
    """

    cumulative_total: int = 0
    look_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_batch(
        self, batch_size: int, look: int, time_index: str, **kwargs: Any
    ) -> None:
        """
        Add a batch of observations and update cumulative totals.

        Args:
            batch_size: Size of new batch (scheme-specific interpretation)
            look: Current analysis look
            time_index: Time identifier for this batch
            **kwargs: Additional scheme-specific data
        """
        self.cumulative_total += batch_size

        history_entry = {
            "look": look,
            "time_index": time_index,
            "batch_size": batch_size,
            "cumulative_total": self.cumulative_total,
            **kwargs,  # Allow scheme-specific extensions
        }

        self.look_history.append(history_entry)

    def get_total(self) -> int:
        """Get total sample size."""
        return self.cumulative_total

    def get_look_summary(self) -> Dict[str, List[Any]]:
        """
        Get summary of sample sizes by look.

        Returns:
            Dictionary with lists of sample sizes and cumulative totals
        """
        if not self.look_history:
            return {}

        return {
            "looks": [h["look"] for h in self.look_history],
            "time_indices": [h["time_index"] for h in self.look_history],
            "batch_sizes": [h["batch_size"] for h in self.look_history],
            "cumulative_sizes": [h["cumulative_total"] for h in self.look_history],
        }
