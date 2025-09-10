"""
earlysign.methods.group_sequential.adaptive
===========================================

Adaptive group sequential testing components for handling variable sample sizes
and information time re-estimation.

This module provides utilities for adapting group sequential designs when:
- Daily sample sizes vary unpredictably
- Wall-clock time decisions are required
- Information fractions need re-estimation

Components:
- `AdaptiveInfoTime`: Re-estimate information fractions based on observed data
- `AdaptiveGSTBoundary`: Boundary component that uses adaptive information time
- `SampleSizeTracker`: Track cumulative sample sizes across looks

Examples
--------
>>> from earlysign.stats.methods.group_sequential.adaptive import AdaptiveInfoTime
>>> adaptive = AdaptiveInfoTime(initial_target=1000, looks=5)
>>> # At look 3, observed 600 samples instead of planned 600 (3/5 * 1000)
>>> info_fraction = adaptive.get_info_fraction(current_look=3, observed_n=800)
>>> print(f"Adapted information fraction: {info_fraction:.3f}")
Adapted information fraction: 0.800
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

from earlysign.core.components import Criteria
from earlysign.core.ledger import Ledger
from earlysign.core.names import Namespace, GstBoundaryTag
from earlysign.stats.schemes.two_proportions.statistics import LanDeMetsBoundary


@dataclass
class AdaptiveInfoTime:
    """
    Handles information time re-estimation based on observed sample sizes.

    This class provides adaptive group sequential testing where information
    fractions are re-estimated based on actual observed sample sizes rather
    than pre-planned targets.

    Attributes:
        initial_target: Initial planned sample size per arm
        looks: Number of planned analysis looks
        min_increment: Minimum increment between consecutive information fractions
        max_info_fraction: Maximum allowed information fraction (default 1.0)

    Examples:
        >>> adaptive = AdaptiveInfoTime(initial_target=1000, looks=5)
        >>> # Planned fractions: [0.2, 0.4, 0.6, 0.8, 1.0]
        >>> # At look 2, observed 500 samples instead of planned 400
        >>> adapted_t = adaptive.get_info_fraction(2, 500)
        >>> print(f"Adapted t={adapted_t:.3f} vs planned t=0.400")
        Adapted t=0.500 vs planned t=0.400
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
            observed_n: Observed sample size (per arm or total, consistently)

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
            # Use planned fraction for first look
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


@dataclass(kw_only=True)
class AdaptiveGSTBoundary(Criteria):
    """
    Adaptive group sequential boundary that re-estimates information time.

    This component combines LanDeMetsBoundary with AdaptiveInfoTime to provide
    boundaries that adapt to observed sample sizes rather than fixed planning.

    Attributes:
        alpha_total: Total Type I error rate
        style: Spending function style ("obf", "pocock")
        adaptive_info_time: AdaptiveInfoTime instance for re-estimation
        tag_crit: Tag for criteria events in ledger
    """

    alpha_total: float
    style: str = "obf"
    adaptive_info_time: AdaptiveInfoTime = field(
        default_factory=lambda: AdaptiveInfoTime(initial_target=1000, looks=5)
    )

    def step(
        self, ledger: Ledger, experiment_id: str, step_key: str, time_index: str
    ) -> None:
        """
        Update boundary using adaptive information time.

        This method:
        1. Determines current look from step_key
        2. Gets observed sample size from ledger observations
        3. Re-estimates information fraction
        4. Computes boundary using adapted information time
        """
        # Extract look number from step_key (e.g., "look_3" -> 3)
        try:
            current_look = int(step_key.replace("look_", ""))
        except ValueError:
            # Fallback: assume sequential numbering
            current_look = 1

        # Get observed sample size (simplified - would need proper aggregation)
        observed_n = self._get_observed_sample_size(ledger, experiment_id)

        # Re-estimate information fraction
        adapted_t = self.adaptive_info_time.get_info_fraction(current_look, observed_n)

        # Create boundary component with adapted information time
        boundary = LanDeMetsBoundary(
            alpha_total=self.alpha_total,
            t=adapted_t,
            style=self.style,
            tag_crit="crit:gst",
        )

        # Execute boundary step
        boundary.step(ledger, experiment_id, step_key, time_index)

        # Also log adaptation information
        ledger.write_event(
            time_index=time_index,
            namespace=Namespace.STATS,
            kind="adapted",
            experiment_id=experiment_id,
            step_key=step_key,
            payload_type="AdaptiveInfoTime",
            payload={
                "current_look": current_look,
                "observed_n": observed_n,
                "planned_t": (
                    self.adaptive_info_time.planned_fractions[current_look - 1]
                    if current_look <= len(self.adaptive_info_time.planned_fractions)
                    else 1.0
                ),
                "adapted_t": adapted_t,
                "adaptation_ratio": (
                    adapted_t
                    / self.adaptive_info_time.planned_fractions[current_look - 1]
                    if current_look <= len(self.adaptive_info_time.planned_fractions)
                    and self.adaptive_info_time.planned_fractions[current_look - 1] > 0
                    else 1.0
                ),
            },
            tag="adapt:info_time",
        )

    def _get_observed_sample_size(self, ledger: Ledger, experiment_id: str) -> int:
        """
        Get observed sample size from ledger observations.

        This is a simplified implementation that would need to be enhanced
        to properly aggregate observation batches.
        """
        # Get latest observation
        latest_obs = ledger.latest(namespace=Namespace.OBS, experiment_id=experiment_id)

        if latest_obs and latest_obs.payload:
            # Extract sample size from observation payload
            payload = latest_obs.payload
            if isinstance(payload, dict):
                n_a = payload.get("nA", 0)
                n_b = payload.get("nB", 0)
                return int(n_a + n_b)

        # Fallback
        return 500


@dataclass
class SampleSizeTracker:
    """
    Utility class to track cumulative sample sizes across analysis looks.

    This helper class maintains running totals of sample sizes and can be
    used with AdaptiveInfoTime for more accurate information time estimation.
    """

    cumulative_n_a: int = 0
    cumulative_n_b: int = 0
    look_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_batch(self, n_a: int, n_b: int, look: int, time_index: str) -> None:
        """
        Add a batch of observations and update cumulative totals.

        Args:
            n_a, n_b: New observations for groups A and B
            look: Current analysis look
            time_index: Time identifier for this batch
        """
        self.cumulative_n_a += n_a
        self.cumulative_n_b += n_b

        self.look_history.append(
            {
                "look": look,
                "time_index": time_index,
                "batch_n_a": n_a,
                "batch_n_b": n_b,
                "cumulative_n_a": self.cumulative_n_a,
                "cumulative_n_b": self.cumulative_n_b,
                "total_n": self.cumulative_n_a + self.cumulative_n_b,
            }
        )

    def get_total_n(self) -> int:
        """Get total sample size across both groups."""
        return self.cumulative_n_a + self.cumulative_n_b

    def get_n_per_arm(self) -> float:
        """Get average sample size per arm."""
        return (self.cumulative_n_a + self.cumulative_n_b) / 2.0

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
            "batch_sizes": [h["batch_n_a"] + h["batch_n_b"] for h in self.look_history],
            "cumulative_sizes": [h["total_n"] for h in self.look_history],
        }
