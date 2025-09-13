"""
earlysign.stats.schemes.two_proportions.common
=============================================

Common utilities and data structures for two-proportions testing.

This module provides shared functionality used across different testing
methods within the two-proportions scheme, including data structures,
aggregation functions, and basic computations.
"""

from __future__ import annotations
from typing import TypedDict, Union, List, Dict, Any, Optional

from earlysign.core.ledger import Ledger
from earlysign.core.names import Namespace, ExperimentId


# --- Type Definitions ---


class WaldZPayload(TypedDict):
    """Payload structure for Wald Z-statistic events."""

    z: float
    se: float
    nA: int
    nB: int
    mA: int
    mB: int
    pA_hat: float
    pB_hat: float


class GstBoundaryPayload(TypedDict):
    """Payload structure for GST boundary events."""

    upper: float
    lower: float
    info_time: float
    alpha_i: float


class TwoPropObsBatch(TypedDict):
    """Payload structure for two-proportions observation batch."""

    nA: int  # Total trials in group A
    nB: int  # Total trials in group B
    mA: int  # Successes in group A
    mB: int  # Successes in group B


# --- Aggregation Functions ---


def reduce_counts(
    ledger: Ledger,
    experiment_id: Union[ExperimentId, str],
    step_key: Optional[str] = None,
) -> tuple[int, int, int, int]:
    """
    Aggregate observation counts from the ledger for two-proportions testing.

    Args:
        ledger: Ledger instance to query
        experiment_id: Experiment identifier
        step_key: Optional step key to filter by (if None, aggregates all)

    Returns:
        Tuple of (nA, nB, mA, mB) - total and success counts for both groups
    """
    # Base query for observation events
    obs_filter = (ledger.table.namespace == str(Namespace.OBS)) & (
        ledger.table.experiment_id == str(experiment_id)
    )

    # Add step key filter if specified
    if step_key is not None:
        obs_filter &= ledger.table.step_key == str(step_key)

    obs_query = ledger.table.filter(obs_filter)
    obs_results = obs_query.execute()

    if obs_results.empty:
        return 0, 0, 0, 0

    # Aggregate counts across all observations
    records = ledger.unwrap_results(obs_results)

    nA_total, nB_total, mA_total, mB_total = 0, 0, 0, 0

    for record in records:
        payload = record["payload"]
        if isinstance(payload, dict):
            nA_total += payload.get("nA", 0)
            nB_total += payload.get("nB", 0)
            mA_total += payload.get("mA", 0)
            mB_total += payload.get("mB", 0)

    return nA_total, nB_total, mA_total, mB_total


def get_latest_statistic(
    ledger: Ledger, experiment_id: Union[ExperimentId, str], stat_tag: str
) -> Optional[Dict[str, Any]]:
    """
    Get the latest statistic event with the given tag.

    Args:
        ledger: Ledger instance to query
        experiment_id: Experiment identifier
        stat_tag: Tag for the statistic type

    Returns:
        Latest statistic payload or None if not found
    """
    stat_query = (
        ledger.table.filter(
            (ledger.table.namespace == str(Namespace.STATS))
            & (ledger.table.tag == stat_tag)
            & (ledger.table.experiment_id == str(experiment_id))
        )
        .order_by(ledger.table.timestamp.desc())
        .limit(1)
    )

    stat_results = stat_query.execute()

    if stat_results.empty:
        return None

    records = ledger.unwrap_results(stat_results)
    return records[0]["payload"] if records else None


def get_latest_criteria(
    ledger: Ledger, experiment_id: Union[ExperimentId, str], criteria_tag: str
) -> Optional[Dict[str, Any]]:
    """
    Get the latest criteria event with the given tag.

    Args:
        ledger: Ledger instance to query
        experiment_id: Experiment identifier
        criteria_tag: Tag for the criteria type

    Returns:
        Latest criteria payload or None if not found
    """
    criteria_query = (
        ledger.table.filter(
            (ledger.table.namespace == str(Namespace.CRITERIA))
            & (ledger.table.tag == criteria_tag)
            & (ledger.table.experiment_id == str(experiment_id))
        )
        .order_by(ledger.table.timestamp.desc())
        .limit(1)
    )

    criteria_results = criteria_query.execute()

    if criteria_results.empty:
        return None

    records = ledger.unwrap_results(criteria_results)
    return records[0]["payload"] if records else None


# --- Sample Size Tracking for Two Proportions ---


class TwoProportionsSampleTracker:
    """
    Sample size tracker specialized for two-proportions testing.

    Extends the generic SampleSizeTracker with two-proportions specific
    functionality, tracking separate counts for each group.
    """

    def __init__(self) -> None:
        self.cumulative_nA = 0
        self.cumulative_nB = 0
        self.cumulative_mA = 0  # Successes in A
        self.cumulative_mB = 0  # Successes in B
        self.look_history: List[Dict[str, Any]] = []

    def add_batch(
        self, nA: int, nB: int, mA: int, mB: int, look: int, time_index: str
    ) -> None:
        """
        Add a batch of two-proportions observations.

        Args:
            nA, nB: New trials for groups A and B
            mA, mB: New successes for groups A and B
            look: Current analysis look
            time_index: Time identifier for this batch
        """
        self.cumulative_nA += nA
        self.cumulative_nB += nB
        self.cumulative_mA += mA
        self.cumulative_mB += mB

        self.look_history.append(
            {
                "look": look,
                "time_index": time_index,
                "batch_nA": nA,
                "batch_nB": nB,
                "batch_mA": mA,
                "batch_mB": mB,
                "cumulative_nA": self.cumulative_nA,
                "cumulative_nB": self.cumulative_nB,
                "cumulative_mA": self.cumulative_mA,
                "cumulative_mB": self.cumulative_mB,
                "total_n": self.cumulative_nA + self.cumulative_nB,
                "pA_hat": self.cumulative_mA / max(self.cumulative_nA, 1),
                "pB_hat": self.cumulative_mB / max(self.cumulative_nB, 1),
            }
        )

    def get_current_counts(self) -> tuple[int, int, int, int]:
        """Get current cumulative counts as (nA, nB, mA, mB)."""
        return (
            self.cumulative_nA,
            self.cumulative_nB,
            self.cumulative_mA,
            self.cumulative_mB,
        )

    def get_total_n(self) -> int:
        """Get total sample size across both groups."""
        return self.cumulative_nA + self.cumulative_nB

    def get_n_per_arm(self) -> float:
        """Get average sample size per arm."""
        return (self.cumulative_nA + self.cumulative_nB) / 2.0

    def get_current_rates(self) -> tuple[float, float]:
        """Get current success rates for both groups."""
        pA = self.cumulative_mA / max(self.cumulative_nA, 1)
        pB = self.cumulative_mB / max(self.cumulative_nB, 1)
        return pA, pB


# Statistical functions for two-proportion testing
def wald_z_from_counts(
    nA: int, nB: int, mA: int, mB: int
) -> tuple[float, float, float, float]:
    """Return (z, pA_hat, pB_hat, se) using unpooled SE.

    Computes the Wald Z-statistic for comparing two binomial proportions
    using unpooled standard error estimation.

    Args:
        nA: Total trials in group A
        nB: Total trials in group B
        mA: Successes in group A
        mB: Successes in group B

    Returns:
        Tuple of (z_statistic, pA_hat, pB_hat, standard_error)

    Note:
        We define Z = (pB_hat - pA_hat) / SE so that a *better variant B* yields positive Z.
    """
    import math

    if min(nA, nB) == 0:
        return 0.0, 0.0, 0.0, float("inf")

    pA_hat, pB_hat = mA / max(nA, 1), mB / max(nB, 1)
    var = pA_hat * (1 - pA_hat) / max(nA, 1) + pB_hat * (1 - pB_hat) / max(nB, 1)
    se = math.sqrt(var) if var > 0 else float("inf")
    diff = pB_hat - pA_hat  # variant minus baseline
    z = diff / se if se not in (0.0, float("inf")) else 0.0

    return z, pA_hat, pB_hat, se


def pooled_proportion_se(nA: int, nB: int, mA: int, mB: int) -> float:
    """Compute pooled standard error for difference of two proportions.

    Uses pooled variance estimator assuming equal population proportions.

    Args:
        nA: Total trials in group A
        nB: Total trials in group B
        mA: Successes in group A
        mB: Successes in group B

    Returns:
        Pooled standard error of pA - pB
    """
    import math

    if min(nA, nB) == 0:
        return float("inf")

    n_total = nA + nB
    p_pooled = (mA + mB) / max(n_total, 1)
    var_pooled = p_pooled * (1 - p_pooled) * (1 / max(nA, 1) + 1 / max(nB, 1))

    return math.sqrt(var_pooled) if var_pooled > 0 else float("inf")


def unpooled_proportion_se(nA: int, nB: int, mA: int, mB: int) -> float:
    """Compute unpooled standard error for difference of two proportions.

    Uses separate variance estimators for each group.

    Args:
        nA: Total trials in group A
        nB: Total trials in group B
        mA: Successes in group A
        mB: Successes in group B

    Returns:
        Unpooled standard error of pA - pB
    """
    import math

    if min(nA, nB) == 0:
        return float("inf")

    pA_hat = mA / max(nA, 1)
    pB_hat = mB / max(nB, 1)
    var = pA_hat * (1 - pA_hat) / max(nA, 1) + pB_hat * (1 - pB_hat) / max(nB, 1)

    return math.sqrt(var) if var > 0 else float("inf")
