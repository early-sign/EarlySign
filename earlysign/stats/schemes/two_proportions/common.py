"""
earlysign.stats.schemes.two_proportions.common
=============================================

Common utilities and data structures for two-proportions testing.

This module provides shared functionality used across different testing
methods within the two-proportions scheme, including data structures,
aggregation functions, and basic computations.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict, Union, List, Dict, Any, Optional, Tuple

from earlysign.core.components import Observer
from earlysign.core.ledger import Ledger, PayloadTypeRegistry, PayloadType
from earlysign.core.ledger import Namespace


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


class TwoPropObsBatch(TypedDict):
    """Payload structure for two-proportions observation batch."""

    nA: int  # Total trials in group A
    nB: int  # Total trials in group B
    mA: int  # Successes in group A
    mB: int  # Successes in group B


# --- Data Classes ---


@dataclass(frozen=True)
class TwoPropObsBatchData:
    """Typed observation batch data for two-proportions testing."""

    nA: int
    nB: int
    mA: int
    mB: int


@dataclass(frozen=True)
class TwoPropDesign:
    """Design specification for two-proportions experiments."""

    design_id: str
    version: int
    alpha: float
    power: float
    pA: float
    pB: float
    looks: int
    spending: str
    t_grid: Tuple[float, ...]
    note: str = ""


@dataclass
class ObservationBatch:
    """
    A batch of observations for two-proportions testing.

    Provides validation, accumulation, and transformation capabilities
    for binomial observations before registration in the ledger.
    """

    group_a_successes: int = 0
    group_a_total: int = 0
    group_b_successes: int = 0
    group_b_total: int = 0

    # Optional metadata
    timestamp: Optional[datetime] = None
    source_info: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)

    def add_group_a_observations(self, successes: int, total: int) -> None:
        """Add observations for group A."""
        if successes < 0 or total < 0:
            self.validation_errors.append("Counts cannot be negative")
        if successes > total:
            self.validation_errors.append("Successes cannot exceed total for group A")

        self.group_a_successes += successes
        self.group_a_total += total

    def add_group_b_observations(self, successes: int, total: int) -> None:
        """Add observations for group B."""
        if successes < 0 or total < 0:
            self.validation_errors.append("Counts cannot be negative")
        if successes > total:
            self.validation_errors.append("Successes cannot exceed total for group B")

        self.group_b_successes += successes
        self.group_b_total += total

    def add_individual_observation(self, group: str, success: bool) -> None:
        """Add a single observation to the specified group."""
        if group.lower() in ("a", "group_a", "control"):
            self.group_a_total += 1
            if success:
                self.group_a_successes += 1
        elif group.lower() in ("b", "group_b", "treatment"):
            self.group_b_total += 1
            if success:
                self.group_b_successes += 1
        else:
            self.validation_errors.append(f"Unknown group: {group}")

    def validate(self) -> bool:
        """Validate the batch and return True if valid."""
        # Clear previous validation errors for counts
        self.validation_errors = [
            e for e in self.validation_errors if "cannot" not in e
        ]

        if self.group_a_successes < 0 or self.group_a_total < 0:
            self.validation_errors.append("Group A counts cannot be negative")
        if self.group_b_successes < 0 or self.group_b_total < 0:
            self.validation_errors.append("Group B counts cannot be negative")
        if self.group_a_successes > self.group_a_total:
            self.validation_errors.append("Group A successes cannot exceed total")
        if self.group_b_successes > self.group_b_total:
            self.validation_errors.append("Group B successes cannot exceed total")

        return len(self.validation_errors) == 0

    def is_empty(self) -> bool:
        """Check if the batch has any observations."""
        return self.group_a_total == 0 and self.group_b_total == 0

    def get_rates(self) -> Dict[str, float]:
        """Get success rates for both groups."""
        return {
            "group_a_rate": (
                self.group_a_successes / self.group_a_total
                if self.group_a_total > 0
                else 0.0
            ),
            "group_b_rate": (
                self.group_b_successes / self.group_b_total
                if self.group_b_total > 0
                else 0.0
            ),
        }

    def to_payload(self) -> Dict[str, Any]:
        """Convert to payload format for ledger registration."""
        return {
            "nA": self.group_a_total,
            "nB": self.group_b_total,
            "mA": self.group_a_successes,
            "mB": self.group_b_successes,
        }

    def reset(self) -> None:
        """Reset the batch to empty state."""
        self.group_a_successes = 0
        self.group_a_total = 0
        self.group_b_successes = 0
        self.group_b_total = 0
        self.validation_errors.clear()
        self.source_info.clear()


@dataclass(kw_only=True)
class TwoPropObservation(Observer):
    """
    Data observation component for two-proportions experiments.

    Handles validation, transformation, and registration of binomial observations
    with full event-sourcing integration and audit trail capabilities.

    Parameters
    ----------
    auto_validate : bool, default=True
        Whether to automatically validate batches before registration
    require_both_groups : bool, default=False
        Whether to require observations from both groups in each batch
    tag_obs : str, default="obs"
        Tag to use for observation events
    """

    auto_validate: bool = True
    require_both_groups: bool = False

    # Current batch being built (for step-based interface)
    current_batch: Optional[ObservationBatch] = field(default=None, init=False)

    def create_batch(self, timestamp: Optional[datetime] = None) -> ObservationBatch:
        """Create a new observation batch."""
        batch = ObservationBatch()
        if timestamp:
            batch.timestamp = timestamp
        return batch

    def register_batch(
        self,
        ledger: Ledger,
        experiment_id: str,
        step_key: str,
        time_index: str,
        batch: ObservationBatch,
        force: bool = False,
    ) -> bool:
        """
        Register an observation batch to the ledger.

        Parameters
        ----------
        ledger : Ledger
            The event store
        experiment_id : str
            Experiment identifier
        step_key : str
            Step identifier
        time_index : str
            Time index for this observation
        batch : ObservationBatch
            The batch to register
        force : bool, default=False
            Skip validation if True

        Returns
        -------
        bool
            True if registration succeeded, False otherwise
        """
        if not force and self.auto_validate and not batch.validate():
            return False

        if not force and batch.is_empty():
            return False

        if not force and self.require_both_groups:
            if batch.group_a_total == 0 or batch.group_b_total == 0:
                batch.validation_errors.append("Both groups must have observations")
                return False

        # Register to ledger
        ledger.write_event(
            time_index=str(time_index),
            namespace=self.ns_obs,
            kind="observation",
            experiment_id=str(experiment_id),
            step_key=str(step_key),
            payload_type="TwoPropObsBatch",
            payload=batch.to_payload(),
            tag=self.tag_obs,
            ts=batch.timestamp or datetime.now(timezone.utc),
        )

        return True

    def step(
        self, ledger: Ledger, experiment_id: str, step_key: str, time_index: str
    ) -> None:
        """
        Component step interface - registers current_batch if present.

        This allows the observation component to be used in component pipelines alongside
        statistics, criteria, and signalers.
        """
        if self.current_batch is not None:
            success = self.register_batch(
                ledger, experiment_id, step_key, time_index, self.current_batch
            )
            if not success:
                # Could emit an error event here
                pass

    def ingest_from_dict(
        self, data: Dict[str, Any], batch: Optional[ObservationBatch] = None
    ) -> ObservationBatch:
        """
        Ingest data from a dictionary format.

        Expected formats:
        - {"group_a_success": 8, "group_a_total": 20, "group_b_success": 12, "group_b_total": 20}
        - {"nA": 20, "mA": 8, "nB": 20, "mB": 12}  # Ledger format
        - {"observations": [{"group": "a", "success": true}, ...]}  # Individual observations
        """
        if batch is None:
            batch = self.create_batch()

        # Handle different input formats
        if "group_a_success" in data and "group_a_total" in data:
            batch.add_group_a_observations(
                data["group_a_success"], data["group_a_total"]
            )
            if "group_b_success" in data and "group_b_total" in data:
                batch.add_group_b_observations(
                    data["group_b_success"], data["group_b_total"]
                )

        elif "nA" in data and "mA" in data:
            # Ledger format
            batch.add_group_a_observations(data.get("mA", 0), data.get("nA", 0))
            if "nB" in data and "mB" in data:
                batch.add_group_b_observations(data.get("mB", 0), data.get("nB", 0))

        elif "observations" in data:
            # Individual observations format
            for obs in data["observations"]:
                if "group" in obs and "success" in obs:
                    batch.add_individual_observation(obs["group"], obs["success"])

        else:
            batch.validation_errors.append(
                f"Unrecognized data format: {list(data.keys())}"
            )

        return batch

    def ingest_from_arrays(
        self,
        group_a_outcomes: List[bool],
        group_b_outcomes: List[bool],
        batch: Optional[ObservationBatch] = None,
    ) -> ObservationBatch:
        """Ingest from arrays of boolean outcomes."""
        if batch is None:
            batch = self.create_batch()

        # Count successes and totals
        batch.add_group_a_observations(
            successes=sum(group_a_outcomes), total=len(group_a_outcomes)
        )
        batch.add_group_b_observations(
            successes=sum(group_b_outcomes), total=len(group_b_outcomes)
        )

        return batch


# --- Aggregation Functions ---


def reduce_counts(
    ledger: Ledger,
    experiment_id: str,
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
        ledger.table.entity.startswith(str(experiment_id) + "#")
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
    ledger: Ledger, experiment_id: str, stat_tag: str
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
            & (ledger.table.entity.startswith(str(experiment_id) + "#"))
        )
        .order_by(ledger.table.ts.desc())
        .limit(1)
    )

    stat_results = stat_query.execute()

    if stat_results.empty:
        return None

    records = ledger.unwrap_results(stat_results)
    return records[0]["payload"] if records else None


def get_latest_criteria(
    ledger: Ledger, experiment_id: str, criteria_tag: str
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
            & (ledger.table.entity.startswith(str(experiment_id) + "#"))
        )
        .order_by(ledger.table.ts.desc())
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


# --- Payload Type Handlers ---


def _register_two_prop_payloads() -> None:
    """Register two-proportions payload types for efficient structured queries."""

    # TwoPropObsBatch handler
    class TwoPropObsBatchHandler(PayloadType):
        def wrap(self, data: Union[TwoPropObsBatchData, Dict[str, Any]]) -> str:
            if isinstance(data, dict):
                payload = {
                    "nA": data["nA"],
                    "nB": data["nB"],
                    "mA": data["mA"],
                    "mB": data["mB"],
                }
            else:
                payload = {
                    "nA": data.nA,
                    "nB": data.nB,
                    "mA": data.mA,
                    "mB": data.mB,
                }
            return PayloadTypeRegistry._default_handler.wrap(payload)

        def unwrap(self, json_str: str) -> TwoPropObsBatchData:
            payload = PayloadTypeRegistry._default_handler.unwrap(json_str)
            return TwoPropObsBatchData(
                nA=payload["nA"],
                nB=payload["nB"],
                mA=payload["mA"],
                mB=payload["mB"],
            )

    PayloadTypeRegistry.register("TwoPropObsBatch", TwoPropObsBatchHandler())

    # WaldZ payload handler
    class WaldZPayloadHandler(PayloadType):
        def wrap(self, data: Union[WaldZPayload, Dict[str, Any]]) -> str:
            return PayloadTypeRegistry._default_handler.wrap(dict(data))

        def unwrap(self, json_str: str) -> WaldZPayload:
            payload = PayloadTypeRegistry._default_handler.unwrap(json_str)
            return WaldZPayload(
                z=payload["z"],
                se=payload["se"],
                nA=payload["nA"],
                nB=payload["nB"],
                mA=payload["mA"],
                mB=payload["mB"],
                pA_hat=payload["pA_hat"],
                pB_hat=payload["pB_hat"],
            )

    PayloadTypeRegistry.register("WaldZPayload", WaldZPayloadHandler())


# Auto-register payload types
_register_two_prop_payloads()
