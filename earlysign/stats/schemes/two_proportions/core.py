"""
earlysign.stats.schemes.two_proportions.core
============================================

Core data structures and utilities for two-proportions testing.

This module provides the foundational elements for comparing two binomial
proportions:

- Type definitions and payload schemas
- Data validation and ingestion components
- Data aggregation utilities from the event store

All components follow event-sourcing principles with immutable events
and complete audit trails for regulatory compliance.

Examples
--------
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.stats.schemes.two_proportions.core import TwoPropObservation, ObservationBatch
>>>
>>> # Create observation component
>>> observation = TwoPropObservation()
>>> conn = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(conn, "test")
>>>
>>> # Create and populate a batch
>>> batch = ObservationBatch()
>>> batch.add_group_a_observations(successes=8, total=20)
>>> batch.add_group_b_observations(successes=12, total=20)
>>>
>>> # Validate and register the batch
>>> result = observation.register_batch(ledger, "exp1", "step1", "t1", batch)
>>> result  # Should return True if successful
True
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Tuple, TypedDict

from earlysign.core.components import Observer
from earlysign.core.ledger import Ledger, PayloadTypeRegistry, PayloadType
from earlysign.core.names import ExperimentId, StepKey, TimeIndex, Namespace


# --- Type Definitions and Payload Schemas ---


class WaldZPayload(TypedDict):
    """Payload for Wald Z-statistic results."""

    z: float
    se: float
    nA: int
    nB: int
    mA: int
    mB: int
    pA_hat: float
    pB_hat: float


class GstBoundaryPayload(TypedDict):
    """Payload for GST boundary computation results."""

    upper: float
    lower: float
    info_time: float
    alpha_i: float


@dataclass(frozen=True)
class TwoPropObsBatch:
    """Typed observation batch for two-proportions testing."""

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


# --- Data Validation and Ingestion ---


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
        experiment_id: Union[str, ExperimentId],
        step_key: Union[str, StepKey],
        time_index: Union[str, TimeIndex],
        batch: ObservationBatch,
        force: bool = False,
    ) -> bool:
        """
        Register an observation batch to the ledger.

        Parameters
        ----------
        ledger : Ledger
            The event store
        experiment_id : str or ExperimentId
            Experiment identifier
        step_key : str or StepKey
            Step identifier
        time_index : str or TimeIndex
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


# --- Data Aggregation Utilities ---


def reduce_counts(ledger: Ledger, *, experiment_id: str) -> Tuple[int, int, int, int]:
    """
    Aggregate counts from all observation records for the experiment.

    Parameters
    ----------
    ledger : Ledger
        The event store to read from
    experiment_id : str
        Experiment identifier to filter by

    Returns
    -------
    tuple[int, int, int, int]
        (nA, nB, mA, mB) - total counts and successes for groups A and B

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.core.names import Namespace
    >>> from earlysign.stats.schemes.two_proportions.core import TwoPropObsBatch
    >>> conn = ibis.duckdb.connect(":memory:")
    >>> L = Ledger(conn, "test")
    >>> L.write_event(time_index="t1", namespace=Namespace.OBS, kind="observation",
    ...               experiment_id="exp#1", step_key="s1",
    ...               payload_type="TwoPropObsBatch", payload={"nA":5,"nB":6,"mA":1,"mB":0})
    >>> reduce_counts(L, experiment_id="exp#1")
    (5, 6, 1, 0)
    """
    observations = ledger.table.filter(
        (ledger.table.namespace == str(Namespace.OBS))
        & (ledger.table.entity.startswith(f"{experiment_id}#"))
    )

    # Use select with JSON extraction to sum counts directly
    result = observations.select(
        nA=observations.payload["nA"].int.sum(),
        nB=observations.payload["nB"].int.sum(),
        mA=observations.payload["mA"].int.sum(),
        mB=observations.payload["mB"].int.sum(),
    ).execute()

    if result.empty:
        return 0, 0, 0, 0

    row = result.iloc[0]
    return (
        int(row["nA"] or 0),
        int(row["nB"] or 0),
        int(row["mA"] or 0),
        int(row["mB"] or 0),
    )


# Register payload encoders/decoders for two-proportions types
def _register_two_prop_payloads() -> None:
    """Register two-proportions payload types for efficient structured queries."""

    # TwoPropObsBatch handler
    class TwoPropObsBatchHandler(PayloadType):
        def wrap(self, data: Union[TwoPropObsBatch, Dict[str, Any]]) -> str:
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

        def unwrap(self, json_str: str) -> TwoPropObsBatch:
            payload = PayloadTypeRegistry._default_handler.unwrap(json_str)
            return TwoPropObsBatch(
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
