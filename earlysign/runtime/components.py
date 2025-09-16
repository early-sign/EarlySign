"""
earlysign.runtime.components
===========================

Reusable components for experiment lifecycle management.

This module provides common patterns that can be shared across different
statistical methods, reducing code duplication while preserving the flexibility
needed for method-specific logic.

Components:
- PersistentExperiment: Design persistence and recovery from ledger
- ObservationRecorder: Typed observation recording and querying
- ExperimentReporter: Standard progress reporting structure
- SignalTracker: Signal storage and alert management

Examples:
--------
>>> import ibis
>>> from earlysign.runtime.components import PersistentExperiment, ObservationRecorder
>>> from earlysign.tests.util import create_test_connection

>>> class MyExperiment(PersistentExperiment, ObservationRecorder):
...     alpha: float = 0.05
...
...     def get_design_payload(self) -> Dict[str, Any]:
...         return {"alpha": self.alpha}
...
...     def apply_recovered_design(self, design: Dict[str, Any]) -> None:
...         self.alpha = design.get("alpha", 0.05)

>>> conn = create_test_connection()
>>> exp = MyExperiment(experiment_id="test", ledger=conn)
>>> exp.is_design_persisted()
True
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum

import ibis

from earlysign.core.ledger import Ledger
from earlysign.core.components import Namespace


class SignalType(str, Enum):
    """Standard signal types for experiments."""

    STOP_SUCCESS = "stop_success"
    STOP_FUTILITY = "stop_futility"
    CONTINUE = "continue"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    ERROR = "error"


@dataclass
class PersistentExperiment(ABC):
    """
    Base class for experiments with design persistence and recovery.

    Handles the common pattern of:
    1. Persisting experiment design parameters to ledger
    2. Recovering experiment state from ledger
    3. Managing ledger connections

    Subclasses must implement:
    - get_design_payload(): Return design parameters to persist
    - apply_recovered_design(): Apply recovered design to instance
    """

    experiment_id: str
    ledger: Union[Ledger, ibis.BaseBackend]

    # Internal state
    _ledger: Ledger = field(init=False)
    _design_recovered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize ledger connection and ensure design persistence."""
        # Handle ledger setup
        if isinstance(self.ledger, Ledger):
            self._ledger = self.ledger
        else:
            # Assume it's an ibis connection
            self._ledger = Ledger(self.ledger, f"ledger_{self.experiment_id}")

        self._ledger.ensure()

        # Persist design if this is initial creation
        if not self._design_recovered:
            self._persist_design()

        # Recover state from ledger if needed
        self._recover_design_if_needed()

    @classmethod
    def from_ledger(
        cls, experiment_id: str, ledger: Union[Ledger, ibis.BaseBackend], **kwargs
    ) -> "PersistentExperiment":
        """
        Recover an existing experiment from the ledger.

        Args:
            experiment_id: The experiment identifier
            ledger: Ledger instance or ibis connection
            **kwargs: Additional constructor arguments

        Returns:
            Recovered experiment instance

        Raises:
            ValueError: If no design found in ledger
        """
        instance = cls.__new__(cls)
        instance.experiment_id = experiment_id
        instance.ledger = ledger
        instance._design_recovered = True

        # Handle ledger setup
        if isinstance(ledger, Ledger):
            instance._ledger = ledger
        else:
            instance._ledger = Ledger(ledger, f"ledger_{experiment_id}")

        instance._ledger.ensure()

        # Recover design from ledger
        design = instance._get_design_from_ledger()
        if not design:
            raise ValueError(f"No design found for experiment {experiment_id}")

        # Apply recovered design
        instance.apply_recovered_design(design)

        # Set any additional kwargs
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        return instance

    def is_design_persisted(self) -> bool:
        """Check if the experiment design is already persisted in the ledger."""
        design = self._get_design_from_ledger()
        return design is not None

    @abstractmethod
    def get_design_payload(self) -> Dict[str, Any]:
        """
        Return the design parameters to persist.

        Returns:
            Dictionary of design parameters
        """
        pass

    @abstractmethod
    def apply_recovered_design(self, design: Dict[str, Any]) -> None:
        """
        Apply recovered design parameters to this instance.

        Args:
            design: Design parameters recovered from ledger
        """
        pass

    def _persist_design(self) -> None:
        """Persist experiment design parameters to the ledger."""
        design_payload = self.get_design_payload()
        design_payload["created_at"] = datetime.now(timezone.utc).isoformat()

        self._ledger.insert_event(
            payload_type="ExperimentDesign",
            payload=design_payload,
            labels={
                "namespace": Namespace.DESIGN.value,
                "kind": "design",
                "experiment_id": self.experiment_id,
                "tag": self._get_design_tag(),
            },
        )

    def _get_design_tag(self) -> str:
        """Get tag for design events. Override for custom tagging."""
        return "experiment_design"

    def _get_design_from_ledger(self) -> Optional[Dict[str, Any]]:
        """Recover experiment design from the ledger."""
        query = (
            self._ledger.t.filter(
                self._ledger.t.labels["namespace"].str == Namespace.DESIGN.value
            )
            .filter(self._ledger.t.labels["experiment_id"].str == self.experiment_id)
            .filter(self._ledger.t.labels["kind"].str == "design")
            .filter(self._ledger.t.payload_type == "ExperimentDesign")
            .order_by(self._ledger.t.ts.desc())
            .limit(1)
        )

        try:
            results = query.execute()
            if len(results) > 0:
                row = results.to_dict("records")[0]
                return dict(row["payload"])
        except Exception:
            return None

        return None

    def _recover_design_if_needed(self) -> None:
        """Recover design from ledger if this instance was created normally."""
        if not self._design_recovered:
            design = self._get_design_from_ledger()
            if design:
                self.apply_recovered_design(design)


@dataclass
class ObservationRecorder:
    """
    Handles typed observation recording and querying.

    Provides standardized methods for:
    - Recording observations with proper timestamps and structure
    - Querying observation counts with flexible filtering
    - Managing observation payload types

    Requires: experiment_id and _ledger attributes (from PersistentExperiment)
    """

    def record_observations(
        self,
        observations: List[Dict[str, Any]],
        payload_type: str,
        kind: str,
        namespace: str = Namespace.OBS.value,
        extra_labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a batch of observations.

        Args:
            observations: List of observation dictionaries
            payload_type: Type identifier for the payload
            kind: Kind of observation (e.g., "ab_observation")
            namespace: Namespace for the observations
            extra_labels: Additional labels to attach
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        for obs in observations:
            # Ensure observation has timestamp
            obs_payload = dict(obs)
            if "timestamp" not in obs_payload:
                obs_payload["timestamp"] = timestamp

            # Build labels
            labels = {
                "namespace": namespace,
                "kind": kind,
                "experiment_id": getattr(self, "experiment_id"),
            }
            if extra_labels:
                labels.update(extra_labels)

            # Add any additional labels from the observation itself
            if "group" in obs_payload:
                labels["group"] = obs_payload["group"]

            getattr(self, "_ledger").insert_event(
                payload_type=payload_type, payload=obs_payload, labels=labels
            )

    def get_observation_counts(
        self,
        kind: str,
        payload_type: str,
        group_by: Optional[str] = None,
        count_field: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> Union[int, Dict[str, Any]]:
        """
        Get aggregated counts of observations.

        Args:
            kind: Kind of observation to query
            payload_type: Payload type to filter by
            group_by: Field to group results by (e.g., "group")
            count_field: Field to count occurrences of
            filter_conditions: Additional filter conditions

        Returns:
            Integer count if not grouped, dictionary of counts if grouped
        """
        # Build base query
        ledger = getattr(self, "_ledger")
        query = (
            ledger.t.filter(ledger.t.labels["namespace"].str == Namespace.OBS.value)
            .filter(
                ledger.t.labels["experiment_id"].str == getattr(self, "experiment_id")
            )
            .filter(ledger.t.labels["kind"].str == kind)
            .filter(ledger.t.payload_type == payload_type)
        )

        # Apply additional filters
        if filter_conditions:
            for field, value in filter_conditions.items():
                if field.startswith("labels."):
                    label_key = field[7:]  # Remove "labels." prefix
                    query = query.filter(ledger.t.labels[label_key].str == str(value))

        try:
            results = query.execute()
            records = results.to_dict("records")

            if not group_by:
                # Simple count
                if count_field:
                    return sum(1 for r in records if r["payload"].get(count_field))
                return len(records)

            # Grouped counts
            grouped = {}
            for record in records:
                group_value = record["payload"].get(group_by)
                if group_value not in grouped:
                    grouped[group_value] = {"total": 0}

                grouped[group_value]["total"] += 1

                # Count specific field if requested
                if count_field and record["payload"].get(count_field):
                    if count_field not in grouped[group_value]:
                        grouped[group_value][count_field] = 0
                    grouped[group_value][count_field] += 1

            return grouped

        except Exception:
            return {} if group_by else 0


@dataclass
class SignalTracker:
    """
    Handles signal storage and alert management.

    Provides standardized methods for:
    - Storing analysis signals with metadata
    - Retrieving latest signals
    - Checking for alerts based on signal patterns

    Requires: experiment_id and _ledger attributes (from PersistentExperiment)
    """

    def store_signal(
        self,
        signal_type: str,
        decision: str,
        metadata: Dict[str, Any],
        namespace: str = Namespace.SIGNALS.value,
    ) -> None:
        """
        Store an analysis signal.

        Args:
            signal_type: Type of signal (e.g., "ab", "guardrail")
            decision: Decision made (e.g., SignalType.CONTINUE.value)
            metadata: Additional signal metadata
            namespace: Namespace for signals
        """
        payload = {
            "decision": decision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }

        getattr(self, "_ledger").insert_event(
            payload_type=f"{signal_type.title()}Signal",
            payload=payload,
            labels={
                "namespace": namespace,
                "kind": f"{signal_type}_signal",
                "experiment_id": getattr(self, "experiment_id"),
                "tag": f"{signal_type}_decision",
            },
        )

    def get_latest_signal(self, signal_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest signal of specified type.

        Args:
            signal_type: Type of signal to retrieve

        Returns:
            Latest signal data or None
        """
        kind_filter = f"{signal_type}_signal"

        ledger = getattr(self, "_ledger")
        query = (
            ledger.t.filter(ledger.t.labels["namespace"].str == Namespace.SIGNALS.value)
            .filter(
                ledger.t.labels["experiment_id"].str == getattr(self, "experiment_id")
            )
            .filter(ledger.t.labels["kind"].str == kind_filter)
            .order_by(ledger.t.ts.desc())
            .limit(1)
        )

        try:
            results = query.execute()
            if len(results) > 0:
                row = results.to_dict("records")[0]
                payload = dict(row["payload"])
                payload["timestamp"] = row["ts"]
                return payload
        except Exception:
            return None

        return None

    def check_standard_alerts(
        self, alert_conditions: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Check for alerts based on signal patterns.

        Args:
            alert_conditions: Dictionary mapping signal types to alert-worthy decisions

        Returns:
            List of alert dictionaries
        """
        alerts = []

        for signal_type, alert_decisions in alert_conditions.items():
            latest_signal = self.get_latest_signal(signal_type)
            if latest_signal and latest_signal.get("decision") in alert_decisions:
                alerts.append(
                    {
                        "type": f"{signal_type}_alert",
                        "decision": latest_signal["decision"],
                        "signal_type": signal_type,
                        "message": f"{signal_type} signal: {latest_signal['decision']}",
                        "timestamp": latest_signal.get("timestamp"),
                        "metadata": {
                            k: v
                            for k, v in latest_signal.items()
                            if k not in ["decision", "timestamp"]
                        },
                    }
                )

        return alerts


@dataclass
class ExperimentReporter:
    """
    Provides standardized progress reporting structure.

    Builds consistent progress reports with:
    - Experiment metadata
    - Current status summary
    - Signal history
    - Alert status

    Requires: experiment_id attribute (from PersistentExperiment)
    """

    def get_base_report_structure(self) -> Dict[str, Any]:
        """
        Get the base structure for progress reports.

        Returns:
            Base report structure with common fields
        """
        return {
            "experiment_id": getattr(self, "experiment_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": {},
            "alerts": [],
            "metadata": {},
        }

    def build_signal_summary(self, signal_types: List[str]) -> Dict[str, Any]:
        """
        Build summary of latest signals for each type.

        Args:
            signal_types: List of signal types to include

        Returns:
            Dictionary with latest signal info for each type
        """
        summary = {}

        for signal_type in signal_types:
            # Requires get_latest_signal method (from SignalTracker mixin)
            latest = getattr(self, "get_latest_signal")(signal_type)
            summary[signal_type] = {
                "latest_decision": (
                    latest.get("decision", "no_analysis") if latest else "no_analysis"
                ),
                "timestamp": latest.get("timestamp") if latest else None,
                "metadata": (
                    {
                        k: v
                        for k, v in latest.items()
                        if k not in ["decision", "timestamp"]
                    }
                    if latest
                    else {}
                ),
            }

        return summary


# Convenience mixin that combines all components
@dataclass
class ExperimentFramework(
    PersistentExperiment, ObservationRecorder, SignalTracker, ExperimentReporter
):
    """
    Complete framework combining all reusable experiment components.

    Provides the full set of common functionality:
    - Design persistence and recovery
    - Observation recording and querying
    - Signal storage and alerts
    - Progress reporting

    Subclasses only need to implement:
    - get_design_payload()
    - apply_recovered_design()
    - Method-specific analysis logic
    """

    pass
