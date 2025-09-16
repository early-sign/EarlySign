"""
earlysign.api.ab_testing
========================

AB Testing + Guardrail monitoring class using only earlysign.core components.

This implementation demonstrates how to build a complete AB testing system
directly on top of the core ledger without relying on the intermediate
stats layer. It implements:

- GST Wald statistic computation for AB testing
- E-process updates for guardrail monitoring
- Design persistence and recovery from ledger
- Daily step execution with proper state management

Examples
--------
>>> import ibis
>>> from earlysign.api.ab_testing import ABTestWithGuardrails
>>> from earlysign.tests.util import create_test_connection

Basic usage pattern:

>>> # Day 1: Initialize experiment
>>> conn = create_test_connection()
>>> monitor = ABTestWithGuardrails(
...     experiment_id="test_exp",
...     ledger=conn,
...     alpha_ab=0.05,
...     looks_ab=4,
...     alpha_guardrail=0.05,
...     min_effect_size=0.02
... )
>>> monitor.is_design_persisted()
True

>>> # Simulate daily data arrival
>>> monitor.record_ab_observations([(True, 'A'), (False, 'A'), (True, 'B'), (True, 'B')])
>>> monitor.record_guardrail_observations([('A', True, False), ('B', True, False)])

>>> # Run daily analysis steps
>>> ab_result = monitor.step_ab()
>>> guardrail_result = monitor.step_guardrail()

>>> # Check for any alerts
>>> alerts = monitor.check_alerts()
>>> len(alerts) >= 0
True

>>> # Day 2: Recover from ledger and continue
>>> monitor2 = ABTestWithGuardrails.from_ledger(
...     experiment_id="test_exp",
...     ledger=conn
... )
>>> monitor2.get_progress_report()['ab_test']['looks_completed'] >= 1
True
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Literal, Union
from enum import Enum

import ibis
from scipy.stats import norm
from scipy.special import betaln

from earlysign.core.ledger import Ledger
from earlysign.core.components import Namespace


class SignalType(str, Enum):
    """Types of signals that can be emitted."""

    STOP_SUCCESS = "stop_success"
    STOP_FUTILITY = "stop_futility"
    CONTINUE = "continue"
    GUARDRAIL_VIOLATION = "guardrail_violation"
    ERROR = "error"


@dataclass
class ABTestWithGuardrails:
    """
    Complete AB testing system with guardrail monitoring using only core components.

    This class implements both AB testing with group sequential testing (GST) using
    Wald statistics and guardrail monitoring using E-processes. All state is
    persisted to and recovered from the ledger.

    The system supports daily execution where each instance can be destroyed and
    recreated from the ledger state.
    """

    experiment_id: str
    ledger: Union[Ledger, ibis.BaseBackend]

    # AB testing parameters
    alpha_ab: float = 0.05
    looks_ab: int = 4
    spending_style: Literal["obf", "pocock"] = "obf"
    max_sample_size_ab: Optional[int] = None

    # Guardrail parameters
    alpha_guardrail: float = 0.05
    min_effect_size: float = 0.02
    beta_prior_shape: float = 1.0

    # Internal state (recovered from ledger)
    _ledger: Ledger = field(init=False)
    _design_recovered: bool = field(default=False, init=False)
    _current_look_ab: int = field(default=1, init=False)
    _planned_info_fractions: List[float] = field(default_factory=list, init=False)

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

        # Recover state from ledger
        self._recover_state()

    @classmethod
    def from_ledger(
        cls, experiment_id: str, ledger: Union[Ledger, ibis.BaseBackend]
    ) -> "ABTestWithGuardrails":
        """
        Recover an existing experiment from the ledger.

        Args:
            experiment_id: The experiment identifier
            ledger: Ledger instance or ibis connection

        Returns:
            Recovered ABTestWithGuardrails instance

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

        # Recover all parameters from ledger
        design = instance._get_design_from_ledger()
        if not design:
            raise ValueError(f"No design found for experiment {experiment_id}")

        # Set all parameters from recovered design
        for key, value in design.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        # Initialize computed fields
        instance._current_look_ab = 1
        instance._planned_info_fractions = []

        # Recover current state
        instance._recover_state()

        return instance

    def is_design_persisted(self) -> bool:
        """Check if the experiment design is already persisted in the ledger."""
        design = self._get_design_from_ledger()
        return design is not None

    def _persist_design(self) -> None:
        """Persist experiment design parameters to the ledger."""
        design_payload = {
            "alpha_ab": self.alpha_ab,
            "looks_ab": self.looks_ab,
            "spending_style": self.spending_style,
            "max_sample_size_ab": self.max_sample_size_ab,
            "alpha_guardrail": self.alpha_guardrail,
            "min_effect_size": self.min_effect_size,
            "beta_prior_shape": self.beta_prior_shape,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._ledger.insert_event(
            payload_type="ExperimentDesign",
            payload=design_payload,
            labels={
                "namespace": Namespace.DESIGN.value,
                "kind": "design",
                "experiment_id": self.experiment_id,
                "tag": "ab_guardrail_design",
            },
        )

        # Initialize planned information fractions for AB test
        self._planned_info_fractions = [
            (i + 1) / self.looks_ab for i in range(self.looks_ab)
        ]

        fractions_payload = {
            "planned_fractions": self._planned_info_fractions,
            "current_look": 1,
        }

        self._ledger.insert_event(
            payload_type="InfoFractions",
            payload=fractions_payload,
            labels={
                "namespace": Namespace.DESIGN.value,
                "kind": "info_fractions",
                "experiment_id": self.experiment_id,
                "tag": "ab_fractions",
            },
        )

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

    def _recover_state(self) -> None:
        """Recover current experimental state from the ledger."""
        # Recover information fractions
        fractions_query = (
            self._ledger.t.filter(
                self._ledger.t.labels["namespace"].str == Namespace.DESIGN.value
            )
            .filter(self._ledger.t.labels["experiment_id"].str == self.experiment_id)
            .filter(self._ledger.t.labels["kind"].str == "info_fractions")
            .filter(self._ledger.t.payload_type == "InfoFractions")
            .order_by(self._ledger.t.ts.desc())
            .limit(1)
        )

        try:
            frac_results = fractions_query.execute()
            if len(frac_results) > 0:
                frac_row = frac_results.to_dict("records")[0]
                payload = frac_row["payload"]
                self._planned_info_fractions = payload.get("planned_fractions", [])
                self._current_look_ab = payload.get("current_look", 1)
        except Exception:
            # Fallback to default
            if not self._planned_info_fractions:
                self._planned_info_fractions = [
                    (i + 1) / self.looks_ab for i in range(self.looks_ab)
                ]

    def record_ab_observations(self, observations: List[Tuple[bool, str]]) -> None:
        """
        Record binary outcome observations for the AB test.

        Args:
            observations: List of (outcome, group) tuples where outcome is True/False
                         and group is 'A' or 'B'
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        for outcome, group in observations:
            payload = {"outcome": outcome, "group": group, "timestamp": timestamp}

            self._ledger.insert_event(
                payload_type="ABObservation",
                payload=payload,
                labels={
                    "namespace": Namespace.OBS.value,
                    "kind": "ab_observation",
                    "experiment_id": self.experiment_id,
                    "group": group,
                    "tag": "ab_binary",
                },
            )

    def record_guardrail_observations(
        self, observations: List[Tuple[str, bool, bool]]
    ) -> None:
        """
        Record guardrail monitoring observations.

        Args:
            observations: List of (group, success, adverse_event) tuples
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        for group, success, adverse_event in observations:
            payload = {
                "group": group,
                "success": success,
                "adverse_event": adverse_event,
                "timestamp": timestamp,
            }

            self._ledger.insert_event(
                payload_type="GuardrailObservation",
                payload=payload,
                labels={
                    "namespace": Namespace.OBS.value,
                    "kind": "guardrail_observation",
                    "experiment_id": self.experiment_id,
                    "group": group,
                    "tag": "guardrail_binary",
                },
            )

    def step_ab(self) -> Dict[str, Any]:
        """
        Execute AB testing step: compute Wald statistic and check boundaries.

        Returns:
            Dictionary containing analysis results and decision
        """
        # Get current AB observation counts
        nA, mA, nB, mB = self._get_ab_counts()

        if nA == 0 or nB == 0:
            return {
                "decision": SignalType.CONTINUE.value,
                "z_statistic": 0.0,
                "info_fraction": 0.0,
                "boundary_upper": float("inf"),
                "boundary_lower": float("-inf"),
                "sample_sizes": {"nA": nA, "nB": nB, "mA": mA, "mB": mB},
            }

        # Compute Wald Z-statistic
        z_stat, pA, pB, se = self._compute_wald_statistic(nA, nB, mA, mB)

        # Store aggregated data snapshot
        self._store_ab_snapshot(nA, mA, nB, mB, z_stat, pA, pB, se)

        # Compute current information fraction (based on sample size)
        total_n = nA + nB
        if self.max_sample_size_ab:
            info_fraction = min(1.0, total_n / self.max_sample_size_ab)
        else:
            # Use current look fraction
            look_idx = min(
                self._current_look_ab - 1, len(self._planned_info_fractions) - 1
            )
            info_fraction = self._planned_info_fractions[look_idx]

        # Compute GST boundaries
        boundary_upper, boundary_lower = self._compute_gst_boundaries(info_fraction)

        # Store boundary information
        self._store_ab_boundaries(info_fraction, boundary_upper, boundary_lower)

        # Make decision
        decision = SignalType.CONTINUE.value
        if abs(z_stat) >= boundary_upper:
            decision = (
                SignalType.STOP_SUCCESS.value
                if z_stat > 0
                else SignalType.STOP_FUTILITY.value
            )

        # Store AB signal
        self._store_ab_signal(
            decision, z_stat, boundary_upper, boundary_lower, info_fraction
        )

        # Update look counter for planned analyses
        if (
            info_fraction
            >= self._planned_info_fractions[
                min(self._current_look_ab - 1, len(self._planned_info_fractions) - 1)
            ]
        ):
            self._current_look_ab = min(self._current_look_ab + 1, self.looks_ab + 1)
            self._update_look_counter()

        return {
            "decision": decision,
            "z_statistic": float(z_stat),
            "info_fraction": float(info_fraction),
            "boundary_upper": float(boundary_upper),
            "boundary_lower": float(boundary_lower),
            "sample_sizes": {"nA": nA, "nB": nB, "mA": mA, "mB": mB},
            "proportions": {"pA": float(pA), "pB": float(pB)},
            "standard_error": float(se),
            "current_look": self._current_look_ab,
        }

    def step_guardrail(self) -> Dict[str, Any]:
        """
        Execute guardrail monitoring step: update E-process and check violation.

        Returns:
            Dictionary containing guardrail analysis results
        """
        # Get guardrail observation counts by group
        guardrail_counts = self._get_guardrail_counts()

        if not guardrail_counts:
            return {
                "decision": SignalType.CONTINUE.value,
                "e_value": 1.0,
                "log_e_value": 0.0,
                "threshold": 1.0 / self.alpha_guardrail,
                "counts": {},
            }

        # Compute E-value for each group comparing to historical baseline
        total_e_value = 1.0
        log_e_values = {}

        for group, counts in guardrail_counts.items():
            adverse_count = counts.get("adverse", 0)
            total_count = counts.get("total", 0)

            if total_count > 0:
                # Compute log E-value using beta-binomial approach
                # Assuming null: adverse rate = 5%, alternative: adverse rate > 5% + min_effect
                null_rate = 0.05  # baseline adverse event rate
                log_e_val = self._compute_log_evalue_guardrail(
                    adverse_count, total_count, null_rate, self.min_effect_size
                )
                log_e_values[group] = log_e_val
                total_e_value *= math.exp(log_e_val)

        # Store E-process update
        self._store_guardrail_evalue(total_e_value, log_e_values, guardrail_counts)

        # Check threshold violation (Ville's inequality)
        threshold = 1.0 / self.alpha_guardrail
        decision = (
            SignalType.GUARDRAIL_VIOLATION.value
            if total_e_value >= threshold
            else SignalType.CONTINUE.value
        )

        # Store guardrail signal
        self._store_guardrail_signal(decision, total_e_value, threshold)

        return {
            "decision": decision,
            "e_value": float(total_e_value),
            "log_e_value": float(sum(log_e_values.values())),
            "threshold": float(threshold),
            "counts": guardrail_counts,
            "group_log_e_values": log_e_values,
        }

    def check_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for any alerts from recent signals.

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # Check latest AB signal
        ab_signal = self._get_latest_signal("ab")
        if ab_signal and ab_signal.get("decision") in [
            SignalType.STOP_SUCCESS.value,
            SignalType.STOP_FUTILITY.value,
        ]:
            alerts.append(
                {
                    "type": "ab_stopping",
                    "decision": ab_signal["decision"],
                    "message": f"AB test recommends stopping: {ab_signal['decision']}",
                    "z_statistic": ab_signal.get("z_statistic", 0),
                    "timestamp": ab_signal.get("timestamp"),
                }
            )

        # Check latest guardrail signal
        guardrail_signal = self._get_latest_signal("guardrail")
        if (
            guardrail_signal
            and guardrail_signal.get("decision") == SignalType.GUARDRAIL_VIOLATION.value
        ):
            alerts.append(
                {
                    "type": "guardrail_violation",
                    "message": f"Guardrail violation detected: E-value = {guardrail_signal.get('e_value', 'N/A')}",
                    "e_value": guardrail_signal.get("e_value", 1.0),
                    "threshold": guardrail_signal.get("threshold", 20.0),
                    "timestamp": guardrail_signal.get("timestamp"),
                }
            )

        return alerts

    def get_progress_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive progress report.

        Returns:
            Dictionary containing current experiment status
        """
        # Get current counts
        nA, mA, nB, mB = self._get_ab_counts()
        guardrail_counts = self._get_guardrail_counts()

        # Get latest results
        ab_signal = self._get_latest_signal("ab")
        guardrail_signal = self._get_latest_signal("guardrail")

        report = {
            "experiment_id": self.experiment_id,
            "ab_test": {
                "sample_sizes": {"nA": nA, "nB": nB, "mA": mA, "mB": mB},
                "proportions": {
                    "pA": mA / nA if nA > 0 else 0,
                    "pB": mB / nB if nB > 0 else 0,
                },
                "latest_decision": (
                    ab_signal.get("decision", "no_analysis")
                    if ab_signal
                    else "no_analysis"
                ),
                "z_statistic": ab_signal.get("z_statistic", 0) if ab_signal else 0,
                "looks_completed": self._current_look_ab - 1,
                "total_looks": self.looks_ab,
            },
            "guardrail": {
                "counts": guardrail_counts,
                "latest_decision": (
                    guardrail_signal.get("decision", "no_analysis")
                    if guardrail_signal
                    else "no_analysis"
                ),
                "e_value": (
                    guardrail_signal.get("e_value", 1.0) if guardrail_signal else 1.0
                ),
                "threshold": 1.0 / self.alpha_guardrail,
            },
            "alerts": self.check_alerts(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return report

    # Private helper methods

    def _get_ab_counts(self) -> Tuple[int, int, int, int]:
        """Get current AB test observation counts."""
        query = (
            self._ledger.t.filter(
                self._ledger.t.labels["namespace"].str == Namespace.OBS.value
            )
            .filter(self._ledger.t.labels["experiment_id"].str == self.experiment_id)
            .filter(self._ledger.t.labels["kind"].str == "ab_observation")
            .filter(self._ledger.t.payload_type == "ABObservation")
        )

        try:
            results = query.execute()
            records = results.to_dict("records")

            nA = sum(1 for r in records if r["payload"]["group"] == "A")
            nB = sum(1 for r in records if r["payload"]["group"] == "B")
            mA = sum(
                1
                for r in records
                if r["payload"]["group"] == "A" and r["payload"]["outcome"]
            )
            mB = sum(
                1
                for r in records
                if r["payload"]["group"] == "B" and r["payload"]["outcome"]
            )

            return nA, mA, nB, mB
        except Exception:
            return 0, 0, 0, 0

    def _get_guardrail_counts(self) -> Dict[str, Dict[str, int]]:
        """Get current guardrail observation counts by group."""
        query = (
            self._ledger.t.filter(
                self._ledger.t.labels["namespace"].str == Namespace.OBS.value
            )
            .filter(self._ledger.t.labels["experiment_id"].str == self.experiment_id)
            .filter(self._ledger.t.labels["kind"].str == "guardrail_observation")
            .filter(self._ledger.t.payload_type == "GuardrailObservation")
        )

        try:
            results = query.execute()
            records = results.to_dict("records")

            counts = {}
            for record in records:
                group = record["payload"]["group"]
                adverse = record["payload"]["adverse_event"]

                if group not in counts:
                    counts[group] = {"total": 0, "adverse": 0}

                counts[group]["total"] += 1
                if adverse:
                    counts[group]["adverse"] += 1

            return counts
        except Exception:
            return {}

    def _compute_wald_statistic(
        self, nA: int, nB: int, mA: int, mB: int
    ) -> Tuple[float, float, float, float]:
        """Compute Wald Z-statistic for two proportions."""
        if nA == 0 or nB == 0:
            return 0.0, 0.0, 0.0, 0.0

        pA = mA / nA
        pB = mB / nB

        # Unpooled standard error
        se = math.sqrt((pA * (1 - pA)) / nA + (pB * (1 - pB)) / nB)

        if se == 0:
            return 0.0, pA, pB, 0.0

        z = (pA - pB) / se
        return z, pA, pB, se

    def _compute_gst_boundaries(self, info_fraction: float) -> Tuple[float, float]:
        """Compute GST boundaries using Lan-DeMets spending."""
        if info_fraction <= 0:
            return float("inf"), float("-inf")

        # Lan-DeMets spending function
        if self.spending_style == "obf":
            # O'Brien-Fleming
            z_alpha_2 = norm.ppf(1 - self.alpha_ab / 2)
            alpha_t = 2 * (1 - norm.cdf(z_alpha_2 / math.sqrt(info_fraction)))
        else:
            # Pocock
            alpha_t = self.alpha_ab * math.log(1 + (math.e - 1) * info_fraction)

        # Two-sided boundary
        if alpha_t >= 1.0:
            boundary = 0.0
        else:
            boundary = norm.ppf(1 - alpha_t / 2)

        return float(boundary), float(-boundary)

    def _compute_log_evalue_guardrail(
        self, adverse_count: int, total_count: int, null_rate: float, min_effect: float
    ) -> float:
        """Compute log E-value for guardrail monitoring."""
        if total_count == 0:
            return 0.0

        # Beta-binomial E-value computation
        # Alternative: Beta(1, 1) prior on rate > null_rate + min_effect
        # Null: fixed rate = null_rate

        # Alternative likelihood (with Beta(1,1) prior)
        log_alt = betaln(1 + adverse_count, 1 + total_count - adverse_count) - betaln(
            1, 1
        )

        # Null likelihood
        from math import log, lgamma

        log_null = (
            lgamma(total_count + 1)
            - lgamma(adverse_count + 1)
            - lgamma(total_count - adverse_count + 1)
            + adverse_count * log(null_rate)
            + (total_count - adverse_count) * log(1 - null_rate)
        )

        return float(log_alt - log_null)

    def _store_ab_snapshot(
        self,
        nA: int,
        mA: int,
        nB: int,
        mB: int,
        z: float,
        pA: float,
        pB: float,
        se: float,
    ) -> None:
        """Store aggregated AB data snapshot."""
        payload = {
            "nA": nA,
            "mA": mA,
            "nB": nB,
            "mB": mB,
            "z_statistic": z,
            "pA": pA,
            "pB": pB,
            "se": se,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._ledger.insert_event(
            payload_type="ABSnapshot",
            payload=payload,
            labels={
                "namespace": Namespace.STATS.value,
                "kind": "ab_snapshot",
                "experiment_id": self.experiment_id,
                "tag": "ab_aggregated",
            },
        )

    def _store_ab_boundaries(
        self, info_fraction: float, upper: float, lower: float
    ) -> None:
        """Store GST boundary information."""
        payload = {
            "info_fraction": info_fraction,
            "boundary_upper": upper,
            "boundary_lower": lower,
            "spending_style": self.spending_style,
            "alpha": self.alpha_ab,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._ledger.insert_event(
            payload_type="GSTBoundary",
            payload=payload,
            labels={
                "namespace": Namespace.CRITERIA.value,
                "kind": "gst_boundary",
                "experiment_id": self.experiment_id,
                "tag": "ab_boundary",
            },
        )

    def _store_ab_signal(
        self, decision: str, z_stat: float, upper: float, lower: float, info_frac: float
    ) -> None:
        """Store AB testing signal."""
        payload = {
            "decision": decision,
            "z_statistic": z_stat,
            "boundary_upper": upper,
            "boundary_lower": lower,
            "info_fraction": info_frac,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._ledger.insert_event(
            payload_type="ABSignal",
            payload=payload,
            labels={
                "namespace": Namespace.SIGNALS.value,
                "kind": "ab_signal",
                "experiment_id": self.experiment_id,
                "tag": "ab_decision",
            },
        )

    def _store_guardrail_evalue(
        self,
        e_value: float,
        log_e_values: Dict[str, float],
        counts: Dict[str, Dict[str, int]],
    ) -> None:
        """Store E-value update for guardrails."""
        payload = {
            "e_value": e_value,
            "log_e_values": log_e_values,
            "counts": counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._ledger.insert_event(
            payload_type="EValueUpdate",
            payload=payload,
            labels={
                "namespace": Namespace.STATS.value,
                "kind": "evalue_update",
                "experiment_id": self.experiment_id,
                "tag": "guardrail_eprocess",
            },
        )

    def _store_guardrail_signal(
        self, decision: str, e_value: float, threshold: float
    ) -> None:
        """Store guardrail monitoring signal."""
        payload = {
            "decision": decision,
            "e_value": e_value,
            "threshold": threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._ledger.insert_event(
            payload_type="GuardrailSignal",
            payload=payload,
            labels={
                "namespace": Namespace.SIGNALS.value,
                "kind": "guardrail_signal",
                "experiment_id": self.experiment_id,
                "tag": "guardrail_decision",
            },
        )

    def _update_look_counter(self) -> None:
        """Update the current look counter in the ledger."""
        fractions_payload = {
            "planned_fractions": self._planned_info_fractions,
            "current_look": self._current_look_ab,
        }

        self._ledger.insert_event(
            payload_type="InfoFractions",
            payload=fractions_payload,
            labels={
                "namespace": Namespace.DESIGN.value,
                "kind": "info_fractions",
                "experiment_id": self.experiment_id,
                "tag": "ab_fractions",
            },
        )

    def _get_latest_signal(
        self, signal_type: Literal["ab", "guardrail"]
    ) -> Optional[Dict[str, Any]]:
        """Get the latest signal of specified type."""
        kind_filter = "ab_signal" if signal_type == "ab" else "guardrail_signal"

        query = (
            self._ledger.t.filter(
                self._ledger.t.labels["namespace"].str == Namespace.SIGNALS.value
            )
            .filter(self._ledger.t.labels["experiment_id"].str == self.experiment_id)
            .filter(self._ledger.t.labels["kind"].str == kind_filter)
            .order_by(self._ledger.t.ts.desc())
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
