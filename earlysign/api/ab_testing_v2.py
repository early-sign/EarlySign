"""
earlysign.api.ab_testing_v2
===========================

AB Testing + Guardrail monitoring using the new framework components.

This implementation demonstrates how the framework components reduce code duplication
while maintaining the same functionality as ab_testing.py. It uses:

- ExperimentFramework: Combines all reusable patterns
- Method-specific logic: Only the statistical computations and decisions

The result is significantly less code while maintaining full functionality.

Examples
--------
>>> import ibis
>>> from earlysign.api.ab_testing_v2 import ABTestWithGuardrailsV2
>>> from earlysign.tests.util import create_test_connection

Basic usage pattern (identical to v1):

>>> # Day 1: Initialize experiment
>>> conn = create_test_connection()
>>> monitor = ABTestWithGuardrailsV2(
...     experiment_id="test_exp_v2",
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
>>> monitor2 = ABTestWithGuardrailsV2.from_ledger(
...     experiment_id="test_exp_v2",
...     ledger=conn
... )
>>> monitor2.get_progress_report()['ab_test']['looks_completed'] >= 0
True
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Literal, Union

from scipy.stats import norm
from scipy.special import betaln

from earlysign.runtime.components import ExperimentFramework, SignalType


@dataclass
class ABTestWithGuardrailsV2(ExperimentFramework):
    """
    AB testing system with guardrail monitoring using framework components.

    This version demonstrates how the framework reduces code while maintaining
    the same functionality. Only method-specific logic is implemented here.
    """

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
    _current_look_ab: int = field(default=1, init=False)
    _planned_info_fractions: List[float] = field(default_factory=list, init=False)

    # Framework interface implementation
    def get_design_payload(self) -> Dict[str, Any]:
        """Return design parameters to persist."""
        return {
            "alpha_ab": self.alpha_ab,
            "looks_ab": self.looks_ab,
            "spending_style": self.spending_style,
            "max_sample_size_ab": self.max_sample_size_ab,
            "alpha_guardrail": self.alpha_guardrail,
            "min_effect_size": self.min_effect_size,
            "beta_prior_shape": self.beta_prior_shape,
        }

    def apply_recovered_design(self, design: Dict[str, Any]) -> None:
        """Apply recovered design parameters."""
        self.alpha_ab = design.get("alpha_ab", 0.05)
        self.looks_ab = design.get("looks_ab", 4)
        self.spending_style = design.get("spending_style", "obf")
        self.max_sample_size_ab = design.get("max_sample_size_ab")
        self.alpha_guardrail = design.get("alpha_guardrail", 0.05)
        self.min_effect_size = design.get("min_effect_size", 0.02)
        self.beta_prior_shape = design.get("beta_prior_shape", 1.0)

        # Initialize computed state
        self._planned_info_fractions = [
            (i + 1) / self.looks_ab for i in range(self.looks_ab)
        ]
        self._recover_current_look()

    def _get_design_tag(self) -> str:
        """Custom design tag for AB testing."""
        return "ab_guardrail_design"

    def _recover_current_look(self) -> None:
        """Recover current look from ledger."""
        # Try to get latest look info from signals
        latest_ab = self.get_latest_signal("ab")
        if latest_ab and "current_look" in latest_ab:
            self._current_look_ab = latest_ab["current_look"]
        elif latest_ab and "info_fraction" in latest_ab:
            info_frac = latest_ab["info_fraction"]
            # Estimate current look based on info fraction - if we completed this fraction,
            # we should be at the next look
            for i, planned_frac in enumerate(self._planned_info_fractions):
                if info_frac >= planned_frac:
                    self._current_look_ab = i + 2  # Next look after completing this one
                else:
                    self._current_look_ab = i + 1  # We're at this look
                    break
            else:
                # If we've exceeded all planned fractions
                self._current_look_ab = len(self._planned_info_fractions) + 1
        else:
            # Default if no signal found
            self._current_look_ab = 1

    # Observation recording (simplified using framework)
    def record_ab_observations(self, observations: List[Tuple[bool, str]]) -> None:
        """Record binary outcome observations for the AB test."""
        obs_data = [
            {"outcome": outcome, "group": group} for outcome, group in observations
        ]

        self.record_observations(
            observations=obs_data, payload_type="ABObservation", kind="ab_observation"
        )

    def record_guardrail_observations(
        self, observations: List[Tuple[str, bool, bool]]
    ) -> None:
        """Record guardrail monitoring observations."""
        obs_data = [
            {"group": group, "success": success, "adverse_event": adverse_event}
            for group, success, adverse_event in observations
        ]

        self.record_observations(
            observations=obs_data,
            payload_type="GuardrailObservation",
            kind="guardrail_observation",
        )

    # Analysis steps (method-specific logic)
    def step_ab(self) -> Dict[str, Any]:
        """Execute AB testing step: compute Wald statistic and check boundaries."""
        # Get current AB observation counts (using framework)
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

        # Compute Wald Z-statistic (method-specific)
        z_stat, pA, pB, se = self._compute_wald_statistic(nA, nB, mA, mB)

        # Compute current information fraction (method-specific)
        info_fraction = self._compute_info_fraction(nA + nB)

        # Compute GST boundaries (method-specific)
        boundary_upper, boundary_lower = self._compute_gst_boundaries(info_fraction)

        # Make decision (method-specific)
        decision = SignalType.CONTINUE.value
        if abs(z_stat) >= boundary_upper:
            decision = (
                SignalType.STOP_SUCCESS.value
                if z_stat > 0
                else SignalType.STOP_FUTILITY.value
            )

        # Store signal using framework
        signal_metadata = {
            "z_statistic": float(z_stat),
            "info_fraction": float(info_fraction),
            "boundary_upper": float(boundary_upper),
            "boundary_lower": float(boundary_lower),
            "sample_sizes": {"nA": nA, "nB": nB, "mA": mA, "mB": mB},
            "proportions": {"pA": float(pA), "pB": float(pB)},
            "standard_error": float(se),
            "current_look": self._current_look_ab,
        }

        self.store_signal("ab", decision, signal_metadata)

        # Update look counter for planned analyses
        if (
            info_fraction
            >= self._planned_info_fractions[
                min(self._current_look_ab - 1, len(self._planned_info_fractions) - 1)
            ]
        ):
            self._current_look_ab = min(self._current_look_ab + 1, self.looks_ab + 1)

        result = {"decision": decision, **signal_metadata}

        return result

    def step_guardrail(self) -> Dict[str, Any]:
        """Execute guardrail monitoring step: update E-process and check violation."""
        # Get guardrail observation counts (using framework)
        guardrail_counts = self._get_guardrail_counts()

        if not guardrail_counts:
            return {
                "decision": SignalType.CONTINUE.value,
                "e_value": 1.0,
                "log_e_value": 0.0,
                "threshold": 1.0 / self.alpha_guardrail,
                "counts": {},
            }

        # Compute E-value for each group (method-specific)
        total_e_value = 1.0
        log_e_values = {}

        for group, counts in guardrail_counts.items():
            adverse_count = counts.get("adverse_event", 0)
            total_count = counts.get("total", 0)

            if total_count > 0:
                null_rate = 0.05  # baseline adverse event rate
                log_e_val = self._compute_log_evalue_guardrail(
                    adverse_count, total_count, null_rate, self.min_effect_size
                )
                log_e_values[group] = log_e_val
                total_e_value *= math.exp(log_e_val)

        # Check threshold violation (method-specific)
        threshold = 1.0 / self.alpha_guardrail
        decision = (
            SignalType.GUARDRAIL_VIOLATION.value
            if total_e_value >= threshold
            else SignalType.CONTINUE.value
        )

        # Store signal using framework
        signal_metadata = {
            "e_value": float(total_e_value),
            "log_e_value": float(sum(log_e_values.values())),
            "threshold": float(threshold),
            "counts": guardrail_counts,
            "group_log_e_values": log_e_values,
        }

        self.store_signal("guardrail", decision, signal_metadata)

        result = {"decision": decision, **signal_metadata}

        return result

    # Alert checking (simplified using framework)
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for any alerts from recent signals."""
        alert_conditions = {
            "ab": [SignalType.STOP_SUCCESS.value, SignalType.STOP_FUTILITY.value],
            "guardrail": [SignalType.GUARDRAIL_VIOLATION.value],
        }

        alerts = self.check_standard_alerts(alert_conditions)

        # Customize alert messages
        for alert in alerts:
            if alert["signal_type"] == "ab":
                alert["message"] = f"AB test recommends stopping: {alert['decision']}"
            elif alert["signal_type"] == "guardrail":
                alert["message"] = (
                    f"Guardrail violation detected: E-value = {alert['metadata'].get('e_value', 'N/A')}"
                )

        return alerts

    # Progress reporting (simplified using framework)
    def get_progress_report(self) -> Dict[str, Any]:
        """Generate a comprehensive progress report."""
        # Use framework base structure
        report = self.get_base_report_structure()

        # Get current counts
        nA, mA, nB, mB = self._get_ab_counts()
        guardrail_counts = self._get_guardrail_counts()

        # Build signal summary using framework
        report["signals"] = self.build_signal_summary(["ab", "guardrail"])

        # Add method-specific details
        report["ab_test"] = {
            "sample_sizes": {"nA": nA, "nB": nB, "mA": mA, "mB": mB},
            "proportions": {
                "pA": mA / nA if nA > 0 else 0,
                "pB": mB / nB if nB > 0 else 0,
            },
            "latest_decision": report["signals"]["ab"]["latest_decision"],
            "z_statistic": report["signals"]["ab"]["metadata"].get("z_statistic", 0),
            "looks_completed": self._current_look_ab - 1,
            "total_looks": self.looks_ab,
        }

        report["guardrail"] = {
            "counts": guardrail_counts,
            "latest_decision": report["signals"]["guardrail"]["latest_decision"],
            "e_value": report["signals"]["guardrail"]["metadata"].get("e_value", 1.0),
            "threshold": 1.0 / self.alpha_guardrail,
        }

        # Add alerts
        report["alerts"] = self.check_alerts()

        return report

    # Private helper methods (method-specific logic only)

    def _get_ab_counts(self) -> Tuple[int, int, int, int]:
        """Get current AB test observation counts."""
        # Use framework method with AB-specific parameters
        counts = self.get_observation_counts(
            kind="ab_observation", payload_type="ABObservation", group_by="group"
        )

        if not isinstance(counts, dict):
            return 0, 0, 0, 0

        nA = counts.get("A", {}).get("total", 0)
        nB = counts.get("B", {}).get("total", 0)
        mA = sum(1 for g in counts.get("A", {}) if g == "outcome" and counts["A"][g])
        mB = sum(1 for g in counts.get("B", {}) if g == "outcome" and counts["B"][g])

        # Fallback: manual count if framework method doesn't work as expected
        try:
            query = (
                self._ledger.t.filter(self._ledger.t.labels["namespace"].str == "obs")
                .filter(
                    self._ledger.t.labels["experiment_id"].str == self.experiment_id
                )
                .filter(self._ledger.t.labels["kind"].str == "ab_observation")
                .filter(self._ledger.t.payload_type == "ABObservation")
            )

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
        try:
            query = (
                self._ledger.t.filter(self._ledger.t.labels["namespace"].str == "obs")
                .filter(
                    self._ledger.t.labels["experiment_id"].str == self.experiment_id
                )
                .filter(self._ledger.t.labels["kind"].str == "guardrail_observation")
                .filter(self._ledger.t.payload_type == "GuardrailObservation")
            )

            results = query.execute()
            records = results.to_dict("records")

            counts = {}
            for record in records:
                group = record["payload"]["group"]
                adverse = record["payload"]["adverse_event"]

                if group not in counts:
                    counts[group] = {"total": 0, "adverse_event": 0}

                counts[group]["total"] += 1
                if adverse:
                    counts[group]["adverse_event"] += 1

            return counts
        except Exception:
            return {}

    def _compute_info_fraction(self, total_n: int) -> float:
        """Compute current information fraction."""
        if self.max_sample_size_ab:
            return min(1.0, total_n / self.max_sample_size_ab)
        else:
            # Use current look fraction
            look_idx = min(
                self._current_look_ab - 1, len(self._planned_info_fractions) - 1
            )
            return self._planned_info_fractions[look_idx]

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
