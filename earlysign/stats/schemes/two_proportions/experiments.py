"""
earlysign.stats.schemes.two_proportions.experiments
===================================================

Experiment templates and orchestration for two-proportions testing.

This module provides complete experiment implementations that coordinate
components for comparing two binomial proportions using different statistical
methodologies:

**Base Template:**
- `TwoPropTemplate`: Common functionality for binomial comparison experiments

**Group Sequential Testing:**
- `TwoPropGSTTemplate`: GST with error spending functions

**Safe Testing:**
- `TwoPropSafeTemplate`: Safe testing with beta-binomial e-processes

Each template handles component configuration, design registration, and
result extraction while maintaining complete event-sourcing audit trails.

Examples
--------
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>> from earlysign.stats.schemes.two_proportions.experiments import TwoPropGSTTemplate

GST experiment:
>>> gst_exp = TwoPropGSTTemplate("test_gst", alpha_total=0.05, looks=4)
>>> ledger = PolarsLedger()
>>> gst_exp.setup(ledger)
>>> # Add observations and run analysis steps...

Safe Testing experiment:
>>> from earlysign.stats.schemes.two_proportions.experiments import TwoPropSafeTemplate
>>> safe_exp = TwoPropSafeTemplate("test_safe", alpha_total=0.05)
>>> safe_exp.setup(ledger)
>>> # Add observations and run analysis steps...
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union

from earlysign.core.ledger import Ledger
from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex
from earlysign.core.components import Statistic, Criteria, Signaler
from earlysign.runtime.experiment_template import ExperimentTemplate, AnalysisResult
from earlysign.stats.schemes.two_proportions.core import TwoPropObservation
from earlysign.stats.schemes.two_proportions.statistics import (
    # GST components
    WaldZStatistic,
    LanDeMetsBoundary,
    PeekSignaler,
    # Safe testing components
    BetaBinomialEValue,
    SafeThreshold,
    SafeSignaler,
)


# --- Base Template ---


class TwoPropTemplate(ExperimentTemplate):
    """
    Base class for two-proportions experiment templates.

    Provides common functionality for binomial comparison experiments
    while allowing different statistical methods to be plugged in.

    Attributes:
        experiment_id: Unique identifier for the experiment
        alpha_total: Total Type I error rate (default 0.05)
    """

    def __init__(
        self,
        experiment_id: str,
        alpha_total: float = 0.05,
    ):
        super().__init__(experiment_id)
        self.alpha_total = alpha_total

    def _populate_batch(self, batch: Any, **kwargs: Any) -> None:
        """Handle two-proportions specific batch population."""
        # Support multiple input formats
        if "group_a_success" in kwargs and "group_a_total" in kwargs:
            batch.add_group_a_observations(
                kwargs["group_a_success"], kwargs["group_a_total"]
            )
            if "group_b_success" in kwargs and "group_b_total" in kwargs:
                batch.add_group_b_observations(
                    kwargs["group_b_success"], kwargs["group_b_total"]
                )
        elif "data" in kwargs:
            # Support dictionary format
            batch = self.components["observation"].ingest_from_dict(
                kwargs["data"], batch
            )
        else:
            raise ValueError("Unsupported observation format for TwoPropTemplate")

    def get_summary(self) -> Dict[str, Any]:
        """Enhanced summary for two-proportions experiments."""
        summary = super().get_summary()
        summary.update(
            {
                "alpha_total": self.alpha_total,
                "experiment_type": "two_proportions",
            }
        )

        if self._is_setup and self.ledger:
            # Add observation counts
            obs_events = list(
                self.ledger.iter_ns(
                    namespace=Namespace.OBS, experiment_id=self.experiment_id
                )
            )

            total_nA = sum(event.payload.get("nA", 0) for event in obs_events)
            total_nB = sum(event.payload.get("nB", 0) for event in obs_events)
            total_mA = sum(event.payload.get("mA", 0) for event in obs_events)
            total_mB = sum(event.payload.get("mB", 0) for event in obs_events)

            summary.update(
                {
                    "total_observations": total_nA + total_nB,
                    "group_a": {
                        "total": total_nA,
                        "successes": total_mA,
                        "rate": total_mA / total_nA if total_nA > 0 else 0.0,
                    },
                    "group_b": {
                        "total": total_nB,
                        "successes": total_mB,
                        "rate": total_mB / total_nB if total_nB > 0 else 0.0,
                    },
                }
            )

        return summary


# --- Group Sequential Testing Template ---


class TwoPropGSTTemplate(TwoPropTemplate):
    """
    Two-proportions Group Sequential Testing experiment template.

    Implements group sequential testing with error spending functions
    for comparing two binomial proportions. Uses Wald Z-statistics
    with Lan-DeMets spending boundaries.

    Attributes:
        experiment_id: Unique identifier for the experiment
        alpha_total: Total Type I error rate (default 0.05)
        looks: Number of planned interim analyses (default 4)
        spending_function: Error spending style ("obf" or "pocock")
    """

    def __init__(
        self,
        experiment_id: str,
        alpha_total: float = 0.05,
        looks: int = 4,
        spending_function: str = "obf",  # "obf" or "pocock"
    ):
        super().__init__(experiment_id, alpha_total)
        self.looks = looks
        self.spending_function = spending_function

    def configure_components(self) -> Dict[str, Any]:
        """Configure GST components."""
        return {
            "observation": TwoPropObservation(),
            "statistic": self._create_statistic(),
            "criteria": self._create_criteria(),
            "signaler": self._create_signaler(),
        }

    def _create_statistic(self) -> Statistic:
        """Create Z-statistic for proportions difference."""
        return WaldZStatistic()

    def _create_criteria(self) -> Criteria:
        """Create GST error spending criteria."""
        return LanDeMetsBoundary(
            alpha_total=self.alpha_total,
            t=1.0,  # This will need to be calculated dynamically
            style=self.spending_function,
        )

    def _create_signaler(self) -> Signaler:
        """Create signaler for GST decisions."""
        return PeekSignaler()

    def register_design(self, ledger: Ledger) -> None:
        """Register the GST experimental design."""
        design_payload = {
            "method": "group_sequential_testing",
            "alpha_total": self.alpha_total,
            "looks": self.looks,
            "spending_function": self.spending_function,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        ledger.write_event(
            namespace=Namespace.DESIGN,
            kind="experiment_design",
            payload_type="gst_design",
            experiment_id=str(self.experiment_id),
            step_key="design",
            time_index="t0",
            payload=design_payload,
        )

    def extract_results(self, ledger: Ledger) -> AnalysisResult:
        """Extract GST analysis results."""
        # Get the latest signal event
        signal_events = list(
            ledger.iter_ns(
                namespace=Namespace.SIGNALS, experiment_id=str(self.experiment_id)
            )
        )

        if not signal_events:
            raise ValueError("No signal events found")

        latest_signal = signal_events[-1]
        should_stop = latest_signal.payload.get("action") == "stop"

        # Get corresponding statistic and criteria events
        statistic_events = list(
            ledger.iter_ns(
                namespace=Namespace.STATS, experiment_id=str(self.experiment_id)
            )
        )
        criteria_events = list(
            ledger.iter_ns(
                namespace=Namespace.CRITERIA, experiment_id=str(self.experiment_id)
            )
        )

        latest_stat = statistic_events[-1] if statistic_events else None
        latest_criteria = criteria_events[-1] if criteria_events else None

        statistic_value = latest_stat.payload.get("z", 0.0) if latest_stat else 0.0
        threshold_value = (
            latest_criteria.payload.get("upper", None) if latest_criteria else None
        )

        # Calculate sample proportions from observations
        obs_events = list(
            ledger.iter_ns(
                namespace=Namespace.OBS, experiment_id=str(self.experiment_id)
            )
        )

        total_nA = sum(event.payload.get("nA", 0) for event in obs_events)
        total_nB = sum(event.payload.get("nB", 0) for event in obs_events)
        total_mA = sum(event.payload.get("mA", 0) for event in obs_events)
        total_mB = sum(event.payload.get("mB", 0) for event in obs_events)

        return AnalysisResult(
            should_stop=should_stop,
            statistic_value=statistic_value,
            threshold_value=threshold_value,
            look_number=len(signal_events),
            group_a_rate=total_mA / total_nA if total_nA > 0 else 0.0,
            group_b_rate=total_mB / total_nB if total_nB > 0 else 0.0,
            total_sample_size=total_nA + total_nB,
            additional_metrics={
                "z_statistic": statistic_value,
                "se": latest_stat.payload.get("se", 0.0) if latest_stat else 0.0,
                "spending_function": self.spending_function,
                "looks_completed": len(signal_events),
                "total_looks": self.looks,
                "info_fraction": (
                    latest_criteria.payload.get("info_time", 0.0)
                    if latest_criteria
                    else 0.0
                ),
            },
            statistic_event=latest_stat,
            criteria_event=latest_criteria,
            signal_event=latest_signal,
        )


# --- Safe Testing Template ---


class TwoPropSafeTemplate(TwoPropTemplate):
    """
    Two-proportions Safe Testing experiment template.

    Implements safe testing with beta-binomial e-processes for comparing
    two binomial proportions. Provides anytime-valid inference without
    traditional sequential testing corrections.

    Attributes:
        experiment_id: Unique identifier for the experiment
        alpha_total: Total Type I error rate (default 0.05)
        prior_params: Beta prior parameters (default uniform: alpha=1, beta=1)
    """

    def __init__(
        self,
        experiment_id: str,
        alpha_total: float = 0.05,
        prior_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__(experiment_id, alpha_total)
        self.prior_params = prior_params or {"alpha": 1.0, "beta": 1.0}

    def configure_components(self) -> Dict[str, Any]:
        """Configure Safe Testing components."""
        return {
            "observation": TwoPropObservation(),
            "statistic": self._create_statistic(),
            "criteria": self._create_criteria(),
            "signaler": self._create_signaler(),
        }

    def _create_statistic(self) -> Statistic:
        """Create beta-binomial e-process statistic."""
        return BetaBinomialEValue(
            alpha_prior=self.prior_params["alpha"], beta_prior=self.prior_params["beta"]
        )

    def _create_criteria(self) -> Criteria:
        """Create Safe Testing criteria."""
        return SafeThreshold(alpha_level=self.alpha_total)

    def _create_signaler(self) -> Signaler:
        """Create signaler for Safe Testing decisions."""
        return SafeSignaler()

    def register_design(self, ledger: Ledger) -> None:
        """Register the Safe Testing experimental design."""
        design_payload = {
            "method": "safe_testing",
            "alpha_total": self.alpha_total,
            "prior_params": self.prior_params,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        ledger.write_event(
            namespace=Namespace.DESIGN,
            kind="experiment_design",
            payload_type="safe_design",
            experiment_id=str(self.experiment_id),
            step_key="design",
            time_index="t0",
            payload=design_payload,
        )

    def extract_results(self, ledger: Ledger) -> AnalysisResult:
        """Extract Safe Testing analysis results."""
        # Get the latest signal event
        signal_events = list(
            ledger.iter_ns(
                namespace=Namespace.SIGNALS, experiment_id=str(self.experiment_id)
            )
        )

        if not signal_events:
            raise ValueError("No signal events found")

        latest_signal = signal_events[-1]
        should_stop = latest_signal.payload.get("action") == "stop"

        # Get corresponding statistic and criteria events
        statistic_events = list(
            ledger.iter_ns(
                namespace=Namespace.STATS, experiment_id=str(self.experiment_id)
            )
        )
        criteria_events = list(
            ledger.iter_ns(
                namespace=Namespace.CRITERIA, experiment_id=str(self.experiment_id)
            )
        )

        latest_stat = statistic_events[-1] if statistic_events else None
        latest_criteria = criteria_events[-1] if criteria_events else None

        e_value = latest_stat.payload.get("e_value", 1.0) if latest_stat else 1.0
        threshold_value = (
            latest_criteria.payload.get("threshold", None) if latest_criteria else None
        )

        # Calculate sample proportions from observations
        obs_events = list(
            ledger.iter_ns(
                namespace=Namespace.OBS, experiment_id=str(self.experiment_id)
            )
        )

        total_nA = sum(event.payload.get("nA", 0) for event in obs_events)
        total_nB = sum(event.payload.get("nB", 0) for event in obs_events)
        total_mA = sum(event.payload.get("mA", 0) for event in obs_events)
        total_mB = sum(event.payload.get("mB", 0) for event in obs_events)

        return AnalysisResult(
            should_stop=should_stop,
            statistic_value=e_value,
            threshold_value=threshold_value,
            look_number=len(signal_events),
            group_a_rate=total_mA / total_nA if total_nA > 0 else 0.0,
            group_b_rate=total_mB / total_nB if total_nB > 0 else 0.0,
            total_sample_size=total_nA + total_nB,
            additional_metrics={
                "e_value": e_value,
                "log_e_value": math.log(e_value) if e_value > 0 else float("-inf"),
                "prior_params": self.prior_params,
                "anytime_valid": True,
            },
            statistic_event=latest_stat,
            criteria_event=latest_criteria,
            signal_event=latest_signal,
        )
