"""
earlysign.schemes.two_proportions.safe
=====================================

Safe Testing implementations for two-proportions experiments.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional, Union, List, Dict, Any

from earlysign.core.ledger import Ledger
from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex
from earlysign.core.components import Statistic, Criteria, Signaler
from earlysign.runtime.experiment_module import AnalysisResult
from earlysign.stats.schemes.two_proportions.modules import TwoPropModule
from earlysign.stats.schemes.two_proportions.ingest import TwoPropObservation


class TwoPropSafeModule(TwoPropModule):
    """Two-proportions Safe Testing experiment module.

    Implements safe testing with beta-binomial e-processes
    for comparing two binomial proportions.
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
        from earlysign.stats.schemes.two_proportions.safe_components import (
            BetaBinomialEValue,
        )

        return BetaBinomialEValue(
            alpha_prior=self.prior_params["alpha"], beta_prior=self.prior_params["beta"]
        )

    def _create_criteria(self) -> Criteria:
        """Create Safe Testing criteria."""
        from earlysign.stats.schemes.two_proportions.safe_components import (
            SafeThreshold,
        )

        return SafeThreshold(alpha_level=self.alpha_total)

    def _create_signaler(self) -> Signaler:
        """Create signaler for Safe Testing decisions."""
        from earlysign.stats.schemes.two_proportions.safe_components import SafeSignaler

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
        should_stop = latest_signal.payload.get("stop_decision", False)

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

        # Calculate sample proportions
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
            look_number=self._current_look,
            group_a_rate=total_mA / total_nA if total_nA > 0 else 0.0,
            group_b_rate=total_mB / total_nB if total_nB > 0 else 0.0,
            total_sample_size=total_nA + total_nB,
            additional_metrics={
                "e_value": e_value,
                "log_e_value": math.log(e_value) if e_value > 0 else float("-inf"),
                "prior_params": self.prior_params,
            },
            statistic_event=latest_stat,
            criteria_event=latest_criteria,
            signal_event=latest_signal,
        )
