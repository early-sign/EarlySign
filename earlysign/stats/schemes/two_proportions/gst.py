"""
earlysign.schemes.two_proportions.gst
====================================

Group Sequential Testing implementations for two-proportions experiments.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional, Union, List, Dict, Any

from earlysign.core.ledger import Ledger
from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex
from earlysign.core.components import Statistic, Criteria, Signaler
from earlysign.runtime.experiment_template import AnalysisResult
from earlysign.stats.schemes.two_proportions.modules import TwoPropTemplate
from earlysign.stats.schemes.two_proportions.ingest import TwoPropObservation


class TwoPropGSTTemplate(TwoPropTemplate):
    """Two-proportions Group Sequential Testing experiment module.

    Implements group sequential testing with error spending functions
    for comparing two binomial proportions.
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
        from earlysign.stats.schemes.two_proportions.gst_components import (
            WaldZStatistic,
        )

        return WaldZStatistic()

    def _create_criteria(self) -> Criteria:
        """Create GST error spending criteria."""
        from earlysign.stats.schemes.two_proportions.gst_components import (
            LanDeMetsBoundary,
        )

        return LanDeMetsBoundary(
            alpha_total=self.alpha_total,
            t=1.0,  # This will need to be calculated dynamically
            style=self.spending_function,
        )

    def _create_signaler(self) -> Signaler:
        """Create signaler for GST decisions."""
        from earlysign.stats.schemes.two_proportions.gst_components import (
            PeekSignaler,
        )

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

        statistic_value = latest_stat.payload.get("value", 0.0) if latest_stat else 0.0
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
            statistic_value=statistic_value,
            threshold_value=threshold_value,
            look_number=self._current_look,
            group_a_rate=total_mA / total_nA if total_nA > 0 else 0.0,
            group_b_rate=total_mB / total_nB if total_nB > 0 else 0.0,
            total_sample_size=total_nA + total_nB,
            additional_metrics={
                "p_value": latest_signal.payload.get("p_value"),
                "spending_function": self.spending_function,
                "looks_completed": self._current_look,
                "total_looks": self.looks,
            },
            statistic_event=latest_stat,
            criteria_event=latest_criteria,
            signal_event=latest_signal,
        )
