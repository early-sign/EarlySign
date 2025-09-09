"""
earlysign.schemes.two_proportions.modules
=========================================

Base template classes for two-proportions experiments.

Provides common functionality for binomial comparison experiments
while allowing different statistical methods to be plugged in.

Examples
--------
>>> from earlysign.stats.schemes.two_proportions.modules import TwoPropTemplate
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>>
>>> class MyTwoPropExperiment(TwoPropTemplate):
...     def configure_components(self): return {...}
...     def extract_results(self, ledger): return AnalysisResult(...)
...     def register_design(self, ledger): pass
>>>
>>> # template = MyTwoPropExperiment("test", alpha_total=0.05)
>>> # template.setup(PolarsLedger())
"""

from __future__ import annotations
from typing import Dict, Any, List

from earlysign.runtime.experiment_template import ExperimentTemplate, AnalysisResult
from earlysign.core.names import Namespace
from earlysign.core.ledger import Ledger


class TwoPropTemplate(ExperimentTemplate):
    """Base class for two-proportions experiment templates."""

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
