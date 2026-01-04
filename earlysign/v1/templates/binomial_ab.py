"""
Binomial A/B Testing Template
=============================

This module provides a standard template for running sequential A/B tests with binary outcomes.

Usage
-----
The following example demonstrates how to set up and run a sequential A/B test
simulating a scenario with a 20% baseline conversion rate and a relative 10% lift (Treatment = 22%).

For demonstration, we prepare the following datastream.

    >>> # We simulate a stream where Treatment actually has the lift (p=0.25 vs p=0.20)
    >>> stream = BinomialStream(
    ...     n_per_batch=100,
    ...     p_control=0.20,
    ...     p_treatment=0.25,
    ...     seed=42
    ... )

Then, we initialize the template and run the experiment.

    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.v1.methods.group_sequential.protocol import GSTProtocol
    >>> from earlysign.v1.methods.group_sequential.protocol_designer import ProtocolDesigner
    >>> from earlysign.v1.templates.binomial_ab import BinomialABTemplate
    >>> from earlysign.v1.tests.util import BinomialStream

    >>> # 1. Setup Environment (In-memory DuckDB)
    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="test_experiment_001")

    >>> # 2. Define Protocol using Designer
    >>> # Scenario: Detecting a 10% relative lift (20% -> 22%) with 80% power.
    >>> designer = ProtocolDesigner.from_dict({
    ...     "model": "canonical_gaussian",
    ...     "model_params": {"rng_seed": 42}
    ... })
    >>> protocol = designer.plan_binomial_ab(
    ...     alpha=0.05,
    ...     power=0.8,
    ...     delta=0.02, # 20% -> 22%
    ...     p_control=0.20,
    ...     k=2,
    ...     shape_type="obrien_fleming"
    ... )
    >>> print(f"Designed Max Sample Size: {protocol.n_max}")
    Designed Max Sample Size: 12623

    >>> # 3. Initialize Template and Save the designed protocol
    >>> template = BinomialABTemplate(ledger)
    >>> template.set_protocol(protocol)

    >>> # 4. Run Experiment
    >>> for batch in stream:
    ...     result = template.update(batch)
    ...     # Check if we crossed a boundary or stopped for futility
    ...     if result['status'] != "CONTINUE":
    ...         break

    >>> # 5. Generate Final Report
    >>> final_result = template.report_result()
    >>> print(f"Final Status: {final_result['final_status']}")
    Final Status: STOP_EFFICACY
    >>> print(f"Is Rejected: {final_result['is_rejected']}")
    Is Rejected: True

In practice, each iteration may run in a different process.
To support this use case, the Template object can be destroyed after each iteration and re-instantiated.
"""

from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel

from earlysign.v1.framework.session import Session
from earlysign.v1.methods.actions import Decision, Ingest, UpdateProtocol
from earlysign.v1.methods.binomial import BinomialSummaryFact
from earlysign.v1.methods.group_sequential.binomial import (
    BinomialZProjector,
)
from earlysign.v1.methods.group_sequential.protocol import GSTProtocol
from earlysign.v1.methods.group_sequential.report import (
    ABDecisionRecord,
    BinomialFinalProjector,
    BinomialProgressProjector,
)

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger


class BinomialABTemplate:
    """
    Standard orchestration for a Binomial A/B test using Group Sequential Design.
    """

    def __init__(self, ledger: "Ledger"):
        self.ledger = ledger

    def set_protocol(self, protocol: GSTProtocol):
        """
        Persists the trial protocol to the ledger.
        This handles both initial intent and realized designs.
        """
        with Session(self.ledger) as sess:
            UpdateProtocol(sess, protocol)

    def update(self, batch: List[BaseModel]) -> Dict[str, Any]:
        """
        Orchestrates a single minibatch update cycle:
        Ingest -> [Read -> Analyze -> Decide -> Snapshot].
        """
        from earlysign.v1.framework.projector import ProtocolProjector

        # 1. Ingest Data
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    Ingest(sess, item)

        # 2. Analysis
        with Session(self.ledger) as sess:
            # Reconstruct Protocol from Ledger
            p = sess.Read(ProtocolProjector(GSTProtocol)).data

            summary_c = sess.Read(
                BinomialSummaryFact(identity="summary_c", filter_arm="C")
            ).data
            summary_t = sess.Read(
                BinomialSummaryFact(identity="summary_t", filter_arm="T")
            ).data

            cumulative_n = summary_c.n + summary_t.n
            info_frac = cumulative_n / p.n_max if p.n_max > 0 else 0

            result = {
                "info_frac": info_frac,
                "look": None,
                "z_stat": None,
                "is_rejected": False,
                "status": "CONTINUE",
            }

            # 3. Check if Look is due
            # We determine the "current" look by comparing cumulative N with milestones.
            look_num = None
            boundary = None
            for i, m in enumerate(p.milestones):
                if info_frac >= m:
                    look_num = i + 1
                    boundary = p.boundaries[i]

            if look_num:
                # Execution layer: Pure statistical calculation
                analysis_traced = sess.Read(BinomialZProjector(boundary))
                calc_res = analysis_traced.data

                # Record Decision if rejected or final
                status = "CONTINUE"
                if calc_res.is_rejected:
                    status = "STOP_EFFICACY"
                    Decision(
                        sess,
                        ABDecisionRecord(
                            status="STOP", message=f"Rejected at Look {look_num}"
                        ),
                        trace=analysis_traced.trace,
                    )
                elif look_num == len(p.milestones):
                    status = "STOP_FINAL"

                result.update(
                    {
                        "look": look_num,
                        "z_stat": calc_res.z_stat,
                        "is_rejected": calc_res.is_rejected,
                        "status": status,
                    }
                )

            return result

    def report_progress(self) -> Dict[str, Any]:
        """
        Returns the current progress report.
        Reconstructs state via BinomialProgressProjector.
        """
        with Session(self.ledger) as sess:
            return sess.Read(BinomialProgressProjector()).data.model_dump()

    def report_result(self) -> Dict[str, Any]:
        """Returns the final study report."""
        with Session(self.ledger) as sess:
            return sess.Read(BinomialFinalProjector()).data.model_dump()

    def backtest(self, batches: Any) -> Dict[str, Any]:
        """
        Historical Analysis: Replays data and stops immediately on a stopping decision.
        Returns a FinalReport.

        Args:
           batches: Iterator yielding `BatchObservation` objects or lists of them.
                    Each `BatchObservation` must have `n`, `success`, and `arm`.
        """
        for i, batch in enumerate(batches):
            res = self.update(batch if isinstance(batch, list) else [batch])
            if res.get("status") == "STOP_EFFICACY":
                return self.report_result()

        return self.report_result()

    def plot_result(self) -> Any:
        """
        Generates a summary plot of the GST results.

        Returns:
            matplotlib.figure.Figure: The generated plot figure.
        """
        from earlysign.v1.framework.projector import ProtocolProjector
        from earlysign.v1.methods.group_sequential.protocol import GSTProtocol
        from earlysign.v1.methods.group_sequential.report import (
            plot_gst_summary,
            reconstruct_binomial_z_history,
        )

        with Session(self.ledger) as sess:
            protocol_res = sess.Read(ProtocolProjector(GSTProtocol))
            p = protocol_res.data

            # Reconstruct History
            history_n, history_z = reconstruct_binomial_z_history(sess.table, p)

            # Generate Plot
            return plot_gst_summary(p, history_n, history_z)
