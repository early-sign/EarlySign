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
    >>> from earlysign.v1.tests.util import BinomialStream
    >>> stream = BinomialStream(
    ...     n_per_batch=100,
    ...     p_control=0.20,
    ...     p_treatment=0.25,
    ...     seed=42
    ... )

Then, we initialize the template and run the experiment.

    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.v1.templates.binomial_ab import BinomialABTemplate, BinomialABTaskSpec
    >>> import earlysign.schema.ES3.GST as GST
    >>> from earlysign.v1.tests.util import BinomialStream

    >>> # 1. Setup Environment (In-memory DuckDB)
    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="test_experiment_001")

    >>> # 2. Define Task and Design Protocol
    >>> # Scenario: Detecting a 10% relative lift (20% -> 22%) with 80% power.
    >>> task = BinomialABTaskSpec(
    ...     arms=["control", "treatment"],
    ...     efficacy=GST.EfficacyRequirement(alpha=0.05),
    ...     futility=GST.FutilityRequirement(power=0.8),
    ...     hypotheses=GST.HypothesisSpec(
    ...         h_null="Difference <= 0",
    ...         h_alt="Difference > 0.02",
    ...         test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
    ...         target_effect=GST.BinaryEffectSize(
    ...             proportions={"control": 0.20, "treatment": 0.22}
    ...         )
    ...     )
    ... )

    >>> # 3. Initialize Template and Design
    >>> template = BinomialABTemplate(ledger)
    >>> protocol = template.design(
    ...     task=task,
    ...     looks=2,
    ...     spending_function="obrien_fleming",
    ...     designer_params={"model": "canonical_gaussian", "model_params": {"rng_seed": 42}}
    ... )

    >>> print(f"Designed Max Sample Size: {int(protocol.method.efficacy.schedule.interim_points[-1])}")
    Designed Max Sample Size: 12623

    >>> # 4. Save the designed protocol
    >>> template.set_protocol(protocol)

    >>> # 4. Run Experiment
    >>> for batch in stream:
    ...     template.update(batch)
    ...     result = template.report_progress()
    ...     # Check if we crossed a boundary or stopped for futility
    ...     if result['status'] != "CONTINUE":
    ...         break

    >>> # 5. Generate Final Report
    >>> final_result = template.report_result()
    >>> print(f"Final Status: {final_result['final_status']}")
    Final Status: DecisionStatus.STOP_EFFICACY
    >>> print(f"Is Rejected: {final_result['is_rejected']}")
    Is Rejected: True

In practice, each iteration may run in a different process.
To support this use case, the Template object can be destroyed after each iteration and re-instantiated.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

import earlysign.schema.ES3.GST as GST
from earlysign.schema.ES3.GST.Log.Analysis import DecisionStatus
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.protocol import AutoNameMixin
from earlysign.v1.framework.session import Session
from earlysign.v1.framework.trace import Traced
from earlysign.v1.framework.write_models import WriteModel
from earlysign.v1.methods.actions import Decision, Ingest, UpdateProtocol
from earlysign.v1.methods.binomial import BinomialSummaryFact
from earlysign.v1.methods.group_sequential.binomial import (
    BinomialGSTEngine,
    BinomialTestResult,
)
from earlysign.v1.methods.group_sequential.protocol_designer import (
    ProtocolDesigner,
)
from earlysign.v1.methods.group_sequential.report import (
    ABDecisionRecord,
    BinomialFinalProjector,
    BinomialProgressProjector,
    plot_gst_summary,
    reconstruct_binomial_z_history,
)

# --- ES3 Protocol Manifest ---


class BinomialABTaskSpec(GST.TaskSpec):
    response_type: Literal["binary"] = "binary"
    # Design Requirements
    efficacy: GST.EfficacyRequirement = GST.EfficacyRequirement(alpha=0.025)
    futility: GST.FutilityRequirement = GST.FutilityRequirement(
        power=0.8, binding=False
    )

    hypotheses: GST.HypothesisSpec


class BinomialABMethodSpec(GST.MethodSpec):
    # Efficacy Stopping Rule
    efficacy: GST.StoppingRule
    # Futility Stopping Rule
    futility: Optional[GST.StoppingRule] = None


class BinomialABProtocol(GST.Protocol, AutoNameMixin):
    task: BinomialABTaskSpec
    method: BinomialABMethodSpec
    name: str = Field(default="")


if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger


class BinomialABTemplate:
    """
    Standard orchestration for a Binomial A/B test using Group Sequential Design.
    """

    @classmethod
    def design(
        cls,
        task: BinomialABTaskSpec,
        looks: int,
        spending_function: str = "obrien_fleming",
        designer_params: Optional[Dict[str, Any]] = None,
    ) -> BinomialABProtocol:
        """
        Designs a Binomial A/B protocol based on the provided TaskSpec.

        Args:
            task: The generic task specification containing requirements (alpha, power, delta).
            looks: Number of interim looks (K).
            spending_function: Shape of the boundary (e.g., 'obrien_fleming').
            designer_params: Optional params for ProtocolDesigner (e.g., model type).

        Returns:
            A populated BinomialABProtocol with the calculated schedule.
        """
        designer = ProtocolDesigner.from_dict(designer_params or {})

        # Delegate logic to Designer
        method_spec_base = designer.method_from_task_spec(
            task=task,
            params={"looks": looks, "spending_function": spending_function},
        )

        if not method_spec_base.efficacy:
            raise ValueError("Designed method is missing efficacy rule.")

        # Wrap in specific Protocol Method Spec
        method_spec = BinomialABMethodSpec(
            kind="group_sequential",
            efficacy=method_spec_base.efficacy,
            futility=method_spec_base.futility,
        )

        return BinomialABProtocol(task=task, method=method_spec)

    def __init__(self, ledger: "Ledger"):
        self.ledger = ledger

    def set_protocol(self, protocol: BinomialABProtocol) -> None:
        """
        Persists the trial protocol to the ledger.
        This handles both initial intent and realized designs.
        """
        with Session(self.ledger) as sess:
            UpdateProtocol(sess, protocol)

    def update(self, batch: List[BaseModel]) -> None:
        """
        Orchestrates a single minibatch update cycle:
        Ingest -> [Read -> Analyze -> Decide -> Snapshot].
        """
        # 1. Ingest Data
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    Ingest(sess, item)

        # 2. Analysis
        with Session(self.ledger) as sess:
            # Reconstruct Protocol from Ledger
            protocol = sess.Read(ProtocolProjector(BinomialABProtocol))

            summary_c = sess.Read(
                BinomialSummaryFact(identity="summary_c", filter_arm="C")
            )
            summary_t = sess.Read(
                BinomialSummaryFact(identity="summary_t", filter_arm="T")
            )

            # 3. Engine Execution

            # Use CallAndCommit to execute logic and persist result with scientific lineage
            result: Traced[BinomialTestResult] = WriteModel.CallAndCommit(
                sess,
                BinomialTestResult,
                BinomialGSTEngine(protocol.data).run,
                summary_c=summary_c,
                summary_t=summary_t,
                protocol=protocol,
            )

            # Record Decision
            # Record Decision
            if result.data.status in (
                DecisionStatus.STOP_EFFICACY,
                DecisionStatus.STOP_FUTILITY,
            ):
                Decision(
                    sess,
                    ABDecisionRecord(
                        status=result.data.status,
                        message=f"Stopped: {result.data.status} at Look {result.data.look}",
                    ),
                    # Use the trace from the calculation result which includes dependencies
                    trace=result.trace,
                )

    def report_progress(self) -> Dict[str, Any]:
        """
        Returns the current progress report.
        Reconstructs state via BinomialProgressProjector.
        """
        with Session(self.ledger) as sess:
            return sess.Read(
                BinomialProgressProjector(protocol_type=BinomialABProtocol)
            ).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """Returns the final study report."""
        with Session(self.ledger) as sess:
            return sess.Read(
                BinomialFinalProjector(protocol_type=BinomialABProtocol)
            ).data.model_dump(mode="json")

    def backtest(self, batches: Any) -> Dict[str, Any]:
        """
        Historical Analysis: Replays data and stops immediately on a stopping decision.
        Returns a FinalReport.

        Args:
           batches: Iterator yielding `BatchObservation` objects or lists of them.
                    Each `BatchObservation` must have `n`, `success`, and `arm`.
        """
        for i, batch in enumerate(batches):
            self.update(batch if isinstance(batch, list) else [batch])
            prog = self.report_progress()
            if prog.get("decision") != DecisionStatus.CONTINUE:
                return self.report_result()

        return self.report_result()

    def plot_result(self) -> Any:
        """
        Generates a summary plot of the GST results.

        Returns:
            matplotlib.figure.Figure: The generated plot figure.
        """
        with Session(self.ledger) as sess:
            protocol_res = sess.Read(ProtocolProjector(BinomialABProtocol))
            p = protocol_res.data

            # Retrieve Z-statistic history for visualization
            # Use sess.table to respect snapshot isolation
            history_n, history_z = reconstruct_binomial_z_history(sess.table, p)

            # Generate Plot
            return plot_gst_summary(p, history_n, history_z)
