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
    >>> from earlysign.schema.ES3.GST.Log import DecisionStatus

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
    ...     futility=GST.FutilityRequirement(power=0.8, binding=True),
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
    ...     designer_params={"model": "canonical_joint", "model_params": {"rng_seed": 42}}
    ... )

    >>> print(f"Designed Max Sample Size: {int(protocol.method.efficacy.schedule.interim_points[-1])}")
    Designed Max Sample Size: 12861

    >>> # 4. Save the designed protocol
    >>> template.set_protocol(protocol)

    >>> # 5. Run Experiment
    >>> for batch in stream:
    ...     template.update(batch)
    ...     result = template.report_progress()
    ...     # Check if we crossed a boundary or stopped for futility
    ...     if result['status'] != DecisionStatus.CONTINUE_:
    ...         break

    >>> # 6. Generate Final Report
    >>> final_result = template.report_result()
    >>> print(f"Final Status: {final_result['final_status']}")
    Final Status: stop_plan_end_reached
    >>> print(f"Is Rejected: {final_result['is_rejected']}")
    Is Rejected: False

In practice, each iteration may run in a different process.
To support this use case, the Template object can be destroyed after each iteration and re-instantiated.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

import earlysign.schema.ES3.GST as GST
from earlysign.schema.ES3.Binomial import ArmMetrics, ArmStatus
from earlysign.schema.ES3.GST.Log import Decision, DecisionStatus
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.protocol_mixin import AutoNameMixin
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.binomial import Scoreboard
from earlysign.v1.methods.group_sequential.execution.binomial import BinomialGSTEngine
from earlysign.v1.methods.group_sequential.execution.entities import InterimAnalyses
from earlysign.v1.methods.group_sequential.plan.protocol_design import ProtocolDesigner
from earlysign.v1.methods.group_sequential.reporting.projectors import (
    FinalProjector,
    ProgressProjector,
)
from earlysign.v1.methods.group_sequential.reporting.visualization import (
    plot_gst_summary,
)


class BinomialABTaskSpec(GST.TaskSpec):
    response_type: GST.ResponseType = GST.ResponseType.BINARY
    # Design Requirements
    efficacy: GST.EfficacyRequirement
    futility: GST.FutilityRequirement

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
        """
        with Session(self.ledger) as sess:
            sess.Commit(protocol)

    def update(self, batch: List[BaseModel]) -> None:
        """
        Orchestrates a single minibatch update cycle:
        Ingest -> [Read -> Analyze -> Decide -> Snapshot].
        """
        # 1. Ingest Data
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.Commit(item, trace=[])

        # 2. Analysis
        with Session(self.ledger) as sess:
            # Reconstruct Protocol from Ledger
            protocol = sess.Read(ProtocolProjector(BinomialABProtocol))

            metrics = sess.Read(Scoreboard(identity="metrics")).data
            summary_c = metrics.arms.get(
                "C",
                ArmStatus(
                    metrics=ArmMetrics(n=0, successes=0, p_hat=0.0), is_active=True
                ),
            ).metrics
            summary_t = metrics.arms.get(
                "T",
                ArmStatus(
                    metrics=ArmMetrics(n=0, successes=0, p_hat=0.0), is_active=True
                ),
            ).metrics

            # 3. Engine Execution - compute result
            engine = BinomialGSTEngine(protocol.data)
            test_result = engine.run(
                summary_c=summary_c,
                summary_t=summary_t,
                protocol=protocol.data,
            )

            # 4. Commit the result (trace comes from session's accumulated reads)
            sess.Commit(test_result)

            # 5. Record Decision if stopping
            if test_result.status in (
                DecisionStatus.STOP_EFFICACY,
                DecisionStatus.STOP_FUTILITY,
            ):
                DecisionAction(
                    sess,
                    Decision(
                        status=test_result.status,
                        message=f"Stopped: {test_result.status} at Look {test_result.look}",
                    ),
                    # Uses implicit session.trace from the Reads
                )

    def report_progress(self) -> Dict[str, Any]:
        """
        Returns the current progress report.
        """
        with Session(self.ledger) as sess:
            report = sess.Read(ProgressProjector()).data
            return report.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """Returns the final study report."""
        with Session(self.ledger) as sess:
            return sess.Read(FinalProjector()).data.model_dump(mode="json")

    def backtest(self, batches: Any) -> Dict[str, Any]:
        """
        Historical Analysis: Replays data and stops immediately on a stopping decision.

        Args:
           batches: Iterator yielding `ArmData` objects or lists of them.
                    Each `ArmData` must have `n`, `success`, and `arm`.
        """
        for i, batch in enumerate(batches):
            self.update(batch if isinstance(batch, list) else [batch])
            prog = self.report_progress()
            if prog.get("status") != DecisionStatus.CONTINUE_:
                return self.report_result()

        return self.report_result()

    def plot_result(self) -> Any:
        """
        Generates a summary plot of the GST results.

        Returns:
            matplotlib.figure.Figure: The generated plot figure.
        """
        with Session(self.ledger) as sess:
            protocol = sess.Read(ProtocolProjector(BinomialABProtocol)).data

            # Retrieve trajectory from InterimAnalyses entity
            trajectory = sess.Read(InterimAnalyses(identity="interim_analyses")).data

            # Extract history from trajectory
            history_n: List[int] = []
            history_z: List[float] = []

            for _, state in trajectory:
                history_n.append(state.sample_n)
                history_z.append(state.z_stat)

            # Generate Plot
            return plot_gst_summary(protocol, history_n, history_z)
