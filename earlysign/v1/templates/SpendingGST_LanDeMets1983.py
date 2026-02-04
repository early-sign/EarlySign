"""Binomial A/B Testing Template.

This module provides a standard template for running sequential A/B tests with
binary outcomes.

Examples:
    >>> import ibis, duckdb  # noqa: F401
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.v1.templates.SpendingGST_LanDeMets1983 import LanDeMets1983Template, LanDeMets1983TaskSpec
    >>> import earlysign.schema.ES3.GST as GST
    >>> from earlysign.schema.ES3.GST.Log import DecisionStatus
    >>> from earlysign.v1.tests.util import BinomialStream
    >>>
    >>> # Setup
    >>> # We use an in-memory DuckDB ledger for this example. In production, you would
    >>> # typically connect to a persistent database.
    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>>
    >>> # Bind ledger for experiment
    >>> ledger = ledger.bind(experiment_id="example_001")
    >>>
    >>> # Next, we define the experimental task. Here we are testing for a 10% relative lift
    >>> # (from 20% to 22% conversion rate) with standard error control (alpha=0.05, power=0.80).
    >>> # Define task
    >>> task = LanDeMets1983TaskSpec(
    ...     arms=["control", "treatment"],
    ...     efficacy=GST.EfficacyRequirement(alpha=0.05),
    ...     futility=GST.FutilityRequirement(power=0.8, binding=True),
    ...     hypotheses=GST.HypothesisSpec(
    ...         h_null_description="Difference <= 0",
    ...         h_alt_description="Difference > 0.02",
    ...         test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
    ...         target_effect=GST.BinaryEffectSize(
    ...             proportions={"control": 0.20, "treatment": 0.22}
    ...         )
    ...     )
    ... )
    >>>
    >>> # Now we design the protocol with 2 interim looks using the O'Brien-Fleming spending function.
    >>> # The designer calculates the required sample size and decision boundaries.
    >>> protocol = LanDeMets1983Template.design(
    ...     task=task,
    ...     looks=2,
    ...     spending_function="obrien_fleming",
    ...     designer_params={"model": "canonical_joint", "model_params": {"rng_seed": 42}}
    ... )
    >>> print(f"Designed Max Sample Size: {int(protocol.method.stopping_policy.timer.max_sample_size)}")
    Designed Max Sample Size: 10163
    >>>
    >>> # With the protocol designed, we initialize the template and persist it to the ledger.
    >>> template = LanDeMets1983Template(ledger)
    >>> template.set_protocol(protocol)
    >>>
    >>> # For demonstration, we simulate a data stream where the treatment actually has a larger
    >>> # effect than designed for (p=0.25 vs p=0.20, a 25% relative lift). The arm names in
    >>> # the stream must match those defined in the protocol.
    >>> stream = BinomialStream(
    ...     n_per_batch=1000,
    ...     arms={"control": 0.20, "treatment": 0.25},
    ...     n_max=13000,
    ...     seed=42
    ... )
    >>>
    >>> # We run the experiment by iterating through data batches. After each update, we check
    >>> # if a stopping boundary has been crossed.
    >>> for batch in stream:
    ...     template.update(batch)
    ...     progress = template.report_progress()
    ...     if progress['status'] != DecisionStatus.CONTINUE_:
    ...         break
    >>>
    >>> # Finally, we generate the final report to see the study outcome.
    >>>
    >>> final = template.report_result()
    >>> print(f"Final Status: {final['final_status']}")
    Final Status: stop_efficacy
    >>> print(f"Is Rejected: {final['is_rejected']}")
    Is Rejected: True
    >>>
    >>> # In practice, each iteration may run in a different process.
    >>> # To support this use case, the Template object can be destroyed after each iteration and re-instantiated.
    >>>
    >>> # Test: Protocol Deserialization from JSON
    >>> protocol_json = '{"name": "earlysign.v1.templates.SpendingGST_LanDeMets1983.LanDeMets1983Protocol", "ES3_version": "v1.0.0", "task": {"kind": "group_sequential", "arms": ["C", "T"], "response_type": "binary", "hypotheses": {"h_null_description": "Diff <= 0", "h_alt_description": "Diff > 0.05", "test_logic": {"kind": "superiority", "superiority_margin": 0.05}, "target_effect": {"type": "binary", "proportions": {"C": 0.09, "T": 0.14}}}, "efficacy": {"alpha": 0.025}, "futility": {"power": 0.8, "binding": false}}, "method": {"kind": "group_sequential", "stopping_policy": {"statistic": {"kind": "two_arm_binomial_z"}, "strategy": {"kind": "alpha_beta_spending", "statistical_model": {"kind": "canonical_gaussian"}, "alpha_spending_fn": {"family": "obrien_fleming", "params": null}, "beta_spending_fn": {"family": "obrien_fleming", "params": null}, "alpha_budget": 0.025, "beta_budget": 0.2, "alpha_binding": true, "beta_binding": false}, "timer": {"kind": "sample_size", "unit": "individuals", "max_sample_size": 1255}, "schedule": {"kind": "fixed", "analyses": [0.5, 1.0]}}, "adaptation": null}}'
    >>> protocol_from_json = LanDeMets1983Protocol.model_validate_json(protocol_json)
    >>> protocol_from_json.task.arms
    ['C', 'T']
    >>> protocol_from_json.method.stopping_policy.statistic.kind
    'two_arm_binomial_z'
    >>> final['is_rejected']
    True
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

import earlysign.schema.ES3.GST as GST
from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult
from earlysign.v1.framework.projector import ProtocolProjector
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
from earlysign.v1.templates.base import AutoNameMixin, TemplateBase


class LanDeMets1983TaskSpec(GST.TaskSpec):
    response_type: GST.ResponseType = GST.ResponseType.BINARY
    # Design Requirements
    efficacy: GST.EfficacyRequirement
    futility: GST.FutilityRequirement

    hypotheses: GST.HypothesisSpec


class LanDeMets1983Protocol(GST.Protocol, AutoNameMixin):
    task: LanDeMets1983TaskSpec
    method: GST.MethodSpec
    name: str = Field(default="")


class LanDeMets1983Template(TemplateBase[LanDeMets1983Protocol]):
    """Standard orchestrator for Binomial A/B tests (Lan & DeMets 1983).

    Provides high-level methods for designing, Updating, and reporting
    sequential A/B tests with binary outcomes using Alpha Spending functions.

    Reference:
        Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries
        for clinical trials. Biometrika, 70(3), 659–663.
    """

    _protocol_class = LanDeMets1983Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        task: LanDeMets1983TaskSpec,
        looks: int,
        spending_function: str = "obrien_fleming",
        spending_params: Optional[Dict[str, Any]] = None,
        designer_params: Optional[Dict[str, Any]] = None,
    ) -> LanDeMets1983Protocol:
        """
        Designs a Binomial A/B protocol based on the provided TaskSpec.

        Args:
            task: The generic task specification containing requirements (alpha, power, delta).
            looks: Number of interim looks (K).
            spending_function: Shape of the boundary (e.g., 'obrien_fleming').
            designer_params: Optional params for ProtocolDesigner (e.g., model type).

        Returns:
            A populated LanDeMets1983Protocol with the calculated schedule.
        """
        designer = ProtocolDesigner.from_dict(designer_params or {})

        # Delegate logic to Designer
        method_spec = designer.method_from_task_spec(
            task=task,
            params={
                "looks": looks,
                "spending_function": spending_function,
                "spending_params": spending_params,
            },
        )

        return LanDeMets1983Protocol(
            task=task,
            method=method_spec,
        )

    def update(self, batch: List[BaseModel]) -> None:
        """
        Orchestrates a single minibatch update cycle:
        Ingest -> [Read -> Analyze -> Decide -> Snapshot].
        """
        # 1. Ingest Data
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.commit(item, trace=[])

        # 2. Analysis
        with Session(self.ledger) as sess:
            # Reconstruct Protocol from Ledger
            protocol = sess.read(ProtocolProjector(LanDeMets1983Protocol))

            metrics = sess.read(Scoreboard(identity="metrics"))

            # 3. Engine Execution - compute result
            # The engine extracts arm names from protocol.task.arms internally.
            engine = BinomialGSTEngine(protocol.data)

            # 4. Commit the result via CallAndCommit to automate lineage tracking.
            # This ensures causality between the input metrics and the LookResult.
            # Decision flow is handled downstream in callers (e.g. by checking status).
            sess.call_and_commit(
                LookResult,
                engine.run,
                metrics=metrics,
            )

    def report_progress(self) -> Dict[str, Any]:
        """
        Returns the current progress report.
        """
        with Session(self.ledger) as sess:
            report = sess.read(ProgressProjector()).data
            return report.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """Returns the final study report."""
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")

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
            protocol = sess.read(ProtocolProjector(LanDeMets1983Protocol))

            # Retrieve trajectory from InterimAnalyses entity
            trajectory = sess.read(InterimAnalyses(identity="interim_analyses"))

            # Extract history from trajectory
            history_n: List[int] = []
            history_z: List[float] = []

            for _, state in trajectory.data:
                history_n.append(state.sample_n)
                history_z.append(state.z_stat)

            # Generate Plot
            return plot_gst_summary(protocol, history_n, history_z)
