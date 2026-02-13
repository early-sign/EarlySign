"""Binomial A/B Testing Template.

This module provides a standard template for running sequential A/B tests with
binary outcomes using Alpha Spending functions.

Reference:
    Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with
    Applications to Clinical Trials. Chapman and Hall/CRC.

Example:
    >>> import ibis, duckdb  # noqa: F401
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.v1.templates.GST_Spending_JennisonTurnbull2000 import JennisonTurnbull2000Template, JennisonTurnbull2000TaskSpec
    >>> import earlysign.schema.ES3.Base as ES3_BASE
    >>> import earlysign.schema.ES3.GST as GST
    >>> from earlysign.schema.ES3.GST.Log import DecisionStatus
    >>> from earlysign.v1.tests.util import BinomialStream
    >>> import numpy as np
    >>>
    >>> # Setup
    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="example_001")
    >>>
    >>> # Design
    >>> protocol = JennisonTurnbull2000Template.design(
    ...     p_control=0.20, p_treatment=0.25, alpha=0.05, power=0.8, looks=2
    ... )
    >>>
    >>> # Initialize and Run
    >>> template = JennisonTurnbull2000Template(ledger)
    >>> template.set_protocol(protocol)
    >>>
    >>> stream = BinomialStream(n_per_batch=600, arms={"control": 0.20, "treatment": 0.25}, n_max=13000, seed=42)
    >>> for batch in stream:
    ...     template.update(batch)
    ...     prog = template.report_progress()
    ...     if prog['is_milestone']:
    ...         print(f"Look {prog['look']}: Z={prog['z_stat']:.2f}, Boundary={prog['efficacy_boundary']:.2f}")
    ...     if prog['status'] != DecisionStatus.CONTINUE_:
    ...         break
    Look 1: Z=0.70, Boundary=1.92
    Look 2: Z=1.89, Boundary=1.75
    >>>
    >>> final = template.report_result()
    >>> print(f"Final Status: {final['final_status']}")
    Final Status: stop_efficacy
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

import earlysign.schema.ES3.Base as ES3_BASE
import earlysign.schema.ES3.GST as GST
from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.GST import (
    AbsoluteDifference,
    EffectMeasure,
    OddsRatio,
    RelativeRisk,
)
from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.binomial import Scoreboard
from earlysign.v1.methods.group_sequential.execution.binomial import (
    BinomialGSTEngine,
)
from earlysign.v1.methods.group_sequential.execution.entities import (
    InterimAnalyses,
)
from earlysign.v1.methods.group_sequential.execution.trigger_strategies import (
    get_pending_look_trigger,
)
from earlysign.v1.methods.group_sequential.plan.protocol_design import (
    ProtocolDesigner,
)
from earlysign.v1.methods.group_sequential.reporting.projectors import (
    FinalProjector,
    ProgressProjector,
)
from earlysign.v1.methods.group_sequential.reporting.visualization import (
    plot_gst_summary,
)
from earlysign.v1.templates.base import AutoNameMixin, TemplateBase


class JennisonTurnbull2000TaskSpec(GST.TaskSpec):
    response_type: GST.ResponseType = GST.ResponseType.BINARY
    # Design Requirements
    efficacy: GST.EfficacyRequirement
    futility: GST.FutilityRequirement

    hypotheses: GST.HypothesisSpec


class JennisonTurnbull2000Protocol(GST.Protocol, AutoNameMixin):
    task: JennisonTurnbull2000TaskSpec
    method: GST.MethodSpec
    name: str = Field(default="")


def _resolve_p_treatment(
    p_control: float, effect_spec: Union[EffectMeasure, Dict[str, Any]]
) -> float:
    """Helper to resolve p_treatment from p_control and an effect measure."""
    if isinstance(effect_spec, dict):
        effect_spec = TypeAdapter(EffectMeasure).validate_python(effect_spec)

    match effect_spec:
        case AbsoluteDifference(value=delta):
            return p_control + delta
        case OddsRatio(value=or_val):
            # OR = (p1/(1-p1)) / (p0/(1-p0))
            if not (0 < p_control < 1):
                raise ValueError("p_control must be between 0 and 1 for Odds Ratio")
            odds_c = p_control / (1.0 - p_control)
            odds_t = or_val * odds_c
            return odds_t / (1.0 + odds_t)
        case RelativeRisk(value=rr_val):
            return p_control * rr_val
        case _:
            raise ValueError(f"Unknown effect type: {type(effect_spec)}")


class JennisonTurnbull2000Template(TemplateBase[JennisonTurnbull2000Protocol]):
    """Standard orchestrator for Binomial A/B tests (Jennison & Turnbull 2000).

    Provides high-level methods for designing, Updating, and reporting
    sequential A/B tests with binary outcomes using Alpha Spending functions.

    Reference:
        Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with
        Applications to Clinical Trials. Chapman and Hall/CRC.
    """

    _protocol_class = JennisonTurnbull2000Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        looks: int,
        alpha: float,
        power: float,
        task: Optional[JennisonTurnbull2000TaskSpec] = None,
        spending_function: str = "obrien_fleming",
        spending_params: Optional[Dict[str, Any]] = None,
        designer_params: Optional[Dict[str, Any]] = None,
        scheduling: (
            Literal["equidistant", "asn_minimizer"] | List[float]
        ) = "asn_minimizer",
        # Binomial params (convenience helpers if task is None)
        p_control: Optional[float] = None,
        p_treatment: Optional[float] = None,
        effect_spec: Optional[Union[EffectMeasure, Dict[str, Any]]] = None,
    ) -> JennisonTurnbull2000Protocol:
        """
        Designs a Binomial A/B protocol.

        Supports both direct TaskSpec input or scalar parameters for convenience.
        Also supports advanced scheduling options including ASN minimization.

        Args:
            task: The generic task specification. If None, p_control/p_treatment are used.
            looks: Number of interim looks (K).
            spending_function: Shape of the boundary (e.g., 'obrien_fleming').
            spending_params: Parameters for the spending function.
            designer_params: Optional params for ProtocolDesigner (e.g., model type).
            scheduling: Schedule type or list of information fractions.
                        - 'equidistant': standard equal spacing (default)
                        - 'asn_minimizer': optimizes spacing to minimize ASN under H1
                        - List[float]: explicit list of information fractions (0 < t <= 1)
            p_control: Baseline proportion (if task is None).
            p_treatment: Target proportion (if task is None).
            effect_spec: Structured target effect size (e.g. delta, OR, RR).

        Returns:
            A populated JennisonTurnbull2000Protocol.

        Examples:
            >>> import numpy as np
            >>> from earlysign.v1.templates.GST_Spending_JennisonTurnbull2000 import JennisonTurnbull2000Template
            >>> import earlysign.schema.ES3.Base as ES3_BASE
            >>>
            >>> # --- Example 1: Standard Equidistant Schedule ---
            >>> protocol_eq = JennisonTurnbull2000Template.design(
            ...     p_control=0.20, p_treatment=0.22, alpha=0.05, power=0.8, looks=3,
            ...     scheduling="equidistant"
            ... )
            >>> np.allclose(protocol_eq.method.stopping_policy.schedule.analyses, [1/3, 2/3, 1.0])
            True
            >>>
            >>> # --- Example 2: Using structured effect_spec (delta) ---
            >>> protocol_delta = JennisonTurnbull2000Template.design(
            ...     p_control=0.20, effect_spec={"kind": "absolute_difference", "value": 0.02},
            ...     alpha=0.05, power=0.8, looks=3
            ... )
            >>> protocol_delta.task.hypotheses.target_effect.proportions["treatment"]
            0.22
        """
        # 1. Construct/Validate Task
        if task is None:
            if p_control is None:
                raise ValueError("p_control must be provided if task is None")

            if p_treatment is None:
                if effect_spec is None:
                    raise ValueError(
                        "Either p_treatment or effect_spec must be provided if task is None"
                    )
                p_treatment = _resolve_p_treatment(p_control, effect_spec)

            task = JennisonTurnbull2000TaskSpec(
                arms=ES3_BASE.TwoArmComparison(
                    control_arm_name="control",
                    treatment_arm_name="treatment",
                ),
                response_type=GST.ResponseType.BINARY,
                efficacy=GST.EfficacyRequirement(alpha=alpha),
                futility=GST.FutilityRequirement(power=power),
                hypotheses=GST.HypothesisSpec(
                    h_null_description="Difference <= 0",
                    h_alt_description=f"Difference > {p_treatment - p_control:.4f}",
                    test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                    target_effect=GST.BinaryEffectSize(
                        proportions={"control": p_control, "treatment": p_treatment}
                    ),
                ),
            )

        # 2. Prepare Designer
        designer_params = designer_params or {"model": "canonical_joint"}
        designer = ProtocolDesigner.from_dict(designer_params)
        rng_seed = designer_params.get("model_params", {}).get("rng_seed", 42)

        # 3. Design Strategy Components using common ProtocolDesigner
        proportions = task.hypotheses.target_effect.proportions
        arms = task.arms
        if not isinstance(arms, ES3_BASE.TwoArmComparison):
            raise NotImplementedError(
                f"GST on {type(arms).__name__} is not yet supported in this template. "
                "Currently, only TwoArmComparison is supported."
            )
        arm_0, arm_1 = arms.control_arm_name, arms.treatment_arm_name

        method_spec, _ = designer.design_gs_binomial(
            alpha=task.efficacy.alpha,
            power=task.futility.power if task.futility else 0.8,
            p_control=proportions[arm_0],
            p_treatment=proportions[arm_1],
            looks=looks,
            scheduling=scheduling,
            spending_function=spending_function,
            spending_params=spending_params,
            futility=(task.futility is not None),
            futility_binding=task.futility.binding if task.futility else False,
            tails=1,
            rng_seed=rng_seed,
        )

        return JennisonTurnbull2000Protocol(
            task=task,
            method=method_spec,
        )

    def update(self, batch: List[BaseModel]) -> None:
        """
        Orchestrates a single minibatch update cycle:
        Ingest -> [Read -> Analyze -> Decide -> Snapshot].
        """
        # 1. Ingest Data
        with Session(self.ledger) as sess:
            if batch:
                # Validate arm names
                protocol_traced = sess.read(
                    ProtocolProjector(JennisonTurnbull2000Protocol)
                )
                arms = protocol_traced.data.task.arms
                if isinstance(arms, ES3_BASE.TwoArmComparison):
                    allowed_arms = {arms.control_arm_name, arms.treatment_arm_name}
                else:
                    allowed_arms = set()
                for item in batch:
                    arm_name = getattr(item, "arm", None)
                    if arm_name and arm_name not in allowed_arms:
                        warnings.warn(
                            f"Received data for unexpected arm '{arm_name}'. "
                            f"Expected arms: {allowed_arms}",
                            UserWarning,
                        )
                    sess.commit(item, trace=[])

            # 2. Analysis & Trigger check
            # Reconstruct Protocol from Ledger
            protocol_traced = sess.read(ProtocolProjector(JennisonTurnbull2000Protocol))
            metrics = sess.read(Scoreboard(identity="metrics"))
            trajectory = sess.read(InterimAnalyses(identity="interim_analyses"))

            # 3. Check if an analysis is "due"
            trigger = get_pending_look_trigger(protocol_traced, metrics, trajectory)

            if trigger:
                # 4. Engine Execution - compute result using trajectory history
                engine = BinomialGSTEngine(protocol_traced.data)

                sess.call_and_commit(
                    LookResult,
                    engine.run,
                    metrics=metrics,
                    history=trajectory,
                    trigger=trigger,
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
                break

        return self.report_result()

    def backtest_from_table(
        self,
        table: Any,
        *,
        arm_col: str = "arm",
        n_col: str = "n",
        success_col: str = "success",
        order_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Historical Analysis from an Ibis table.
        Replays data from the table and stops immediately on a stopping decision.

        Args:
            table: Ibis table containing historical data.
            arm_col: Column name for the arm identifier.
            n_col: Column name for the number of trials.
            success_col: Column name for the number of successes.
            order_by: Column name to order the data by.
        """
        from earlysign.schema.ES3.Binomial import ArmData

        # 1. Project and order
        if order_by:
            # We must include order_by in our select if we want to order by it,
            # or just use the original table's column.
            data_table = table.select(
                arm=table[arm_col],
                n=table[n_col],
                success=table[success_col],
                _order=table[order_by],
            ).order_by("_order")
        else:
            data_table = table.select(
                arm=table[arm_col],
                n=table[n_col],
                success=table[success_col],
            )

        # 2. Replay & Stop
        df = data_table.execute()

        for _, row in df.iterrows():
            batch = [
                ArmData(
                    arm=str(row["arm"]),
                    n=int(row["n"]),
                    success=int(row["success"]),
                )
            ]
            self.update(batch)
            prog = self.report_progress()
            if prog.get("status") != DecisionStatus.CONTINUE_:
                break

        return self.report_result()

    def plot_result(self) -> Any:
        """
        Generates a summary plot of the GST results.

        Returns:
            matplotlib.figure.Figure: The generated plot figure.
        """
        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(JennisonTurnbull2000Protocol))

            # Retrieve trajectory from InterimAnalyses entity
            trajectory = sess.read(InterimAnalyses(identity="interim_analyses"))

            # Generate Plot
            return plot_gst_summary(
                protocol.data,
                title="GST Monitoring",
                full_history=trajectory.data,
            )
