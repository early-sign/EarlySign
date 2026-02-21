"""Classic Group Sequential Testing Controller.

This module provides a controller for "Classic" Group Sequential Tests (GST) based on
standard boundary shapes like Pocock, O'Brien-Fleming, and Wang-Tsiatis power
families. Unlike the Alpha-Spending approach, these designs use fixed boundary
shape constants determined by the total number of looks and alpha/power requirements.
"""

from typing import Any, Dict, Literal, Optional, Sequence, Union, cast

from pydantic import BaseModel, Field

import earlysign.schema.ES3.Base as ES3_BASE
import earlysign.schema.ES3.GST as GST
from earlysign.core.ledger import Ledger
from earlysign.core.util.logging import get_logger
from earlysign.framework.controller import (
    AutoNameMixin,
    Controller,
    RichDisplayMixin,
)
from earlysign.framework.projector import ProtocolProjector
from earlysign.framework.session import Session
from earlysign.framework.trace import Traced
from earlysign.methods.continuous import Scoreboard as ContinuousScoreboard
from earlysign.methods.group_sequential.core.model import (
    NumericalIntegrationConfig,
    SimulationConfig,
)
from earlysign.methods.group_sequential.design.protocol_design import (
    ProtocolDesigner,
)
from earlysign.methods.group_sequential.engine.engine import GroupSequentialEngine
from earlysign.methods.group_sequential.engine.entities import (
    InterimAnalyses,
)
from earlysign.methods.group_sequential.engine.trigger_strategies import (
    get_pending_look_trigger,
)
from earlysign.methods.group_sequential.reporting.projectors import (
    BacktestProjector,
    FinalProjector,
    ProgressProjector,
)
from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult


class ClassicTaskSpec(GST.TaskSpec):
    """Task specification for Classic GST."""

    pass


class ClassicProtocol(GST.Protocol, AutoNameMixin, RichDisplayMixin):
    """Protocol for Classic GST."""

    task: ClassicTaskSpec
    method: GST.MethodSpec
    name: str = Field(default="")


class ClassicGSTController(Controller[ClassicProtocol]):
    """Orchestrator for Classic Group Sequential Tests (Pocock, OBF, Wang-Tsiatis).

    This controller supports designs that are defined by a fixed boundary shape
    parameter (Delta for Wang-Tsiatis) rather than an Alpha Spending Function.
    """

    _protocol_class = ClassicProtocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        type: Literal["pocock", "obrien_fleming", "wang_tsiatis"],
        alpha: float,
        power: float,
        k: int,
        delta: Optional[float] = None,  # Can be derived if arm values are provided
        p_control: Optional[float] = None,
        p_treatment: Optional[float] = None,
        sigma: Optional[float] = None,
        mu_control: Optional[float] = 0.0,
        mu_treatment: Optional[float] = None,
        wang_tsiatis_delta: Optional[float] = None,
        tails: int = 2,
        arms: int = 2,
        seed: int = 42,
        control_arm_name: str = "control",
        treatment_arm_name: str = "treatment",
        method: Literal["simulation", "numerical_integration"] = "simulation",
        method_config: Optional[
            Union[SimulationConfig, NumericalIntegrationConfig]
        ] = None,
    ) -> ClassicProtocol:
        """Designs a Classic GST Protocol.

        Args:
            type: Design type ("pocock", "obrien_fleming", "wang_tsiatis").
            alpha: Type I error rate.
            power: Target power.
            k: Number of looks.
            delta: Detectable effect size (absolute difference). Optional if explicit arm values are provided.
            p_control: Control arm proportion (for Binomial designs).
            p_treatment: Treatment arm proportion (for Binomial designs).
            sigma: Standard deviation (for Continuous designs).
            mu_control: Control arm mean (for Continuous designs). Defaults to 0.0.
            mu_treatment: Treatment arm mean (for Continuous designs).
            wang_tsiatis_delta: Delta parameter for Wang-Tsiatis family.
            tails: Number of tails (1 or 2).
            arms: Number of arms (1 or 2).
            seed: Random seed for simulation/calculation.
            method: Method for boundary solving ('simulation' or 'numerical_integration').
            method_config: Configuration object.

        Returns:
            A populated ClassicProtocol.
        """
        # 1. Resolve Parameters via Match/Case
        # We determine response_type, effect_size, and calculated_delta
        response_type: GST.ResponseType
        eff_size: GST.EffectSizeUnion
        calc_delta: float

        match (p_control, p_treatment, delta, sigma):
            # --- Binomial Cases ---
            case (float() as pc, float() as pt, _, _):
                # Explicit Proportions
                calc_delta = pt - pc
                response_type = GST.ResponseType.BINARY
                eff_props = (
                    {control_arm_name: pc, treatment_arm_name: pt}
                    if arms == 2
                    else {treatment_arm_name: pt}
                )
                eff_size = GST.BinaryEffectSize(proportions=eff_props)

            case (float() as pc, None, float() as d, _):
                # Control + Delta
                calc_delta = d
                pt = pc + d
                response_type = GST.ResponseType.BINARY
                eff_props = (
                    {control_arm_name: pc, treatment_arm_name: pt}
                    if arms == 2
                    else {treatment_arm_name: pt}
                )
                eff_size = GST.BinaryEffectSize(proportions=eff_props)

            # --- Continuous Cases ---
            case (None, None, _, float() as s):
                # Continuous (sigma provided)
                response_type = GST.ResponseType.CONTINUOUS

                # Resolve Means
                mc = mu_control if mu_control is not None else 0.0

                if mu_treatment is not None:
                    mt = mu_treatment
                    calc_delta = mt - mc
                elif delta is not None:
                    calc_delta = delta
                    mt = mc + delta
                else:
                    raise ValueError(
                        "For Continuous designs, must provide either `delta` or `mu_treatment`."
                    )

                means = (
                    {control_arm_name: mc, treatment_arm_name: mt}
                    if arms == 2
                    else {treatment_arm_name: mt}
                )
                eff_size = GST.ContinuousEffectSize(means=means, standard_deviation=s)

            case _:
                raise ValueError(
                    "Invalid parameter combination. Provide either:\n"
                    "1. (p_control, p_treatment) or (p_control, delta) for Binomial.\n"
                    "2. (sigma) plus (mu_treatment, mu_control) or (delta) for Continuous."
                )

        # 2. Setup Designer
        designer = ProtocolDesigner.from_dict(
            {"model": "canonical_joint", "model_params": {"rng_seed": seed}}
        )

        arm_names = (
            [control_arm_name, treatment_arm_name]
            if arms == 2
            else [treatment_arm_name]
        )

        # 3. Design via common logic
        method_spec, n_total = designer.design_gs_classic(
            alpha=alpha,
            power=power,
            delta=calc_delta,
            looks=k,
            type=type,
            p_control=p_control,
            sigma=sigma,
            wang_tsiatis_delta=wang_tsiatis_delta or 0.25,
            tails=tails,
            arm_names=arm_names,
            rng_seed=seed,
            method=method,
            method_config=method_config,
        )

        # 4. Assemble Task Spec
        task_arms: ES3_BASE.ArmStructure
        if arms == 1:
            task_arms = ES3_BASE.SingleArm(arm_name="treatment")
        else:
            task_arms = ES3_BASE.TwoArmComparison(
                control_arm_name=control_arm_name, treatment_arm_name=treatment_arm_name
            )

        task = ClassicTaskSpec(
            arms=task_arms,
            response_type=response_type,
            hypotheses=GST.HypothesisSpec(
                h_null_description="No Difference",
                h_alt_description=f"Difference {calc_delta}",
                test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                target_effect=eff_size,
            ),
            efficacy=GST.EfficacyRequirement(alpha=alpha),
            futility=GST.FutilityRequirement(power=power),
        )

        return ClassicProtocol(task=task, method=method_spec)

    def update(self, batch: Sequence[BaseModel]) -> None:
        """Updates the experiment with new data."""
        # Reuse standard logic: Commit -> (Read -> Engine -> Output)
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(ClassicProtocol))
            # The original code had a specific BinomialScoreboard import.
            # Now we use the generic Scoreboard and rely on response_type.
            metrics: Traced[Any]
            if protocol.data.task.response_type == GST.ResponseType.BINARY:
                from earlysign.methods.binomial import (
                    Scoreboard as BinomialScoreboard,
                )

                metrics = sess.read(BinomialScoreboard(identity="metrics"))
            else:
                metrics = sess.read(ContinuousScoreboard(identity="metrics"))

            trajectory = sess.read(InterimAnalyses(identity="interim_analyses"))
            trigger = get_pending_look_trigger(
                protocol=Traced(data=protocol.data, trace=protocol.trace),
                metrics=Traced(data=metrics.data, trace=metrics.trace),
                history=Traced(data=trajectory.data, trace=trajectory.trace),
            )

            if trigger:
                if protocol.data.task.response_type == GST.ResponseType.BINARY:
                    engine = GroupSequentialEngine(protocol.data)
                else:
                    # GroupSequentialEngine is polymorphic and handles Continuous types
                    engine = GroupSequentialEngine(protocol.data)

                sess.call_and_commit(
                    LookResult,
                    engine.run,
                    identity="interim_analyses",
                    metrics=metrics.data,
                    history=trajectory.data,
                    trigger=trigger.data,
                )

    def report_progress(self) -> Dict[str, Any]:
        """Returns the current progress report."""
        with Session(self.ledger) as sess:
            # Reusing standard ProgressProjector
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
           batches: Iterator yielding `BinomialArmData` objects or lists of them.
        """
        logger = get_logger(__name__)
        for i, batch in enumerate(batches):
            self.update(batch if isinstance(batch, list) else [batch])
            prog = self.report_progress()

            if prog.get("status") != DecisionStatus.CONTINUE:
                logger.info(
                    f"Stopping criterion met at index {i}: {prog.get('status')}"
                )
                break

        return self.report_result()

    def backtest_from_table(
        self,
        table: Any,
        *,
        arm_col: str = "arm",
        total_col: str = "total",
        success_col: str = "success",
        order_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Historical Analysis from an Ibis table.
        Replays data from the table and stops immediately on a stopping decision.
        """
        from earlysign.schema.ES3.Binomial import BinomialArmData

        # 1. Project and order
        if order_by:
            data_table = table.select(
                arm=table[arm_col],
                total=table[total_col],
                success=table[success_col],
                _order=table[order_by],
            ).order_by("_order")
        else:
            data_table = table.select(
                arm=table[arm_col],
                total=table[total_col],
                success=table[success_col],
            )

        # 2. Replay & Stop
        df = data_table.execute()
        logger = get_logger(__name__)

        for i, (_, row) in enumerate(df.iterrows()):
            batch = [
                BinomialArmData(
                    arm=str(row["arm"]),
                    total=int(row["total"]),
                    success=int(row["success"]),
                )
            ]
            self.update(batch)
            prog = self.report_progress()

            if prog.get("status") != DecisionStatus.CONTINUE:
                logger.info(f"Stopping criterion met at row {i}: {prog.get('status')}")
                break

        # Calculate efficiency report
        with Session(self.ledger) as sess:
            total_data_points = int(df[total_col].sum())
            report = sess.read(BacktestProjector(total_samples=total_data_points)).data
            res = report.model_dump(mode="json")
            # Flatten final_report for compatibility
            fr = res.pop("final_report")
            res.update(fr)
        return res

    @classmethod
    def describe_protocol_instance(cls, protocol: BaseModel) -> str:
        """Summarizes the Classic GST design (Pocock, OBF, etc)."""
        from string import Template

        tpl = Template(
            """
Design: Classic Group Sequential Test ($design_type)
==================================================
Task: $task_name
Response Type: $response_type
Arms: $arms_desc
Requirements: Alpha=$alpha, Power=$power

Stopping Policy:
  Method: $method_kind
  Looks (K): $looks
  Analyses: $analyses
"""
        )

        classic_protocol = cast(ClassicProtocol, protocol)
        task = classic_protocol.task
        method = classic_protocol.method

        # Extract details
        design_type = "Unknown"
        strategy = method.stopping_policy.strategy
        if hasattr(strategy, "kind"):
            design_type = strategy.kind.replace("_", " ").title()

        arms_desc = "N/A"
        if isinstance(task.arms, ES3_BASE.TwoArmComparison):
            arms_desc = (
                f"{task.arms.control_arm_name} vs {task.arms.treatment_arm_name}"
            )
        elif isinstance(task.arms, ES3_BASE.SingleArm):
            arms_desc = f"Single Arm: {task.arms.arm_name}"

        policy = method.stopping_policy
        looks_str: str = "N/A"
        analyses = "N/A"
        if hasattr(policy.schedule, "analyses"):
            analyses_list = getattr(policy.schedule, "analyses")
            looks_str = str(len(analyses_list))
            analyses = ", ".join([f"{t:.2f}" for t in analyses_list])

        return tpl.substitute(
            design_type=design_type,
            task_name=classic_protocol.name or "Unnamed Classic GST",
            response_type=task.response_type,
            arms_desc=arms_desc,
            alpha=f"{task.efficacy.alpha:.4f}" if task.efficacy else "N/A",
            power=f"{task.futility.power:.4f}" if task.futility else "N/A",
            method_kind=method.kind,
            looks=looks_str,
            analyses=analyses,
        ).strip()
