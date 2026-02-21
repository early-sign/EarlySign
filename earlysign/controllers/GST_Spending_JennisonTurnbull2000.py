"""Binomial A/B Testing Controller.

This module provides a standard controller for running sequential A/B tests with
binary outcomes using Alpha Spending functions.

Reference:
    Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with
    Applications to Clinical Trials. Chapman and Hall/CRC.

Example:
    >>> import ibis, duckdb  # noqa: F401
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.controllers.GST_Spending_JennisonTurnbull2000 import JennisonTurnbull2000Controller, JennisonTurnbull2000TaskSpec
    >>> import earlysign.schema.ES3.Base as ES3_BASE
    >>> import earlysign.schema.ES3.GST as GST
    >>> from earlysign.schema.ES3.GST.Log import DecisionStatus
    >>> from earlysign.tests.util import BinomialStream
    >>> from earlysign.schema.ES3.Binomial import BinomialArmData
    >>> import numpy as np
    >>> # Setup
    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="example_001")
    >>>
    >>> # Scenario:
    >>> # You are a Product Manager at Acme Corp launching a new checkout flow (v2).
    >>> # You hope to increase the "Payment Success Rate" by 5% (absolute) over the baseline (v1).
    >>> # You have a fixed budget for 4 weeks of testing, with interim analyses every week
    >>> # to potentially stop early for overwhelming efficacy (shipping the win) or futility (saving traffic).
    >>>
    >>> # Design: 4 looks (weekly), target lift 20% -> 25% (Huge win)
    >>> protocol = JennisonTurnbull2000Controller.design(
    ...     p_control=0.20, p_treatment=0.25, alpha=0.05, power=0.8, looks=4,
    ...     method="simulation"
    ... )
    >>>
    >>> # Initialize and Run
    >>> controller = JennisonTurnbull2000Controller(ledger, rng_seed=42)
    >>> controller.set_protocol(protocol)
    >>>
    >>> # Simulate Weekly Batches
    >>> # We manually define batches to show the progression towards a "win".
    >>> # Week 1: 300 users, slight lift (20% vs 23%) -> Continue
    >>> # Week 2: 300 users, strong lift (20% vs 28%) -> Significance!
    >>> batch_w1 = [BinomialArmData(total=300, success=60, arm="control"), BinomialArmData(total=300, success=69, arm="treatment")]
    >>> batch_w2 = [BinomialArmData(total=300, success=60, arm="control"), BinomialArmData(total=300, success=84, arm="treatment")]
    >>>
    >>> # Update Week 1
    >>> controller.update(batch_w1)
    >>> prog = controller.report_progress()
    >>> print(f"Week 1: Z={prog['z_stat']:.2f}, Status={prog['status']}")
    Week 1: Z=0.89, Status=continue
    >>>
    >>> # Update Week 2
    >>> controller.update(batch_w2)
    >>> prog = controller.report_progress()
    >>> print(f"Week 2: Z={prog['z_stat']:.2f}, Status={prog['status']}")
    Week 2: Z=2.27, Status=stop_efficacy
    >>>
    >>> final = controller.report_result()
    >>> print(f"Final Status: {final['final_status']}")
    Final Status: stop_efficacy
"""

import warnings
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

from pydantic import BaseModel, Field, TypeAdapter

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
from earlysign.framework.session import BacktestSession, Session
from earlysign.methods.binomial import Scoreboard
from earlysign.methods.group_sequential.core.model import (
    NumericalIntegrationConfig,
    SimulationConfig,
)
from earlysign.methods.group_sequential.design.protocol_design import (
    ProtocolDesigner,
)
from earlysign.methods.group_sequential.engine.engine import (
    GroupSequentialEngine,
)
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
from earlysign.methods.group_sequential.reporting.visualization import (
    plot_gst_summary,
)
from earlysign.schema.ES3.GST import (
    AbsoluteDifference,
    EffectMeasure,
    OddsRatio,
    RelativeImprovement,
    RelativeRisk,
)
from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult


class JennisonTurnbull2000TaskSpec(GST.TaskSpec):
    response_type: GST.ResponseType = GST.ResponseType.BINARY
    # Design Requirements
    efficacy: GST.EfficacyRequirement
    futility: GST.FutilityRequirement

    hypotheses: GST.HypothesisSpec


class JennisonTurnbull2000Protocol(GST.Protocol, AutoNameMixin, RichDisplayMixin):
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
        case RelativeImprovement(value=ri_val):
            return p_control * (1.0 + ri_val)
        case _:
            raise ValueError(f"Unknown effect type: {type(effect_spec)}")


class JennisonTurnbull2000Controller(Controller[JennisonTurnbull2000Protocol]):
    """Standard orchestrator for Binomial A/B tests (Jennison & Turnbull 2000).

    Provides high-level methods for designing, Updating, and reporting
    sequential A/B tests with binary outcomes using Alpha Spending functions.

    Reference:
        Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with
        Applications to Clinical Trials. Chapman and Hall/CRC.
    """

    _protocol_class = JennisonTurnbull2000Protocol

    def __init__(self, ledger: Ledger, rng_seed: int = 42):
        self.ledger = ledger
        self.rng_seed = rng_seed

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
        control_arm_name: str = "control",
        treatment_arm_name: str = "treatment",
        allocation_ratios: Optional[Dict[str, float]] = None,
        method: Literal["simulation", "numerical_integration"] = "simulation",
        method_config: Optional[
            Union[SimulationConfig, NumericalIntegrationConfig]
        ] = None,
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
            method: Method for boundary solving ('simulation' or 'numerical_integration').
                    Simulation is significantly faster for larger designs.
            method_config: Configuration object.

        Returns:
            A populated JennisonTurnbull2000Protocol.

        Examples:
            >>> import numpy as np
            >>> from earlysign.controllers.GST_Spending_JennisonTurnbull2000 import JennisonTurnbull2000Controller
            >>> import earlysign.schema.ES3.Base as ES3_BASE
            >>>
            >>> from earlysign.methods.group_sequential.shared.design_utils import get_info_times
            >>>
            >>> # --- Example 1: Standard Equidistant Schedule ---
            >>> protocol_eq = JennisonTurnbull2000Controller.design(
            ...     p_control=0.20, p_treatment=0.22, alpha=0.05, power=0.8, looks=3,
            ...     scheduling="equidistant", designer_params={"model_params": {"rng_seed": 42}}
            ... )
            >>> np.allclose(get_info_times(protocol_eq.method.stopping_policy.schedule), [1/3, 2/3, 1.0])
            True
            >>>
            >>> # --- Example 2: Using structured effect_spec (delta) ---
            >>> protocol_delta = JennisonTurnbull2000Controller.design(
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

            if allocation_ratios and len(allocation_ratios) > 1:
                arms: ES3_BASE.ArmStructure = ES3_BASE.MultiArmComparison(
                    control_arm_name=control_arm_name,
                    treatment_arm_names=list(allocation_ratios.keys()),
                    allocation_ratios=allocation_ratios,
                )
                props = {control_arm_name: p_control}
                props.update({arm: p_treatment for arm in allocation_ratios.keys()})
            else:
                ratio = 1.0
                if allocation_ratios:
                    ratio = list(allocation_ratios.values())[0]

                # TwoArmComparison using allocation_ratios
                two_arm_ratios = None
                if ratio != 1.0:
                    two_arm_ratios = {treatment_arm_name: ratio}
                elif allocation_ratios:
                    two_arm_ratios = allocation_ratios

                arms = ES3_BASE.TwoArmComparison(
                    control_arm_name=control_arm_name,
                    treatment_arm_name=treatment_arm_name,
                    allocation_ratios=two_arm_ratios,
                )
                props = {
                    control_arm_name: p_control,
                    treatment_arm_name: p_treatment,
                }

            task = JennisonTurnbull2000TaskSpec(
                arms=arms,
                response_type=GST.ResponseType.BINARY,
                efficacy=GST.EfficacyRequirement(alpha=alpha),
                futility=GST.FutilityRequirement(power=power),
                hypotheses=GST.HypothesisSpec(
                    h_null_description="Difference <= 0",
                    h_alt_description=f"Difference > {p_treatment - p_control:.4f}",
                    test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                    target_effect=GST.BinaryEffectSize(proportions=props),
                ),
            )

        # 2. Prepare Designer
        designer_params = designer_params or {"model": "canonical_joint"}
        designer = ProtocolDesigner.from_dict(designer_params)
        rng_seed = designer_params.get("model_params", {}).get("rng_seed", 42)

        # 3. Design Strategy Components using common ProtocolDesigner
        from earlysign.schema.ES3.GST import BinaryEffectSize

        hypotheses = task.hypotheses
        if isinstance(hypotheses.target_effect, BinaryEffectSize):
            proportions = hypotheses.target_effect.proportions
        else:
            raise ValueError(
                "JennisonTurnbull2000Controller only supports Binary outcomes."
            )
        arms = task.arms

        allocation_ratios = None
        control_arm_name = "control"
        treatment_arm_name = "treatment"

        if isinstance(arms, ES3_BASE.TwoArmComparison):
            control_arm_name = arms.control_arm_name
            treatment_arm_name = arms.treatment_arm_name
            allocation_ratios = arms.allocation_ratios
        elif isinstance(arms, ES3_BASE.MultiArmComparison):
            control_arm_name = arms.control_arm_name
            treatment_arm_name = arms.treatment_arm_names[0]
            allocation_ratios = arms.allocation_ratios
        else:
            raise NotImplementedError(
                f"GST on {type(arms).__name__} is not yet supported in this controller. "
                "Currently, only TwoArmComparison or MultiArmComparison is supported."
            )

        arm_0, arm_1 = control_arm_name, treatment_arm_name

        method_spec, _ = designer.design_gs_binomial(
            alpha=task.efficacy.alpha,
            power=task.futility.power if task.futility else 0.8,
            p_control=proportions[arm_0],
            p_treatment=proportions[arm_1],
            looks=looks,
            scheduling=cast(Any, scheduling),
            spending_function=spending_function,
            spending_params=spending_params,
            futility=(task.futility is not None),
            futility_binding=bool(task.futility.binding) if task.futility else False,
            tails=1,
            rng_seed=rng_seed,
            allocation_ratios=allocation_ratios,
            control_arm_name=control_arm_name,
            treatment_arm_name=treatment_arm_name,
            method=method,
            method_config=method_config,
        )

        return JennisonTurnbull2000Protocol(
            task=task,
            method=method_spec,
        )

    def update(
        self, batch: Sequence[BaseModel], session: Optional[Session] = None
    ) -> None:
        """
        Orchestrates a single minibatch update cycle:
        Ingest -> [Read -> Analyze -> Decide -> Snapshot].
        """
        # 1. Ingest Data
        # Re-use provided session if available (optimization for backtests)
        if session is not None:
            self._update_with_session(session, batch)
        else:
            with Session(self.ledger) as sess:
                self._update_with_session(sess, batch)

    def _update_with_session(self, sess: Session, batch: Sequence[BaseModel]) -> None:
        """Internal helper for protocol-consistent update logic."""
        if batch:
            # Validate arm names
            protocol_traced = sess.read(ProtocolProjector(JennisonTurnbull2000Protocol))
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
            engine = GroupSequentialEngine(protocol_traced.data, rng_seed=self.rng_seed)

            sess.call_and_commit(
                LookResult,
                engine.run,
                identity="interim_analyses",
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
           batches: Iterator yielding `BinomialArmData` objects or lists of them.
                    Each `BinomialArmData` must have `total`, `success`, and `arm`.
        """
        logger = get_logger(__name__)

        # Use BacktestSession for high-performance replay
        with BacktestSession(
            self.ledger, invariant_projectors=[ProtocolProjector]
        ) as sess:
            for i, batch in enumerate(batches):
                self.update(batch if isinstance(batch, list) else [batch], session=sess)
                # Report progress is now essentially free (cached)
                report = sess.read(ProgressProjector()).data
                prog = report.model_dump(mode="json")

                if prog.get("status") != DecisionStatus.CONTINUE:
                    logger.info(
                        f"Stopping criterion met at index {i}: {prog.get('status')}"
                    )
                    break

        # Calculate efficiency report
        # We need total count from batches. If it's an iterator, we might not have it unless we pre-calculated.
        # But for backtest_from_table, we do.
        return self.report_result()

    def backtest_from_table(
        self,
        table: Any,
        *,
        arm_col: str = "arm",
        total_col: str = "total",  # Use total_col to match base class
        success_col: str = "success",
        order_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Historical Analysis from an Ibis table.
        Replays data from the table and stops immediately on a stopping decision.

        Args:
            table: Ibis table containing historical data.
            arm_col: Column name for the arm identifier.
            total_col: Column name for the number of trials.
            success_col: Column name for the number of successes.
            order_by: Column name to order the data by.
        """
        from earlysign.schema.ES3.Binomial import BinomialArmData

        # 1. Project and order
        if order_by:
            # We must include order_by in our select if we want to order by it,
            # or just use the original table's column.
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

        # Use BacktestSession for high-performance replay
        with BacktestSession(
            self.ledger, invariant_projectors=[ProtocolProjector]
        ) as sess:
            for i, (_, row) in enumerate(df.iterrows()):
                batch = [
                    BinomialArmData(
                        arm=str(row["arm"]),
                        total=int(row["total"]),
                        success=int(row["success"]),
                    )
                ]
                self.update(batch, session=sess)

                # Performance note: sess.read(ProgressProjector) is cheap here due to caching
                report = sess.read(ProgressProjector()).data
                prog = report.model_dump(mode="json")

                if prog.get("status") != DecisionStatus.CONTINUE:
                    logger.info(
                        f"Stopping criterion met at row {i}: {prog.get('status')}"
                    )
                    break

            # Calculate efficiency report at the end
            # We use sum of 'total' for actual sample size efficiency
            total_data_points = int(df[total_col].sum())
            backtest_report_data = sess.read(
                BacktestProjector(total_samples=total_data_points)
            ).data
            bt_res = backtest_report_data.model_dump(mode="json")
            # Flatten final_report for compatibility
            fr = bt_res.pop("final_report")
            bt_res.update(fr)
            return bt_res

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

    @classmethod
    def describe_protocol_instance(cls, protocol: BaseModel) -> str:
        """Summarizes the Jennison & Turnbull (2000) design."""
        from string import Template

        tpl = Template(
            """
Design: Jennison & Turnbull (2000) - Binomial Sequential A/B Test
================================================================
Task: $task_name
Arms: $control vs $treatment
Target Rates: $p_control vs $p_treatment (Delta: $delta)
Requirements: Alpha=$alpha, Power=$power

Stopping Policy:
  Method: $method_kind
  Looks: $looks
  Analyses (Info Fracs): $analyses
  Spending: $spending_fn
"""
        )

        jt_protocol = cast(JennisonTurnbull2000Protocol, protocol)
        task = jt_protocol.task
        method = jt_protocol.method

        # Extract details
        arms = task.arms
        ctrl_arm = (
            arms.control_arm_name
            if isinstance(arms, ES3_BASE.TwoArmComparison)
            else "control"
        )
        trtm_arm = (
            arms.treatment_arm_name
            if isinstance(arms, ES3_BASE.TwoArmComparison)
            else "treatment"
        )

        from earlysign.schema.ES3.GST import BinaryEffectSize

        hypotheses = task.hypotheses
        p0 = 0.0
        p1 = 0.0
        if isinstance(hypotheses.target_effect, BinaryEffectSize):
            p0 = hypotheses.target_effect.proportions.get(ctrl_arm, 0)
            p1 = hypotheses.target_effect.proportions.get(trtm_arm, 0)

        policy = method.stopping_policy
        strategy = policy.strategy

        spending_fn = "Unknown"
        from earlysign.schema.ES3.GST import (
            AlphaBetaSpendingStrategy,
        )

        if isinstance(strategy, AlphaBetaSpendingStrategy):
            spending_fn = f"Alpha: {strategy.alpha_spending_fn.family}, Beta: {strategy.beta_spending_fn.family}"
        elif hasattr(strategy, "spending_fn"):
            spending_fn = strategy.spending_fn.family

        looks_val: Union[int, str] = "N/A"
        analyses = "N/A"
        if hasattr(policy.schedule, "analyses"):
            analyses_list = getattr(policy.schedule, "analyses")
            looks_val = len(analyses_list)
            analyses = ", ".join([f"{t:.2f}" for t in analyses_list])

        return tpl.substitute(
            task_name=jt_protocol.name or "Unnamed GST",
            control=ctrl_arm,
            treatment=trtm_arm,
            p_control=f"{p0:.4f}",
            p_treatment=f"{p1:.4f}",
            delta=f"{p1-p0:.4f}",
            alpha=f"{task.efficacy.alpha:.4f}" if task.efficacy else "N/A",
            power=f"{task.futility.power:.4f}" if task.futility else "N/A",
            method_kind=method.kind,
            looks=str(looks_val),
            analyses=analyses,
            spending_fn=spending_fn,
        ).strip()
