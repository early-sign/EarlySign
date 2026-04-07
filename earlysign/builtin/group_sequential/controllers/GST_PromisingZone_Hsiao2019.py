"""Optimal Promising Zone Adaptive Design Controller (Hsiao 2019).

This controller implements the "Optimal Promising Zone" adaptive design for Binomial A/B testing
based on the method described in:

    Hsiao, S. T., Liu, L., & Mehta, C. R. (2019). Optimal promising zone designs.
    Biometrical Journal, 61(5), 1175–1186.

Key Features:
  - Preserves the Type I error rate using the "Promising Zone" principle with
    an unweighted (conventional) test statistic.
  - Increases sample size only when interim results fall into a pre-defined "Promising Zone"
    (typically defined by Conditional Power boundaries, e.g., [0.5, 0.9]).
  - Optimizes the Conditional Power (CP) calculation using the *design* effect size (delta_min)
    rather than the observed effect size, ensuring robustness and optimality.

References:
    Hsiao, S. T., Liu, L., & Mehta, C. R. (2019). Optimal promising zone designs.
    Biometrical Journal, 61(5), 1175–1186.

    Mehta, C. R., & Pocock, S. J. (2011). Adaptive increase in sample size when interim results
    are promising: A practical guide with examples. Statistics in Medicine, 30(28), 3267–3284.

Examples:
    >>> import ibis, duckdb  # noqa: F401
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.builtin.group_sequential.controllers.GST_PromisingZone_Hsiao2019 import Hsiao2019Controller
    >>> import earlysign.schema.ES3.Base as ES3_BASE
    >>> from earlysign.schema.ES3.Binomial import BinomialArmData
    >>>
    >>> # 1. Setup
    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="doctest_hsiao2019")
    >>>
    >>> # 2. Design "Optimal Promising Zone"
    >>> # 2 Looks, Unfavorable zone < 0.5 CP, Promising [0.5, 0.9], Favorable > 0.9
    >>> protocol = Hsiao2019Controller.design(
    ...     p_control=0.10,
    ...     p_treatment=0.14,  # delta=0.04
    ...     alpha=0.025,
    ...     power=0.8,
    ...     looks=2,
    ...     conditional_power_min=0.5,        # Lower bound of promising zone
    ...     conditional_power_max=0.9,        # Upper bound of promising zone
    ...     target_conditional_power=0.9,     # Target CP for resizing
    ... )
    >>> controller = Hsiao2019Controller(ledger)
    >>> controller.set_protocol(protocol)
    >>>
    >>> # 3. Update with "Promising" data
    >>> # Control: 30/300 (10%), Treatment: 42/300 (14%) -> Null diff
    >>> # Wait, we need "promising" result (CP ~ 0.6).
    >>> # Let's simulate a Z-score that yields CP in [0.5, 0.9] based on *design* effect.
    >>> # Design Effect = 0.04.
    >>> # If observed is roughly congruent or slightly less, CP might be moderate.
    >>> # For doctest simplicity, we manually inject data to hit the zone.
    >>> batch = [
    ...     BinomialArmData(total=300, success=30, arm="control"),
    ...     BinomialArmData(total=300, success=48, arm="treatment") # 16% -> +6% benefit observed
    ... ]
    >>> controller.update(batch)
    >>>
    >>> # 4. Report
    >>> report = controller.report_progress()
    >>> # Check if adaptation happened (max_sample_size might increase)
    >>> # original N approx 780 per arm? Total ~1500?
    >>> # If CP is promising, N should increase.
"""

import warnings
from typing import Any, Dict, Literal, Optional, Sequence, Union, cast

from pydantic import BaseModel

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.builtin.group_sequential import schema as GST
from earlysign.builtin.group_sequential.core.model import (
    NumericalIntegrationConfig,
    SimulationConfig,
)
from earlysign.builtin.group_sequential.engine.engine import (
    GroupSequentialEngine,
)
from earlysign.builtin.group_sequential.engine.entities import InterimAnalyses
from earlysign.builtin.group_sequential.engine.sample_size_reestimation import (
    ConditionalPowerAdaptationEngine as PromisingZoneAdaptationEngine,
)
from earlysign.builtin.group_sequential.reporting.projectors import (
    BacktestProjector,
    FinalProjector,
)
from earlysign.builtin.group_sequential.reporting.visualization import (
    plot_gst_summary,
)
from earlysign.builtin.group_sequential.schema import (
    AdaptationLog,
    DecisionStatus,
    GSTMethodSpec,
    GSTProtocol,
    GSTTaskSpec,
    PromisingZoneSpec,
    PromisingZoneStatus,
)
from earlysign.core.ledger import Ledger
from earlysign.core.util.logging import get_logger
from earlysign.framework.controller import Controller, RichDisplayMixin
from earlysign.framework.projector import ProtocolProjector
from earlysign.framework.session import Session


class Hsiao2019Protocol(GSTProtocol, RichDisplayMixin):
    """Protocol for Optimal Promising Zone Design (Hsiao et al 2019)."""

    task: GSTTaskSpec
    method: GSTMethodSpec

    # Specific configuration for Hsiao methodology
    conditional_power_min: float = 0.5
    conditional_power_max: float = 0.9
    target_conditional_power: float = 0.9


Hsiao2019Protocol.model_rebuild()


class Hsiao2019Controller(Controller[Hsiao2019Protocol]):
    """Orchestrator for Optimal Promising Zone Designs (Hsiao et al., 2019).

    This implementation uses the **Unweighted** test statistic for the final analysis,
    permitted by the "Promising Zone" design constraints (Chen, DeMets, Lan 2004; Mehta & Pocock 2011).
    """

    _protocol_class = Hsiao2019Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def describe_protocol_instance(cls, protocol: BaseModel) -> str:
        if not isinstance(protocol, Hsiao2019Protocol):
            raise TypeError("Protocol must be an instance of Hsiao2019Protocol")
        return f"Hsiao2019 Optimal Promising Zone Design (CP: [{protocol.conditional_power_min}, {protocol.conditional_power_max}], Target CP: {protocol.target_conditional_power})"

    @classmethod
    def design(
        cls,
        looks: int,
        alpha: float,
        power: float,
        task: Optional[GST.GSTTaskSpec] = None,
        # Promising Zone Parameters
        conditional_power_min: float = 0.5,
        conditional_power_max: float = 0.9,
        target_conditional_power: float = 0.9,
        # max_sample_size_cap: float = 2.0, # Not strictly enforced in protocol schema yet
        spending_function: str = "obrien_fleming",
        spending_params: Optional[Dict[str, Any]] = None,
        designer_params: Optional[Dict[str, Any]] = None,
        # Binomial params (convenience)
        p_control: Optional[float] = None,
        p_treatment: Optional[float] = None,
        control_arm_name: str = "control",
        treatment_arm_name: str = "treatment",
        method: Literal["simulation", "numerical_integration"] = "simulation",
        method_config: Optional[
            Union[SimulationConfig, NumericalIntegrationConfig]
        ] = None,
    ) -> Hsiao2019Protocol:
        """
        Designs the protocol.

        Args:
             conditional_power_min: Minimum Conditional Power to be considered "Promising".
             conditional_power_max: Maximum Conditional Power to be considered "Promising" (above this is "Favorable").
             target_conditional_power: Target Conditional Power to achieve when increasing sample size.
             method: Method for boundary solving ('simulation' or 'numerical_integration').
             method_config: Configuration object.
        """
        from earlysign.builtin.group_sequential.design.protocol_design import (
            ProtocolDesigner,
        )

        # 1. Construct/Validate Task
        if task is None:
            if p_control is None or p_treatment is None:
                raise ValueError(
                    "Must provide either 'task' or 'p_control'/'p_treatment'."
                )

            delta = p_treatment - p_control
            task = GST.GSTTaskSpec(
                arms=ES3_BASE.TwoArmComparison(
                    control_arm_name=control_arm_name,
                    treatment_arm_name=treatment_arm_name,
                ),
                response_type=GST.ResponseType.BINARY,
                efficacy=GST.EfficacyRequirement(alpha=alpha),
                futility=GST.FutilityRequirement(power=power),
                hypotheses=GST.HypothesisSpec(
                    h_null_description="diff <= 0",
                    h_alt_description=f"diff > {delta}",
                    test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                    target_effect=GST.BinaryEffectSize(
                        proportions=[p_control, p_treatment]
                    ),
                ),
            )

        # 2. Design
        designer_params = designer_params or {"model": "canonical_joint"}
        designer = ProtocolDesigner.from_dict(designer_params)

        # We use standard GSD design parameters initially
        method_spec = designer.method_from_task_spec(
            task,
            params={
                "looks": looks,
                "spending_function": spending_function,
                "spending_params": spending_params,
                "ssr_method": "hsiao_2019",
                "method": method,
                "method_config": method_config,
            },
        )

        # Attach AdaptationSpec with target power for reference
        promising_spec = PromisingZoneSpec(
            conditional_power_threshold_min=conditional_power_min,
            conditional_power_threshold_max=conditional_power_max,
            target_conditional_power=target_conditional_power,
        )

        ssr_spec = GST.SampleSizeReestimationSpec(
            method=GST.Method.CONDITIONAL_POWER,
            promising_zone=promising_spec,
            use_weighted_statistic=False,
            n_range=[0, 1000000],
            target_power=target_conditional_power,
        )

        if method_spec.adaptation is None:
            method_spec.adaptation = GST.AdaptationSpec()

        # Assign SSR spec to the composition container
        method_spec.adaptation.sample_size_reestimation = ssr_spec

        # Create Protocol
        protocol = Hsiao2019Protocol(
            name="Optimal Promising Zone Design (Hsiao 2019)",
            task=task,
            method=method_spec,
            conditional_power_min=conditional_power_min,
            conditional_power_max=conditional_power_max,
            target_conditional_power=target_conditional_power,
        )

        return protocol

    def update(self, batch: Sequence[BaseModel]) -> None:
        """
        Run update cycle with Adaptation using Hsiao et al (2019) logic.
        """
        from earlysign.builtin.group_sequential.engine.trigger_strategies import (
            get_pending_look_trigger,
        )
        from earlysign.parts.trackers.binomial import Scoreboard

        # 1. Ingest
        with Session(self.ledger) as sess:
            if batch:
                # Validate arm names
                protocol_traced = sess.read(ProtocolProjector(Hsiao2019Protocol))
                arms = protocol_traced.data.task.arms
                if not isinstance(arms, ES3_BASE.TwoArmComparison):
                    raise NotImplementedError("Only TwoArmComparison is supported.")
                allowed_arms = {arms.control_arm_name, arms.treatment_arm_name}
                for item in batch:
                    arm_name = getattr(item, "arm", None)
                    if arm_name and arm_name not in allowed_arms:
                        warnings.warn(f"Unexpected arm '{arm_name}'", UserWarning)
                    sess.commit(item, trace=[])

            # 2. Analysis
            protocol_traced = sess.read(ProtocolProjector(Hsiao2019Protocol))
            current_protocol = protocol_traced.data

            metrics = sess.read(Scoreboard(identity="metrics"))
            trajectory = sess.read(InterimAnalyses(identity="interim_analyses"))

            # 3. Standard GSD Engine
            from earlysign.builtin.group_sequential.schema import (
                GSTMethodSpec,
                GSTProtocol,
                GSTTaskSpec,
            )

            gst_protocol = GSTProtocol(
                name="Hsiao-Runtime",
                task=current_protocol.task,
                method=current_protocol.method,
            )
            engine = GroupSequentialEngine(gst_protocol)

            trigger_info = get_pending_look_trigger(
                protocol=protocol_traced,
                metrics=metrics,
                history=trajectory,
            )

            look_result = None
            if trigger_info:
                result = engine.run(
                    metrics=metrics.data,
                    history=trajectory.data,
                    trigger=trigger_info.data,
                )
                sess.commit(result, identity="interim_analyses")
                look_result = result

            # 4. Adaptation Logic
            if not look_result:
                return

            if look_result.status == DecisionStatus.CONTINUE:
                # Use Hsiao parameters from protocol
                cp_min = current_protocol.conditional_power_min
                cp_max = current_protocol.conditional_power_max
                target_conditional_power = current_protocol.target_conditional_power

                adapter = PromisingZoneAdaptationEngine()
                adaptation_log = adapter.check_and_adapt(
                    look_result,
                    gst_protocol,
                )

                sess.commit(adaptation_log)

                if (
                    adaptation_log.promising_zone_status
                    == PromisingZoneStatus.PROMISING
                    and adaptation_log.recommended_sample_size
                ):
                    # APPLY ADAPTATION: Update Protocol
                    new_protocol = adapter.replan_sample_size(
                        gst_protocol,
                        adaptation_log,
                        look_result,
                        target_conditional_power,
                    )

                    updated_hsiao_protocol = Hsiao2019Protocol(
                        name="Hsiao 2019 Protocol",
                        task=cast(GSTTaskSpec, new_protocol.task),
                        method=cast(GSTMethodSpec, new_protocol.method),
                        conditional_power_min=cp_min,
                        conditional_power_max=cp_max,
                        target_conditional_power=target_conditional_power,
                    )

                    sess.commit(updated_hsiao_protocol)

    def report_progress(self) -> Dict[str, Any]:
        """Report intermediate progress."""
        from earlysign.builtin.group_sequential.reporting.projectors import (
            ProgressProjector,
        )

        with Session(self.ledger) as sess:
            return sess.read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")

    def backtest(self, batches: Any) -> Dict[str, Any]:
        """
        Historical Analysis: Replays data and stops immediately on a stopping decision.
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

    def plot_result(self) -> Any:
        with Session(self.ledger) as sess:
            # 1. Read Protocol
            protocol_traced = sess.read(ProtocolProjector(GSTProtocol))
            gst_protocol = protocol_traced.data

            # Repackage as Hsiao2019Protocol
            Hsiao2019Protocol(
                name="Hsiao 2019 Protocol",
                task=gst_protocol.task,
                method=gst_protocol.method,
            )

            trajectory = sess.read(InterimAnalyses(identity="interim_analyses")).data

            logs_df = self.ledger.t.filter(
                self.ledger.t.type == "AdaptationLog"
            ).execute()

            adaptation_logs = []
            if not logs_df.empty:
                for _, row in logs_df.iterrows():
                    adaptation_logs.append(AdaptationLog.model_validate(row["payload"]))

            return plot_gst_summary(
                gst_protocol,
                title="Optimal Promising Zone Design (Hsiao et al. 2019)",
                adaptation_logs=adaptation_logs,
                full_history=trajectory,
            )
