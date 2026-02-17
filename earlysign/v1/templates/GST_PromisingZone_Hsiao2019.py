"""Optimal Promising Zone Design Template (Hsiao et al., 2019).

This template implements the "Optimal Promising Zone" adaptive design for Binomial A/B testing
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
    >>> from earlysign.v1.templates.GST_PromisingZone_Hsiao2019 import Hsiao2019Template
    >>> import earlysign.schema.ES3.Base as ES3_BASE
    >>> from earlysign.schema.ES3.Binomial import ArmData
    >>>
    >>> # 1. Setup
    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="doctest_hsiao2019")
    >>>
    >>> # 2. Design "Optimal Promising Zone"
    >>> # 2 Looks, Unfavorable zone < 0.5 CP, Promising [0.5, 0.9], Favorable > 0.9
    >>> protocol = Hsiao2019Template.design(
    ...     p_control=0.10,
    ...     p_treatment=0.14,  # delta=0.04
    ...     alpha=0.025,
    ...     power=0.8,
    ...     looks=2,
    ...     cp_min=0.5,        # Lower bound of promising zone
    ...     cp_max=0.9,        # Upper bound of promising zone
    ...     target_cp=0.9,     # Target CP for resizing
    ... )
    >>> template = Hsiao2019Template(ledger)
    >>> template.set_protocol(protocol)
    >>>
    >>> # 3. Update with "Promising" data
    >>> # Control: 30/300 (10%), Treatment: 42/300 (14%) -> Null diff
    >>> # Wait, we need "promising" result (CP ~ 0.6).
    >>> # Let's simulate a Z-score that yields CP in [0.5, 0.9] based on *design* effect.
    >>> # Design Effect = 0.04.
    >>> # If observed is roughly congruent or slightly less, CP might be moderate.
    >>> # For doctest simplicity, we manually inject data to hit the zone.
    >>> batch = [
    ...     ArmData(n=300, success=30, arm="control"),
    ...     ArmData(n=300, success=48, arm="treatment") # 16% -> +6% benefit observed
    ... ]
    >>> template.update(batch)
    >>>
    >>> # 4. Report
    >>> report = template.report_progress()
    >>> # Check if adaptation happened (max_sample_size might increase)
    >>> # original N approx 780 per arm? Total ~1500?
    >>> # If CP is promising, N should increase.
"""

import warnings
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

import earlysign.schema.ES3.Base as ES3_BASE
import earlysign.schema.ES3.GST as GST
from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.GST.Log import (
    AdaptationLog,
    DecisionStatus,
    LookResult,
    PromisingZoneStatus,
)
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.framework.template import TemplateBase
from earlysign.v1.methods.group_sequential.execution.binomial import (
    BinomialGSTEngine,
)
from earlysign.v1.methods.group_sequential.execution.entities import InterimAnalyses
from earlysign.v1.methods.group_sequential.execution.sample_size_reestimation import (
    ConditionalPowerAdaptationEngine as PromisingZoneAdaptationEngine,
)
from earlysign.v1.methods.group_sequential.reporting.projectors import (
    FinalProjector,
    ProgressProjector,
)
from earlysign.v1.methods.group_sequential.reporting.visualization import (
    plot_gst_summary,
)


class Hsiao2019Protocol(BaseModel):
    """Protocol for Optimal Promising Zone Design (Hsiao et al 2019)."""

    task: GST.TaskSpec
    method: GST.MethodSpec

    # Specific configuration for Hsiao methodology
    cp_min: float = 0.5
    cp_max: float = 0.9
    target_cp: float = 0.9


Hsiao2019Protocol.model_rebuild()


class Hsiao2019Template(TemplateBase[Hsiao2019Protocol]):
    """Orchestrator for Optimal Promising Zone Designs (Hsiao et al., 2019).

    This implementation uses the **Unweighted** test statistic for the final analysis,
    permitted by the "Promising Zone" design constraints (Chen, DeMets, Lan 2004; Mehta & Pocock 2011).
    """

    _protocol_class = Hsiao2019Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        looks: int,
        alpha: float,
        power: float,
        task: Optional[GST.TaskSpec] = None,
        # Promising Zone Parameters
        cp_min: float = 0.5,
        cp_max: float = 0.9,
        target_cp: float = 0.9,
        # max_sample_size_cap: float = 2.0, # Not strictly enforced in protocol schema yet
        spending_function: str = "obrien_fleming",
        spending_params: Optional[Dict[str, Any]] = None,
        designer_params: Optional[Dict[str, Any]] = None,
        # Binomial params (convenience)
        p_control: Optional[float] = None,
        p_treatment: Optional[float] = None,
    ) -> Hsiao2019Protocol:
        """
        Designs the protocol.

        Args:
             cp_min: Minimum Conditional Power to be considered "Promising".
             cp_max: Maximum Conditional Power to be considered "Promising" (above this is "Favorable").
             target_cp: Target Conditional Power to achieve when increasing sample size.
        """
        from earlysign.v1.methods.group_sequential.plan.protocol_design import (
            ProtocolDesigner,
        )

        # 1. Construct/Validate Task
        if task is None:
            if p_control is None or p_treatment is None:
                raise ValueError(
                    "Must provide either 'task' or 'p_control'/'p_treatment'."
                )

            delta = p_treatment - p_control
            task = GST.TaskSpec(
                arms=ES3_BASE.TwoArmComparison(
                    control_arm_name="control",
                    treatment_arm_name="treatment",
                ),
                response_type=GST.ResponseType.BINARY,
                efficacy=GST.EfficacyRequirement(alpha=alpha),
                futility=GST.FutilityRequirement(power=power),
                hypotheses=GST.HypothesisSpec(
                    h_null_description="diff <= 0",
                    h_alt_description=f"diff > {delta}",
                    test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                    target_effect=GST.BinaryEffectSize(
                        proportions={"control": p_control, "treatment": p_treatment}
                    ),
                ),
            )

        # 2. Design
        designer_params = designer_params or {"model": "canonical_joint"}
        designer = ProtocolDesigner.from_dict(designer_params)

        # We use standard GSD design parameters initially
        method = designer.method_from_task_spec(
            task,
            params={
                "looks": looks,
                "spending_function": spending_function,
                "spending_params": spending_params,
                "ssr_method": "hsiao_2019",
            },
        )

        # Attach AdaptationSpec with target power for reference
        method.adaptation = GST.SampleSizeReestimationSpec(
            n_range=[0, 1000000],
            target_power=target_cp,
            method="conditional_power",
            use_weighted_statistic=False,
        )

        # Create Protocol
        protocol = Hsiao2019Protocol(
            task=task, method=method, cp_min=cp_min, cp_max=cp_max, target_cp=target_cp
        )

        return protocol

    def update(self, batch: List[BaseModel]) -> None:
        """
        Run update cycle with Adaptation using Hsiao et al (2019) logic.
        """
        from earlysign.v1.methods.binomial import Scoreboard

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
            history = sess.read(InterimAnalyses(identity="interim_analyses"))

            # 3. Standard GSD Engine
            # This engine will respect use_weighted_statistic=False if snapshot exists
            gst_protocol = GST.Protocol(
                name="Hsiao2019-Runtime",
                task=current_protocol.task,
                method=current_protocol.method,
            )
            engine = BinomialGSTEngine(gst_protocol)

            sess.call_and_commit(
                LookResult,
                engine.run,
                metrics=metrics,
                history=history,
            )

            # 4. Adaptation Logic
            trajectory = sess.read(InterimAnalyses(identity="interim_analyses")).data
            if not trajectory:
                return

            look_result = trajectory[-1][1]

            if look_result.status == DecisionStatus.CONTINUE_:
                # Use Hsiao parameters from protocol
                cp_min = current_protocol.cp_min
                cp_max = current_protocol.cp_max
                target_cp = current_protocol.target_cp

                adapter = PromisingZoneAdaptationEngine()
                adaptation_log = adapter.check_and_adapt(
                    look_result,
                    gst_protocol,
                    cp_threshold_min=cp_min,
                    cp_threshold_max=cp_max,
                )

                sess.commit(adaptation_log)

                if (
                    adaptation_log.promising_zone_status
                    == PromisingZoneStatus.PROMISING
                    and adaptation_log.recommended_sample_size
                ):
                    # APPLY ADAPTATION: Update Protocol
                    new_protocol = adapter.replan_sample_size(
                        gst_protocol, adaptation_log, look_result, target_cp
                    )

                    updated_hsiao_protocol = Hsiao2019Protocol(
                        task=new_protocol.task,
                        method=new_protocol.method,
                        cp_min=cp_min,
                        cp_max=cp_max,
                        target_cp=target_cp,
                    )

                    sess.commit(updated_hsiao_protocol)

    def report_progress(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            report = sess.read(ProgressProjector()).data.model_dump(mode="json")
            protocol_wrapper = sess.read(ProtocolProjector(Hsiao2019Protocol)).data
            report["max_sample_size"] = (
                protocol_wrapper.method.stopping_policy.timer.max_sample_size
            )
            # Add Hsiao specific info
            report["promising_zone"] = [
                protocol_wrapper.cp_min,
                protocol_wrapper.cp_max,
            ]
            return report

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")

    def plot_result(self) -> Any:
        with Session(self.ledger) as sess:
            protocol_wrapper = sess.read(ProtocolProjector(Hsiao2019Protocol)).data
            # Convert wrapper to GST.Protocol for plotting function compatibility
            gst_protocol = GST.Protocol(
                task=protocol_wrapper.task, method=protocol_wrapper.method
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
                full_history=trajectory.data,
            )
