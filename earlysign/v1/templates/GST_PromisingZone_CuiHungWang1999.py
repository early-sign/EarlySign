"""Promising Zone Adaptive Design Template (Cui-Hung-Wang).

This template implements a "Promising Zone" adaptive design for Binomial A/B testing
based on the method described in:

    Cui, L., Hung, H. M., & Wang, S. J. (1999). Modification of sample size in
    group sequential clinical trials. Biometrics, 55(3), 853–857.

It allows for an interim Sample Size Re-estimation (SSR) if the results fall into
a "Promising Zone" (neither efficacious nor futile, but showing promise).

Key Features:
    - Group Sequential Design (GSD) foundation (e.g., O'Brien-Fleming).
    - Conditional Power (CP) calculation at interim looks.
    - SSR rule: Increase sample size to recover power if CP is within the promising zone.
    - Weighted test statistic (Chen-DeMets-Lan) to preserve Type-1 error rate after adaptation.

References:
    Cui, L., Hung, H. M., & Wang, S. J. (1999). Modification of sample size in
    group sequential clinical trials. Biometrics, 55(3), 853–857.

    In practice, each iteration may run in a different process.
    To support this use case, the Template object can be destroyed after each iteration and re-instantiated.

Examples:
    >>> import ibis, duckdb  # noqa: F401
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.v1.templates.GST_PromisingZone_CuiHungWang1999 import CuiHungWang1999Template
    >>> from earlysign.schema.ES3.Binomial import ArmData
    >>> from earlysign.schema.ES3.GST.Log import DecisionStatus
    >>>
    >>> # 1. Setup
    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="doctest_chw1999")
    >>>
    >>> # 2. Design with Promising Zone
    >>> # 2 Looks, initial N=1000.
    >>> protocol = CuiHungWang1999Template.design(
    ...     p_control=0.10,
    ...     p_treatment=0.13,
    ...     alpha=0.025,
    ...     power=0.8,
    ...     looks=2,
    ...     spending_function="obrien_fleming",
    ...     designer_params={"model": "canonical_joint", "model_params": {"rng_seed": 42}}
    ... )
    >>> template = CuiHungWang1999Template(ledger)
    >>> template.set_protocol(protocol)
    >>>
    >>> # 3. Update with "Promising" data (Conditional Power ~ 0.6)
    >>> # Need roughly half the data.
    >>> # Control: 50/500 (10%), Treatment: 65/500 (13%) -> Z ~ 1.5
    >>> # This should fall into the promising zone if configured right.
    >>> batch = [
    ...     ArmData(n=500, success=50, arm="C"),
    ...     ArmData(n=500, success=65, arm="T")
    ... ]
    >>> template.update(batch)
    >>>
    >>> # 5. Verify status and SSR trigger
    >>> report = template.report_progress()
    >>> report["status"]
    'continue'
    >>> report["max_sample_size"]
    3194
"""

import warnings
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

import earlysign.schema.ES3.GST as GST
from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.GST.Log import (
    AdaptationLog,
    DecisionStatus,
    PromisingZoneStatus,
)
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.group_sequential.execution.binomial import (
    BinomialGSTEngine,
)
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
from earlysign.v1.templates.base import TemplateBase


class CuiHungWang1999Protocol(BaseModel):
    """Protocol for Promising Zone Adaptive Design (Cui-Hung-Wang 1999)."""

    task: GST.TaskSpec
    method: GST.MethodSpec


CuiHungWang1999Protocol.model_rebuild()


class CuiHungWang1999Template(TemplateBase[CuiHungWang1999Protocol]):
    """Orchestrator for Promising Zone Adaptive Designs (Cui-Hung-Wang).

    Logic:
    1. Standard Group Sequential Test (Binomial).
    2. If not stopped: Check Promising Zone.
    3. If Promising: Calc new N (SSR) -> Update Protocol -> Commit Adaptation Log.
    4. Next Looks: Use Cui-Hung-Wang weighted statistic (if needed/implemented) or adjusted boundaries.
       *Our current engine implementation uses the CHW adjusted statistic logic.*
    """

    _protocol_class = CuiHungWang1999Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        task: Optional[GST.TaskSpec] = None,
        looks: int = 2,
        spending_function: str = "obrien_fleming",
        spending_params: Optional[Dict[str, Any]] = None,
        designer_params: Optional[Dict[str, Any]] = None,
        # Binomial params (convenience)
        p_control: Optional[float] = None,
        p_treatment: Optional[float] = None,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> CuiHungWang1999Protocol:
        """
        Designs the protocol. Supports both TaskSpec and scalar inputs.

        Args:
             task: Generic task specification.
             looks: Number of looks.
             spending_function: Spending function family.
             spending_params: Spending function parameters.
             designer_params: Designer configuration.
             p_control, p_treatment, alpha, power: Scalar inputs if task is None.
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
                arms=["control", "treatment"],
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
        method = designer.method_from_task_spec(
            task,
            params={
                "looks": looks,
                "spending_function": spending_function,
                "spending_params": spending_params,
                "ssr_method": "cui_hung_wang",
            },
        )
        return CuiHungWang1999Protocol(task=task, method=method)

    def update(self, batch: List[BaseModel]) -> None:
        """
        Run update cycle with Adaptation check.

        Args:
            batch: Data batch.
        """
        from earlysign.v1.methods.binomial import Scoreboard

        # 1. Ingest
        if batch:
            with Session(self.ledger) as sess:
                # Validate arm names
                protocol = sess.read(ProtocolProjector(CuiHungWang1999Protocol)).data
                allowed_arms = set(protocol.task.arms)
                for item in batch:
                    arm_name = getattr(item, "arm", None)
                    if arm_name and arm_name not in allowed_arms:
                        warnings.warn(
                            f"Received data for unexpected arm '{arm_name}'. "
                            f"Expected arms: {allowed_arms}",
                            UserWarning,
                        )
                    sess.commit(item, trace=[])

        # 2. Analysis & Adaptation
        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(CuiHungWang1999Protocol)).data
            metrics = sess.read(Scoreboard(identity="metrics")).data

            # 3. Standard GSD Engine
            engine = BinomialGSTEngine(protocol)

            look_result = engine.run(metrics)

            # Commit the LookResult
            sess.commit(look_result)

            # 4. Adaptation Logic (only if continuing)
            if look_result.status == DecisionStatus.CONTINUE_:
                adapter = PromisingZoneAdaptationEngine()
                adaptation_log = adapter.check_and_adapt(look_result, protocol)

                sess.commit(adaptation_log)

                if (
                    adaptation_log.promising_zone_status
                    == PromisingZoneStatus.PROMISING
                    and adaptation_log.recommended_sample_size
                ):
                    # APPLY ADAPTATION: Update Protocol
                    new_protocol = adapter.replan_sample_size(
                        protocol, adaptation_log, look_result
                    )
                    sess.commit(new_protocol)

    def report_progress(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            report = sess.read(ProgressProjector()).data.model_dump(mode="json")
            protocol = sess.read(ProtocolProjector(CuiHungWang1999Protocol)).data
            # Enrich with current trial constraints
            report["max_sample_size"] = (
                protocol.method.stopping_policy.timer.max_sample_size
            )
            return report

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")

    def plot_result(self) -> Any:
        from earlysign.v1.framework.entity import InterimAnalyses

        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(CuiHungWang1999Protocol)).data
            trajectory = sess.read(InterimAnalyses(identity="interim_analyses")).data

            # Read Adaptation Logs
            logs_df = self.ledger.t.filter(
                self.ledger.t.type == "AdaptationLog"
            ).execute()
            adaptation_logs = []
            if not logs_df.empty:
                for _, row in logs_df.iterrows():
                    adaptation_logs.append(AdaptationLog.model_validate(row["payload"]))

            history_n = []
            history_z = []
            for _, state in trajectory.data:
                history_n.append(state.sample_n)
                history_z.append(state.z_stat)

            return plot_gst_summary(
                protocol,
                history_n,
                history_z,
                title="Promising Zone Design Monitoring",
                adaptation_logs=adaptation_logs,
            )
