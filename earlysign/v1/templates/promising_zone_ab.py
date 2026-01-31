"""
Promising Zone Adaptive Design Template (Cui-Hung-Wang)
=======================================================

This template implements a "Promising Zone" adaptive design for Binomial A/B testing.
It allows for an interim Sample Size Re-estimation (SSR) if the results fall into
a "promising" region (between efficacy and futility boundaries), increasing conditional power.

Reference
---------
Cui, L., Hung, H. M., & Wang, S. J. (1999). Modification of sample size in group sequential clinical trials. Biometrics, 55(3), 853–857.

Usage
-----
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.v1.templates.promising_zone_ab import PromisingZoneABTemplate
>>> from earlysign.schema.ES3.Binomial import ArmData
>>> from earlysign.schema.ES3.GST.Log import DecisionStatus

# 1. Setup Ledger
>>> conn = ibis.connect("duckdb://:memory:")
>>> ledger = Ledger(conn, "events")
>>> ledger.ensure()

# 2. Design a protocol (Promising Zone enabled)
>>> protocol = PromisingZoneABTemplate.design_binomial(
...     p_control=0.5, p_treatment=0.6, alpha=0.025, power=0.8, looks=2,
...     designer_params={"model_params": {"rng_seed": 42}}
... )
>>> protocol.method.stopping_policy.timer.max_sample_size
798

# 3. Initialize Template
>>> template = PromisingZoneABTemplate(ledger)
>>> template.set_protocol(protocol)

# 4. Simulate an interim look that lands in 'Promising Zone'
# (Note: Z=1.0 at info_frac=0.5 is promising for this design)
>>> batch = [
...     ArmData(arm="control", n=250, success=125),
...     ArmData(arm="treatment", n=250, success=147)
... ]
>>> template.update(batch)

# 5. Verify that Sample Size Re-estimation (SSR) was triggered
>>> progress = template.report_progress()
>>> progress["status"] == DecisionStatus.CONTINUE_
True
>>> progress["max_sample_size"]  # Increased from the original design
1025
"""

from typing import Any, Dict, List, Optional

import numpy as np
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
from earlysign.v1.methods.binomial import Scoreboard
from earlysign.v1.methods.group_sequential.adaptation.engine import (
    ConditionalPowerAdaptationEngine,
)
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


class PromisingZoneABProtocol(GST.Protocol):
    """Protocol for Promising Zone Adaptive Design."""

    task: GST.TaskSpec
    method: GST.MethodSpec


PromisingZoneABProtocol.model_rebuild()


class PromisingZoneABTemplate:
    """
    Orchestrator for Promising Zone Adaptive Designs (Cui-Hung-Wang).

    Logic:
    1. Standard Group Sequential Test (Binomial).
    2. If not stopped: Check Promising Zone.
    3. If Promising: Re-calculate Sample Size and UPDATE Protocol.
    """

    @classmethod
    def design_binomial(
        cls,
        p_control: float,
        p_treatment: float,
        alpha: float = 0.05,
        power: float = 0.8,
        looks: int = 2,
        spending_function: str = "obrien_fleming",
        designer_params: Optional[Dict[str, Any]] = None,
    ) -> PromisingZoneABProtocol:
        """
        High-level API for Designing a Binomial A/B test with Promising Zone support.

        Args:
            p_control: Baseline proportion (e.g. 0.20)
            p_treatment: Target proportion (e.g. 0.22)
            alpha: Significance level (one-sided)
            power: Target power
            looks: Number of looks
            spending_function: "obrien_fleming" or "pocock"
        """
        delta = p_treatment - p_control
        if designer_params is None:
            designer_params = {"model": "canonical_joint"}
        elif "model" not in designer_params:
            designer_params["model"] = "canonical_joint"

        designer = ProtocolDesigner.from_dict(designer_params)
        protocol = designer.plan_binomial_ab(
            alpha=alpha,
            power=power,
            delta=delta,
            k=looks,
            p_control=p_control,
            spending_fn=None,  # Handled by designer factory via name
        )
        # Force the name and ensure it uses the specific wrapper class
        protocol.name = "Promising Zone Binomial AB"
        return PromisingZoneABProtocol.model_validate(protocol.model_dump())

    @classmethod
    def design(
        cls,
        task: GST.TaskSpec,
        looks: int,
        spending_function: str = "obrien_fleming",
        spending_params: Optional[Dict[str, Any]] = None,
        designer_params: Optional[Dict[str, Any]] = None,
    ) -> PromisingZoneABProtocol:
        """
        Designs the protocol using a generic TaskSpec.
        """
        designer = ProtocolDesigner.from_dict(designer_params or {})

        method_spec = designer.method_from_task_spec(
            task=task,
            params={
                "looks": looks,
                "spending_function": spending_function,
                "spending_params": spending_params,
            },
        )

        return PromisingZoneABProtocol(
            task=task,
            method=method_spec,
        )

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_protocol(self, protocol: PromisingZoneABProtocol) -> None:
        """Sets the initial protocol."""
        protocol = PromisingZoneABProtocol.model_validate(protocol)
        with Session(self.ledger) as sess:
            sess.Commit(protocol)

    def update(self, batch: List[BaseModel]) -> None:
        """
        Run update cycle with Adaptation check.

        Args:
            batch: Data batch.
        """
        # 1. Ingest
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.Commit(item, trace=[])

        # 2. Analysis & Adaptation
        with Session(self.ledger) as sess:
            # Read LATEST protocol (may have been updated)
            # Note: ProtocolProjector picks up the latest record of type PromisingZoneABProtocol
            # If we subclassed, we need to be careful. The projector filters by type.
            # If we simply commit a new GST.Protocol, it might not leverage the custom class?
            # Ideally, we commit PromisingZoneABProtocol instances.
            protocol = sess.Read(ProtocolProjector(PromisingZoneABProtocol)).data
            metrics = sess.Read(Scoreboard(identity="metrics")).data

            # 3. Standard GSD Engine
            engine = BinomialGSTEngine(protocol)

            # Helper logic to capture result before committing?
            # CallAndCommit hides the return value inside the Session internals until committed?
            # Accessing 'engine.run()' result inside Session requires careful handling if we want to branch logic *here*.
            # Actually, Session.CallAndCommit returns the produced records or we can just call the function directly?
            # Session.CallAndCommit is for tracing.

            # For Adaptation, we need the LookResult *now*.
            look_result = engine.run(metrics)

            # Commit the LookResult
            sess.Commit(look_result)

            # 4. Adaptation Logic (only if continuing and not final)
            status = str(look_result.status)
            if status == DecisionStatus.CONTINUE_:

                # Derive assumed effect from Protocol
                assumed_effect = self._get_assumed_effect(protocol)

                # Check Promising Zone
                adaptation_log = ConditionalPowerAdaptationEngine.assess_promising_zone(
                    look_result, protocol, assumed_effect=assumed_effect
                )

                sess.Commit(adaptation_log)

                if (
                    adaptation_log.promising_zone_status
                    == PromisingZoneStatus.PROMISING
                ):
                    # REPLAN
                    new_protocol = ConditionalPowerAdaptationEngine.replan_sample_size(
                        protocol, adaptation_log
                    )

                    # Check if actually changed
                    old_n = getattr(
                        protocol.method.stopping_policy.timer, "max_sample_size"
                    )
                    new_n = getattr(
                        new_protocol.method.stopping_policy.timer, "max_sample_size"
                    )

                    if new_n != old_n:
                        # Commit NEW Protocol
                        # Cast to PromisingZoneABProtocol to keep projector happy
                        new_proto_wrapper = PromisingZoneABProtocol.model_validate(
                            new_protocol.model_dump()
                        )
                        sess.Commit(new_proto_wrapper)

    def _get_assumed_effect(self, protocol: PromisingZoneABProtocol) -> float:
        """Derives standardized drift (delta) from protocol assumptions."""
        task = protocol.task
        if not isinstance(task.hypotheses.target_effect, GST.BinaryEffectSize):
            return 0.5  # Fallback

        props = task.hypotheses.target_effect.proportions
        # Assuming treatment and control keys exist or taking first two
        p_vals = list(props.values())
        if len(p_vals) < 2:
            return 0.5
        p_c = props.get("control", p_vals[0])
        p_t = props.get("treatment", p_vals[1])

        delta = abs(p_t - p_c)
        sigma2 = p_c * (1.0 - p_c)
        n_max = protocol.method.stopping_policy.timer.max_sample_size

        # drift = delta * sqrt(n_max / (4 * sigma2))
        if sigma2 <= 0:
            return 0.5
        return float(delta * np.sqrt(n_max / (4.0 * sigma2)))

    def report_progress(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            report = sess.Read(ProgressProjector()).data.model_dump(mode="json")
            protocol = sess.Read(ProtocolProjector(PromisingZoneABProtocol)).data
            # Enrich with current trial constraints
            report["max_sample_size"] = (
                protocol.method.stopping_policy.timer.max_sample_size
            )
            return report

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.Read(FinalProjector()).data.model_dump(mode="json")

    def plot_result(self) -> Any:
        with Session(self.ledger) as sess:
            protocol = sess.Read(ProtocolProjector(PromisingZoneABProtocol)).data
            trajectory = sess.Read(InterimAnalyses(identity="interim_analyses")).data

            # Read Adaptation Logs
            # We need a Projector for Logs? Or just raw filter?
            # Since AdaptationLog is a Log, we can filter ledger directly or generic Projector?
            # Let's do raw filter for now as we don't have AdaptationProjector yet.
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
