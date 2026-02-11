"""Classic Group Sequential Testing Template.

This module provides a template for "Classic" Group Sequential Tests (GST) based on
standard boundary shapes like Pocock, O'Brien-Fleming, and Wang-Tsiatis power
families. Unlike the Alpha-Spending approach, these designs use fixed boundary
shape constants determined by the total number of looks and alpha/power requirements.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

import earlysign.schema.ES3.Base as ES3_BASE
import earlysign.schema.ES3.GST as GST
from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.GST.Log import LookResult
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.binomial import Scoreboard
from earlysign.v1.methods.group_sequential.execution.binomial import BinomialGSTEngine
from earlysign.v1.methods.group_sequential.plan.protocol_design import (
    ProtocolDesigner,
)
from earlysign.v1.methods.group_sequential.reporting.projectors import (
    FinalProjector,
    ProgressProjector,
)
from earlysign.v1.templates.base import AutoNameMixin, TemplateBase


class ClassicTaskSpec(GST.TaskSpec):
    """Task specification for Classic GST."""

    pass


class ClassicProtocol(GST.Protocol, AutoNameMixin):
    """Protocol for Classic GST."""

    task: ClassicTaskSpec
    method: GST.MethodSpec
    name: str = Field(default="")


class ClassicGSTTemplate(TemplateBase[ClassicProtocol]):
    """Orchestrator for Classic Group Sequential Tests (Pocock, OBF, Wang-Tsiatis).

    This template supports designs that are defined by a fixed boundary shape
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
        delta: float,  # Effect size (difference in proportions or means)
        k: int,
        p_control: Optional[float] = None,  # Required for Binomial Sample Size
        sigma: Optional[float] = None,  # Required for Continuous Sample Size
        wang_tsiatis_delta: Optional[float] = None,
        tails: int = 2,
        arms: int = 2,  # 1 (Paired/Single) or 2 (Two-sample)
        seed: int = 42,
    ) -> ClassicProtocol:
        """Designs a Classic GST Protocol.

        Args:
            type: Design type ("pocock", "obrien_fleming", "wang_tsiatis").
            alpha: Type I error rate.
            power: Target power.
            delta: Detectable effect size (absolute difference).
            k: Number of looks.
            p_control: Control arm proportion (for Binomial designs).
            sigma: Standard deviation (for Continuous designs).
            wang_tsiatis_delta: Delta parameter for Wang-Tsiatis family.
            tails: Number of tails (1 or 2).
            arms: Number of arms (1 or 2).
            seed: Random seed for simulation/calculation.

        Returns:
            A populated ClassicProtocol.
        """
        # 1. Setup Designer
        designer = ProtocolDesigner.from_dict(
            {"model": "canonical_joint", "model_params": {"rng_seed": seed}}
        )

        if arms != 2:
            raise NotImplementedError(
                f"Classic GST with {arms} arms is not yet supported in this template. "
                "Currently, only 2-arm (Two-sample) comparisons are supported."
            )

        # 2. Design via common logic
        method_spec, n_max = designer.design_gs_classic(
            alpha=alpha,
            power=power,
            delta=delta,
            looks=k,
            type=type,
            p_control=p_control,
            sigma=sigma,
            wang_tsiatis_delta=wang_tsiatis_delta or 0.25,
            tails=tails,
            arms=arms,
            rng_seed=seed,
        )

        # 3. Assemble Task Spec
        if p_control is not None:
            response_type = GST.ResponseType.BINARY
            eff_size = GST.BinaryEffectSize(
                proportions={"control": p_control, "treatment": p_control + delta}
            )
        else:
            response_type = GST.ResponseType.CONTINUOUS
            eff_size = GST.ContinuousEffectSize(
                means={"control": 0.0, "treatment": delta}, standard_deviation=sigma
            )

        if arms == 1:
            task_arms = ES3_BASE.SingleArm(arm_name="treatment")
        else:
            task_arms = ES3_BASE.TwoArmComparison(
                control_arm_name="control", treatment_arm_name="treatment"
            )

        task = ClassicTaskSpec(
            arms=task_arms,
            response_type=response_type,
            hypotheses=GST.HypothesisSpec(
                h_null_description="No Difference",
                h_alt_description=f"Difference {delta}",
                test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                target_effect=eff_size,
            ),
            efficacy=GST.EfficacyRequirement(alpha=alpha),
            futility=GST.FutilityRequirement(power=power),
        )

        return ClassicProtocol(task=task, method=method_spec)

    def update(self, batch: List[BaseModel]) -> None:
        """Updates the experiment with new data."""
        # Reuse standard logic: Commit -> (Read -> Engine -> Output)
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(ClassicProtocol))
            metrics = sess.read(Scoreboard(identity="metrics"))

            # NOTE: Currently we only support Binomial Execution in this template for simplicity
            # To support Continuous, we would need to inspect protocol.task.response_type
            if protocol.task.response_type == GST.ResponseType.BINARY:
                engine = BinomialGSTEngine(protocol.data)
                sess.call_and_commit(LookResult, engine.run, metrics=metrics)
            else:
                # Placeholder for Continuous Engine binding
                pass

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
