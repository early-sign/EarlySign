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
                    {"control": pc, "treatment": pt} if arms == 2 else {"treatment": pt}
                )
                eff_size = GST.BinaryEffectSize(proportions=eff_props)

            case (float() as pc, None, float() as d, _):
                # Control + Delta
                calc_delta = d
                pt = pc + d
                response_type = GST.ResponseType.BINARY
                eff_props = (
                    {"control": pc, "treatment": pt} if arms == 2 else {"treatment": pt}
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
                    {"control": mc, "treatment": mt} if arms == 2 else {"treatment": mt}
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

        # 3. Design via common logic
        method_spec, n_max = designer.design_gs_classic(
            alpha=alpha,
            power=power,
            delta=calc_delta,
            looks=k,
            type=type,
            p_control=p_control,
            sigma=sigma,
            wang_tsiatis_delta=wang_tsiatis_delta or 0.25,
            tails=tails,
            arms=arms,
            rng_seed=seed,
        )

        # 4. Assemble Task Spec
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
                h_alt_description=f"Difference {calc_delta}",
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
