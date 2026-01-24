from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.Binomial import ArmData
from earlysign.schema.ES3.YEAST import MethodSpec, Protocol, TaskSpec
from earlysign.schema.ES3.YEAST.Log import Boundary as BoundarySchema, DecisionStatus
from earlysign.v1.methods.YEAST.boundary import Boundary


class BinomialYeastTaskSpec(BaseModel):
    """
    User-facing Task Specification for Binomial YEAST.
    This mimics GST.TaskSpec structure for consistency in the template.
    """

    kind: Literal["yeast"] = "yeast"
    arms: List[str]
    response_type: str = "binary"
    hypotheses: Dict[str, Any]  # Simplified for template input compatibility


class BinomialYeastTemplate:
    """
    Template for YEAST (Your Evidence Accumulation Sequential Test) on Binomial data.
    """

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.protocol: Optional[Protocol] = None
        self._boundary: Optional[BoundarySchema] = None
        self._cumulative_counts: Dict[str, Dict[str, int]] = {}

    def set_protocol(self, protocol: Protocol) -> None:
        self.protocol = protocol
        # Calculate/Project boundary
        val = Boundary.calculate(protocol)
        self._boundary = BoundarySchema(value=val)

    @classmethod
    def design(
        cls,
        task: BinomialYeastTaskSpec,
        significance_level: float,
        expected_num_observations: int,
        increment_std: Optional[float] = None,
    ) -> Protocol:
        """
        Design a YEAST protocol from parameters.
        """
        # Default increment_std logic mirroring schema description
        if increment_std is None:
            # sqrt(0.5) approx 0.70710678
            increment_std = 0.70710678

        method = MethodSpec(
            kind="yeast",
            significance_level=significance_level,
            expected_num_observations=expected_num_observations,
            increment_std=increment_std,
        )

        return Protocol(
            name="Binomial YEAST Protocol",
            task=TaskSpec(kind="yeast", arms=task.arms),
            method=method,
        )

    def update(self, batch: List[ArmData]) -> None:
        """
        Update the experiment with a batch of data.
        Logic: Calculate trajectory and check boundary.
        """
        if not self.protocol or not self._boundary:
            raise RuntimeError("Protocol not set")

        for data in batch:
            arm = data.arm
            if arm not in self._cumulative_counts:
                self._cumulative_counts[arm] = {"n": 0, "success": 0}
            self._cumulative_counts[arm]["n"] += data.n
            self._cumulative_counts[arm]["success"] += data.success

    def report_progress(self) -> Dict[str, Any]:
        """
        Report current status.
        """
        if not self._cumulative_counts:
            return {"status": DecisionStatus.CONTINUE_, "trajectory": 0.0}

        # Calculate trajectory from cumulative counts
        control = self._cumulative_counts.get("control")
        treatment = self._cumulative_counts.get("treatment")

        traj = 0.0
        if control and treatment:
            traj = float(treatment["success"] - control["success"])

        boundary_val = self._boundary.value if self._boundary else float("inf")
        is_crossed = traj > boundary_val

        status = (
            DecisionStatus.STOP_EFFICACY if is_crossed else DecisionStatus.CONTINUE_
        )

        return {
            "status": status,
            "trajectory": traj,
            "efficacy_boundary": boundary_val,
            "is_efficacy_crossed": is_crossed,
        }

    def report_result(self) -> Dict[str, Any]:
        progress = self.report_progress()
        return {
            "is_rejected": progress["status"] == DecisionStatus.STOP_EFFICACY,
            "final_status": progress["status"],
        }
