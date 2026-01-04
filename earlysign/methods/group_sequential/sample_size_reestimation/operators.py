"""Ledger operators for adaptive group-sequential sample-size re-estimation."""

from dataclasses import dataclass
from typing import Dict

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.methods.adaptive_group_sequential.conditional_update import (
    conditional_power,
    promising_zone_decision,
    update_remaining_boundaries,
)
from earlysign.methods.adaptive_group_sequential.types import BindingMode
from earlysign.methods.group_sequential.sample_size_reestimation.records import (
    ConditionalPowerRecord,
    DesignUpdateDecisionRecord,
    UpdatedBoundariesRecord,
)


class ConditionalPowerCalculation(LedgerOp):
    """
    Calculate conditional power at an interim analysis.

    Reads observed Z-statistic from ledger, computes conditional power
    given assumed future effect size, and writes result back to ledger.
    """

    out_id: str

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        observed_z_id: str,
        current_info_time: float,
        final_info_time: float,
        final_efficacy_bound: float,
        assumed_effect: float,
        variance: float = 1.0,
    ):
        super().__init__(
            ledger,
            out_id=out_id,
            observed_z_id=observed_z_id,
            current_info_time=current_info_time,
            final_info_time=final_info_time,
            final_efficacy_bound=final_efficacy_bound,
            assumed_effect=assumed_effect,
            variance=variance,
        )

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        cp: ConditionalPowerRecord

    outputs: Outputs

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"cp": ConditionalPowerRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.cp

        obs_id = str(getattr(self, "observed_z_id"))
        t = self.ledger.t
        result = (
            t.filter(t.labels["id"] == obs_id)
            .order_by(t.ts.desc())
            .limit(1)
            .select(z=t.payload["z"].cast("float64"))
            .execute()
        )
        if len(result) == 0:
            raise ValueError(f"No observed Z record found with id={obs_id}")
        observed_z = float(result.iloc[0]["z"])

        cp = conditional_power(
            observed_z=observed_z,
            current_info_time=getattr(self, "current_info_time"),
            final_info_time=getattr(self, "final_info_time"),
            final_efficacy_bound=getattr(self, "final_efficacy_bound"),
            assumed_effect=getattr(self, "assumed_effect"),
            variance=getattr(self, "variance"),
        )

        out.insert(
            {
                "conditional_power": cp,
                "observed_z": observed_z,
                "current_info_time": getattr(self, "current_info_time"),
                "final_info_time": getattr(self, "final_info_time"),
                "assumed_effect": getattr(self, "assumed_effect"),
            }
        )


class PromisingZoneDecision(LedgerOp):
    """Make continuation decision based on conditional power in promising zone."""

    out_id: str

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        observed_z_id: str,
        current_info_time: float,
        current_efficacy_bound: float,
        current_futility_bound: float,
        final_info_time: float,
        final_efficacy_bound: float,
        assumed_effect: float,
        cp_threshold: float = 0.80,
    ):
        super().__init__(
            ledger,
            out_id=out_id,
            observed_z_id=observed_z_id,
            current_info_time=current_info_time,
            current_efficacy_bound=current_efficacy_bound,
            current_futility_bound=current_futility_bound,
            final_info_time=final_info_time,
            final_efficacy_bound=final_efficacy_bound,
            assumed_effect=assumed_effect,
            cp_threshold=cp_threshold,
        )

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        decision: DesignUpdateDecisionRecord

    outputs: Outputs

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {
            "decision": DesignUpdateDecisionRecord(name=self.out_id, ledger=self.ledger)
        }

    def run(self) -> None:
        out = self.outputs.decision

        obs_id = str(getattr(self, "observed_z_id"))
        t = self.ledger.t
        result = (
            t.filter(t.labels["id"] == obs_id)
            .order_by(t.ts.desc())
            .limit(1)
            .select(z=t.payload["z"].cast("float64"))
            .execute()
        )
        if len(result) == 0:
            raise ValueError(f"No observed Z record found with id={obs_id}")
        observed_z = float(result.iloc[0]["z"])

        decision_result = promising_zone_decision(
            observed_z=observed_z,
            current_info_time=getattr(self, "current_info_time"),
            current_efficacy_bound=getattr(self, "current_efficacy_bound"),
            current_futility_bound=getattr(self, "current_futility_bound"),
            final_info_time=getattr(self, "final_info_time"),
            final_efficacy_bound=getattr(self, "final_efficacy_bound"),
            assumed_effect=getattr(self, "assumed_effect"),
            cp_threshold=getattr(self, "cp_threshold"),
        )

        out.insert(
            {
                "decision": decision_result["decision"],
                "conditional_power": decision_result["conditional_power"],
                "in_promising_zone": decision_result["in_promising_zone"],
                "recommendation": decision_result["recommendation"],
                "observed_z": observed_z,
                "current_info_time": getattr(self, "current_info_time"),
            }
        )


class UpdateRemainingBoundaries(LedgerOp):
    """
    Recompute boundaries for remaining analyses after interim update.

    When trial design is modified at interim analysis, remaining boundaries
    must be recalculated to maintain overall Type I error control using
    alpha spending approach.
    """

    out_id: str

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        current_info_time: float,
        remaining_info_times: list[float],
        alpha: float,
        alpha_gamma: float,
        cumulative_alpha_spent: float,
        binding_mode: str = "non_binding",
    ):
        super().__init__(
            ledger,
            out_id=out_id,
            current_info_time=current_info_time,
            remaining_info_times=remaining_info_times,
            alpha=alpha,
            alpha_gamma=alpha_gamma,
            cumulative_alpha_spent=cumulative_alpha_spent,
            binding_mode=binding_mode,
        )

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        updated_boundaries: UpdatedBoundariesRecord

    outputs: Outputs

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {
            "updated_boundaries": UpdatedBoundariesRecord(
                name=self.out_id, ledger=self.ledger
            )
        }

    def run(self) -> None:
        import numpy as np

        out = self.outputs.updated_boundaries

        binding_mode_value: BindingMode
        if getattr(self, "binding_mode") == "binding":
            binding_mode_value = "binding"
        else:
            binding_mode_value = "non_binding"

        result = update_remaining_boundaries(
            current_info_time=getattr(self, "current_info_time"),
            remaining_info_times=np.array(getattr(self, "remaining_info_times")),
            alpha=getattr(self, "alpha"),
            alpha_gamma=getattr(self, "alpha_gamma"),
            cumulative_alpha_spent=getattr(self, "cumulative_alpha_spent"),
            binding_mode=binding_mode_value,
        )

        out.insert(
            {
                "remaining_info_times": result["remaining_info_times"].tolist(),
                "upper_bounds": result["upper_bounds"].tolist(),
                "lower_bounds": result["lower_bounds"].tolist(),
                "alpha_remaining": result["alpha_remaining"],
                "alpha_increments": result["alpha_increments"].tolist(),
            }
        )
