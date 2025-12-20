"""
Conditional update operator for adaptive group sequential testing.

Implements promising-zone adaptive designs where trial design is modified
based on interim analysis results and conditional power calculations.
"""

from dataclasses import dataclass
from typing import Dict

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.integration.execution.methods.adaptive_group_sequential.records.conditional import (
    ConditionalPowerRecord,
    DesignUpdateDecisionRecord,
    UpdatedBoundariesRecord,
)
from earlysign.methods.adaptive_group_sequential.conditional_update import (
    BindingMode,
    conditional_power,
    promising_zone_decision,
    update_remaining_boundaries,
)


class ConditionalPowerCalculation(LedgerOp):
    """
    Calculate conditional power at an interim analysis.

    Reads observed Z-statistic from ledger, computes conditional power
    given assumed future effect size, and writes result back to ledger.

    Parameters
    ----------
    ledger : Ledger
        Scoped ledger instance.
    out_id : str
        ID of ConditionalPowerRecord to create.
    observed_z_id : str
        ID of record containing observed Z-statistic.
    current_info_time : float
        Current information time (0 < t < 1).
    final_info_time : float
        Information time at final analysis (typically 1.0).
    final_efficacy_bound : float
        Efficacy boundary at final analysis (Z-scale).
    assumed_effect : float
        Assumed standardized effect size for remaining data.
    variance : float, default=1.0
        Variance parameter.

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>> # Assume observed_z record exists
    >>> op = ConditionalPowerCalculation(
    ...     ledger,
    ...     out_id="cp1",
    ...     observed_z_id="z_interim",
    ...     current_info_time=0.5,
    ...     final_info_time=1.0,
    ...     final_efficacy_bound=1.96,
    ...     assumed_effect=0.5
    ... )
    >>> # op.run()  # Would compute CP and write to ledger
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

        # Read observed Z-statistic from ledger
        # Query the ledger table for the record with the specified ID
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

        # Compute conditional power
        cp = conditional_power(
            observed_z=observed_z,
            current_info_time=getattr(self, "current_info_time"),
            final_info_time=getattr(self, "final_info_time"),
            final_efficacy_bound=getattr(self, "final_efficacy_bound"),
            assumed_effect=getattr(self, "assumed_effect"),
            variance=getattr(self, "variance"),
        )

        # Write result to ledger
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
    """
    Make continuation decision based on conditional power in promising zone.

    Evaluates whether observed interim results warrant continuing the trial,
    stopping for futility/efficacy, or modifying the design.

    Parameters
    ----------
    ledger : Ledger
        Scoped ledger instance.
    out_id : str
        ID of DesignUpdateDecisionRecord to create.
    observed_z_id : str
        ID of record containing observed Z-statistic.
    current_info_time : float
        Current information time.
    current_efficacy_bound : float
        Efficacy boundary at current analysis (Z-scale).
    current_futility_bound : float
        Futility boundary at current analysis (Z-scale).
    final_info_time : float
        Information time at final analysis.
    final_efficacy_bound : float
        Efficacy boundary at final analysis (Z-scale).
    assumed_effect : float
        Assumed effect size for conditional power calculation.
    cp_threshold : float, default=0.80
        Minimum conditional power threshold for continuation.

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>> op = PromisingZoneDecision(
    ...     ledger,
    ...     out_id="decision1",
    ...     observed_z_id="z_interim",
    ...     current_info_time=0.5,
    ...     current_efficacy_bound=2.5,
    ...     current_futility_bound=0.0,
    ...     final_info_time=1.0,
    ...     final_efficacy_bound=1.96,
    ...     assumed_effect=0.5,
    ...     cp_threshold=0.80
    ... )
    >>> # op.run()  # Would compute decision and write to ledger
    """

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

        # Read observed Z-statistic from ledger
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

        # Make decision
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

        # Write decision to ledger
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

    Parameters
    ----------
    ledger : Ledger
        Scoped ledger instance.
    out_id : str
        ID of updated boundaries record to create.
    current_info_time : float
        Current information time where update occurs.
    remaining_info_times : list of float
        Information times for remaining analyses (including final).
    alpha : float
        Overall significance level (one-sided).
    alpha_gamma : float
        Shape parameter for HSD spending function.
    cumulative_alpha_spent : float
        Alpha already spent up to current_info_time.
    binding_mode : str, default="non_binding"
        Whether futility boundaries are binding ("binding" or "non_binding").

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>> op = UpdateRemainingBoundaries(
    ...     ledger,
    ...     out_id="updated_boundaries",
    ...     current_info_time=0.5,
    ...     remaining_info_times=[0.75, 1.0],
    ...     alpha=0.025,
    ...     alpha_gamma=-4.0,
    ...     cumulative_alpha_spent=0.005
    ... )
    >>> # op.run()  # Would compute updated boundaries and write to ledger
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

        # Normalize binding mode flag
        binding_mode_value: BindingMode
        if getattr(self, "binding_mode") == "binding":
            binding_mode_value = "binding"
        else:
            binding_mode_value = "non_binding"

        # Compute updated boundaries
        result = update_remaining_boundaries(
            current_info_time=getattr(self, "current_info_time"),
            remaining_info_times=np.array(getattr(self, "remaining_info_times")),
            alpha=getattr(self, "alpha"),
            alpha_gamma=getattr(self, "alpha_gamma"),
            cumulative_alpha_spent=getattr(self, "cumulative_alpha_spent"),
            binding_mode=binding_mode_value,
        )

        # Write updated boundaries to ledger
        out.insert(
            {
                "remaining_info_times": result["remaining_info_times"].tolist(),
                "upper_bounds": result["upper_bounds"].tolist(),
                "lower_bounds": result["lower_bounds"].tolist(),
                "alpha_remaining": result["alpha_remaining"],
                "alpha_increments": result["alpha_increments"].tolist(),
            }
        )
