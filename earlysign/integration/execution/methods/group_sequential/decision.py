"""Decision records and operators for group sequential testing."""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord, QueryMixin
from earlysign.integration.execution.methods.group_sequential.boundary import (
    GroupSequentialBoundaryRecord,
)
from earlysign.integration.execution.methods.group_sequential.information_time import (
    InformationTimeRecord,
)
from earlysign.integration.execution.schemes.two_proportions.wald_z import (
    WaldZStatisticRecord,
)
from earlysign.methods.group_sequential.boundary import (
    convert_statistic_scale,
)


class GroupSequentialDecisionSignalRecord(LedgerRecord, QueryMixin):
    """Stop/continue signal produced by comparing a statistic to boundaries."""

    schema = {
        "signal": str,
        "reason": str,
        "value": (float | None, None),
        "value_scale": (str | None, None),
        "upper": (float | None, None),
        "lower": (float | None, None),
        "scale": (str | None, None),
        "statistic_type": (str | None, None),
        "statistic_scale": (str | None, None),
        "info_time": (float | None, None),
    }


def _decide(
    value: float, upper: float, lower: float
) -> Tuple[str, Literal["efficacy", "futility", "none"]]:
    """Compare value to boundaries and return (signal, reason)."""
    if value >= upper:
        return "stop_efficacy", "efficacy"
    if value <= lower:
        return "stop_futility", "futility"
    return "continue", "none"


class GSDecision(LedgerOp):
    """Scale-aware decision against a GroupSequentialBoundaryRecord."""

    out_id: str
    boundary: GroupSequentialBoundaryRecord
    info: Optional[InformationTimeRecord]

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        decision: GroupSequentialDecisionSignalRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        boundary: GroupSequentialBoundaryRecord,
        out_id: str,
        value: float,
        value_scale: str = "z",
        info: Optional[InformationTimeRecord] = None,
    ):
        super().__init__(
            ledger,
            boundary=boundary,
            out_id=out_id,
            value=value,
            value_scale=value_scale,
            info=info,
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {
            "decision": GroupSequentialDecisionSignalRecord(
                name=self.out_id, ledger=self.ledger
            )
        }

    def run(self) -> None:
        out = self.outputs.decision
        boundary = self.boundary
        value_in = float(getattr(self, "value"))
        value_scale = str(getattr(self, "value_scale")).lower()

        # Read latest boundary
        try:
            boundary_payload = boundary.latest_payload()
        except LookupError:
            return

        upper_val = boundary_payload.get("upper")
        lower_val = boundary_payload.get("lower")
        upper = float(upper_val) if upper_val is not None else float("inf")
        lower = float(lower_val) if lower_val is not None else float("-inf")
        bscale = str(boundary_payload.get("scale", "z"))
        statistic_type = boundary_payload.get("statistic_type")
        statistic_scale = boundary_payload.get("statistic_scale")
        t_raw = boundary_payload.get("info_time")
        t = float(t_raw) if t_raw is not None else None

        # Fall back to info record if needed
        if t is None and getattr(self, "info", None) is not None:
            info = self.info
            if info is not None:
                try:
                    info_payload = info.latest_payload()
                except LookupError:
                    info_payload = None
                if info_payload is not None:
                    t = float(info_payload.get("info_time", 0.0))

        # Convert value to boundary scale using essentials
        if value_scale != bscale:
            if t is None:
                raise ValueError(
                    "Information time required for scale conversion between z and bm"
                )
            v = convert_statistic_scale(
                value_in, from_scale=value_scale, to_scale=bscale, info_time=t
            )
        else:
            v = value_in

        # Make decision
        signal, reason = _decide(v, upper=upper, lower=lower)

        payload = {
            "signal": signal,
            "reason": reason,
            "value": float(v),
            "value_scale": bscale,
            "upper": float(upper),
            "lower": float(lower),
            "scale": bscale,
        }
        if statistic_type is not None:
            payload["statistic_type"] = statistic_type
        if statistic_scale is not None:
            payload["statistic_scale"] = statistic_scale
        if t is not None:
            payload["info_time"] = float(t)
        out.insert(payload)


# Alias for backward compatibility
Decision = GSDecision


class GSDecisionFromWaldZ(LedgerOp):
    """Decision operator that reads Wald Z-statistic from a record."""

    out_id: str
    wald: WaldZStatisticRecord
    boundary: GroupSequentialBoundaryRecord

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        """Outputs for this operator."""

        decision: GroupSequentialDecisionSignalRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        wald: WaldZStatisticRecord,
        boundary: GroupSequentialBoundaryRecord,
        out_id: str,
        value_scale: str = "z",
        info: Optional[InformationTimeRecord] = None,
    ):
        super().__init__(
            ledger,
            wald=wald,
            boundary=boundary,
            out_id=out_id,
            value_scale=value_scale,
            info=info,
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {
            "decision": GroupSequentialDecisionSignalRecord(
                name=self.out_id, ledger=self.ledger
            )
        }

    def run(self) -> None:
        wald = self.wald
        boundary = self.boundary
        info = getattr(self, "info", None)
        value_scale = str(getattr(self, "value_scale", "z"))

        # Read latest Wald Z
        try:
            wald_payload = wald.latest_payload()
        except LookupError:
            return
        z_val = float(wald_payload["wald_z"])

        # Delegate to GSDecision
        gs_decision = GSDecision(
            self.ledger,
            boundary=boundary,
            out_id=self.out_id,
            value=z_val,
            value_scale=value_scale,
            info=info,
        )
        gs_decision.run()
