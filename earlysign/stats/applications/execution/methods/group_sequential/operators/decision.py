"""
Decision operators for group sequential testing.

Compares statistics against boundaries and emits stop/continue signals.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.applications.execution.methods.group_sequential.records.boundary import (
    GroupSequentialBoundaryRecord,
)
from earlysign.stats.applications.execution.methods.group_sequential.records.decision import (
    GroupSequentialDecisionSignalRecord,
)
from earlysign.stats.applications.execution.methods.group_sequential.records.info import (
    InformationTimeRecord,
)
from earlysign.stats.applications.execution.methods.group_sequential.records.statistics import (
    WaldZStatisticRecord,
)
from earlysign.stats.essentials.methods.group_sequential.boundary import (
    convert_statistic_scale,
)


def _decide(
    value: float, upper: float, lower: float
) -> Tuple[str, Literal["efficacy", "futility", "none"]]:
    """
    Compare value to boundaries and return signal and reason.

    Parameters
    ----------
    value : float
        Test statistic value (on boundary scale).
    upper : float
        Upper (efficacy) boundary.
    lower : float
        Lower (futility) boundary.

    Returns
    -------
    signal : str
        "stop_efficacy", "stop_futility", or "continue"
    reason : str
        "efficacy", "futility", or "none"

    Examples
    --------
    >>> _decide(3.0, upper=2.5, lower=-2.5)
    ('stop_efficacy', 'efficacy')
    >>> _decide(-3.0, upper=2.5, lower=-2.5)
    ('stop_futility', 'futility')
    >>> _decide(0.0, upper=2.5, lower=-2.5)
    ('continue', 'none')
    """
    if value >= upper:
        return "stop_efficacy", "efficacy"
    if value <= lower:
        return "stop_futility", "futility"
    return "continue", "none"


class GSDecision(LedgerOp):
    """
    Scale-aware decision against a GroupSequentialBoundaryRecord.

    Compares a statistic value to efficacy and futility boundaries,
    handling scale conversions automatically.

    Parameters
    ----------
    ledger : Ledger
        Scoped ledger instance.
    boundary : GroupSequentialBoundaryRecord (attached)
        Boundary record to compare against.
    out_id : str
        ID of decision signal record to create.
    value : float
        Test statistic value.
    value_scale : {"z", "bm"}, default="z"
        Scale of the input value.
    info : InformationTimeRecord, optional (attached)
        Information time record (used if boundary doesn't contain info_time).

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.stats.applications.execution.methods.group_sequential.records.boundary import (
    ...     GroupSequentialBoundaryRecord
    ... )
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>> boundary = GroupSequentialBoundaryRecord(name="bound1").attach(ledger)
    >>> boundary.insert({"upper": 2.5, "lower": -2.5, "scale": "z", "info_time": 0.5,
    ...                  "alpha": 0.05, "tails": 2})
    >>> op = GSDecision(ledger, boundary=boundary, out_id="decision1",
    ...                 value=3.0, value_scale="z")
    >>> op.run()
    """

    out_id: str
    boundary: GroupSequentialBoundaryRecord
    info: Optional[InformationTimeRecord]

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        """Outputs for this operator."""

        decision: GroupSequentialDecisionSignalRecord

    # Type annotation for outputs - enables type inference!
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
    """
    Decision operator that reads Wald Z-statistic from a record.

    Convenience operator for the common case where the test statistic
    is a Wald Z already in the ledger.

    Parameters
    ----------
    ledger : Ledger
        Scoped ledger instance.
    wald : LedgerRecord (attached)
        Record containing Wald Z-statistic in payload["z"].
    boundary : GroupSequentialBoundaryRecord (attached)
        Boundary record to compare against.
    out_id : str
        ID of decision signal record to create.

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.stats.applications.execution.methods.group_sequential.records.boundary import GroupSequentialBoundaryRecord
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>> wald_rec = WaldZStatisticRecord(name="wald1").attach(ledger)
    >>> wald_rec.insert({"wald_z": 3.0})
    >>> boundary = GroupSequentialBoundaryRecord(name="bound1").attach(ledger)
    >>> boundary.insert({"upper": 2.5, "lower": -2.5, "scale": "z",
    ...                  "info_time": 0.5, "alpha": 0.05, "tails": 2})
    >>> op = GSDecisionFromWaldZ(ledger, wald=wald_rec, boundary=boundary, out_id="decision1")
    >>> op.run()
    """

    out_id: str
    wald: WaldZStatisticRecord
    boundary: GroupSequentialBoundaryRecord

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        """Outputs for this operator."""

        decision: GroupSequentialDecisionSignalRecord

    # Type annotation for outputs - enables type inference!
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
