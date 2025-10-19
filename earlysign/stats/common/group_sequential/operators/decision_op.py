"""
Decision operators for group sequential testing.

Compares statistics against boundaries and emits stop/continue signals.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.group_sequential.essentials.conversions import convert_scale
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialBoundaryRecord,
    GroupSequentialDecisionSignalRecord,
    InformationTimeRecord,
)
from earlysign.stats.schemes.two_proportions.records import WaldZStatisticRecord


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


class GSDecision(LedgerOperator):
    """
    Scale-aware decision against a GroupSequentialBoundaryRecord.

    Compares a statistic value to efficacy and futility boundaries,
    handling scale conversions automatically.

    Parameters
    ----------
    scoped : Ledger
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
    >>> from earlysign.stats.common.group_sequential.records import (
    ...     GroupSequentialBoundaryRecord
    ... )
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>> boundary = GroupSequentialBoundaryRecord(id="bound1").attach(ledger)
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
        scoped: Ledger,
        *,
        boundary: GroupSequentialBoundaryRecord,
        out_id: str,
        value: float,
        value_scale: str = "z",
        info: Optional[InformationTimeRecord] = None,
    ):
        super().__init__(
            scoped,
            boundary=boundary,
            out_id=out_id,
            value=value,
            value_scale=value_scale,
            info=info,
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"decision": GroupSequentialDecisionSignalRecord(id=self.out_id)}

    def run(self) -> None:
        out = self.outputs.decision
        boundary = self.boundary
        value_in = float(getattr(self, "value"))
        value_scale = str(getattr(self, "value_scale")).lower()

        # Read latest boundary
        bdf = (
            boundary.latest()
            .select(
                upper=boundary.t.payload["upper"].cast("float64"),
                lower=boundary.t.payload["lower"].cast("float64"),
                scale=boundary.t.payload["scale"],
                info_time=boundary.t.payload["info_time"].cast("float64"),
            )
            .execute()
        )
        if len(bdf) == 0:
            return

        upper = float(bdf.iloc[0]["upper"])
        lower = float(bdf.iloc[0]["lower"])
        bscale = str(bdf.iloc[0]["scale"])
        t = float(bdf.iloc[0]["info_time"]) if "info_time" in bdf.columns else None

        # Fall back to info record if needed
        if t is None and getattr(self, "info", None) is not None:
            info = self.info
            if info is not None:
                idf = (
                    info.latest()
                    .select(info_time=info.t.payload["info_time"].cast("float64"))
                    .execute()
                )
                if len(idf) > 0:
                    t = float(idf.iloc[0]["info_time"])

        # Convert value to boundary scale using essentials
        if value_scale != bscale:
            if t is None:
                raise ValueError(
                    "Information time required for scale conversion between z and bm"
                )
            v = convert_scale(
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
        if t is not None:
            payload["info_time"] = float(t)

        out.insert(payload)


# Alias for backward compatibility
Decision = GSDecision


class GSDecisionFromWaldZ(LedgerOperator):
    """
    Decision operator that reads Wald Z-statistic from a record.

    Convenience operator for the common case where the test statistic
    is a Wald Z already in the ledger.

    Parameters
    ----------
    scoped : Ledger
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
    >>> from earlysign.stats.common.group_sequential.records import GroupSequentialBoundaryRecord
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>> wald_rec = WaldZStatisticRecord(id="wald1").attach(ledger)
    >>> wald_rec.insert({"z": 3.0})
    >>> boundary = GroupSequentialBoundaryRecord(id="bound1").attach(ledger)
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
        scoped: Ledger,
        *,
        wald: WaldZStatisticRecord,
        boundary: GroupSequentialBoundaryRecord,
        out_id: str,
    ):
        super().__init__(scoped, wald=wald, boundary=boundary, out_id=out_id)

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"decision": GroupSequentialDecisionSignalRecord(id=self.out_id)}

    def run(self) -> None:
        wald = self.wald
        boundary = self.boundary

        # Read latest Wald Z
        wdf = wald.latest().select("wald_z").execute()
        if len(wdf) == 0:
            return
        z_val = float(wdf.iloc[0]["wald_z"])

        # Delegate to GSDecision
        gs_decision = GSDecision(
            self.scoped,
            boundary=boundary,
            out_id=self.out_id,
            value=z_val,
            value_scale="z",
        )
        gs_decision.run()
