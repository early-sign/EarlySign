"""
Group Sequential (Interim Analysis) glue for one-mean (Gaussian).

Use together with:
- OneMeanSummaryRecord (+ your own loader to insert n/mean snapshots)
- ZMeanKnownVar (to compute Z from summary)
- InformationTime* (to write info_time)
- GroupSequentialDesignRecord + BoundaryFromDesign (to obtain boundaries)
- Decision (scale-aware; compares Z/BM value to boundary)
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

from earlysign.applications.execution.methods.group_sequential.records.boundary import (
    GroupSequentialBoundaryRecord,
)
from earlysign.applications.execution.methods.group_sequential.records.decision import (
    GroupSequentialDecisionSignalRecord,
)
from earlysign.applications.execution.methods.group_sequential.records.info import (
    InformationTimeRecord,
)
from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.essentials.methods.group_sequential.boundary import (
    convert_statistic_scale,
)
from earlysign.stats.schemes.one_mean.records import ZMeanKnownVarRecord


def _decide(value: float, upper: float, lower: float) -> tuple[str, str]:
    """Compare value to (upper, lower) and return (signal, reason)."""
    if value >= upper:
        return "stop_efficacy", "efficacy"
    if value <= lower:
        return "stop_futility", "futility"
    return "continue", "none"


class GSDecisionFromZMean(LedgerOp):
    """
    Compare Z (one-mean known var) to GS boundary with scale handling.

    __init__ parameters
    -------------------
    zstat    : ZMeanKnownVarRecord             # attached input
    boundary : GroupSequentialBoundaryRecord   # attached input
    out_id   : str
    value_scale : {"z","bm"} = "z"             # Z is on "z" by default
    info : InformationTimeRecord = None
    """

    out_id: str
    zstat: ZMeanKnownVarRecord
    boundary: GroupSequentialBoundaryRecord
    info: Optional[InformationTimeRecord]

    def __init__(
        self,
        ledger: Ledger,
        *,
        zstat: ZMeanKnownVarRecord,
        boundary: GroupSequentialBoundaryRecord,
        out_id: str,
        value_scale: str = "z",
        info: Optional[InformationTimeRecord] = None,
    ):
        super().__init__(
            ledger,
            zstat=zstat,
            boundary=boundary,
            out_id=out_id,
            value_scale=value_scale,
            info=info,
        )

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        decision: GroupSequentialDecisionSignalRecord

    outputs: Outputs

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {
            "decision": GroupSequentialDecisionSignalRecord(
                name=self.out_id, ledger=self.ledger
            )
        }

    def run(self) -> None:
        out = self.outputs.decision
        zrec: ZMeanKnownVarRecord = self.zstat
        boundary: GroupSequentialBoundaryRecord = self.boundary
        value_scale = str(getattr(self, "value_scale")).lower()

        zdf = zrec.latest().select(z=zrec.t.payload["z"].cast("float64")).execute()
        if len(zdf) == 0:
            return
        z_raw = float(zdf.iloc[0]["z"])

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
        t = (
            float(bdf.iloc[0]["info_time"])
            if not math.isnan(bdf.iloc[0]["info_time"])
            else None
        )

        if (
            (value_scale != bscale)
            and (t is None)
            and getattr(self, "info", None) is None
        ):
            raise ValueError("Information time required to convert scales but missing.")
        if (t is None) and getattr(self, "info", None) is not None:
            info = self.info
            if info is not None:
                idf = (
                    info.latest()
                    .select(info_time=info.t.payload["info_time"].cast("float64"))
                    .execute()
                )
                if len(idf) > 0:
                    t = float(idf.iloc[0]["info_time"])

        v = convert_statistic_scale(
            z_raw, from_scale=value_scale, to_scale=bscale, info_time=(t or 0.0)
        )
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
