"""
Group Sequential (Interim Analysis) glue for one-mean (Gaussian).

Use together with:
- OneMeanSummaryRecord (+ your own loader to insert n/mean snapshots)
- ZMeanKnownVar (to compute Z from summary)
- InformationTime* (to write info_time)
- GroupSequentialDesign + BoundaryFromDesign (to obtain boundaries)
- Decision (scale-aware; compares Z/BM value to boundary)
"""

import math
from typing import Dict, Optional

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.group_sequential.decision import _decide, convert_scale
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialBoundaryRecord,
    GroupSequentialDecisionSignalRecord,
    InformationTimeRecord,
)
from earlysign.stats.schemes.one_mean.records import ZMeanKnownVarRecord


class GSDecisionFromZMean(LedgerOperator):
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

    def __init__(
        self,
        scoped: Ledger,
        *,
        zstat: ZMeanKnownVarRecord,
        boundary: GroupSequentialBoundaryRecord,
        out_id: str,
        value_scale: str = "z",
        info: Optional[InformationTimeRecord] = None,
    ):
        super().__init__(
            scoped,
            zstat=zstat,
            boundary=boundary,
            out_id=out_id,
            value_scale=value_scale,
            info=info,
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"decision": GroupSequentialDecisionSignalRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        out = self.outputs["decision"]
        zrec: ZMeanKnownVarRecord = self.zstat  # type: ignore[attr-defined]
        boundary: GroupSequentialBoundaryRecord = self.boundary  # type: ignore[attr-defined]
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
            info: InformationTimeRecord = self.info  # type: ignore[attr-defined]
            idf = (
                info.latest()
                .select(info_time=info.t.payload["info_time"].cast("float64"))
                .execute()
            )
            if len(idf) > 0:
                t = float(idf.iloc[0]["info_time"])

        v = convert_scale(
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
