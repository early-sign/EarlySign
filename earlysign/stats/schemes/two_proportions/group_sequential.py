"""
Two-proportions (A vs B) glue for Group Sequential (Interim Analysis).

This module provides a convenience decision operator that:
- reads the latest Wald Z statistic,
- reads the latest boundary (which may be on "z" or "bm" scale),
- (optionally) uses information time if scale conversion is needed,
- emits an efficacy/futility/continue decision.

It does **not** compute boundaries nor statistics by itself; those remain modular:
- Use `WaldZStatistic` (operators.py) to write a WaldZStatisticRecord.
- Use `InformationTime*` and `GroupSequentialDesign` + `BoundaryFromDesign`
  to produce a GroupSequentialBoundaryRecord.
"""

import math
from typing import Dict, Optional

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.group_sequential.essentials.conversions import convert_scale
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialBoundaryRecord,
    GroupSequentialDecisionSignalRecord,
    InformationTimeRecord,
)
from earlysign.stats.schemes.two_proportions.records import (
    WaldZStatisticRecord,
)


def _decide(value: float, upper: float, lower: float) -> tuple[str, str]:
    """Compare value to (upper, lower) and return (signal, reason)."""
    if value >= upper:
        return "stop_efficacy", "efficacy"
    if value <= lower:
        return "stop_futility", "futility"
    return "continue", "none"


class GSDecisionFromWaldZ(LedgerOperator):
    """
    Compare Wald Z to GS boundary with scale handling, output a decision row.

    __init__ parameters
    -------------------
    wald     : WaldZStatisticRecord               # attached input
    boundary : GroupSequentialBoundaryRecord      # attached input
    out_id   : str
    value_scale : {"z","bm"} = "z"
        Scale of the statistic to be compared. WaldZ is on the Z-scale by definition,
        so the default is "z".
    info : InformationTimeRecord = None
        Used for scale conversion if needed (e.g., boundary on "bm" scale).
    """

    def __init__(
        self,
        scoped: Ledger,
        *,
        wald: WaldZStatisticRecord,
        boundary: GroupSequentialBoundaryRecord,
        out_id: str,
        value_scale: str = "z",
        info: Optional[InformationTimeRecord] = None,
    ):
        super().__init__(
            scoped,
            wald=wald,
            boundary=boundary,
            out_id=out_id,
            value_scale=value_scale,
            info=info,
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"decision": GroupSequentialDecisionSignalRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        out = self.outputs["decision"]
        wald: WaldZStatisticRecord = self.wald  # type: ignore[attr-defined]
        boundary: GroupSequentialBoundaryRecord = self.boundary  # type: ignore[attr-defined]
        value_scale = str(getattr(self, "value_scale")).lower()

        # 1) read latest Wald Z
        z_raw = float(wald.latest().execute().iloc[0]["wald_z"])

        # 2) read latest boundary
        bdf = boundary.latest().execute().iloc[0]
        if math.isnan(bdf["info_time"]):
            raise ValueError("Info time cannot be nan.")
        upper = float(bdf["upper"])
        lower = float(bdf["lower"])
        bscale = str(bdf["scale"])
        t = float(bdf["info_time"])

        # If boundary lacks info_time, try provided info record
        if (t is None or not (0.0 <= t <= 1.0)) and getattr(
            self, "info", None
        ) is not None:
            info: InformationTimeRecord = self.info  # type: ignore[attr-defined]
            idf = (
                info.latest()
                .select(info_time=info.t.payload["info_time"].cast("float64"))
                .execute()
            )
            if len(idf) > 0:
                t = float(idf.iloc[0]["info_time"])

        if (value_scale != bscale) and (t is None):
            raise ValueError(
                "Information time is required to convert from '{}' to '{}' scale.".format(
                    value_scale, bscale
                )
            )

        # 3) scale conversion as needed and decide
        v = convert_scale(
            z_raw, from_scale=value_scale, to_scale=bscale, info_time=(t or 0.0)
        )
        signal, reason = _decide(v, upper=upper, lower=lower)

        payload = {
            "signal": signal,
            "reason": reason,
            "value": float(v),  # on boundary scale
            "value_scale": bscale,
            "upper": float(upper),
            "lower": float(lower),
            "scale": bscale,
        }
        if t is not None:
            payload["info_time"] = float(t)
        out.insert(payload)
