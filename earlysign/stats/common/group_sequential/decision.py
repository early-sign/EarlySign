"""
Generic decision operators and helpers for Group Sequential procedures.

Features
--------
- Scale-aware comparison:
  * Boundary scale can be "z" (standardized statistic) or "bm" (Brownian, B(t)=Z*sqrt(t))
  * The operator accepts a value with an explicit `value_scale`, and converts to the
    boundary scale using the current information time.
- Efficacy / Futility split:
  * signal = "stop_efficacy" if value >= upper
  * signal = "stop_futility" if value <= lower
  * signal = "continue" otherwise

Inputs
------
- `boundary` : GroupSequentialBoundaryRecord (attached)
- (optional) `info`: InformationTimeRecord (attached) used when boundary payload
   does not carry `info_time`. Prefer boundary payload if available.

Output
------
- GroupSequentialDecisionSignalRecord (id specified by `out_id` in __init__)
  payload example:
    {
      "signal": "stop_efficacy" | "stop_futility" | "continue",
      "value": 2.13,                 # value on the *boundary* scale
      "value_scale": "z",            # boundary scale
      "upper": 2.5,
      "lower": -2.5,
      "scale": "z",
      "info_time": 0.5
    }
"""

import math
from typing import Dict, Literal, Optional, Tuple

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialBoundaryRecord,
    GroupSequentialDecisionSignalRecord,
    InformationTimeRecord,
)


def convert_scale(
    value: float, *, from_scale: str, to_scale: str, info_time: float
) -> float:
    """
    Convert a statistic between "z" and "bm" scales using information time.

    Parameters
    ----------
    value : float
        Input statistic value.
    from_scale : {"z","bm"}
    to_scale   : {"z","bm"}
    info_time  : float in [0,1]

    Returns
    -------
    float
        Converted value on `to_scale`.

    Notes
    -----
    Brownian mapping uses B(t) = Z * sqrt(t). Inverse is Z = B(t) / sqrt(t).

    Examples
    --------
    >>> round(convert_scale(2.0, from_scale="z", to_scale="bm", info_time=0.25), 6)
    1.0
    >>> round(convert_scale(1.0, from_scale="bm", to_scale="z", info_time=0.25), 6)
    2.0
    >>> round(convert_scale(1.23, from_scale="z", to_scale="z", info_time=0.7), 6)
    1.23
    """
    fs = str(from_scale).lower()
    ts = str(to_scale).lower()
    t = float(info_time)
    if not (0.0 <= t <= 1.0):
        raise ValueError("`info_time` must be in [0,1].")
    if fs not in ("z", "bm") or ts not in ("z", "bm"):
        raise ValueError(
            f"`from_scale`/`to_scale` must be 'z' or 'bm'. Got: fs={fs}, ts={ts}"
        )
    if fs == ts:
        return float(value)
    if fs == "z" and ts == "bm":
        return float(value) * math.sqrt(max(t, 0.0))
    if fs == "bm" and ts == "z":
        denom = math.sqrt(max(t, 1e-12))
        return float(value) / denom
    # unreachable
    return float(value)


def _decide(
    value: float, upper: float, lower: float
) -> Tuple[str, Literal["efficacy", "futility", "none"]]:
    """
    Compare value to (upper, lower) and return (signal, reason).

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


class Decision(LedgerOperator):
    """
    Scale-aware decision against a GroupSequentialBoundaryRecord.

    __init__ parameters
    -------------------
    boundary : GroupSequentialBoundaryRecord   # attached input
    out_id   : str
    value    : float                            # statistic value (on `value_scale`)
    value_scale : {"z","bm"} = "z"              # scale of `value`
    info     : InformationTimeRecord = None     # optional; used when boundary lacks info_time
    """

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
        return {"decision": GroupSequentialDecisionSignalRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        out = self.outputs["decision"]
        boundary = self.boundary  # type: ignore[attr-defined]
        value_in = float(getattr(self, "value"))
        value_scale = str(getattr(self, "value_scale")).lower()

        # Read boundary (latest)
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

        # If boundary payload doesn't carry info_time, try info record
        if (t is None or not (0.0 <= t <= 1.0)) and getattr(
            self, "info", None
        ) is not None:
            info = self.info  # type: ignore[attr-defined]
            idf = (
                info.latest()
                .select(info_time=info.t.payload["info_time"].cast("float64"))
                .execute()
            )
            if len(idf) > 0:
                t = float(idf.iloc[0]["info_time"])

        if t is None:
            # for z<->z we don't need t; for others we do
            if (value_scale != bscale) and (t is None):
                raise ValueError(
                    "Information time is required to convert between 'z' and 'bm' scales."
                )

        # convert to boundary scale and decide
        v = convert_scale(
            value_in, from_scale=value_scale, to_scale=bscale, info_time=(t or 0.0)
        )
        signal, reason = _decide(v, upper=upper, lower=lower)

        payload = {
            "signal": signal,
            "reason": reason,  # "efficacy" | "futility" | "none"
            "value": float(v),  # value on boundary scale
            "value_scale": bscale,  # same as boundary scale
            "upper": float(upper),
            "lower": float(lower),
            "scale": bscale,
        }
        if t is not None:
            payload["info_time"] = float(t)
        out.insert(payload)
