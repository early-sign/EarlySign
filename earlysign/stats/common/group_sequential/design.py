"""
Group Sequential Design & Boundary (scheme-agnostic).

Key extensions
--------------
- Boundary scale: "z" (standardized) or "bm" (Brownian, B(t)=Z*sqrt(t))
- Separate arms: efficacy (upper) and futility (lower)
  * futility.mode: "none" | "symmetric" | "fixed_z"
    - "none"       : lower = -inf
    - "symmetric"  : lower = -upper (two-sided only)
    - "fixed_z"    : use provided per-look or scalar Z cutoff(s); negative direction by convention

Design schema (suggested)
-------------------------
{
  "alpha": 0.05,
  "tails": 2,
  "scale": "z" | "bm",                         # default "z"
  "efficacy": {
     "style": "alpha_spending" | "significance_level",
     # if alpha_spending:
     "family": "obf" | "pocock",
     # if significance_level:
     "alpha_levels": [0.01, 0.01, ...] | {1:0.01, 2:0.01, ...}
  },
  "futility": {
     "mode": "none" | "symmetric" | "fixed_z",
     "binding": false,                          # metadata only (non-binding default)
     # if fixed_z:
     "z": -0.5 | {-1: -0.5, 2: -0.2, ...}      # scalar or per-look; negative means below zero
  }
}

Back-compat
-----------
If top-level {"style":..., "family":..., "alpha_levels":...} is given,
it is interpreted as efficacy.* .
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator
from earlysign.framework.records import LedgerRecord, QueryMixin
from earlysign.stats.common.group_sequential.alpha_spending import (
    cumulative_to_nominal_z,
    obf_spending,
    pocock_spending,
    z_to_brownian,
)
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialBoundaryRecord,
    InformationTimeRecord,
)
from earlysign.stats.common.group_sequential.significance_level import (
    level_to_nominal_z,
)


class GroupSequentialDesignRecord(LedgerRecord, QueryMixin):
    """Design record for Group Sequential Testing.

    Payload example:
        {
            "alpha": 0.05,
            "tails": 2,
            "scale": "z",
            "efficacy": {"style": "alpha_spending", "family": "obf", "alpha_levels": [ ... ]},
            "futility": {"mode": "symmetric"}  # or {"mode": "none"} / {"mode": "binding", ...}
        }
    """

    payload_type: str = "GroupSequential/Design"
    schema = {
        "alpha": float,
        "tails": int,
        "scale": str,
        "efficacy": dict,
        "futility": dict,
    }


def _get_alpha_level_for_look(
    alpha_levels: Mapping[int, float] | List[float], look: int
) -> float:
    if isinstance(alpha_levels, dict):
        if look not in alpha_levels:
            raise KeyError(f"alpha_levels missing look={look}")
        return float(alpha_levels[look])
    idx = look - 1
    if idx < 0 or idx >= len(alpha_levels):
        raise IndexError("look out of range for alpha_levels list.")
    return float(alpha_levels[idx])


def _resolve_efficacy_upper_z(
    design: Mapping[str, Any], t: float, look: Optional[int]
) -> float:
    """Return the efficacy upper boundary on Z-scale."""
    # Accept both new nested design["efficacy"] and legacy top-level keys.
    eff = dict(design.get("efficacy") or {})
    style = (eff.get("style") or design.get("style") or "").lower()
    tails = int(design["tails"])
    alpha = float(design["alpha"])

    if style == "alpha_spending":
        family = (eff.get("family") or design.get("family") or "obf").lower()
        if family == "obf":
            a_cum = obf_spending(t, alpha=alpha, tails=tails)
        elif family == "pocock":
            a_cum = pocock_spending(t, alpha=alpha)
        else:
            raise ValueError("Unknown efficacy.family: {}".format(family))
        up, _ = cumulative_to_nominal_z(a_cum, tails=tails)
        return float(up)

    if style == "significance_level":
        if look is None:
            raise ValueError("`look` is required for significance_level efficacy.")
        alpha_levels = eff.get("alpha_levels") or design.get("alpha_levels")
        if alpha_levels is None:
            raise ValueError("efficacy.alpha_levels required for significance_level.")
        a_look = _get_alpha_level_for_look(alpha_levels, look)
        up, _ = level_to_nominal_z(a_look, tails=tails)
        return float(up)

    raise ValueError("Unknown efficacy.style: {}".format(style))


def _resolve_futility_lower_z(
    design: Mapping[str, Any], upper_z: float, look: Optional[int]
) -> float:
    """Return the futility lower boundary on Z-scale (may be -inf)."""
    fut = dict(design.get("futility") or {})
    mode = (fut.get("mode") or "symmetric").lower()
    tails = int(design["tails"])

    if mode == "none":
        return float("-inf")

    if mode == "symmetric":
        if tails == 2:
            return float(-abs(upper_z))
        else:
            return float("-inf")

    if mode == "fixed_z":
        zspec = fut.get("z")
        if zspec is None:
            raise ValueError("futility.fixed_z requires key 'z'.")
        if isinstance(zspec, dict):
            if look is None:
                raise ValueError("futility.fixed_z with dict requires `look`.")
            if look not in zspec:
                raise KeyError(f"futility.fixed_z missing look={look}")
            zval = float(zspec[look])
        else:
            zval = float(zspec)
        # Convention: futility is below zero; ensure lower bound
        return float(min(zval, 0.0))

    raise ValueError("Unknown futility.mode: {}".format(mode))


def resolve_boundary_from_design(
    *,
    design: Mapping[str, Any],
    info_time: float,
    look: Optional[int] = None,
) -> Tuple[float, float, str]:
    """
    Compute (upper, lower, scale) boundaries given a design and info_time.

    Returns
    -------
    upper : float
        Efficacy boundary (on requested scale).
    lower : float
        Futility boundary (on requested scale).
    scale : {"z","bm"}
        Scale of the returned boundaries.

    Examples
    --------
    >>> d = {"alpha":0.05,"tails":2,"scale":"z","efficacy":{"style":"alpha_spending","family":"obf"}}
    >>> up, lo, sc = resolve_boundary_from_design(design=d, info_time=0.5, look=2)
    >>> sc
    'z'
    """
    t = float(info_time)
    if not (0.0 <= t <= 1.0):
        raise ValueError("info_time must be in [0,1].")

    # Efficacy on Z-scale
    up_z = _resolve_efficacy_upper_z(design, t, look)
    # Futility on Z-scale
    lo_z = _resolve_futility_lower_z(design, up_z, look)

    # Map to requested scale
    scale = str(design.get("scale", "z")).lower()
    if scale == "z":
        return float(up_z), float(lo_z), "z"
    elif scale == "bm":
        return float(z_to_brownian(up_z, t)), float(z_to_brownian(lo_z, t)), "bm"
    else:
        raise ValueError("design['scale'] must be 'z' or 'bm'.")


class GroupSequentialDesign(LedgerOperator):
    """
    Validate and insert a Group Sequential Design row.

    __init__ parameters
    -------------------
    out_id : str
        ID of GroupSequentialDesignRecord to create.
    design : dict
        See module docstring ('Design schema') for keys.
    """

    REQUIRED_KEYS = ("alpha", "tails")

    def __init__(self, scoped: Ledger, *, out_id: str, design: Dict[str, Any]):
        super().__init__(scoped, out_id=out_id, design=design)

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"design": GroupSequentialDesignRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        out = self.outputs["design"]
        design = dict(getattr(self, "design"))
        # Basic checks
        for k in self.REQUIRED_KEYS:
            if k not in design:
                raise ValueError("Missing required design key: {}".format(k))
        a = float(design["alpha"])
        if not (0.0 < a < 1.0):
            raise ValueError("design['alpha'] must be in (0,1).")
        tails = int(design["tails"])
        if tails not in (1, 2):
            raise ValueError("design['tails'] must be 1 or 2.")
        scale = str(design.get("scale", "z")).lower()
        if scale not in ("z", "bm"):
            raise ValueError("design['scale'] must be 'z' or 'bm'.")

        # If legacy style is given at top-level, normalize into efficacy.*
        if "style" in design and "efficacy" not in design:
            eff = {
                k: design[k] for k in ("style", "family", "alpha_levels") if k in design
            }
            design["efficacy"] = eff
        if "futility" in design:
            mode = str(design["futility"].get("mode", "symmetric")).lower()
            if mode not in ("none", "symmetric", "fixed_z"):
                raise ValueError(
                    "futility.mode must be one of {'none','symmetric','fixed_z'}."
                )

        out.insert(design)


class BoundaryFromDesign(LedgerOperator):
    """
    Read latest Design & InfoTime, then insert a boundary row (efficacy/futility).

    __init__ parameters
    -------------------
    design : GroupSequentialDesignRecord   # attached input
    info   : InformationTimeRecord         # attached input
    out_id : str                           # boundary record id to create
    look   : Optional[int]                 # for significance-level efficacy or fixed futility dict
    """

    design: GroupSequentialDesignRecord
    info: InformationTimeRecord
    out_id: str
    look: Optional[int]

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"boundary": GroupSequentialBoundaryRecord(id=self.out_id)}

    def run(self) -> None:
        design_rec = self.design
        info_rec = self.info
        out = self.outputs["boundary"]
        look = getattr(self, "look", None)

        ddf = design_rec.latest().select(design=design_rec.t.payload).execute()
        if len(ddf) == 0:
            return
        design_payload = ddf.iloc[0]["design"]

        idf = (
            info_rec.latest()
            .select(info_time=info_rec.t.payload["info_time"].cast("float64"))
            .execute()
        )
        if len(idf) == 0:
            return
        t = float(idf.iloc[0]["info_time"])

        up, lo, scale = resolve_boundary_from_design(
            design=design_payload, info_time=t, look=look
        )

        # Compose payload with both flat and structured fields for downstream ease.
        payload = {
            # flat for easy consumption (compatible with simple Decision operators)
            "upper": float(up),  # efficacy
            "lower": float(lo),  # futility
            # structured for richer consumers
            "efficacy": {"upper": float(up)},
            "futility": {
                "lower": float(lo),
                "binding": bool(
                    design_payload.get("futility", {}).get("binding", False)
                ),
                "mode": str(
                    design_payload.get("futility", {}).get("mode", "symmetric")
                ).lower(),
            },
            "scale": str(scale),  # "z" or "bm"
            "alpha": float(design_payload["alpha"]),
            "tails": int(design_payload["tails"]),
            "info_time": float(t),
        }
        if look is not None:
            payload["look"] = int(look)
        out.insert(payload)
