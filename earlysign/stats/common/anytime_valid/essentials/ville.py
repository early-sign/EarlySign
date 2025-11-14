"""
Ville's inequality utilities and operators (scheme-agnostic).

- ville_threshold(alpha) -> 1/alpha
- VilleThreshold        : persist {"alpha", "threshold"}
- VilleDecision         : compare latest E to threshold (with optional futility)

Futility policy
---------------
futility_mode: "none" | "fixed"
  - "none" : never produce futility stops
  - "fixed": stop_futility if E <= futility_tau (0 < tau < 1 suggested)
"""

from dataclasses import dataclass
from typing import Dict, Optional

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.anytime_valid.records import (
    EProcessRecord,
    SafeDecisionRecord,
    SafeDesignRecord,
    VilleThresholdRecord,
)


def ville_threshold(alpha: float) -> float:
    """
    Compute the Ville threshold: 1/alpha.

    Examples
    --------
    >>> ville_threshold(0.05)
    20.0
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("`alpha` must be in (0,1).")
    return 1.0 / float(alpha)


class VilleThreshold(LedgerOp):
    """
    Insert a Ville threshold row: {"alpha": alpha, "threshold": 1/alpha}.

    __init__ parameters
    -------------------
    out_id : str
    alpha  : float in (0,1)
    """

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        threshold: VilleThresholdRecord

    outputs: Outputs

    def __init__(self, scoped: Ledger, *, out_id: str, alpha: float):
        super().__init__(scoped, out_id=out_id, alpha=alpha)

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"threshold": VilleThresholdRecord(name=self.out_id)}

    def run(self) -> None:
        out = self.outputs.threshold
        alpha = float(getattr(self, "alpha"))
        thr = ville_threshold(alpha)
        out.insert({"alpha": alpha, "threshold": float(thr)})


class VilleDecision(LedgerOp):
    """
    Compare the latest E-process value to a Ville threshold.

    __init__ parameters
    -------------------
    eproc          : EProcessRecord              # attached input
    alpha          : float
    threshold_rec  : Optional[VilleThresholdRecord] = None
    futility_mode  : str = "none"                # "none" | "fixed"
    futility_tau   : Optional[float] = None      # used only when mode == "fixed"
    """

    out_id: str
    eproc: EProcessRecord
    alpha: Optional[float]
    threshold_rec: Optional[VilleThresholdRecord]
    design_rec: Optional[SafeDesignRecord]
    futility_mode: str
    futility_tau: Optional[float]

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        decision: SafeDecisionRecord

    outputs: Outputs

    def derived_records(self) -> dict[str, LedgerRecord]:
        return {"decision": SafeDecisionRecord(name=self.out_id)}

    def run(self) -> None:
        out = self.outputs.decision

        eproc: EProcessRecord = getattr(self, "eproc")
        alpha = getattr(self, "alpha", None)
        threshold_rec = getattr(self, "threshold_rec", None)
        design_rec = getattr(self, "design_rec", None)

        edf = (
            eproc.latest()
            .select(
                E=eproc.t.payload["E"].cast("float64"),
                look=eproc.t.payload["look"],
            )
            .execute()
        )
        if len(edf) == 0:
            return
        E = float(edf.iloc[0]["E"])

        # --- resolve threshold ---
        if threshold_rec is not None:
            tdf = (
                threshold_rec.latest()
                .select(
                    T=threshold_rec.t.payload["threshold"].cast("float64"),
                    alpha=threshold_rec.t.payload["alpha"].cast("float64"),
                )
                .execute()
            )
            if len(tdf) == 0:
                return
            T = float(tdf.iloc[0]["T"])
            a = float(tdf.iloc[0]["alpha"])
        elif design_rec is not None:
            ddf = (
                design_rec.latest()
                .select(
                    alpha=design_rec.t.payload["alpha"].cast("float64"),
                )
                .execute()
            )
            if len(ddf) == 0:
                return
            a = float(ddf.iloc[0]["alpha"])
            T = ville_threshold(a)
        elif alpha is not None:
            a = float(alpha)
            T = ville_threshold(a)
        else:
            raise ValueError("Provide `design_rec`, or `threshold_rec`, or `alpha`.")

        # --- futility policy ---
        mode = str(getattr(self, "futility_mode", "none")).lower()
        tau = getattr(self, "futility_tau", None)
        signal, reason = "continue", "none"
        if E >= T:
            signal, reason = "reject", "efficacy"
        else:
            if mode == "fixed" and (tau is not None):
                tau = float(tau)
                if tau <= 0.0:
                    raise ValueError("`futility_tau` must be > 0 for mode='fixed'.")
                if E <= tau:
                    signal, reason = "stop_futility", "futility"

        payload = {
            "criterion": "Ville",
            "signal": signal,
            "reason": reason,
            "E": float(E),
            "threshold": float(T),
            "alpha": float(a),
        }
        look = edf.iloc[0]["look"]
        if look is not None:
            payload["look"] = int(look)

        out.insert(payload)
