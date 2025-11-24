"""Operators for Ville thresholds and decisions (anytime-valid safe testing)."""

from dataclasses import dataclass
from typing import Dict, Optional

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.integration.execution.methods.anytime_valid.records import (
    EProcessRecord,
    SafeDecisionRecord,
    SafeDesignRecord,
    VilleThresholdRecord,
)
from earlysign.stats.methods.anytime_valid.boundary import ville_threshold


class VilleThreshold(LedgerOp):
    """Insert a Ville threshold row: {'alpha': alpha, 'threshold': 1/alpha}."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        threshold: VilleThresholdRecord

    outputs: Outputs

    def __init__(self, ledger: Ledger, *, out_id: str, alpha: float):
        super().__init__(ledger, out_id=out_id, alpha=alpha)

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"threshold": VilleThresholdRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.threshold
        alpha = float(getattr(self, "alpha"))
        thr = ville_threshold(alpha)
        out.insert({"alpha": alpha, "threshold": float(thr)})


class VilleDecision(LedgerOp):
    """Compare the latest E-process value to a Ville threshold."""

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

    def build_outputs(self) -> dict[str, LedgerRecord]:
        return {"decision": SafeDecisionRecord(name=self.out_id, ledger=self.ledger)}

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
            )
            .execute()
        )
        if len(edf) == 0:
            return
        E = float(edf.iloc[0]["E"])

        # resolve threshold
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

        out.insert(payload)
