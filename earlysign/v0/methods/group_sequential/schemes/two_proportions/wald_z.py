"""Wald Z statistic record and operator for two-proportion experiments."""

from dataclasses import dataclass

from earlysign.v0.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.v0.framework.records import LedgerRecord, QueryMixin
from earlysign.v0.methods.group_sequential.schemes.two_proportions.binomial_arms import (
    BinomialArmSnapshot,
)
from earlysign.v0.stats.schemes.two_proportions.statistic.wald_z import (
    compute_binomial_wald_z,
)


class WaldZStatisticRecord(LedgerRecord, QueryMixin):
    """Wald Z statistic record (scheme-agnostic)."""

    schema = {
        "wald_z": (float, ...),
    }


class BinomialWaldZ(LedgerOp):
    """Compute Wald Z for two arms and insert one row."""

    control: BinomialArmSnapshot
    variant: BinomialArmSnapshot
    pooled: bool
    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        """Outputs for this operator."""

        wald: WaldZStatisticRecord

    outputs: Outputs

    def build_outputs(self) -> dict[str, LedgerRecord]:
        return {"wald": WaldZStatisticRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.wald
        pooled = bool(getattr(self, "pooled", True))

        control_payload = self.control.latest_payload()
        variant_payload = self.variant.latest_payload()

        nA = int(control_payload["trial"])
        mA = int(control_payload["success"])
        nB = int(variant_payload["trial"])
        mB = int(variant_payload["success"])
        z = compute_binomial_wald_z(nA=nA, mA=mA, nB=nB, mB=mB, pooled=pooled)

        out.insert({"wald_z": float(z)})
