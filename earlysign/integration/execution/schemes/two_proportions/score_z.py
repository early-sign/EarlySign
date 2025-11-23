"""Score Z statistic record and operator for two-proportion experiments."""

from dataclasses import dataclass

from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord, QueryMixin
from earlysign.integration.execution.schemes.two_proportions.binomial_arms import (
    BinomialArmSnapshot,
)
from earlysign.stats.schemes.two_proportions.wald_z import compute_binomial_wald_z


class ScoreZStatisticRecord(LedgerRecord, QueryMixin):
    """Score Z statistic record for two-proportions."""

    schema = {
        "score_z": (float, ...),
    }


class ScoreZStatistic(LedgerOp):
    """Compute score Z (pooled variance) and insert one row."""

    control: BinomialArmSnapshot
    variant: BinomialArmSnapshot
    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        """Outputs for this operator."""

        score: ScoreZStatisticRecord

    outputs: Outputs

    def build_outputs(self) -> dict[str, LedgerRecord]:
        return {"score": ScoreZStatisticRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.score

        control_payload = self.control.latest_payload()
        variant_payload = self.variant.latest_payload()
        nA = int(control_payload["trial"])
        mA = int(control_payload["success"])
        nB = int(variant_payload["trial"])
        mB = int(variant_payload["success"])
        z = compute_binomial_wald_z(nA=nA, mA=mA, nB=nB, mB=mB, pooled=True)

        payload = {"score_z": float(z)}
        out.insert(payload)
