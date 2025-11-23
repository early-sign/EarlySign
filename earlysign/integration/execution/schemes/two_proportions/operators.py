"""
Two-proportions scheme operator(s).

- BinomialWaldZ  : Z for (pB - pA) with optional pooled variance
- ScoreZStatistic : Z for (pB - pA) using pooled variance under H0 (score test)
"""

from dataclasses import dataclass
from typing import Sequence

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.integration.execution.methods.group_sequential.records.info import (
    InformationTimeRecord,
)
from earlysign.integration.execution.methods.group_sequential.records.statistics import (
    WaldZStatisticRecord,
)
from earlysign.integration.execution.schemes.two_proportions.records import (
    BinomialArmSnapshot,
    ScoreZStatisticRecord,
)
from earlysign.stats.methods.group_sequential.info_time import (
    info_time_from_sample_size,
)
from earlysign.stats.schemes.two_proportions.wald_z import compute_wald_z

ArmRecord = BinomialArmSnapshot


class InformationTime(LedgerOp):
    """
    Insert information-time record from sample counts.

    Parameters
    ----------
    ledger : Ledger
        Scoped ledger instance.
    out_id : str
        ID of InformationTimeRecord to create.
    control : BinomialArmSnapshot
        Record containing cumulative counts for the control arm.
    variants : Sequence[BinomialArmSnapshot]
        Record(s) containing cumulative counts for the comparison arm(s).
    planned_max_n : int
        Maximum total sample size.
    """

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        out_id: str,
        control: BinomialArmSnapshot,
        variants: BinomialArmSnapshot | Sequence[BinomialArmSnapshot],
        planned_max_n: int,
    ):
        super().__init__(
            ledger,
            out_id=out_id,
            control=control,
            variants=tuple(variants) if isinstance(variants, Sequence) else (variants,),
            planned_max_n=planned_max_n,
        )

    def build_outputs(self) -> dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.info
        control_record: ArmRecord = getattr(self, "control")
        variant_records: tuple[ArmRecord, ...] = getattr(self, "variants")
        planned_max_n = getattr(self, "planned_max_n")

        if not variant_records:
            raise ValueError("At least one variant record is required.")

        control_df = (
            control_record.latest(explode=True).select("trial", "success").execute()
        )
        variant_dfs = [
            rec.latest(explode=True).select("trial", "success").execute()
            for rec in variant_records
        ]

        trial_control = int(control_df.iloc[0]["trial"])
        trial_variants = sum(int(df.iloc[0]["trial"]) for df in variant_dfs)
        n_total = trial_control + trial_variants

        t = info_time_from_sample_size(n_current=n_total, n_max=planned_max_n)
        out.insert({"info_time": float(t)})


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

    # Type annotation for outputs - enables type inference!
    outputs: Outputs

    def build_outputs(self) -> dict[str, LedgerRecord]:
        return {"wald": WaldZStatisticRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        control: ArmRecord = self.control
        variant: ArmRecord = self.variant
        out = self.outputs.wald
        pooled = bool(getattr(self, "pooled", True))

        control_df = control.latest(explode=True).select("trial", "success").execute()
        variant_df = variant.latest(explode=True).select("trial", "success").execute()

        nA = int(control_df.iloc[0]["trial"])
        mA = int(control_df.iloc[0]["success"])
        nB = int(variant_df.iloc[0]["trial"])
        mB = int(variant_df.iloc[0]["success"])
        z = compute_wald_z(nA=nA, mA=mA, nB=nB, mB=mB, pooled=pooled)

        out.insert({"wald_z": float(z)})


class ScoreZStatistic(LedgerOp):
    """Compute score Z (pooled variance) and insert one row."""

    control: BinomialArmSnapshot
    variant: BinomialArmSnapshot
    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        """Outputs for this operator."""

        score: ScoreZStatisticRecord

    # Type annotation for outputs - enables type inference!
    outputs: Outputs

    def build_outputs(self) -> dict[str, LedgerRecord]:
        return {"score": ScoreZStatisticRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        control: ArmRecord = self.control
        variant: ArmRecord = self.variant
        out = self.outputs.score

        control_df = control.latest(explode=True).select("trial", "success").execute()
        variant_df = variant.latest(explode=True).select("trial", "success").execute()
        nA = int(control_df.iloc[0]["trial"])
        mA = int(control_df.iloc[0]["success"])
        nB = int(variant_df.iloc[0]["trial"])
        mB = int(variant_df.iloc[0]["success"])
        # Score Z is equivalent to Wald Z with pooled variance
        z = compute_wald_z(nA=nA, mA=mA, nB=nB, mB=mB, pooled=True)

        payload = {"score_z": float(z)}
        out.insert(payload)
