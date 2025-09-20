from dataclasses import dataclass
from typing import Dict

import math
from earlysign.framework.operator import LedgerOperator
from earlysign.stats.schemes.two_binomials.records import (
    BinomialCountsRecord,
    WaldZStatisticRecord,
)


@dataclass
class WaldZStatistic(LedgerOperator):
    """Compute Wald Z for two-proportion test and persist to out record."""

    counts: BinomialCountsRecord
    out: WaldZStatisticRecord

    def run(self) -> None:
        # Read the most recent aggregated row (nA, mA, nB, mB)
        t = self.counts.latest()
        q = t.select(
            nA=t.payload["nA"].cast("int64"),
            mA=t.payload["mA"].cast("int64"),
            nB=t.payload["nB"].cast("int64"),
            mB=t.payload["mB"].cast("int64"),
        )
        df = q.execute()
        if len(df) == 0:
            return

        nA = int(df.iloc[0]["nA"])
        mA = int(df.iloc[0]["mA"])
        nB = int(df.iloc[0]["nB"])
        mB = int(df.iloc[0]["mB"])
        if nA == 0 or nB == 0:
            return

        pA = mA / nA
        pB = mB / nB
        se = math.sqrt((pA * (1 - pA)) / nA + (pB * (1 - pB)) / nB)
        if se == 0:
            return
        z = (pA - pB) / se

        # Write a single row to out
        self.out.insert(
            wald_z=float(z),
            pA=float(pA),
            pB=float(pB),
            se=float(se),
        )

    def derived_records(self) -> Dict[str, WaldZStatisticRecord]:
        return {"wald": self.out}
