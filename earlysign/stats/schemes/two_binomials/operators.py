import math
from dataclasses import dataclass
from typing import Dict, Any
from earlysign.framework.operator import LedgerOperator
from earlysign.framework.record import LedgerRecord
from earlysign.stats.schemes.two_binomials.records import (
    BinomialCountsRecord,
    WaldZStatisticRecord,
)


@dataclass
class WaldZStatistic(LedgerOperator):
    counts: BinomialCountsRecord

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"wald": WaldZStatisticRecord(id=f"{self.counts.id}:waldz")}

    def run(self) -> Any:
        last = self.counts.latest().execute()
        if len(last) == 0:
            return {"z": 0.0, "se": 0.0}
        row = last.to_dict("records")[0]["payload"]
        nA, nB, mA, mB = int(row["nA"]), int(row["nB"]), int(row["mA"]), int(row["mB"])
        if nA == 0 or nB == 0:
            return {"z": 0.0, "se": 0.0}
        pA, pB = mA / nA, mB / nB
        se = math.sqrt((pA * (1 - pA)) / nA + (pB * (1 - pB)) / nB)
        z = 0.0 if se == 0 else (pA - pB) / se
        self.outputs["wald"].insert(
            {
                "z": z,
                "se": se,
                "pA": pA,
                "pB": pB,
                "nA": nA,
                "nB": nB,
                "mA": mA,
                "mB": mB,
            }
        )
        return {"z": z, "se": se}
