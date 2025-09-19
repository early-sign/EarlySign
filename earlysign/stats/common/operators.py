import math
from dataclasses import dataclass
from typing import Dict, Any
from scipy.stats import norm

from earlysign.framework.operator import LedgerOperator
from earlysign.framework.record import LedgerRecord
from earlysign.stats.common.records import (
    InformationTimeRecord,
    GSTBoundaryRecord,
    DecisionSignalRecord,
)
from earlysign.stats.schemes.two_binomials.records import (
    BinomialCountsRecord,
    WaldZStatisticRecord,
)


@dataclass
class InformationTime(LedgerOperator):
    counts: BinomialCountsRecord
    max_sample_size: int

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=f"{self.counts.id}:it")}

    def run(self) -> Any:
        last = self.counts.latest().execute()
        if len(last) == 0 or self.max_sample_size <= 0:
            return {"fraction": 0.0}
        row = last.to_dict("records")[0]["payload"]
        nA, nB = int(row["nA"]), int(row["nB"])
        frac = min(1.0, (nA + nB) / float(self.max_sample_size))
        self.outputs["info"].insert({"fraction": frac})
        return {"fraction": frac}


@dataclass
class GSTBoundary(LedgerOperator):
    info: InformationTimeRecord
    alpha: float = 0.05
    style: str = "obf"

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"boundary": GSTBoundaryRecord(id=f"{self.info.id}:gst")}

    def run(self) -> Any:
        iq = self.info.latest().execute()
        if len(iq) == 0:
            return {"upper": float("inf"), "lower": float("-inf")}
        info_fraction = float(iq.to_dict("records")[0]["payload"]["fraction"])
        if info_fraction <= 0.0:
            up, lo = float("inf"), float("-inf")
        else:
            if self.style == "obf":
                z_alpha_2 = norm.ppf(1 - self.alpha / 2)
                alpha_t = 2 * (1 - norm.cdf(z_alpha_2 / math.sqrt(info_fraction)))
            else:
                alpha_t = self.alpha * math.log(1 + (math.e - 1) * info_fraction)
            up = 0.0 if alpha_t >= 1.0 else float(norm.ppf(1 - alpha_t / 2))
            lo = -up
        self.outputs["boundary"].insert(
            {
                "info_fraction": info_fraction,
                "upper": up,
                "lower": lo,
                "alpha": self.alpha,
                "style": self.style,
            }
        )
        return {"upper": up, "lower": lo}


@dataclass
class Decision(LedgerOperator):
    zstat: WaldZStatisticRecord
    boundary: GSTBoundaryRecord

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"signal": DecisionSignalRecord(id=f"{self.zstat.id}:decision")}

    def run(self) -> Any:
        wz = self.zstat.latest().execute()
        bd = self.boundary.latest().execute()
        if len(wz) == 0 or len(bd) == 0:
            return {"decision": "no_data"}
        z = float(wz.to_dict("records")[0]["payload"]["z"])
        up = float(bd.to_dict("records")[0]["payload"]["upper"])
        decision = "stop_success" if z >= up else "continue"
        self.outputs["signal"].insert({"decision": decision, "z": z, "upper": up})
        return {"decision": decision}
