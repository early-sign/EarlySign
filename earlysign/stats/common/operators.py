from dataclasses import dataclass
from typing import Dict
from math import sqrt
from scipy.stats import norm

from earlysign.framework.operator import LedgerOperator
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
    """Compute information time (here: simple fraction by sample size proxy)."""

    counts: BinomialCountsRecord
    out: InformationTimeRecord
    max_sample_size: int | None = None
    planned_fractions: list[float] | None = None
    current_look: int = 1

    def run(self) -> None:
        # Get n_total (from the latest binomial counts)
        t = self.counts.latest()
        q = t.select(
            nA=t.payload["nA"].cast("int64"),
            nB=t.payload["nB"].cast("int64"),
        )
        df = q.execute()
        if len(df) == 0:
            return
        n_total = int(df.iloc[0]["nA"]) + int(df.iloc[0]["nB"])

        if self.max_sample_size is not None:
            info = min(1.0, n_total / self.max_sample_size)
        else:
            if not self.planned_fractions:
                # Default to equal fractions (example: 4 looks total)
                self.planned_fractions = [(i + 1) / 4 for i in range(4)]
            idx = min(max(self.current_look - 1, 0), len(self.planned_fractions) - 1)
            info = float(self.planned_fractions[idx])

        self.out.insert(info_time=float(info))

    def derived_records(self) -> Dict[str, InformationTimeRecord]:
        return {"info": self.out}


@dataclass
class GSTBoundary(LedgerOperator):
    """Compute GST boundary (O'Brien-Fleming or Pocock-like)."""

    info: InformationTimeRecord
    out: GSTBoundaryRecord
    alpha: float = 0.05
    style: str = "obf"  # "obf" or "pocock"

    def run(self) -> None:
        t = self.info.latest()
        q = t.select(info_time=t.payload["info_time"].cast("float64"))
        df = q.execute()
        if len(df) == 0:
            return
        info = float(df.iloc[0]["info_time"])
        if info <= 0:
            return

        if self.style == "obf":
            # O'Brien–Fleming (Lan-DeMets approximation)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)
            alpha_t = 2 * (1 - norm.cdf(z_alpha_2 / sqrt(info)))
        else:
            # Pocock
            from math import e, log

            alpha_t = self.alpha * log(1 + (e - 1) * info)

        boundary = 0.0 if alpha_t >= 1.0 else float(norm.ppf(1 - alpha_t / 2))
        self.out.insert(
            upper=boundary, lower=-boundary, alpha=float(self.alpha), style=self.style
        )

    def derived_records(self) -> Dict[str, GSTBoundaryRecord]:
        return {"boundary": self.out}


@dataclass
class Decision(LedgerOperator):
    """Gate decision based on Wald Z and boundary + info time."""

    wald: WaldZStatisticRecord
    info: InformationTimeRecord
    bound: GSTBoundaryRecord
    out: DecisionSignalRecord

    def run(self) -> None:
        # Read the three dependencies
        wdf = (
            self.wald.latest()
            .select(wald_z=self.wald.t.payload["wald_z"].cast("float64"))
            .execute()
        )
        idf = (
            self.info.latest()
            .select(info_time=self.info.t.payload["info_time"].cast("float64"))
            .execute()
        )
        bdf = (
            self.bound.latest()
            .select(
                upper=self.bound.t.payload["upper"].cast("float64"),
                lower=self.bound.t.payload["lower"].cast("float64"),
            )
            .execute()
        )

        if len(wdf) == 0 or len(idf) == 0 or len(bdf) == 0:
            return

        z = float(wdf.iloc[0]["wald_z"])
        up = float(bdf.iloc[0]["upper"])
        lo = float(bdf.iloc[0]["lower"])
        decision = "stop" if (z >= up or z <= lo) else "continue"

        self.out.insert(
            signal=decision,
            wald_z=z,
            info_time=float(idf.iloc[0]["info_time"]),
        )

    def derived_records(self) -> Dict[str, DecisionSignalRecord]:
        return {"decision": self.out}
