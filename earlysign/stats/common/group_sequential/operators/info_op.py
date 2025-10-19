"""
Information time operators for group sequential testing.

Wraps information time calculation functions with Ledger operations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.group_sequential.essentials.information import (
    info_time_from_fisher,
    info_time_from_ratio,
    info_time_from_sample_size,
    info_time_from_sd,
    info_time_from_variance,
)
from earlysign.stats.common.group_sequential.records import InformationTimeRecord
from earlysign.stats.schemes.two_proportions.records import (
    BinomialCountsRecord,
    BinomialCountsSnapshotRecord,
)


class InformationTime(LedgerOperator):
    """
    Insert information-time record from sample counts.

    Parameters
    ----------
    scoped : Ledger
        Scoped ledger instance.
    out_id : str
        ID of InformationTimeRecord to create.
    cum_counts : BinomialCountsRecord | BinomialCountsSnapshotRecord (attached)
        Record containing cumulative binomial counts.
    planned_max_n : int
        Maximum total sample size.

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.stats.schemes.two_proportions.records import BinomialCountsSnapshotRecord
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events")
    >>> ledger.ensure()
    >>> counts = BinomialCountsSnapshotRecord(id="counts1").attach(ledger)
    >>> counts.insert({"nA": 100, "mA": 10, "nB": 100, "mB": 15})
    >>> op = InformationTime(ledger, out_id="info1", cum_counts=counts, planned_max_n=400)
    >>> op.run()
    """

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        cum_counts: Union[BinomialCountsRecord, BinomialCountsSnapshotRecord],
        planned_max_n: int,
    ):
        super().__init__(
            scoped,
            out_id=out_id,
            cum_counts=cum_counts,
            planned_max_n=planned_max_n,
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}

    def run(self) -> None:
        out = self.outputs.info
        cum_counts = getattr(self, "cum_counts")
        planned_max_n = getattr(self, "planned_max_n")

        # Read latest counts
        cdf = cum_counts.latest().execute()
        if len(cdf) == 0:
            raise ValueError("No counts data available")

        latest = cdf.iloc[0]
        n_total = int(latest["nA"]) + int(latest["nB"])

        # Use essentials function
        t = info_time_from_sample_size(n_current=n_total, n_max=planned_max_n)

        out.insert({"info_time": float(t)})


class InformationTimeFromRatio(LedgerOperator):
    """Insert t = info_now / info_max."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        info_now: float,
        info_max: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            scoped, out_id=out_id, info_now=info_now, info_max=info_max, look=look
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_ratio(
            info_now=float(getattr(self, "info_now")),
            info_max=float(getattr(self, "info_max")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromVariance(LedgerOperator):
    """Insert t = var_target / var_now."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        var_now: float,
        var_target: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            scoped, out_id=out_id, var_now=var_now, var_target=var_target, look=look
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_variance(
            var_now=float(getattr(self, "var_now")),
            var_target=float(getattr(self, "var_target")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromSD(LedgerOperator):
    """Insert t = (sd_target^2) / (sd_now^2)."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        sd_now: float,
        sd_target: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            scoped, out_id=out_id, sd_now=sd_now, sd_target=sd_target, look=look
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_sd(
            sd_now=float(getattr(self, "sd_now")),
            sd_target=float(getattr(self, "sd_target")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)


class InformationTimeFromFisher(LedgerOperator):
    """Insert t = fisher_now / fisher_max."""

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        fisher_now: float,
        fisher_max: float,
        look: Optional[int] = None,
    ):
        super().__init__(
            scoped,
            out_id=out_id,
            fisher_now=fisher_now,
            fisher_max=fisher_max,
            look=look,
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}

    def run(self) -> None:
        out = self.outputs.info
        t = info_time_from_fisher(
            fisher_now=float(getattr(self, "fisher_now")),
            fisher_max=float(getattr(self, "fisher_max")),
        )
        payload = {"info_time": float(t)}
        if getattr(self, "look", None) is not None:
            payload["look"] = int(getattr(self, "look"))
        out.insert(payload)
