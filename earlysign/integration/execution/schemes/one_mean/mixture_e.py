"""Mixture e-process operators for one-mean (Gaussian, known variance)."""

import math
from dataclasses import dataclass
from typing import Any, Dict

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.integration.execution.schemes.one_mean.records import (
    EProcessRecord,
    OneMeanSummaryRecord,
)
from earlysign.stats.schemes.one_mean.mixture_e import (
    mixture_e_one_mean_known_var,
    one_sided_mixture_e_one_mean_known_var,
)


class MixtureEProcessOneMeanKnownVar(LedgerOp):
    """
    Insert symmetric-alt mixture e-process row for one-mean.

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.integration.execution.schemes.one_mean.records import OneMeanSummaryRecord
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events"); _ = ledger.ensure(); ledger = ledger.bind(experiment_id="demo")
    >>> summary = OneMeanSummaryRecord("s").attach(ledger)
    >>> _ = summary.insert({"n": 10, "mean": 0.1, "look": 1})
    >>> _ = MixtureEProcessOneMeanKnownVar(
    ...     ledger, summary=summary, out_id="e1", sigma2=1.0, tau2=1.0
    ... ).run()
    >>> eproc = EProcessRecord("e1").attach(ledger)
    >>> len(eproc.latest().execute()) == 1
    True
    """

    out_id: str
    summary: OneMeanSummaryRecord

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        eproc: EProcessRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        summary: OneMeanSummaryRecord,
        out_id: str,
        sigma2: float,
        tau2: float = 1.0,
    ):
        super().__init__(
            ledger, summary=summary, out_id=out_id, sigma2=sigma2, tau2=tau2
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"eproc": EProcessRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.eproc
        summary: OneMeanSummaryRecord = self.summary
        sigma2 = float(getattr(self, "sigma2"))
        tau2 = float(getattr(self, "tau2"))

        s = (
            summary.latest()
            .select(
                n=summary.t.payload["n"].cast("int64"),
                mean=summary.t.payload["mean"].cast("float64"),
                look=summary.t.payload["look"].cast("int64"),
            )
            .execute()
        )
        if len(s) == 0:
            return

        n = int(s.iloc[0]["n"])
        ybar = float(s.iloc[0]["mean"])
        e_value = mixture_e_one_mean_known_var(n, ybar, sigma2, tau2)
        payload: Dict[str, Any] = {
            "E": float(e_value),
            "logE": float(math.log(max(e_value, 1e-300))),
            "look": int(s.iloc[0]["look"]),
            "params": {"sigma2": sigma2, "tau2": tau2},
        }
        out.insert(payload)


class OneSidedMixtureEProcessOneMeanKnownVar(LedgerOp):
    """
    Insert one-sided-alt mixture e-process row for one-mean.

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.integration.execution.schemes.one_mean.records import OneMeanSummaryRecord
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events"); _ = ledger.ensure(); ledger = ledger.bind(experiment_id="demo")
    >>> summary = OneMeanSummaryRecord("s2").attach(ledger)
    >>> _ = summary.insert({"n": 8, "mean": 0.05, "look": 1})
    >>> _ = OneSidedMixtureEProcessOneMeanKnownVar(
    ...     ledger, summary=summary, out_id="e2", sigma2=1.0, tau2=1.0
    ... ).run()
    >>> eproc = EProcessRecord("e2").attach(ledger)
    >>> len(eproc.latest().execute()) == 1
    True
    """

    out_id: str
    summary: OneMeanSummaryRecord

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        eproc: EProcessRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        summary: OneMeanSummaryRecord,
        out_id: str,
        sigma2: float,
        tau2: float = 1.0,
    ):
        super().__init__(
            ledger, summary=summary, out_id=out_id, sigma2=sigma2, tau2=tau2
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"eproc": EProcessRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.eproc
        summary: OneMeanSummaryRecord = self.summary
        sigma2 = float(getattr(self, "sigma2"))
        tau2 = float(getattr(self, "tau2"))

        s = (
            summary.latest()
            .select(
                n=summary.t.payload["n"].cast("int64"),
                mean=summary.t.payload["mean"].cast("float64"),
                look=summary.t.payload["look"].cast("int64"),
            )
            .execute()
        )
        if len(s) == 0:
            return

        n = int(s.iloc[0]["n"])
        ybar = float(s.iloc[0]["mean"])
        e_value = one_sided_mixture_e_one_mean_known_var(n, ybar, sigma2, tau2)
        payload: Dict[str, Any] = {
            "E": float(e_value),
            "logE": float(math.log(max(e_value, 1e-300))),
            "look": int(s.iloc[0]["look"]),
            "params": {"sigma2": sigma2, "tau2": tau2, "alt": "one_sided"},
        }
        out.insert(payload)
