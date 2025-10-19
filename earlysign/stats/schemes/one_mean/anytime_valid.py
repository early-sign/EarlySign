"""
Anytime-Valid (Safe) testing for one-mean (Gaussian).

We assume known variance sigma^2 and test H0: mu = 0.

Two mixture-evidence e-processes using the sample mean (sufficient for Gaussian):

Variant 1 (symmetric alt):
- H0: ybar | n ~ Normal(0, sigma^2/n)
- H1: ybar | n ~ Normal(0, sigma^2/n + tau^2)    with tau^2 > 0

Variant 2 (one-sided alt):
- H0: same
- H1: ybar | n ~ NormalPlus(0, sigma^2/n + tau^2)  # half-normal mixture on mu
  Implementation: marginal density is Normal(0, v1) conditioned on mu>=0
  → closed form via Normal CDF factor 2*Phi( (ybar*sqrt(tau^2/v1)) ) * NormalPDF(ybar; 0, v1)
  (derivation uses convolution of N(0,tau^2) truncated at >=0 with Normal(0, sigma^2/n))

Operators
---------
- MixtureEProcessOneMeanKnownVar
- OneSidedMixtureEProcessOneMeanKnownVar
"""

import math
from dataclasses import dataclass
from typing import Any, Dict

from scipy.stats import norm

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.anytime_valid.records import EProcessRecord
from earlysign.stats.schemes.one_mean.records import OneMeanSummaryRecord


def mixture_e_one_mean_known_var(
    n: int, ybar: float, sigma2: float, tau2: float
) -> float:
    """
    Symmetric-alt mixture e-value for one-mean with known variance.

    Under H0: ybar ~ N(0, v0) with v0 = sigma2 / n
    Under H1: ybar ~ N(0, v1) with v1 = sigma2 / n + tau2, tau2 = c * sigma2 / n

    e = f1(ybar) / f0(ybar) = sqrt(v0/v1) * exp(-0.5*ybar^2*(1/v1 - 1/v0))

    Examples
    --------
    >>> round(mixture_e_one_mean_known_var(100, 0.2, 1.0, 1.0), 6)
    0.720823
    """
    if n <= 0 or sigma2 <= 0.0 or tau2 <= 0.0:
        raise ValueError("n>0, sigma2>0, tau2>0 are required.")
    v0 = sigma2 / float(n)
    v1 = v0 + tau2
    return math.sqrt(v0 / v1) * math.exp(-0.5 * (ybar**2) * (1.0 / v1 - 1.0 / v0))


def one_sided_mixture_e_one_mean_known_var(
    n: int, ybar: float, sigma2: float, tau2: float
) -> float:
    """
    One-sided (mu >= 0) mixture e-value.

    The marginal under H1 is the convolution of Normal(mu, v0) with mu~Normal^+(0, tau²).
    Its density at ybar is:
        f1^+(ybar) = 2 * Phi( (tau/sqrt(v1)) * ybar ) * NormalPDF(ybar; 0, v1)
    with v0 = σ²/n, v1 = v0 + τ², τ² = c σ²/n, τ = sqrt(τ²).

    Then e = f1^+(ybar) / f0(ybar).

    Examples
    --------
    >>> round(one_sided_mixture_e_one_mean_known_var(100, 0.2, 1.0, 1.0), 6)
    0.834527
    """
    if n <= 0 or sigma2 <= 0.0 or tau2 <= 0.0:
        raise ValueError("n>0, sigma2>0, tau2>0 are required.")
    v0 = sigma2 / float(n)
    v1 = v0 + tau2
    # Normal PDFs (cancel constants via ratio, but keep explicit for clarity)
    f1 = (
        2.0
        * norm.cdf((math.sqrt(tau2) / math.sqrt(v1)) * ybar)
        * (1.0 / math.sqrt(2.0 * math.pi * v1))
        * math.exp(-(ybar**2) / (2.0 * v1))
    )
    f0 = (1.0 / math.sqrt(2.0 * math.pi * v0)) * math.exp(-(ybar**2) / (2.0 * v0))
    return float(f1 / max(f0, 1e-300))


class MixtureEProcessOneMeanKnownVar(LedgerOperator):
    """Insert symmetric-alt mixture e-process row for one-mean."""

    out_id: str
    summary: OneMeanSummaryRecord

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        eproc: EProcessRecord

    outputs: Outputs

    def __init__(
        self,
        scoped: Ledger,
        *,
        summary: OneMeanSummaryRecord,
        out_id: str,
        sigma2: float,
        tau2: float = 1.0,
    ):
        super().__init__(
            scoped, summary=summary, out_id=out_id, sigma2=sigma2, tau2=tau2
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"eproc": EProcessRecord(id=self.out_id)}

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
        E = mixture_e_one_mean_known_var(n, ybar, sigma2, tau2)
        payload: Dict[str, Any] = {
            "E": float(E),
            "logE": float(math.log(max(E, 1e-300))),
        }
        payload["look"] = int(s.iloc[0]["look"])
        payload["params"] = {"sigma2": sigma2, "tau2": tau2}
        out.insert(payload)


class OneSidedMixtureEProcessOneMeanKnownVar(LedgerOperator):
    """Insert one-sided-alt mixture e-process row for one-mean."""

    out_id: str
    summary: OneMeanSummaryRecord

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        eproc: EProcessRecord

    outputs: Outputs

    def __init__(
        self,
        scoped: Ledger,
        *,
        summary: OneMeanSummaryRecord,
        out_id: str,
        sigma2: float,
        tau2: float = 1.0,
    ):
        super().__init__(
            scoped, summary=summary, out_id=out_id, sigma2=sigma2, tau2=tau2
        )

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"eproc": EProcessRecord(id=self.out_id)}

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
        E = one_sided_mixture_e_one_mean_known_var(n, ybar, sigma2, tau2)
        payload: Dict[str, Any] = {
            "E": float(E),
            "logE": float(math.log(max(E, 1e-300))),
        }
        payload["look"] = int(s.iloc[0]["look"])
        payload["params"] = {"sigma2": sigma2, "tau2": tau2, "alt": "one_sided"}
        out.insert(payload)
