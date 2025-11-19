"""
Anytime-Valid (Safe) testing for two-proportions (A vs B).

We provide two mixture-evidence e-process variants based on beta-binomial marginals:

Variant 1 (symmetric alt, already standard):
- Null (equality):          p_A = p_B = p,   with  p ~ Beta(a0, b0)
- Alternative (independent): p_A ~ Beta(a1, b1), p_B ~ Beta(a1, b1)

Variant 2 (skewed alt to emphasize B > A):
- Null (equality):          p_A = p_B = p,   with p ~ Beta(a0, b0)
- Alternative (independent, skewed):
    p_A ~ Beta(a1A, b1A), p_B ~ Beta(a1B, b1B)  # choose a1B>b1B or similar to tilt upwards

Both define e-values via Bayes factor m_alt(data) / m_null(data); these form e-processes.

Operators
---------
- MixtureEProcessTwoProportions               : symmetric alt
- MixtureEProcessTwoProportionsSkew           : skewed alt (one-sided leaning)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from scipy.special import betaln

from earlysign.applications.execution.schemes.two_proportions.records import (
    BinomialCountsRecord,
)
from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.anytime_valid.records import EProcessRecord


def mixture_e_two_proportions(
    nA: int,
    mA: int,
    nB: int,
    mB: int,
    *,
    prior_null: Tuple[float, float] = (0.5, 0.5),
    prior_alt: Tuple[float, float] = (0.5, 0.5),
) -> float:
    """
    Compute the mixture-evidence e-value for two-proportions.

    Parameters
    ----------
    nA, mA, nB, mB : int
        Trials and successes for A and B.
    prior_null : (a0, b0)
        Beta prior parameters for the common p under H0.
    prior_alt : (a1, b1)
        Beta prior parameters for independent p_A, p_B under H1.

    Examples
    --------
    >>> round(mixture_e_two_proportions(100, 40, 120, 70, prior_null=(0.5,0.5), prior_alt=(0.5,0.5)), 6)
    4.302329
    """
    if not (0 <= mA <= nA and 0 <= mB <= nB):
        raise ValueError("counts must satisfy 0 <= m <= n.")
    a0, b0 = map(float, prior_null)
    a1, b1 = map(float, prior_alt)

    log_alt = (
        betaln(a1 + mA, b1 + (nA - mA))
        - betaln(a1, b1)
        + betaln(a1 + mB, b1 + (nB - mB))
        - betaln(a1, b1)
    )
    log_null = betaln(a0 + (mA + mB), b0 + ((nA + nB) - (mA + mB))) - betaln(a0, b0)
    return float(math.exp(float(log_alt - log_null)))


def mixture_e_two_proportions_skew(
    nA: int,
    mA: int,
    nB: int,
    mB: int,
    *,
    prior_null: Tuple[float, float] = (0.5, 0.5),
    prior_alt_A: Tuple[float, float] = (0.5, 0.5),
    prior_alt_B: Tuple[float, float] = (1.0, 0.5),
) -> float:
    """
    Compute a skewed-alt mixture e-value for two-proportions.

    The alt uses independent Beta priors with different shapes for A and B,
    allowing you to emphasize one-sided effects (e.g., B > A).

    Examples
    --------
    >>> round(mixture_e_two_proportions_skew(100, 40, 120, 70), 6)
    5.154707
    """
    if not (0 <= mA <= nA and 0 <= mB <= nB):
        raise ValueError("counts must satisfy 0 <= m <= n.")
    a0, b0 = map(float, prior_null)
    a1A, b1A = map(float, prior_alt_A)
    a1B, b1B = map(float, prior_alt_B)

    log_alt = (
        betaln(a1A + mA, b1A + (nA - mA))
        - betaln(a1A, b1A)
        + betaln(a1B + mB, b1B + (nB - mB))
        - betaln(a1B, b1B)
    )
    log_null = betaln(a0 + (mA + mB), b0 + ((nA + nB) - (mA + mB))) - betaln(a0, b0)
    return float(math.exp(float(log_alt - log_null)))


class MixtureEProcessTwoProportions(LedgerOp):
    """Insert symmetric-alt mixture e-process row for two-proportions."""

    out_id: str
    counts: BinomialCountsRecord

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        eproc: EProcessRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        counts: BinomialCountsRecord,
        out_id: str,
        prior_null: Tuple[float, float] = (0.5, 0.5),
        prior_alt: Tuple[float, float] = (0.5, 0.5),
    ):
        super().__init__(
            ledger,
            counts=counts,
            out_id=out_id,
            prior_null=prior_null,
            prior_alt=prior_alt,
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"eproc": EProcessRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.eproc
        counts: BinomialCountsRecord = self.counts
        prior_null = tuple(getattr(self, "prior_null"))
        prior_alt = tuple(getattr(self, "prior_alt"))

        cdf = (
            counts.latest()
            .select(
                nA=counts.t.payload["nA"].cast("int64"),
                mA=counts.t.payload["mA"].cast("int64"),
                nB=counts.t.payload["nB"].cast("int64"),
                mB=counts.t.payload["mB"].cast("int64"),
                look=counts.t.payload["look"].cast("int64"),
            )
            .execute()
        )
        if len(cdf) == 0:
            return

        nA, mA, nB, mB = map(int, cdf.iloc[0][["nA", "mA", "nB", "mB"]])
        E = mixture_e_two_proportions(
            nA, mA, nB, mB, prior_null=prior_null, prior_alt=prior_alt
        )
        payload: Dict[str, Any] = {
            "E": float(E),
            "logE": float(math.log(max(E, 1e-300))),
        }
        look = cdf.iloc[0]["look"]
        if look is not None:
            payload["look"] = int(look)
        payload["priors"] = {"null": list(prior_null), "alt": list(prior_alt)}
        out.insert(payload)


class MixtureEProcessTwoProportionsSkew(LedgerOp):
    """Insert skewed-alt mixture e-process row for two-proportions."""

    out_id: str
    counts: BinomialCountsRecord

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        eproc: EProcessRecord

    outputs: Outputs

    def __init__(
        self,
        ledger: Ledger,
        *,
        counts: BinomialCountsRecord,
        out_id: str,
        prior_null: Tuple[float, float] = (0.5, 0.5),
        prior_alt_A: Tuple[float, float] = (0.5, 0.5),
        prior_alt_B: Tuple[float, float] = (1.0, 0.5),
    ):
        super().__init__(
            ledger,
            counts=counts,
            out_id=out_id,
            prior_null=prior_null,
            prior_alt_A=prior_alt_A,
            prior_alt_B=prior_alt_B,
        )

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"eproc": EProcessRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.eproc
        counts: BinomialCountsRecord = self.counts
        prior_null = tuple(getattr(self, "prior_null"))
        prior_alt_A = tuple(getattr(self, "prior_alt_A"))
        prior_alt_B = tuple(getattr(self, "prior_alt_B"))

        cdf = (
            counts.latest()
            .select(
                nA=counts.t.payload["nA"].cast("int64"),
                mA=counts.t.payload["mA"].cast("int64"),
                nB=counts.t.payload["nB"].cast("int64"),
                mB=counts.t.payload["mB"].cast("int64"),
                look=counts.t.payload["look"].cast("int64"),
            )
            .execute()
        )
        if len(cdf) == 0:
            return

        nA, mA, nB, mB = map(int, cdf.iloc[0][["nA", "mA", "nB", "mB"]])
        E = mixture_e_two_proportions_skew(
            nA,
            mA,
            nB,
            mB,
            prior_null=prior_null,
            prior_alt_A=prior_alt_A,
            prior_alt_B=prior_alt_B,
        )
        payload: Dict[str, Any] = {
            "E": float(E),
            "logE": float(math.log(max(E, 1e-300))),
        }
        payload["look"] = int(cdf.iloc[0]["look"])
        payload["priors"] = {
            "null": list(prior_null),
            "alt_A": list(prior_alt_A),
            "alt_B": list(prior_alt_B),
        }
        out.insert(payload)
