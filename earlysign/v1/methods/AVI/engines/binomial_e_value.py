from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from earlysign.schema.ES3.AVI.Log import DecisionStatus, LookResult
from earlysign.schema.ES3.Binomial import Scoreboard as BinomialScoreboard


class EValueResult(BaseModel):
    """Result of an e-value calculation."""

    e_value: float
    is_rejected: bool
    alpha: float


class EProcessProtocol(BaseModel):
    """Protocol for continuous monitoring based on E-processes.

    E-processes allow for anytime-valid testing, where a rejection at any point
    (without a fixed schedule) is scientifically valid.
    """

    alpha: float = Field(0.05, description="Type-1 error rate")
    null_p: float = Field(
        0.5, description="Success probability hypothesized under the Null (H0)."
    )
    alt_p: float = Field(
        ..., description="Success probability hypothesized under the Alternative (H1)."
    )


def compute_binomial_e_value(
    n: int, successes: int, null_p: float, alt_p: float, alpha: float = 0.05
) -> EValueResult:
    r"""Computes a simple likelihood ratio e-value for a binomial test.

    Formula: $E = (alt\_p / null\_p)^S * ((1-alt\_p) / (1-null\_p))^{(n-S)}$

    Args:
        n: Total number of trials.
        successes: Number of successes observed.
        null_p: Probability of success under the null hypothesis.
        alt_p: Probability of success under the alternative hypothesis.
        alpha: Significant level for rejection boundary (1/alpha).

    Returns:
        An `EValueResult` containing the calculated e-value and rejection status.

    Examples:
        >>> res = compute_binomial_e_value(100, 60, 0.5, 0.6)
        >>> res.e_value > 1.0
        True
        >>> res = compute_binomial_e_value(100, 40, 0.5, 0.6)
        >>> res.e_value < 1.0
        True
    """
    # Likelihood under H1 / Likelihood under H0
    if n == 0:
        return EValueResult(e_value=1.0, is_rejected=False, alpha=alpha)

    # Avoid division by zero or log of zero
    null_p = max(min(null_p, 1 - 1e-10), 1e-10)
    alt_p = max(min(alt_p, 1 - 1e-10), 1e-10)

    log_e = successes * np.log(alt_p / null_p) + (n - successes) * np.log(
        (1 - alt_p) / (1 - null_p)
    )

    e_val = float(np.exp(log_e))
    is_rejected = e_val > (1.0 / alpha)

    return EValueResult(e_value=e_val, is_rejected=is_rejected, alpha=alpha)


class BinomialEValueEngine:
    """Engine for 1-sample Binomial E-value monitoring."""

    def __init__(self, protocol: EProcessProtocol):
        self.protocol = protocol

    def run(self, metrics: BinomialScoreboard, **kwargs: Any) -> LookResult:
        """
        Run the engine on aggregated metrics.
        Note: This engine assumes a single aggregated success/n stream
        across all arms if multiple are present, or a specific arm if provided.
        """
        # Sum across all arms for 1-sample test on total success rate
        n_total = sum(a.metrics.n for a in metrics.arms.values())
        s_total = sum(a.metrics.successes for a in metrics.arms.values())

        res = compute_binomial_e_value(
            n=n_total,
            successes=s_total,
            null_p=self.protocol.null_p,
            alt_p=self.protocol.alt_p,
            alpha=self.protocol.alpha,
        )

        status = DecisionStatus.CONTINUE_
        if res.is_rejected:
            status = DecisionStatus.STOP_EFFICACY

        return LookResult(
            sample_n=n_total,
            trajectory=res.e_value,
            boundary=1.0 / self.protocol.alpha,
            is_crossed=res.is_rejected,
            status=status,
        )
