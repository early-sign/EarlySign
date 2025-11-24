"""Mixture-evidence e-process helpers for two-proportion experiments.

These utilities compute beta-binomial Bayes factors that form e-processes for
anytime-valid inference. Two variants are provided:

- ``mixture_e_two_proportions``: symmetric alternative using the same Beta
  prior for both arms.
- ``mixture_e_two_proportions_skew``: asymmetric alternative allowing different
  priors per arm to emphasise one-sided effects.
"""

import math
from typing import Tuple

from scipy.special import betaln

BetaPrior = Tuple[float, float]


def _validate_counts(nA: int, mA: int, nB: int, mB: int) -> None:
    """Ensure counts are internally consistent (allows zero samples)."""

    for label, n, m in (("A", nA, mA), ("B", nB, mB)):
        if n < 0:
            raise ValueError(f"n{label} must be non-negative.")
        if not (0 <= m <= n):
            raise ValueError(f"Counts for arm {label} must satisfy 0 <= m <= n.")


def _log_beta_marginal(prior: BetaPrior, successes: int, failures: int) -> float:
    a, b = map(float, prior)
    return float(betaln(a + successes, b + failures) - betaln(a, b))


def mixture_e_two_proportions(
    nA: int,
    mA: int,
    nB: int,
    mB: int,
    *,
    prior_null: BetaPrior = (0.5, 0.5),
    prior_alt: BetaPrior = (0.5, 0.5),
) -> float:
    """
    Compute the mixture-evidence e-value for two-proportions (symmetric alt).

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

    _validate_counts(nA, mA, nB, mB)
    log_alt = _log_beta_marginal(prior_alt, mA, nA - mA) + _log_beta_marginal(
        prior_alt, mB, nB - mB
    )
    log_null = _log_beta_marginal(prior_null, mA + mB, (nA + nB) - (mA + mB))
    return float(math.exp(log_alt - log_null))


def mixture_e_two_proportions_skew(
    nA: int,
    mA: int,
    nB: int,
    mB: int,
    *,
    prior_null: BetaPrior = (0.5, 0.5),
    prior_alt_A: BetaPrior = (0.5, 0.5),
    prior_alt_B: BetaPrior = (1.0, 0.5),
) -> float:
    """
    Compute a skewed-alt mixture e-value for two-proportions.

    The alternative uses independent Beta priors with different shapes for A
    and B, allowing one-sided emphasis (e.g., B > A).

    Examples
    --------
    >>> round(mixture_e_two_proportions_skew(100, 40, 120, 70), 6)
    5.154707
    """

    _validate_counts(nA, mA, nB, mB)
    log_alt = _log_beta_marginal(prior_alt_A, mA, nA - mA) + _log_beta_marginal(
        prior_alt_B, mB, nB - mB
    )
    log_null = _log_beta_marginal(prior_null, mA + mB, (nA + nB) - (mA + mB))
    return float(math.exp(log_alt - log_null))
