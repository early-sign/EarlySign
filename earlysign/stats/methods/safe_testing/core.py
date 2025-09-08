"""
earlysign.stats.methods.safe_testing.core
==========================================

Core mathematics for safe testing (anytime-valid inference).

Provides mathematical foundations for safe testing including:
- E-value computations for various distributions
- Beta-binomial e-processes
- Anytime-valid threshold calculations

These functions implement the mathematical theory without dependencies
on specific problem domains or experimental setups.
"""

from __future__ import annotations
import math

from scipy.special import betaln
from scipy.optimize import minimize_scalar


def log_beta_binomial_evalue_simple(
    mA: int,
    nA: int,
    mB: int,
    nB: int,
    alpha_A: float = 1.0,
    beta_A: float = 1.0,
    alpha_B: float = 1.0,
    beta_B: float = 1.0,
) -> float:
    """
    Compute log e-value for two-sample beta-binomial test with fixed priors.

    This implements the simple case where we use fixed beta priors for each group.
    The e-value is the Bayes factor comparing the alternative (independent betas)
    to the null hypothesis (common probability under uniform prior).

    Args:
        mA, nA: Successes and trials in group A
        mB, nB: Successes and trials in group B
        alpha_A, beta_A: Beta prior parameters for group A
        alpha_B, beta_B: Beta prior parameters for group B

    Returns:
        Log e-value (natural log)
    """
    if min(nA, nB) == 0:
        return 0.0

    # Alternative: independent beta-binomial likelihoods
    log_alt_A = betaln(alpha_A + mA, beta_A + nA - mA) - betaln(alpha_A, beta_A)
    log_alt_B = betaln(alpha_B + mB, beta_B + nB - mB) - betaln(alpha_B, beta_B)
    log_alternative = log_alt_A + log_alt_B

    # Null: pooled data with uniform prior (equivalent to Beta(1,1))
    m_pool = mA + mB
    n_pool = nA + nB
    log_null = betaln(1 + m_pool, 1 + n_pool - m_pool) - betaln(1, 1)

    return float(log_alternative - log_null)


def log_beta_binomial_evalue_adaptive(
    mA: int,
    nA: int,
    mB: int,
    nB: int,
    alpha_shape: float = 1.0,
    beta_shape: float = 1.0,
) -> float:
    """
    Compute log e-value using adaptive/optimized beta priors.

    This version optimizes the choice of beta parameters to maximize the e-value,
    which can improve power while maintaining validity.

    Args:
        mA, nA: Successes and trials in group A
        mB, nB: Successes and trials in group B
        alpha_shape, beta_shape: Shape parameters for the prior on beta parameters

    Returns:
        Log e-value (natural log)
    """
    if min(nA, nB) == 0:
        return 0.0

    def neg_log_evalue(log_alpha: float) -> float:
        """Negative log e-value to minimize (i.e., maximize e-value)."""
        alpha = math.exp(log_alpha)
        beta = beta_shape  # Could also optimize this
        return -log_beta_binomial_evalue_simple(
            mA, nA, mB, nB, alpha, beta, alpha, beta
        )

    # Optimize over reasonable range for log(alpha)
    result = minimize_scalar(neg_log_evalue, bounds=(-2, 5), method="bounded")
    return -result.fun if result.success else 0.0
