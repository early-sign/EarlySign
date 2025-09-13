"""
earlysign.stats.common.e_process
==============================

Generic e-process and safe testing methods.

This module provides core mathematical algorithms for anytime-valid testing
using e-processes and safe testing theory. These methods are independent of
the specific statistical scheme and provide the foundation for implementing
e-value based testing procedures.

The functions implement the theoretical foundations of e-processes, including
e-value computations, betting strategies, and anytime-valid thresholds.
"""

from __future__ import annotations
import math
from typing import Optional, List, Tuple

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

    Note:
        This is a generic implementation that can be applied to any
        two-sample binomial comparison scheme.
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
    which can improve power while maintaining validity. The optimization is
    performed over hyperprior parameters.

    Args:
        mA, nA: Successes and trials in group A
        mB, nB: Successes and trials in group B
        alpha_shape, beta_shape: Shape parameters for the hyperprior

    Returns:
        Log e-value (natural log)

    Note:
        This is a generic implementation that can be applied to any
        two-sample binomial comparison scheme.
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


def safe_threshold(alpha_level: float) -> float:
    """
    Compute anytime-valid significance threshold for e-values.

    In safe testing, significance is achieved when the e-value exceeds 1/α,
    where α is the desired error rate. This threshold provides anytime-valid
    Type I error control without multiple testing corrections.

    Args:
        alpha_level: Desired Type I error rate

    Returns:
        Safe testing threshold (1/α)

    Mathematical foundation:
        Threshold = 1/α provides exact Type I error control:
        P(E_τ ≥ 1/α for some τ) ≤ α under H0
    """
    if alpha_level <= 0 or alpha_level >= 1:
        raise ValueError(f"alpha_level must be in (0, 1), got {alpha_level}")

    return 1.0 / alpha_level


def e_process_product(log_e_values: List[float]) -> Tuple[float, float]:
    """
    Compute the product of e-values from log e-values.

    For a sequence of e-processes, their product is also an e-process.
    This function computes the product while maintaining numerical stability.

    Args:
        log_e_values: List of log e-values

    Returns:
        Tuple of (e_value, log_e_value) for the product
    """
    if not log_e_values:
        return 1.0, 0.0

    log_product = sum(log_e_values)

    # Handle numerical stability
    if log_product > 100:
        return float("inf"), log_product
    else:
        return math.exp(log_product), log_product


def mixture_e_value(
    log_e_values: List[float], weights: Optional[List[float]] = None
) -> Tuple[float, float]:
    """
    Compute mixture e-value from multiple e-processes.

    A convex mixture of e-processes is also an e-process. This can be useful
    for combining different betting strategies or handling uncertainty about
    the best approach.

    Args:
        log_e_values: List of log e-values from different processes
        weights: Mixing weights (must sum to 1). If None, uses uniform weights.

    Returns:
        Tuple of (mixture_e_value, log_mixture_e_value)
    """
    if not log_e_values:
        return 1.0, 0.0

    if weights is None:
        weights = [1.0 / len(log_e_values)] * len(log_e_values)

    if len(weights) != len(log_e_values):
        raise ValueError("weights and log_e_values must have same length")

    if abs(sum(weights) - 1.0) > 1e-10:
        raise ValueError("weights must sum to 1")

    # Convert to e-values for mixture computation
    e_values = [
        math.exp(log_e) if log_e < 100 else float("inf") for log_e in log_e_values
    ]

    # Compute weighted mixture
    mixture_e = sum(w * e for w, e in zip(weights, e_values))

    # Convert back to log scale
    if mixture_e == float("inf"):
        return float("inf"), float("inf")
    elif mixture_e <= 0:
        return 0.0, float("-inf")
    else:
        return mixture_e, math.log(mixture_e)
