"""Pure statistical functions for Z-tests (Binomial and Continuous).

This module contains the mathematical logic for calculating Z-statistics,
decoupled from the EarlySign framework entities.
"""

from typing import Optional

import numpy as np


def calculate_two_arm_binomial_z(
    n_c: int,
    successes_c: int,
    n_t: int,
    successes_t: int,
    pooled: bool = True,
) -> Optional[float]:
    """Calculate Z-statistic for two-arm binomial comparison.

    Args:
        n_c: Total samples in control arm.
        successes_c: Number of successes in control arm.
        n_t: Total samples in treatment arm.
        successes_t: Number of successes in treatment arm.
        pooled: Whether to use pooled variance (Score test) or unpooled (Wald).

    Returns:
        Z-statistic (float) or None if calculation is invalid (e.g., n=0).
    """
    if n_c == 0 or n_t == 0:
        return None

    p_c = successes_c / n_c
    p_t = successes_t / n_t

    if pooled:
        p_pool = (successes_c + successes_t) / (n_c + n_t)
        if p_pool <= 0 or p_pool >= 1.0:
            return 0.0
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    else:
        # Unpooled (Wald)
        se_c2 = (p_c * (1 - p_c)) / n_c if 0 < p_c < 1 else 0.0
        se_t2 = (p_t * (1 - p_t)) / n_t if 0 < p_t < 1 else 0.0
        se = np.sqrt(se_c2 + se_t2)

    if se <= 0:
        return 0.0

    return float((p_t - p_c) / se)


def calculate_two_arm_continuous_z(
    n_c: int,
    mean_c: float,
    var_c: float,
    n_t: int,
    mean_t: float,
    var_t: float,
    pooled: bool = False,
) -> Optional[float]:
    """Calculate Z-statistic for two-arm continuous comparison.

    Args:
        n_c: Total samples in control arm.
        mean_c: Mean of control arm.
        var_c: Variance of control arm.
        n_t: Total samples in treatment arm.
        mean_t: Mean of treatment arm.
        var_t: Variance of treatment arm.
        pooled: Whether to use pooled variance estimate.

    Returns:
        Z-statistic (float) or None if calculation is invalid (e.g., n < 2).
    """
    if n_c < 2 or n_t < 2:
        return None

    if pooled:
        # Pooled variance estimate
        v_pool = ((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)
        se = np.sqrt(v_pool * (1 / n_c + 1 / n_t))
    else:
        # Unpooled (Welch)
        se = np.sqrt(var_c / n_c + var_t / n_t)

    if se <= 0:
        return 0.0

    return float((mean_t - mean_c) / se)
