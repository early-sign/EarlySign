"""Common utilities for two-proportions testing.

This module provides shared calculation utilities for two-proportion tests
that can be used across both design planning and runtime analysis.
"""

import math
from typing import Tuple


def compute_proportions(nA: int, mA: int, nB: int, mB: int) -> Tuple[float, float]:
    """Compute sample proportions with validation.

    Parameters
    ----------
    nA : int
        Sample size for group A (control)
    mA : int
        Number of successes in group A
    nB : int
        Sample size for group B (treatment)
    mB : int
        Number of successes in group B

    Returns
    -------
    pA, pB : tuple of float
        Sample proportions for groups A and B

    Raises
    ------
    ValueError
        If sample sizes are non-positive or successes exceed sample sizes

    Examples
    --------
    >>> pA, pB = compute_proportions(100, 40, 100, 55)
    >>> round(pA, 2), round(pB, 2)
    (0.4, 0.55)
    """
    if nA <= 0:
        raise ValueError("nA must be positive.")
    if nB <= 0:
        raise ValueError("nB must be positive.")
    if not (0 <= mA <= nA):
        raise ValueError("mA must be in [0, nA].")
    if not (0 <= mB <= nB):
        raise ValueError("mB must be in [0, nB].")

    return float(mA) / float(nA), float(mB) / float(nB)


def compute_pooled_variance(nA: int, mA: int, nB: int, mB: int) -> Tuple[float, float]:
    """Compute pooled proportion and its variance for two-proportion test.

    Uses the pooled estimator under H0: pA = pB.

    Parameters
    ----------
    nA, mA : int
        Sample size and successes for group A
    nB, mB : int
        Sample size and successes for group B

    Returns
    -------
    p_pooled : float
        Pooled proportion estimate
    var_pooled : float
        Pooled variance estimate

    Examples
    --------
    >>> p_pool, var_pool = compute_pooled_variance(100, 40, 100, 55)
    >>> round(p_pool, 3)
    0.475
    >>> var_pool > 0
    True
    """
    pA, pB = compute_proportions(nA, mA, nB, mB)
    p_pooled = (mA + mB) / float(nA + nB)
    var_pooled = p_pooled * (1.0 - p_pooled) * (1.0 / nA + 1.0 / nB)
    return p_pooled, var_pooled


def compute_unpooled_variance(nA: int, mA: int, nB: int, mB: int) -> float:
    """Compute unpooled variance for two-proportion test.

    Uses separate variance estimates for each group.

    Parameters
    ----------
    nA, mA : int
        Sample size and successes for group A
    nB, mB : int
        Sample size and successes for group B

    Returns
    -------
    var_unpooled : float
        Unpooled variance estimate

    Examples
    --------
    >>> var = compute_unpooled_variance(100, 40, 100, 55)
    >>> var > 0
    True
    """
    pA, pB = compute_proportions(nA, mA, nB, mB)
    return pA * (1.0 - pA) / nA + pB * (1.0 - pB) / nB


def compute_wald_z(nA: int, mA: int, nB: int, mB: int, pooled: bool = True) -> float:
    """Compute Wald Z-statistic for difference in proportions.

    Parameters
    ----------
    nA, mA : int
        Sample size and successes for group A (control)
    nB, mB : int
        Sample size and successes for group B (treatment)
    pooled : bool, default True
        Whether to use pooled variance estimate

    Returns
    -------
    z : float
        Wald Z-statistic for (pB - pA)

    Examples
    --------
    >>> z = compute_wald_z(100, 40, 100, 55, pooled=True)
    >>> round(z, 3)
    2.124

    >>> z_unpooled = compute_wald_z(100, 40, 100, 55, pooled=False)
    >>> abs(z_unpooled - z) < 0.1  # Should be similar
    True
    """
    pA, pB = compute_proportions(nA, mA, nB, mB)
    diff = pB - pA

    if pooled:
        _, var = compute_pooled_variance(nA, mA, nB, mB)
    else:
        var = compute_unpooled_variance(nA, mA, nB, mB)

    if var <= 0.0:
        return float("inf") if diff > 0 else float("-inf") if diff < 0 else 0.0

    return diff / math.sqrt(var)


def compute_standard_error(
    nA: int, nB: int, pA: float, pB: float, pooled: bool = True
) -> float:
    """Compute standard error for difference in proportions.

    This is useful for design calculations where proportions are assumed
    rather than observed.

    Parameters
    ----------
    nA, nB : int
        Sample sizes for groups A and B
    pA, pB : float
        Assumed proportions for groups A and B (in [0, 1])
    pooled : bool, default True
        Whether to use pooled variance estimate

    Returns
    -------
    se : float
        Standard error of (pB - pA)

    Examples
    --------
    >>> se = compute_standard_error(100, 100, 0.4, 0.55, pooled=True)
    >>> round(se, 4)
    0.0706

    >>> se_unpooled = compute_standard_error(100, 100, 0.4, 0.55, pooled=False)
    >>> abs(se_unpooled - se) < 0.01
    True
    """
    if not (0.0 <= pA <= 1.0 and 0.0 <= pB <= 1.0):
        raise ValueError("Proportions must be in [0, 1].")
    if nA <= 0 or nB <= 0:
        raise ValueError("Sample sizes must be positive.")

    if pooled:
        p_pooled = (pA + pB) / 2.0
        var = p_pooled * (1.0 - p_pooled) * (1.0 / nA + 1.0 / nB)
    else:
        var = pA * (1.0 - pA) / nA + pB * (1.0 - pB) / nB

    return math.sqrt(var)
