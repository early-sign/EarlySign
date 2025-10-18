"""Common utilities for two-means testing.

This module provides shared calculation utilities for two-sample t-tests
that can be used across both design planning and runtime analysis.
"""

import math


def compute_standard_error(
    nA: int, nB: int, sigma: float, pooled: bool = True
) -> float:
    """Compute standard error for difference in means.

    This is useful for design calculations where the common standard deviation
    is assumed rather than observed.

    Parameters
    ----------
    nA, nB : int
        Sample sizes for groups A and B
    sigma : float
        Assumed common standard deviation (pooled)
    pooled : bool, default True
        Whether to use pooled variance estimate (for this function, always True
        as we assume common sigma)

    Returns
    -------
    se : float
        Standard error of (muB - muA)

    Examples
    --------
    >>> se = compute_standard_error(100, 100, 1.0)
    >>> round(se, 4)
    0.1414

    >>> se2 = compute_standard_error(50, 100, 2.0)
    >>> round(se2, 4)
    0.3464
    """
    if sigma < 0:
        raise ValueError("Standard deviation must be non-negative.")
    if nA <= 0 or nB <= 0:
        raise ValueError("Sample sizes must be positive.")

    return sigma * math.sqrt(1.0 / nA + 1.0 / nB)
