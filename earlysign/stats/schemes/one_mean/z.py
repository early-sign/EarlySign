"""Z-statistic helpers for one-mean tests with known variance."""

import math


def compute_z_mean_known_var(n: int, mean: float, sigma2: float) -> float:
    """
    Z for testing mu=0 with known variance sigma^2 (two-sided by default upstream).

    Z = mean * sqrt(n) / sqrt(sigma^2)

    Examples
    --------
    >>> round(compute_z_mean_known_var(100, 0.2, 1.0), 3)
    2.0
    """

    if n <= 0:
        raise ValueError("n must be positive.")
    if sigma2 <= 0.0:
        raise ValueError("sigma^2 must be positive.")
    return float(mean) * math.sqrt(float(n) / float(sigma2))
