"""
Fixed Sample Design Calculator
==============================

Provides utilities for calculating comparable fixed sample sizes for A/B testing,
serving as a baseline for Group Sequential Designs.
"""

import numpy as np
from scipy.stats import norm


def calculate_fixed_sample_size(
    p_control: float,
    p_treatment: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    sided: int = 1,
) -> int:
    """
    Calculates the Total Sample Size (N) for a fixed design Binomial A/B test
    using the pooled variance approximation (standard Z-test).

    Formula:
        N ~ 4 * sigma^2 * (Z_alpha + Z_beta)^2 / delta^2

    Args:
        p_control: Control arm proportion.
        p_treatment: Treatment arm proportion.
        alpha: Type I error rate (default 0.05).
        power: Power (1 - beta) (default 0.8).
        ratio: Allocation ratio n_t / n_c (default 1.0).
               Currently only 1:1 (ratio=1.0) is robustly supported by this formula.
        sided: 1 for one-sided, 2 for two-sided.

    Returns:
        Total sample size required (Control + Treatment).
    """
    if sided == 1:
        z_alpha = norm.ppf(1 - alpha)
    else:
        z_alpha = norm.ppf(1 - alpha / 2)

    z_beta = norm.ppf(power)

    # Pooled variance approximation
    p_pool = (p_control + p_treatment) / 2
    delta = abs(p_treatment - p_control)

    if delta == 0:
        raise ValueError("Delta cannot be zero for sample size calculation.")

    # Variance for difference of portions: sigma^2_diff = p(1-p)(1/n + 1/n) = 2p(1-p)/n
    # Standard formula for "n per arm":
    # n = 2 * p_pool * (1 - p_pool) * (z_alpha + z_beta)^2 / delta^2
    # Total N = 2 * n
    # So: Total N = 4 * p_pool * (1 - p_pool) * (z_alpha + z_beta)^2 / delta^2

    sigma_sq = p_pool * (1 - p_pool)

    n_total = 4 * sigma_sq * ((z_alpha + z_beta) / delta) ** 2

    return int(np.ceil(n_total))
