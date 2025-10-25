"""
Information time calculation functions for group sequential designs.

This module provides pure functions for calculating information time t ∈ [0, 1],
which represents the proportion of planned information accrued at a given point.

Information time is fundamental to group sequential testing:
- Determines where we are on the spending function α(t)
- Used to compute boundaries at each analysis
- Can be based on sample size, events, variance, or Fisher information

Functions
---------
Sample-based:
    info_time_from_sample_size(n_current, n_max) -> float
    info_time_from_samples_by_group(n_control, n_treatment, n_max_control, n_max_treatment) -> float

Ratio-based (general):
    info_time_from_ratio(info_now, info_max) -> float

Variance-based:
    info_time_from_variance(var_now, var_target) -> float
    info_time_from_sd(sd_now, sd_target) -> float

Fisher information-based:
    info_time_from_fisher(fisher_now, fisher_max) -> float

Examples
--------
>>> # Sample-based information time
>>> t = info_time_from_sample_size(n_current=500, n_max=2000)
>>> t
0.25

>>> # Variance-based (information ∝ 1/variance)
>>> t = info_time_from_variance(var_now=0.04, var_target=0.01)
>>> t
0.25

>>> # Create equally-spaced schedule
>>> import numpy as np
>>> times = choose_t_equally_spaced(n_looks=3)
>>> np.round(times, 2)
array([0.33, 0.67, 1.  ])
"""

from earlysign.stats.common.group_sequential.essentials.conversions import clip01

# =============================================================================
# Core Information Time Calculations
# =============================================================================


def info_time_from_sample_size(n_current: int, n_max: int) -> float:
    """
    Calculate information time from sample sizes.

    Most common method: t = n_current / n_max

    Parameters
    ----------
    n_current : int
        Current total sample size.
    n_max : int
        Planned maximum total sample size.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> info_time_from_sample_size(n_current=100, n_max=400)
    0.25

    >>> info_time_from_sample_size(n_current=400, n_max=400)
    1.0

    >>> # Over-recruitment is clipped to 1.0
    >>> info_time_from_sample_size(n_current=500, n_max=400)
    1.0
    """
    if n_max <= 0:
        raise ValueError(f"n_max must be positive, got {n_max}")
    if n_current < 0:
        raise ValueError(f"n_current must be non-negative, got {n_current}")

    return clip01(float(n_current) / float(n_max))


def info_time_from_samples_by_group(
    n_control: int, n_treatment: int, n_max_control: int, n_max_treatment: int
) -> float:
    """
    Calculate information time from per-group sample sizes.

    For unequal allocation ratios, information time is based on the
    harmonic mean of the two groups' information times (approximately).

    Parameters
    ----------
    n_control : int
        Current control group sample size.
    n_treatment : int
        Current treatment group sample size.
    n_max_control : int
        Planned maximum control group sample size.
    n_max_treatment : int
        Planned maximum treatment group sample size.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> # Equal allocation, equal accrual
    >>> info_time_from_samples_by_group(50, 50, 200, 200)
    0.25

    >>> # Unequal allocation (1:2 ratio)
    >>> info_time_from_samples_by_group(50, 100, 200, 400)
    0.25

    Notes
    -----
    Information for comparing two groups depends on both sample sizes.
    For proportions with equal allocation:
        I(t) ∝ n_control(t) · n_treatment(t) / (n_control(t) + n_treatment(t))

    This function uses the simpler total sample size approach:
        t = (n_control + n_treatment) / (n_max_control + n_max_treatment)
    """
    n_current = n_control + n_treatment
    n_max = n_max_control + n_max_treatment
    return info_time_from_sample_size(n_current, n_max)


def info_time_from_ratio(info_now: float, info_max: float) -> float:
    """
    Generic information time from any information metric ratio.

    Use when you have a direct measure of information (Fisher information,
    precision, number of events, etc.) and want t = info_now / info_max.

    Parameters
    ----------
    info_now : float
        Current information level.
    info_max : float
        Maximum planned information level.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> # Event-based (survival analysis)
    >>> info_time_from_ratio(info_now=75, info_max=300)
    0.25

    >>> # Fisher information
    >>> info_time_from_ratio(info_now=1000.0, info_max=4000.0)
    0.25
    """
    if info_max <= 0:
        raise ValueError(f"info_max must be positive, got {info_max}")
    if info_now < 0:
        raise ValueError(f"info_now must be non-negative, got {info_now}")

    return clip01(float(info_now) / float(info_max))


def info_time_from_variance(var_now: float, var_target: float) -> float:
    """
    Calculate information time from variance (information ∝ 1/variance).

    When variance decreases with sample size, information increases.
    t = var_target / var_now (inverse relationship).

    Parameters
    ----------
    var_now : float
        Current variance estimate.
    var_target : float
        Target (final) variance.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> # Variance decreases with sample size
    >>> info_time_from_variance(var_now=0.04, var_target=0.01)
    0.25

    >>> # At target variance, t=1
    >>> info_time_from_variance(var_now=0.01, var_target=0.01)
    1.0

    Notes
    -----
    Variance-based information time is particularly useful when:
    - Sample sizes vary by group or time
    - Adaptive designs adjust allocation ratios
    - Precision is the primary endpoint consideration
    """
    if var_target <= 0:
        raise ValueError(f"var_target must be positive, got {var_target}")
    if var_now <= 0:
        raise ValueError(f"var_now must be positive, got {var_now}")

    return clip01(float(var_target) / float(var_now))


def info_time_from_sd(sd_now: float, sd_target: float) -> float:
    """
    Calculate information time from standard deviation.

    Converts SD to variance and uses info_time_from_variance.
    t = (sd_target / sd_now)²

    Parameters
    ----------
    sd_now : float
        Current standard deviation estimate.
    sd_target : float
        Target (final) standard deviation.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> # SD decreases from 0.2 to 0.1 (variance 0.04 → 0.01)
    >>> info_time_from_sd(sd_now=0.2, sd_target=0.1)
    0.25

    >>> info_time_from_sd(sd_now=0.1, sd_target=0.1)
    1.0
    """
    if sd_target <= 0:
        raise ValueError(f"sd_target must be positive, got {sd_target}")
    if sd_now <= 0:
        raise ValueError(f"sd_now must be positive, got {sd_now}")

    var_now = sd_now**2
    var_target = sd_target**2
    return info_time_from_variance(var_now, var_target)


def info_time_from_fisher(fisher_now: float, fisher_max: float) -> float:
    """
    Calculate information time from Fisher information.

    Fisher information is a direct measure of statistical information.
    t = fisher_now / fisher_max

    Parameters
    ----------
    fisher_now : float
        Current Fisher information.
    fisher_max : float
        Maximum planned Fisher information.

    Returns
    -------
    float
        Information time in [0, 1].

    Examples
    --------
    >>> info_time_from_fisher(fisher_now=250.0, fisher_max=1000.0)
    0.25
    """
    return info_time_from_ratio(fisher_now, fisher_max)
