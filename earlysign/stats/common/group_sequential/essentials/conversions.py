"""
Scale conversions and utility functions for group sequential testing.

This module provides pure functions for converting between different scales
and computing nominal boundaries from cumulative error spending.

Scales:
-------
- "z": Standardized Z-statistic scale
- "bm": Brownian motion scale B(t) = Z·√t

Functions
---------
Cumulative alpha to boundaries:
    cumulative_to_nominal_z(spent_alpha_cum, tails=2) -> (upper_z, lower_z)
    level_to_nominal_z(alpha_level, tails=2) -> (upper_z, lower_z)

Scale conversions:
    z_to_brownian(z, t) -> float
    brownian_to_z(b, t) -> float
    convert_scale(value, from_scale, to_scale, info_time) -> float

Examples
--------
>>> # Convert cumulative alpha to Z boundaries
>>> upper, lower = cumulative_to_nominal_z(0.01, tails=2)
>>> round(upper, 3), round(lower, 3)
(2.576, -2.576)

>>> # Convert Z to Brownian scale
>>> b = z_to_brownian(z=2.0, t=0.25)
>>> round(b, 3)
1.0

>>> # Convert back to Z
>>> z = brownian_to_z(b=1.0, t=0.25)
>>> round(z, 3)
2.0
"""

import math
from typing import Tuple

from scipy.stats import norm

# =============================================================================
# Cumulative Alpha/Level to Nominal Z Boundaries
# =============================================================================


def cumulative_to_nominal_z(
    spent_alpha_cum: float, *, tails: int = 2
) -> Tuple[float, float]:
    """
    Convert cumulative spent alpha to symmetric nominal Z boundaries.

    Given the cumulative alpha spent up to an information time, compute
    the Z-statistic boundaries that correspond to this spending level.

    Parameters
    ----------
    spent_alpha_cum : float
        Cumulative alpha spent up to current information time, in (0, 1).
    tails : {1, 2}, default=2
        Number of tails for the test.

    Returns
    -------
    upper_z : float
        Upper (efficacy) boundary on Z scale.
    lower_z : float
        Lower (futility) boundary on Z scale. For two-sided tests, this
        is the symmetric negative of upper_z. For one-sided tests, this
        is -inf (no lower bound).

    Examples
    --------
    >>> # Two-sided boundaries for alpha=0.01
    >>> upper, lower = cumulative_to_nominal_z(0.01, tails=2)
    >>> round(upper, 3), round(lower, 3)
    (2.576, -2.576)

    >>> # One-sided boundary for alpha=0.025
    >>> upper, lower = cumulative_to_nominal_z(0.025, tails=1)
    >>> round(upper, 3)
    1.96
    >>> lower
    -inf

    >>> # Very small alpha (early in OBF design)
    >>> upper, lower = cumulative_to_nominal_z(0.0001, tails=2)
    >>> round(upper, 3)
    3.891

    Notes
    -----
    For two-sided tests:
        upper_z = Φ^{-1}(1 - α/2)
        lower_z = -upper_z

    For one-sided tests:
        upper_z = Φ^{-1}(1 - α)
        lower_z = -∞
    """
    if spent_alpha_cum <= 0.0:
        # No alpha spent: infinite efficacy, no futility
        return float("inf"), float("-inf") if tails == 2 else float("-inf")

    if tails == 2:
        z = float(norm.isf(spent_alpha_cum / 2.0))
        return z, -z
    elif tails == 1:
        z = float(norm.isf(spent_alpha_cum))
        return z, float("-inf")
    else:
        raise ValueError(f"Tails must be 1 or 2, got {tails}")


def level_to_nominal_z(alpha_level: float, *, tails: int = 2) -> Tuple[float, float]:
    """
    Convert per-look significance level to symmetric Z boundaries.

    This is similar to cumulative_to_nominal_z but emphasizes that the
    input is a per-look (nominal) significance level rather than cumulative
    spending. The computation is identical.

    Parameters
    ----------
    alpha_level : float
        Per-look significance level in (0, 1).
    tails : {1, 2}, default=2
        Number of tails for the test.

    Returns
    -------
    upper_z : float
        Upper boundary on Z scale.
    lower_z : float
        Lower boundary on Z scale.

    Examples
    --------
    >>> # Common per-look levels
    >>> upper, lower = level_to_nominal_z(0.01, tails=2)
    >>> round(upper, 3), round(lower, 3)
    (2.576, -2.576)

    >>> upper, lower = level_to_nominal_z(0.05, tails=2)
    >>> round(upper, 3), round(lower, 3)
    (1.96, -1.96)
    """
    if alpha_level <= 0.0 or alpha_level >= 1.0:
        raise ValueError(f"Alpha level must be in (0, 1), got {alpha_level}")
    if tails not in (1, 2):
        raise ValueError(f"Tails must be 1 or 2, got {tails}")

    return cumulative_to_nominal_z(alpha_level, tails=tails)


# =============================================================================
# Scale Conversions: Z ↔ Brownian Motion
# =============================================================================


def z_to_brownian(z: float, t: float) -> float:
    """
    Convert Z-statistic to Brownian motion scale.

    The Brownian motion representation is B(t) = Z·√t, where t is the
    information time. This scale is useful for certain boundary calculations
    and visualizations.

    Parameters
    ----------
    z : float
        Z-statistic value (standardized).
    t : float
        Information time in [0, 1].

    Returns
    -------
    float
        Value on Brownian motion scale.

    Examples
    --------
    >>> # Z=2 at 25% information
    >>> b = z_to_brownian(z=2.0, t=0.25)
    >>> round(b, 3)
    1.0

    >>> # Z=3 at 100% information
    >>> b = z_to_brownian(z=3.0, t=1.0)
    >>> round(b, 3)
    3.0

    >>> # At t=0, any finite Z maps to 0
    >>> z_to_brownian(z=100.0, t=0.0)
    0.0

    Notes
    -----
    Under the null hypothesis with no drift, B(t) ~ N(0, t).
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Information time t must be in [0, 1], got {t}")

    return float(z) * math.sqrt(max(t, 0.0))


def brownian_to_z(b: float, t: float) -> float:
    """
    Convert Brownian motion value to Z-statistic scale.

    Inverse of z_to_brownian: Z = B(t) / √t.

    Parameters
    ----------
    b : float
        Value on Brownian motion scale.
    t : float
        Information time in (0, 1]. Must be positive to avoid division by zero.

    Returns
    -------
    float
        Z-statistic value.

    Examples
    --------
    >>> # B=1.0 at 25% information
    >>> z = brownian_to_z(b=1.0, t=0.25)
    >>> round(z, 3)
    2.0

    >>> # B=3.0 at 100% information
    >>> z = brownian_to_z(b=3.0, t=1.0)
    >>> round(z, 3)
    3.0

    Notes
    -----
    Division by √t can be unstable near t=0. Use with caution for very
    small information times.
    """
    if not (0.0 < t <= 1.0):
        raise ValueError(f"Information time t must be in (0, 1], got {t}")

    # Use small epsilon to avoid true division by zero
    denom = math.sqrt(max(t, 1e-12))
    return float(b) / denom


def convert_scale(
    value: float, *, from_scale: str, to_scale: str, info_time: float
) -> float:
    """
    Convert a statistic between "z" and "bm" (Brownian motion) scales.

    This is a general-purpose converter that handles both directions:
    - z → bm: multiply by √t
    - bm → z: divide by √t
    - z → z or bm → bm: identity

    Parameters
    ----------
    value : float
        Input statistic value.
    from_scale : {"z", "bm"}
        Scale of the input value.
    to_scale : {"z", "bm"}
        Desired output scale.
    info_time : float
        Information time in [0, 1]. For bm→z conversion, must be > 0.

    Returns
    -------
    float
        Converted value on to_scale.

    Examples
    --------
    >>> # Z to Brownian
    >>> b = convert_scale(2.0, from_scale="z", to_scale="bm", info_time=0.25)
    >>> round(b, 3)
    1.0

    >>> # Brownian to Z
    >>> z = convert_scale(1.0, from_scale="bm", to_scale="z", info_time=0.25)
    >>> round(z, 3)
    2.0

    >>> # Identity conversion
    >>> z = convert_scale(1.23, from_scale="z", to_scale="z", info_time=0.7)
    >>> round(z, 3)
    1.23

    Raises
    ------
    ValueError
        If scales are not recognized or info_time is out of range.
    """
    fs = str(from_scale).lower()
    ts = str(to_scale).lower()

    if fs not in ("z", "bm"):
        raise ValueError(f"from_scale must be 'z' or 'bm', got {fs}")
    if ts not in ("z", "bm"):
        raise ValueError(f"to_scale must be 'z' or 'bm', got {ts}")
    if not (0.0 <= info_time <= 1.0):
        raise ValueError(f"info_time must be in [0, 1], got {info_time}")

    # Identity conversion
    if fs == ts:
        return float(value)

    # z → bm
    if fs == "z" and ts == "bm":
        return z_to_brownian(value, info_time)

    # bm → z
    if fs == "bm" and ts == "z":
        return brownian_to_z(value, info_time)

    # Unreachable
    return float(value)


# =============================================================================
# Helper: Clip to [0, 1] range (for information time calculations)
# =============================================================================


def clip01(x: float) -> float:
    """
    Clip a float into [0, 1].

    Utility function for ensuring information times stay in valid range.

    Parameters
    ----------
    x : float
        Value to clip.

    Returns
    -------
    float
        Value clipped to [0, 1].

    Examples
    --------
    >>> clip01(-0.1)
    0.0
    >>> clip01(0.5)
    0.5
    >>> clip01(1.5)
    1.0
    """
    return max(0.0, min(1.0, float(x)))
