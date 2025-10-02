"""
Alpha-spending utilities (Lan–DeMets style) and nominal Z mapping.

- obf_spending(t, alpha, tails=2)
- pocock_spending(t, alpha)
- cumulative_to_nominal_z(spent_alpha, tails=2) -> (upper_z, lower_z)
- z_to_brownian(z, t) -> z * sqrt(t)  (map Z-scale boundary to Brownian scale)
"""

import math
from typing import Tuple

from scipy.stats import norm


def obf_spending(t: float, alpha: float, tails: int = 2) -> float:
    """
    Lan–DeMets O'Brien–Fleming cumulative spending α(t).

    Parameters
    ----------
    t : float in [0,1]
        Information time.
    alpha : float in (0,1)
        Overall type-I error.
    tails : {1,2}
        One- or two-sided.

    Notes
    -----
    This implements the *continuous-time* Lan–DeMets OBF approximation:
        α(t) = 2 - 2 Φ(z_{1-α/2} / sqrt(t))  (two-sided)
        α(t) = 1 -   Φ(z_{1-α}   / sqrt(t))  (one-sided)
    which is more stable than discrete K-look allocations.

    Examples
    --------
    >>> round(obf_spending(t=0.25, alpha=0.05, tails=2), 6)
    8.9e-05
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("`t` must be in [0, 1].")
    if not (0.0 < alpha < 1.0):
        raise ValueError("`alpha` must be in (0, 1).")
    if tails not in (1, 2):
        raise ValueError("`tails` must be 1 or 2.")
    # To avoid division by zero at t=0
    t = max(t, 1e-12)

    if norm is None:
        # Very coarse fallback (NOT a faithful OBF spending; dev use only)
        return alpha * min(1.0, math.sqrt(t))

    if tails == 2:
        z = float(norm.isf(alpha / 2.0))
        return float(2.0 - 2.0 * norm.cdf(z / math.sqrt(t)))
    else:
        z = float(norm.isf(alpha))
        return float(1.0 - norm.cdf(z / math.sqrt(t)))


def pocock_spending(t: float, alpha: float) -> float:
    """
    Pocock cumulative spending α(t) ≈ α * ln(1 + (e - 1) t).

    Parameters
    ----------
    t : float in [0,1]
    alpha : float in (0,1)

    Examples
    --------
    >>> round(pocock_spending(t=0.5, alpha=0.05), 6)  # doctest: +ELLIPSIS
    0.03...
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("`t` must be in [0, 1].")
    if not (0.0 < alpha < 1.0):
        raise ValueError("`alpha` must be in (0, 1).")
    return float(alpha * math.log(1.0 + (math.e - 1.0) * t))


def cumulative_to_nominal_z(
    spent_alpha_cum: float, *, tails: int = 2
) -> Tuple[float, float]:
    """
    Map cumulative spent alpha to symmetric nominal Z boundaries.

    Parameters
    ----------
    spent_alpha_cum : float
        Cumulative alpha spent up to current information time.
    tails : {1,2}

    Returns
    -------
    (upper_z, lower_z)

    Examples
    --------
    >>> up, lo = cumulative_to_nominal_z(0.01, tails=2)
    >>> round(up, 3), round(lo, 3)
    (2.576, -2.576)
    """
    if spent_alpha_cum <= 0.0 or norm is None:
        # No alpha spent (or SciPy missing): infinite efficacy and no futility
        return float("inf"), float("-inf") if tails == 2 else float("-inf")
    if tails == 2:
        z = float(norm.isf(spent_alpha_cum / 2.0))
        return z, -z
    elif tails == 1:
        z = float(norm.isf(spent_alpha_cum))
        return z, float("-inf")
    else:
        raise ValueError("`tails` must be 1 or 2.")


def z_to_brownian(z: float, t: float) -> float:
    """
    Convert a Z-scale boundary to Brownian scale: B(t) = Z * sqrt(t).

    Examples
    --------
    >>> round(z_to_brownian(2.0, 0.25), 3)
    1.0
    """
    if t < 0.0 or t > 1.0:
        raise ValueError("`t` must be in [0, 1].")
    return float(z * math.sqrt(max(t, 0.0)))
