"""
Alpha and beta spending functions for group sequential designs.

This module implements spending functions α(t) and β(t) that determine how
error is spent across sequential looks. These are pure functions matching
the functional design system.

Spending functions control the shape of boundaries:
- O'Brien-Fleming (OBF): Conservative early, liberal late
- Pocock: Uniform spending across looks
- Hwang-Shih-DeCani (HSD): Flexible family with shape parameter

Functions
---------
Alpha spending (Type I error):
    obf_spending(t, alpha, tails=2) -> float
    pocock_spending(t, alpha) -> float
    hsd_spending(t, alpha, gamma=-4.0) -> float

Beta spending (Type II error - NEW):
    beta_obf_spending(t, beta) -> float
    beta_pocock_spending(t, beta) -> float
    beta_hsd_spending(t, beta, gamma=-4.0) -> float

Examples
--------
>>> import numpy as np
>>> # O'Brien-Fleming spending at 50% information
>>> alpha = obf_spending(t=0.5, alpha=0.05, tails=2)
>>> round(alpha, 6)
0.005575

>>> # Pocock spending at 50% information
>>> alpha = pocock_spending(t=0.5, alpha=0.05)
>>> round(alpha, 6)
0.031006

>>> # Spending at multiple time points
>>> times = np.array([0.33, 0.67, 1.0])
>>> alphas = np.array([obf_spending(t, 0.05) for t in times])
>>> np.round(alphas, 6)
array([0.000645, 0.016644, 0.05    ])

References
----------
.. [1] Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries
       for clinical trials. Biometrika, 70(3), 659-663.
.. [2] Hwang, I. K., Shih, W. J., & De Cani, J. S. (1990). Group sequential
       designs using a family of type I error probability spending functions.
       Statistics in Medicine, 9(12), 1439-1445.
"""

import math
from typing import Callable

from scipy.stats import norm

# =============================================================================
# Alpha Spending Functions (Type I Error)
# =============================================================================


def obf_spending(t: float, alpha: float, tails: int = 2) -> float:
    """
    O'Brien-Fleming cumulative alpha spending function.

    Implements the Lan-DeMets continuous-time approximation to the
    O'Brien-Fleming boundary. This function spends very little alpha early
    (conservative) and accelerates spending near the end (liberal late).

    Parameters
    ----------
    t : float
        Information time in [0, 1]. Proportion of maximum information accrued.
    alpha : float
        Overall Type I error rate in (0, 1).
    tails : {1, 2}, default=2
        Number of tails for the test.

    Returns
    -------
    float
        Cumulative alpha spent by information time t.

    Notes
    -----
    The O'Brien-Fleming spending function is:
        α(t) = 2 - 2·Φ(z_{1-α/2} / √t)  for two-sided tests
        α(t) = 1 - Φ(z_{1-α} / √t)      for one-sided tests

    where Φ is the standard normal CDF and z_q is the q-quantile.

    Examples
    --------
    >>> # Early look (10% information): very conservative
    >>> round(obf_spending(t=0.1, alpha=0.05, tails=2), 8)
    0.0

    >>> # Mid look (50% information): moderate spending
    >>> round(obf_spending(t=0.5, alpha=0.05, tails=2), 6)
    0.005575

    >>> # Final look (100% information): all alpha spent
    >>> round(obf_spending(t=1.0, alpha=0.05, tails=2), 6)
    0.05

    >>> # One-sided test
    >>> round(obf_spending(t=0.5, alpha=0.025, tails=1), 6)
    0.002787
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Information time t must be in [0, 1], got {t}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
    if tails not in (1, 2):
        raise ValueError(f"Tails must be 1 or 2, got {tails}")

    # Avoid division by zero at t=0
    t = max(t, 1e-12)

    if tails == 2:
        z = float(norm.isf(alpha / 2.0))
        return float(2.0 - 2.0 * norm.cdf(z / math.sqrt(t)))
    else:  # tails == 1
        z = float(norm.isf(alpha))
        return float(1.0 - norm.cdf(z / math.sqrt(t)))


def pocock_spending(t: float, alpha: float) -> float:
    """
    Pocock cumulative alpha spending function.

    Implements approximate Pocock spending using the formula:
        α(t) ≈ α · ln(1 + (e - 1)·t)

    This spends alpha more uniformly across looks compared to O'Brien-Fleming.

    Parameters
    ----------
    t : float
        Information time in [0, 1].
    alpha : float
        Overall Type I error rate in (0, 1).

    Returns
    -------
    float
        Cumulative alpha spent by information time t.

    Examples
    --------
    >>> # Early look: more alpha spent than OBF
    >>> round(pocock_spending(t=0.1, alpha=0.05), 6)
    0.007928

    >>> # Mid look: more uniform spending
    >>> round(pocock_spending(t=0.5, alpha=0.05), 6)
    0.031006

    >>> # Final look: all alpha spent
    >>> round(pocock_spending(t=1.0, alpha=0.05), 6)
    0.05
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Information time t must be in [0, 1], got {t}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

    return float(alpha * math.log(1.0 + (math.e - 1.0) * t))


def hsd_spending(t: float, alpha: float, gamma: float = -4.0) -> float:
    """
    Hwang-Shih-DeCani (HSD) cumulative alpha spending function.

    A flexible family of spending functions parameterized by gamma:
    - gamma = 0: Linear spending α(t) = α·t
    - gamma < 0: O'Brien-Fleming-like (conservative early)
    - gamma > 0: Pocock-like (more uniform)
    - gamma = -4: Close to O'Brien-Fleming

    Parameters
    ----------
    t : float
        Information time in [0, 1].
    alpha : float
        Overall Type I error rate in (0, 1).
    gamma : float, default=-4.0
        Shape parameter controlling spending rate.

    Returns
    -------
    float
        Cumulative alpha spent by information time t.

    Notes
    -----
    The HSD spending function is:
        α(t) = α · (1 - exp(-γ·t)) / (1 - exp(-γ))  for γ ≠ 0
        α(t) = α · t                                 for γ = 0

    Examples
    --------
    >>> # Linear spending (gamma=0)
    >>> round(hsd_spending(t=0.5, alpha=0.05, gamma=0.0), 6)
    0.025

    >>> # O'Brien-Fleming-like (gamma=-4)
    >>> round(hsd_spending(t=0.5, alpha=0.05, gamma=-4.0), 6)
    0.00596

    >>> # Pocock-like (gamma=1)
    >>> round(hsd_spending(t=0.5, alpha=0.05, gamma=1.0), 6)
    0.031123
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Information time t must be in [0, 1], got {t}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

    # Linear spending when gamma ≈ 0
    if abs(gamma) < 1e-12:
        return float(alpha * t)

    # HSD formula
    numerator = 1.0 - math.exp(-gamma * t)
    denominator = 1.0 - math.exp(-gamma)
    return float(alpha * numerator / denominator)


# =============================================================================
# Beta Spending Functions (Type II Error) - NEW
# =============================================================================


def beta_obf_spending(t: float, beta: float) -> float:
    """
    O'Brien-Fleming beta spending for futility boundaries.

    Similar to alpha spending but for Type II error (1 - power).
    Conservative early (little beta spent) and liberal late.

    Parameters
    ----------
    t : float
        Information time in [0, 1].
    beta : float
        Overall Type II error rate (1 - power) in (0, 1).

    Returns
    -------
    float
        Cumulative beta spent by information time t.

    Examples
    --------
    >>> # Beta spending for power=0.90 (beta=0.10)
    >>> round(beta_obf_spending(t=0.5, beta=0.10), 6)
    0.034963

    >>> # Final look: all beta spent
    >>> round(beta_obf_spending(t=1.0, beta=0.10), 6)
    0.1
    """
    # Beta spending uses same shape as alpha spending (one-sided)
    return obf_spending(t, beta, tails=1)


def beta_pocock_spending(t: float, beta: float) -> float:
    """
    Pocock beta spending for futility boundaries.

    Uniform beta spending across looks.

    Parameters
    ----------
    t : float
        Information time in [0, 1].
    beta : float
        Overall Type II error rate (1 - power) in (0, 1).

    Returns
    -------
    float
        Cumulative beta spent by information time t.

    Examples
    --------
    >>> round(beta_pocock_spending(t=0.5, beta=0.10), 6)
    0.062011
    """
    return pocock_spending(t, beta)


def beta_hsd_spending(t: float, beta: float, gamma: float = -4.0) -> float:
    """
    Hwang-Shih-DeCani beta spending for futility boundaries.

    Parameters
    ----------
    t : float
        Information time in [0, 1].
    beta : float
        Overall Type II error rate (1 - power) in (0, 1).
    gamma : float, default=-4.0
        Shape parameter.

    Returns
    -------
    float
        Cumulative beta spent by information time t.

    Examples
    --------
    >>> round(beta_hsd_spending(t=0.5, beta=0.10, gamma=-4.0), 6)
    0.01192
    """
    return hsd_spending(t, beta, gamma)


# =============================================================================
# Helper: Create spending function from family name
# =============================================================================


def get_alpha_spending_function(
    family: str, *, gamma: float = -4.0
) -> Callable[[float, float], float]:
    """
    Get alpha spending function by family name.

    Parameters
    ----------
    family : {"obf", "pocock", "hsd"}
        Spending function family name.
    gamma : float, default=-4.0
        Shape parameter for HSD family.

    Returns
    -------
    Callable[[float, float], float]
        Spending function taking (t, alpha) and returning cumulative alpha.

    Examples
    --------
    >>> fn = get_alpha_spending_function("obf")
    >>> round(fn(0.5, 0.05), 6)
    0.005575
    """
    family_lower = family.lower()
    if family_lower in ("obf", "obrien_fleming", "o'brien-fleming"):
        return obf_spending
    elif family_lower == "pocock":
        return pocock_spending
    elif family_lower == "hsd":
        return lambda t, alpha: hsd_spending(t, alpha, gamma)
    else:
        raise ValueError(
            f"Unknown spending function family: {family}. "
            f"Expected one of: obf, pocock, hsd"
        )


def get_beta_spending_function(
    family: str, *, gamma: float = -4.0
) -> Callable[[float, float], float]:
    """
    Get beta spending function by family name.

    Parameters
    ----------
    family : {"obf", "pocock", "hsd"}
        Spending function family name.
    gamma : float, default=-4.0
        Shape parameter for HSD family.

    Returns
    -------
    Callable[[float, float], float]
        Spending function taking (t, beta) and returning cumulative beta.

    Examples
    --------
    >>> fn = get_beta_spending_function("obf")
    >>> round(fn(0.5, 0.10), 6)
    0.034963
    """
    family_lower = family.lower()
    if family_lower in ("obf", "obrien_fleming", "o'brien-fleming"):
        return beta_obf_spending
    elif family_lower == "pocock":
        return beta_pocock_spending
    elif family_lower == "hsd":
        return lambda t, beta: beta_hsd_spending(t, beta, gamma)
    else:
        raise ValueError(
            f"Unknown spending function family: {family}. "
            f"Expected one of: obf, pocock, hsd"
        )
