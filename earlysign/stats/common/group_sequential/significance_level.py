"""
Per-look significance level → nominal Z boundaries.

- level_to_nominal_z(alpha_level, tails=2) -> (upper_z, lower_z)
"""

from scipy.stats import norm


def level_to_nominal_z(alpha_level: float, *, tails: int = 2) -> tuple[float, float]:
    """
    Map a per-look significance level to symmetric Z boundaries.

    Parameters
    ----------
    alpha_level : float in (0,1)
    tails : {1,2}

    Returns
    -------
    (upper_z, lower_z)

    Examples
    --------
    >>> up, lo = level_to_nominal_z(0.01, tails=2)
    >>> round(up, 3), round(lo, 3)
    (2.576, -2.576)
    """
    if alpha_level <= 0.0 or alpha_level >= 1.0:
        raise ValueError("`alpha_level` must be in (0,1).")
    if tails not in (1, 2):
        raise ValueError("`tails` must be 1 or 2.")
    if norm is None:
        # Fallback: infinite boundaries when exact quantiles are unavailable
        return float("inf"), float("-inf") if tails == 2 else float("-inf")

    if tails == 2:
        z = float(norm.isf(alpha_level / 2.0))
        return z, -z
    else:
        z = float(norm.isf(alpha_level))
        return z, float("-inf")
