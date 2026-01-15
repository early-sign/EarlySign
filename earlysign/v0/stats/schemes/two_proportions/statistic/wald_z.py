"""Wald Z statistic helpers for two-proportion schemes."""

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from earlysign.v0.stats.schemes.two_proportions.util import (
    compute_pooled_variance,
    compute_proportions,
    compute_unpooled_variance,
)


class WaldStatistic(Protocol):
    """Protocol describing the interface for Wald-style statistics."""

    def value(self) -> float:
        """Return the scalar Wald statistic value."""


@dataclass(frozen=True)
class TwoProportionsWaldZ(WaldStatistic):
    """Concrete Wald Z calculator for two-proportion experiments."""

    nA: int
    mA: int
    nB: int
    mB: int
    pooled: bool = True

    def value(self) -> float:
        pA, pB = compute_proportions(self.nA, self.mA, self.nB, self.mB)
        diff = pB - pA

        if self.pooled:
            _, var = compute_pooled_variance(self.nA, self.mA, self.nB, self.mB)
        else:
            var = compute_unpooled_variance(self.nA, self.mA, self.nB, self.mB)

        if var <= 0.0:
            if diff > 0:
                return float("inf")
            if diff < 0:
                return float("-inf")
            return 0.0

        return diff / math.sqrt(var)


def compute_binomial_wald_z(
    nA: int, mA: int, nB: int, mB: int, *, pooled: bool = True
) -> float:
    """Compute Wald Z-statistic for difference in proportions (binomial arms).

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
    float
        Wald Z-statistic for (pB - pA)

    Examples
    --------
    >>> round(compute_binomial_wald_z(100, 40, 100, 55, pooled=True), 6)
    2.123977
    >>> round(compute_binomial_wald_z(100, 40, 100, 55, pooled=False), 6)
    2.148345
    """

    return TwoProportionsWaldZ(
        nA=nA,
        mA=mA,
        nB=nB,
        mB=mB,
        pooled=pooled,
    ).value()


def compute_binomial_wald_z_array(
    nA: np.ndarray,
    mA: np.ndarray,
    nB: np.ndarray,
    mB: np.ndarray,
    *,
    pooled: bool = True,
) -> np.ndarray:
    """Vectorised Wald Z-statistic for many simulations at once.

    Parameters
    ----------
    nA, mA, nB, mB : array-like
        Arrays of cumulative sample sizes and successes for groups A/B.
    pooled : bool, default True
        Whether to use the pooled variance estimate.

    Returns
    -------
    np.ndarray
        Array of Wald Z statistics matching the broadcast of the inputs.

    Examples
    --------
    >>> import numpy as np
    >>> z = compute_binomial_wald_z_array(
    ...     nA=np.array([50, 60]),
    ...     mA=np.array([20, 24]),
    ...     nB=np.array([50, 60]),
    ...     mB=np.array([30, 36]),
    ...     pooled=True,
    ... )
    >>> np.round(z, 4).tolist()
    [2.0, 2.1909]
    """

    nA = np.asarray(nA, dtype=float)
    mA = np.asarray(mA, dtype=float)
    nB = np.asarray(nB, dtype=float)
    mB = np.asarray(mB, dtype=float)

    pA = np.divide(mA, nA, out=np.zeros_like(mA), where=nA > 0)
    pB = np.divide(mB, nB, out=np.zeros_like(mB), where=nB > 0)
    diff = pB - pA

    if pooled:
        total = nA + nB
        p_pool = np.divide(mA + mB, total, out=np.zeros_like(diff), where=total > 0)
        var = (
            p_pool
            * (1.0 - p_pool)
            * (np.divide(1.0, nA, where=nA > 0) + np.divide(1.0, nB, where=nB > 0))
        )
    else:
        var = np.divide(
            pA * (1.0 - pA), nA, out=np.zeros_like(pA), where=nA > 0
        ) + np.divide(pB * (1.0 - pB), nB, out=np.zeros_like(pB), where=nB > 0)

    z = np.empty_like(diff)
    positive = var > 0
    z[positive] = diff[positive] / np.sqrt(var[positive])
    z[~positive & (diff > 0)] = np.inf
    z[~positive & (diff < 0)] = -np.inf
    z[~positive & (diff == 0)] = 0.0
    return np.asarray(z, dtype=float)


# Backwards-compatible aliases
compute_wald_z = compute_binomial_wald_z
compute_wald_z_array = compute_binomial_wald_z_array
