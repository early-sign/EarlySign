"""Wald Z statistic helpers for two-proportion schemes."""

import math
from dataclasses import dataclass
from typing import Protocol

from earlysign.stats.essentials.schemes.two_proportions.util import (
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


def compute_wald_z(nA: int, mA: int, nB: int, mB: int, *, pooled: bool = True) -> float:
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
    float
        Wald Z-statistic for (pB - pA)

    Examples
    --------
    >>> round(compute_wald_z(100, 40, 100, 55, pooled=True), 6)
    2.123977
    >>> round(compute_wald_z(100, 40, 100, 55, pooled=False), 6)
    2.148345
    """

    return TwoProportionsWaldZ(
        nA=nA,
        mA=mA,
        nB=nB,
        mB=mB,
        pooled=pooled,
    ).value()
