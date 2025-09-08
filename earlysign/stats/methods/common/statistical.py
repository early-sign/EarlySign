"""
earlysign.stats.methods.common.statistical
==========================================

Core statistical operations and utilities.

Provides mathematical building blocks for statistical computations,
including proportion testing, standard errors, and basic distributions.
These functions are theory-agnostic and used across multiple methods.
"""

from __future__ import annotations
import math
from typing import Tuple


def wald_z_from_counts(
    nA: int, nB: int, mA: int, mB: int
) -> Tuple[float, float, float, float]:
    """Return (z, pA_hat, pB_hat, se) using unpooled SE.

    Computes the Wald Z-statistic for comparing two binomial proportions
    using unpooled standard error estimation.

    Args:
        nA: Total trials in group A
        nB: Total trials in group B
        mA: Successes in group A
        mB: Successes in group B

    Returns:
        Tuple of (z_statistic, pA_hat, pB_hat, standard_error)

    Note:
        We define Z = (pB_hat - pA_hat) / SE so that a *better variant B* yields positive Z.
    """
    if min(nA, nB) == 0:
        return 0.0, 0.0, 0.0, float("inf")

    pA_hat, pB_hat = mA / max(nA, 1), mB / max(nB, 1)
    var = pA_hat * (1 - pA_hat) / max(nA, 1) + pB_hat * (1 - pB_hat) / max(nB, 1)
    se = math.sqrt(var) if var > 0 else float("inf")
    diff = pB_hat - pA_hat  # variant minus baseline
    z = diff / se if se not in (0.0, float("inf")) else 0.0

    return z, pA_hat, pB_hat, se


def pooled_proportion_se(nA: int, nB: int, mA: int, mB: int) -> float:
    """Compute pooled standard error for two proportions.

    Uses pooled variance estimator assuming equal population proportions.

    Args:
        nA: Total trials in group A
        nB: Total trials in group B
        mA: Successes in group A
        mB: Successes in group B

    Returns:
        Pooled standard error
    """
    if min(nA, nB) == 0:
        return float("inf")

    n_total = nA + nB
    p_pooled = (mA + mB) / max(n_total, 1)
    var_pooled = p_pooled * (1 - p_pooled) * (1 / max(nA, 1) + 1 / max(nB, 1))

    return math.sqrt(var_pooled) if var_pooled > 0 else float("inf")


def unpooled_proportion_se(nA: int, nB: int, mA: int, mB: int) -> float:
    """Compute unpooled standard error for two proportions.

    Uses separate variance estimators for each group.

    Args:
        nA: Total trials in group A
        nB: Total trials in group B
        mA: Successes in group A
        mB: Successes in group B

    Returns:
        Unpooled standard error
    """
    if min(nA, nB) == 0:
        return float("inf")

    pA_hat = mA / max(nA, 1)
    pB_hat = mB / max(nB, 1)
    var = pA_hat * (1 - pA_hat) / max(nA, 1) + pB_hat * (1 - pB_hat) / max(nB, 1)

    return math.sqrt(var) if var > 0 else float("inf")
