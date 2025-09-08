"""
earlysign.stats.methods.group_sequential.core
==============================================

Core mathematics for group sequential testing.

Provides mathematical foundations for sequential testing including:
- Error spending functions (Lan-DeMets, O'Brien-Fleming, Pocock)
- Boundary calculations
- Information fraction computations

These functions implement the mathematical theory without dependencies
on specific problem domains or experimental setups.
"""

from __future__ import annotations
import math
from typing import List, Iterable

from scipy.stats import norm


def lan_demets_spending(alpha_total: float, t: float, style: str) -> float:
    """Lan–DeMets alpha spending (two-sided cumulative) for OBF/Pocock.

    Computes the cumulative Type I error probability spent up to information
    time t using the specified spending function.

    Args:
        alpha_total: Total Type I error budget
        t: Information fraction (0 ≤ t ≤ 1)
        style: Spending function style ("obf", "pocock")

    Returns:
        Cumulative alpha spent at information time t

    References:
        Lan, K.K.G. and DeMets, D.L. (1983). Discrete sequential boundaries for
        clinical trials. Biometrika 70(3), 659-663.
    """
    s = style.lower()
    if s in ("obf", "o'brien", "obrien", "o'brien-fleming"):
        za2 = float(norm.ppf(1 - alpha_total / 2.0))
        return 2.0 - 2.0 * float(norm.cdf(za2 / math.sqrt(max(t, 1e-12))))
    if s in ("pocock",):
        return alpha_total * math.log(1.0 + (math.e - 1.0) * t)
    raise ValueError(f"Unknown spending style: {style}")


def nominal_alpha_increments(
    alpha_total: float, t_grid: Iterable[float], style: str
) -> List[float]:
    """Convert cumulative spending to per-look increments.

    Transforms cumulative alpha spending values into per-analysis increments
    for constructing sequential test boundaries.

    Args:
        alpha_total: Total Type I error budget
        t_grid: Information times for analyses
        style: Spending function style

    Returns:
        List of alpha increments for each analysis
    """
    cum = [lan_demets_spending(alpha_total, t, style) for t in t_grid]
    inc, prev = [], 0.0
    for c in cum:
        inc.append(max(c - prev, 0.0))
        prev = c
    return inc
