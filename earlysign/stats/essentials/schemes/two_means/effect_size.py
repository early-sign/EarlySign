"""Effect size calculator for two-sample mean comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from earlysign.stats.essentials.schemes.protocols import EffectSizeCalculator


def compute_standard_error(
    nA: int, nB: int, sigma: float, pooled: bool = True
) -> float:
    """Compute standard error for difference in means.

    This is useful for design calculations where the common standard deviation
    is assumed rather than observed.

    Parameters
    ----------
    nA, nB : int
        Sample sizes for groups A and B
    sigma : float
        Assumed common standard deviation (pooled)
    pooled : bool, default True
        Whether to use pooled variance estimate (for this function, always True
        as we assume common sigma)

    Returns
    -------
    se : float
        Standard error of (muB - muA)

    Examples
    --------
    >>> se = compute_standard_error(100, 100, 1.0)
    >>> round(se, 4)
    0.1414

    >>> se2 = compute_standard_error(50, 100, 2.0)
    >>> round(se2, 4)
    0.3464
    """
    if sigma < 0:
        raise ValueError("Standard deviation must be non-negative.")
    if nA <= 0 or nB <= 0:
        raise ValueError("Sample sizes must be positive.")

    return sigma * math.sqrt(1.0 / nA + 1.0 / nB)


@dataclass
class MeansEffect:
    """Effect specification for continuous outcomes."""

    mean_control: float = 10.0
    mean_treatment: float | None = None
    effect_size: float | None = 2.0
    std_dev: float = 5.0
    pooled_std: bool = True


@dataclass
class MeansSampleSize:
    """Sample size specification for continuous outcomes."""

    n_per_analysis: int = 100


@dataclass
class MeansEffectSizeCalculator(EffectSizeCalculator):
    """Effect size utilities for continuous outcomes with equal-variance assumption."""

    def standardized_effect(
        self,
        effect: MeansEffect,
        info_time: float,
        *,
        sample_size: MeansSampleSize,
        allocation_ratio: float,
        info_times: np.ndarray,
    ) -> float:
        """Return standardized mean difference at the specified information fraction."""
        mean_control = effect.mean_control
        if effect.mean_treatment is not None:
            mean_treatment = effect.mean_treatment
        else:
            lift = effect.effect_size or 0.0
            mean_treatment = mean_control + lift

        std_dev = effect.std_dev
        n_total_control = sample_size.n_per_analysis * len(info_times)
        n_control_at_t = int(n_total_control * info_time)
        n_treatment_at_t = int(n_control_at_t * allocation_ratio)

        if n_control_at_t == 0 or n_treatment_at_t == 0:
            return 0.0

        se = compute_standard_error(
            n_control_at_t,
            n_treatment_at_t,
            std_dev,
            pooled=effect.pooled_std,
        )
        if se == 0:
            return 0.0

        return float((mean_treatment - mean_control) / se)

    def sample_sizes(
        self,
        sample_size: MeansSampleSize,
        *,
        allocation_ratio: float,
        info_times: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Return planned sample sizes for each interim analysis."""
        n_total_control = sample_size.n_per_analysis * len(info_times)
        n_control = (n_total_control * info_times).astype(int)
        n_treatment = (n_control * allocation_ratio).astype(int)

        return {
            "n_control": n_control,
            "n_treatment": n_treatment,
            "n_total": n_control + n_treatment,
            "info_fraction": info_times,
        }


__all__ = ["MeansEffect", "MeansSampleSize", "MeansEffectSizeCalculator"]
