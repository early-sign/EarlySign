"""
Group sequential utilities for two-sample normal means designs.

This module provides an approximate expected sample size (ASN) calculator
under the canonical normal means assumption. It is deliberately scoped to
the two-means scheme so that applications can depend on a scheme-level API.
"""

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from earlysign.methods.group_sequential.spending import (
    SpendingFunction,
)


class NormalMeansASNCalculator:
    """Approximate ASN calculator for two-arm normal means group sequential tests."""

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        sided: int,
        alternative: float,
        st_dev: float,
        allocation_ratio: float,
        spending: SpendingFunction,
        min_stage_alpha: float = 1e-9,
    ) -> None:
        if sided not in (1, 2):
            raise ValueError("sided must be 1 or 2")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must lie in (0, 1)")
        if not np.isfinite(st_dev) or st_dev <= 0.0:
            raise ValueError("st_dev must be positive")
        if allocation_ratio <= 0.0:
            raise ValueError("allocation_ratio must be positive")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sided = int(sided)
        self.alternative = float(alternative)
        self.st_dev = float(st_dev)
        self.allocation_ratio = float(allocation_ratio)
        self.spending = spending
        self.min_stage_alpha = float(min_stage_alpha)

    def _per_stage_alpha(
        self, information_rates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        cumulative = np.asarray(
            self.spending.cumulative(information_rates), dtype=float
        )
        increments = np.diff(np.concatenate(([0.0], cumulative)))
        increments = np.maximum(increments, self.min_stage_alpha)
        scale = self.alpha / increments.sum()
        return np.asarray(increments * scale, dtype=float)

    def _z_boundaries(
        self, per_stage_alpha: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.asarray(
            self.spending.boundaries_from_stage_alpha(per_stage_alpha), dtype=float
        )

    def _n_max_from_power(self) -> float:
        z_alpha = norm.ppf(1 - (self.alpha / 2.0 if self.sided == 2 else self.alpha))
        z_beta = norm.ppf(1 - self.beta)
        term = (z_alpha + z_beta) * self.st_dev / abs(self.alternative)
        n_per_group = term**2
        n_total = n_per_group * (1.0 + self.allocation_ratio)
        return float(n_total * 1.05)

    def _validate_information_rates(
        self, information_rates: Iterable[float]
    ) -> NDArray[np.float64]:
        rates = np.array(sorted({float(x) for x in information_rates}))
        if rates.size == 0 or rates[-1] != 1.0:
            raise ValueError("information_rates must include 1.0 and be non-empty")
        if rates[0] <= 0.0 or np.any(np.diff(rates) <= 0.0):
            raise ValueError("information_rates must be strictly increasing in (0, 1]")
        return rates

    def evaluate(self, information_rates: Iterable[float]) -> float:
        """Evaluate expected sample size (ASN) for a schedule of information rates."""
        rates = self._validate_information_rates(information_rates)
        per_stage = self._per_stage_alpha(rates)
        boundaries = self._z_boundaries(per_stage)
        n_max = self._n_max_from_power()
        cumulative_n = np.maximum(2.0, rates * n_max)

        sigma_eff = self.st_dev * np.sqrt(1.0 + 1.0 / self.allocation_ratio)
        kappa = (self.alternative / sigma_eff) * np.sqrt(n_max)
        mu = kappa * np.sqrt(rates)

        survival = 1.0
        stagewise = np.zeros_like(rates)

        for idx, (boundary, mean) in enumerate(zip(boundaries, mu)):
            reject_prob = float(np.clip(1.0 - norm.cdf(boundary - mean), 0.0, 1.0))
            stagewise[idx] = survival * reject_prob
            survival *= 1.0 - reject_prob

        expected = float(np.dot(stagewise, cumulative_n) + survival * n_max)
        return expected
