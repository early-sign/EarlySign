"""Canonical Joint Distribution for group sequential statistics.

This module implements the core statistical properties of group sequential
tests based on the multivariate normal 'canonical joint distribution'
as described in Jennison & Turnbull (2000).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import norm
from earlysign.methods.group_sequential.spending import SpendingFunction


class CanonicalJointDistribution:
    """Calculations based on the canonical joint distribution of sequential statistics.

    The properties of a group sequential test depend on the standardized drift
    delta = theta * sqrt(I_max), where theta is the effect size and I_max
    is the maximum information.
    """

    def __init__(self, n_sims: int = 200000, rng_seed: Optional[int] = None):
        self.n_sims = n_sims
        self._rng = np.random.default_rng(rng_seed)

    def _generate_joint_z(self, info_times: np.ndarray) -> np.ndarray:
        """Generate Z-statistics under H0 (drift=0) at given info fractions."""
        k = len(info_times)
        # Covariance for Brownian motion: Cov(Z_i, Z_j) = sqrt(t_min / t_max)
        cov = np.sqrt(np.minimum.outer(info_times, info_times) / np.maximum.outer(info_times, info_times))
        np.fill_diagonal(cov, 1.0)
        return self._rng.multivariate_normal(np.zeros(k), cov, size=self.n_sims)

    def solve_boundary_constant(
        self,
        info_times: Sequence[float],
        alpha: float,
        shape_type: str = "pocock",
        tails: int = 2,
    ) -> float:
        """Solve for the constant 'c' that yields the target alpha for a given shape.

        Args:
            info_times: Cumulative information fractions (t_1, ..., t_K).
            alpha: Target Type I error.
            shape_type: "pocock", "obrien_fleming", or "wang_tsiatis" (with Delta=0.25).
            tails: 1 or 2 (currently implementation focused on symmetric 2-sided).

        Returns:
            The constant c such that P(any |Z_k| > c * shape_k) = alpha.
        """
        t = np.asarray(info_times)
        if shape_type == "pocock":
            c_shape = np.ones_like(t)
        elif shape_type == "obrien_fleming":
            c_shape = 1.0 / np.sqrt(t)
        elif shape_type == "wang_tsiatis":
            # Delta = 0.25 case from Table 3.1
            c_shape = t ** (-0.25)
        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")

        z_sims = self._generate_joint_z(t)
        if tails == 2:
            # We find c such that P(max |Z_k / c_shape_k| > c) = alpha
            normalized_max = np.max(np.abs(z_sims) / c_shape, axis=1)
            return float(np.percentile(normalized_max, 100 * (1 - alpha)))
        else:
            normalized_max = np.max(z_sims / c_shape, axis=1)
            return float(np.percentile(normalized_max, 100 * (1 - alpha)))

    def compute_rejection_probability(
        self,
        info_times: Sequence[float],
        boundaries: Sequence[float],
        drift: float = 0.0,
        tails: int = 2,
    ) -> float:
        """Compute the probability of rejecting H0 (crossing boundaries).

        Args:
            info_times: Actual information fractions.
            boundaries: Z-scale boundaries at each look.
            drift: Standardized drift delta = theta * sqrt(I_max).
            tails: 1 or 2.

        Returns:
            Probability of rejection.
        """
        t = np.asarray(info_times)
        b = np.asarray(boundaries)
        k = len(t)
        
        # Means: E(Z_k) = delta * sqrt(t_k)
        means = drift * np.sqrt(t)
        cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
        np.fill_diagonal(cov, 1.0)
        
        sims = self._rng.multivariate_normal(means, cov, size=self.n_sims * 2)
        if tails == 2:
            rejected = np.any(np.abs(sims) > b, axis=1)
        else:
            rejected = np.any(sims > b, axis=1)
            
        return float(np.mean(rejected))

    def solve_drift(
        self,
        info_times: Sequence[float],
        boundaries: Sequence[float],
        target_power: float,
        tails: int = 2,
    ) -> float:
        """Solve for the standardized drift delta that yields target power.

        Returns delta such that compute_rejection_probability(...) = target_power.
        """
        # Search range for drift: starts around fixed-sample required drift
        # fixed-sample drift = z_alpha/2 + z_beta
        # We assume alpha approx 0.05
        low = 1.0
        high = 10.0
        
        def f(d):
            return self.compute_rejection_probability(info_times, boundaries, drift=d, tails=tails) - target_power

        res = root_scalar(f, bracket=[low, high], xtol=1e-4)
        return float(res.root)

    def solve_boundaries_from_spending(
        self,
        info_times: Sequence[float],
        spending: SpendingFunction,
        tails: int = 2,
    ) -> np.ndarray:
        """Solve for boundaries given a spending function and information schedule.
        
        Args:
            info_times: Cumulative information fractions (t_1, ..., t_K).
            spending: A SpendingFunction implementation.
            tails: 1 or 2 (symmetric).
            
        Returns:
            Array of Z-scale boundaries at each look.
        """
        t = np.asarray(info_times)
        k = len(t)
        cum_alpha = spending.cumulative(t)
        
        z_sims = self._generate_joint_z(t)
        boundaries = np.zeros(k)
        
        for i in range(k):
            target_alpha = cum_alpha[i]
            if i == 0:
                if tails == 2:
                    boundaries[i] = norm.isf(target_alpha / 2.0)
                else:
                    boundaries[i] = norm.isf(target_alpha)
            else:
                # Find b_i such that P(any |Z_j| > b_j for j < i OR (|Z_i| > b_i if tails=2 else Z_i > b_i)) = target_alpha
                if tails == 2:
                    rejected_prev = np.any(np.abs(z_sims[:, :i]) > boundaries[:i], axis=1)
                    def f(b):
                        rejected_curr = np.abs(z_sims[:, i]) > b
                        return np.mean(rejected_prev | rejected_curr) - target_alpha
                else:
                    rejected_prev = np.any(z_sims[:, :i] > boundaries[:i], axis=1)
                    def f(b):
                        rejected_curr = z_sims[:, i] > b
                        return np.mean(rejected_prev | rejected_curr) - target_alpha
                
                res = root_scalar(f, bracket=[0, 10], xtol=1e-5)
                boundaries[i] = res.root
                
        return boundaries

    def solve_boundaries_from_cumulative_alpha(
        self,
        info_times: Sequence[float],
        cum_alpha: Sequence[float],
        tails: int = 2,
    ) -> np.ndarray:
        """Solve for boundaries given an arbitrary cumulative alpha schedule.
        
        Args:
            info_times: Cumulative information fractions.
            cum_alpha: Target cumulative Type I error at each look.
            tails: 1 or 2 (symmetric).
            
        Returns:
            Array of Z-scale boundaries at each look.
        """
        t = np.asarray(info_times)
        k = len(t)
        c_alpha = np.asarray(cum_alpha)
        
        z_sims = self._generate_joint_z(t)
        boundaries = np.zeros(k)
        
        for i in range(k):
            target_alpha = c_alpha[i]
            if i == 0:
                if tails == 2:
                    boundaries[i] = norm.isf(target_alpha / 2.0)
                else:
                    boundaries[i] = norm.isf(target_alpha)
            else:
                if tails == 2:
                    rejected_prev = np.any(np.abs(z_sims[:, :i]) > boundaries[:i], axis=1)
                    def f(b):
                        rejected_curr = np.abs(z_sims[:, i]) > b
                        return np.mean(rejected_prev | rejected_curr) - target_alpha
                else:
                    rejected_prev = np.any(z_sims[:, :i] > boundaries[:i], axis=1)
                    def f(b):
                        rejected_curr = z_sims[:, i] > b
                        return np.mean(rejected_prev | rejected_curr) - target_alpha
                
                res = root_scalar(f, bracket=[0, 10], xtol=1e-5)
                boundaries[i] = res.root
                
        return boundaries

    def evaluate_asn(
        self,
        info_times: Sequence[float],
        boundaries: Sequence[float],
        drift: float = 0.0,
        tails: int = 2,
    ) -> float:
        """Evaluate the expected look number (ASN in terms of looks).
        
        Args:
            info_times: Cumulative information fractions.
            boundaries: Z-scale boundaries.
            drift: Standardized drift.
            tails: 1 or 2.
            
        Returns:
            Expected look at which the trial stops (1.0 to K).
        """
        t = np.asarray(info_times)
        b = np.asarray(boundaries)
        k = len(t)
        
        means = drift * np.sqrt(t)
        cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
        np.fill_diagonal(cov, 1.0)
        
        sims = self._rng.multivariate_normal(means, cov, size=self.n_sims)
        
        if tails == 2:
            reject_matrix = np.abs(sims) > b
        else:
            reject_matrix = sims > b
            
        has_rejected = np.any(reject_matrix, axis=1)
        first_rejected_idx = np.argmax(reject_matrix, axis=1)
        
        # Those that never rejected stop at the last look k.
        stop_look_indices = np.where(has_rejected, first_rejected_idx, k - 1)
        stop_looks = stop_look_indices + 1
        
        return float(np.mean(stop_looks))
