"""Canonical Joint Distribution for group sequential statistics.

This module implements the core statistical properties of group sequential
tests based on the multivariate normal 'canonical joint distribution'
as described in Jennison & Turnbull (2000).
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar
from scipy.stats import norm

from earlysign.methods.group_sequential.spending import SpendingFunction


class CanonicalJointDistribution:
    """Calculations based on the canonical joint distribution of sequential statistics.

    The properties of a group sequential test depend on the standardized drift
    delta = theta * sqrt(I_max), where theta is the effect size and I_max
    is the maximum information.
    """

    def __init__(
        self, n_sims: int = 20000, rng_seed: Optional[int] = None
    ):  # Default to 20k for balanced speed/accuracy
        self.n_sims = n_sims
        self._rng = np.random.default_rng(rng_seed)

    def _generate_joint_z(self, info_times: np.ndarray) -> np.ndarray:
        """Generate Z-statistics under H0 (drift=0) at given info fractions."""
        k = len(info_times)
        # Covariance for Brownian motion: Cov(Z_i, Z_j) = sqrt(t_min / t_max)
        cov = np.sqrt(
            np.minimum.outer(info_times, info_times)
            / np.maximum.outer(info_times, info_times)
        )
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
        len(t)

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

        def f(d: float) -> float:
            return (
                self.compute_rejection_probability(
                    info_times, boundaries, drift=d, tails=tails
                )
                - target_power
            )

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
            target_alpha = float(cum_alpha[i])
            if i == 0:
                if tails == 2:
                    boundaries[i] = norm.isf(target_alpha / 2.0)
                else:
                    boundaries[i] = norm.isf(target_alpha)
            else:
                # Find b_i such that P(any |Z_j| > b_j for j < i OR (|Z_i| > b_i if tails=2 else Z_i > b_i)) = target_alpha
                if tails == 2:
                    rejected_prev = np.any(
                        np.abs(z_sims[:, :i]) > boundaries[:i], axis=1
                    )

                    def f(b: float) -> float:
                        rejected_curr = np.abs(z_sims[:, i]) > b
                        return (
                            float(np.mean(rejected_prev | rejected_curr)) - target_alpha
                        )

                else:
                    rejected_prev = np.any(z_sims[:, :i] > boundaries[:i], axis=1)

                    def f(b: float) -> float:
                        rejected_curr = z_sims[:, i] > b
                        return (
                            float(np.mean(rejected_prev | rejected_curr)) - target_alpha
                        )

                res = root_scalar(f, bracket=[0, 10], xtol=1e-5)
                boundaries[i] = float(res.root)

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
            target_alpha = float(c_alpha[i])
            if i == 0:
                if tails == 2:
                    boundaries[i] = norm.isf(target_alpha / 2.0)
                else:
                    boundaries[i] = norm.isf(target_alpha)
            else:
                if tails == 2:
                    rejected_prev = np.any(
                        np.abs(z_sims[:, :i]) > boundaries[:i], axis=1
                    )

                    def f(b: float) -> float:
                        rejected_curr = np.abs(z_sims[:, i]) > b
                        return (
                            float(np.mean(rejected_prev | rejected_curr)) - target_alpha
                        )

                else:
                    rejected_prev = np.any(z_sims[:, :i] > boundaries[:i], axis=1)

                    def f(b: float) -> float:
                        rejected_curr = z_sims[:, i] > b
                        return (
                            float(np.mean(rejected_prev | rejected_curr)) - target_alpha
                        )

                res = root_scalar(f, bracket=[0, 10], xtol=1e-5)
                boundaries[i] = float(res.root)

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

    def solve_one_sided_asymmetric_boundaries(
        self,
        info_times: Sequence[float],
        alpha_spending: NDArray[np.float64],
        beta_spending: NDArray[np.float64],
        drift: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve for efficacy (a) and futility (b) boundaries for one-sided asymmetric tests.

        Args:
            info_times: Cumulative information fractions.
            alpha_spending: Cumulative alpha spent at each look.
            beta_spending: Cumulative beta spent at each look.
            drift: Standardized drift delta under H1.

        Returns:
            Tuple of (efficacy_boundaries, futility_boundaries).
        """
        t = np.asarray(info_times)
        k = len(t)
        a_cum = np.asarray(alpha_spending)
        b_cum = np.asarray(beta_spending)

        # We need Z_k ~ N(0, BM) for alpha spending (under H0)
        # and Z_k ~ N(delta * sqrt(t_k), BM) for beta spending (under H1)
        z_sims_h0 = self._generate_joint_z(t)
        z_sims_h1 = z_sims_h0 + drift * np.sqrt(t)

        a = np.zeros(k)  # efficacy (upper rejection)
        b = np.full(k, -10.0)  # futility (lower acceptance)

        for i in range(k):
            # 1. Solve for efficacy boundary a[i]
            # Prob(stop for efficacy at look i | not stopped before) = alpha_inc[i]
            # This is cumulative: P(First reach efficacy boundary at look j <= i OR reach futility boundary at look j < i) = alpha_cum[i]
            # However, J&T and most literature define alpha spending such that it's the probability of
            # early rejection UNDER H0. Futility boundaries are often derived under H1.

            # More standard approach (J&T 7.2.1):
            # a[i] is solved under H0 (drift=0).
            # b[i] is solved under H1 (drift=delta).

            # Under H0:
            if i == 0:
                a[i] = norm.isf(a_cum[0])
            else:
                # Efficacy rejection at look i under H0
                # Must not have crossed a_j early (j < i) AND must not have crossed b_j early (j < i)
                stopped_efficacy_prev = np.any(z_sims_h0[:, :i] > a[:i], axis=1)
                stopped_futility_prev = np.any(z_sims_h0[:, :i] < b[:i], axis=1)

                def f_a(val: float) -> float:
                    efficacy_curr = z_sims_h0[:, i] > val
                    # Total rejection prob under H0 so far
                    return float(
                        np.mean(
                            stopped_efficacy_prev
                            | (efficacy_curr & ~stopped_futility_prev)
                        )
                    ) - float(a_cum[i])

                res_a = root_scalar(f_a, bracket=[-10, 10], xtol=1e-5)
                a[i] = float(res_a.root)

            # 2. Solve for futility boundary b[i] under H1
            # Prob(stop for futility at look i | not stopped before) = beta_inc[i]
            # Under H1:
            if i == k - 1:
                # Last look: a[K] = b[K] for a closed test
                b[i] = a[i]
            else:
                if i == 0:

                    def f_b(val: float) -> float:
                        futility_curr = z_sims_h1[:, i] < val
                        return float(np.mean(futility_curr)) - float(b_cum[0])

                else:
                    stopped_efficacy_prev_h1 = np.any(z_sims_h1[:, :i] > a[:i], axis=1)
                    stopped_futility_prev_h1 = np.any(z_sims_h1[:, :i] < b[:i], axis=1)

                    def f_b(val: float) -> float:
                        futility_curr = z_sims_h1[:, i] < val
                        return float(
                            np.mean(
                                stopped_futility_prev_h1
                                | (futility_curr & ~stopped_efficacy_prev_h1)
                            )
                        ) - float(b_cum[i])

                res_b = root_scalar(f_b, bracket=[-10, 10], xtol=1e-5)
                b[i] = float(res_b.root)

        return a, b

    def _compute_ros_boundaries(
        self,
        K: int,
        alpha: float,
        beta: float,
        rho: float,
        ros: float,
        z_h0: np.ndarray,
        alpha_cum: Optional[np.ndarray] = None,
        beta_cum: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute boundaries a and b for a given R_OS inflation factor using fixed Z samples."""
        info_times = np.linspace(1 / K, 1.0, K)
        if alpha_cum is None:
            from earlysign.methods.group_sequential.spending import RhoFamilySpending

            alpha_cum = RhoFamilySpending(alpha=alpha, rho=rho).cumulative(info_times)
        if beta_cum is None:
            from earlysign.methods.group_sequential.spending import RhoFamilySpending

            beta_cum = RhoFamilySpending(alpha=beta, rho=rho).cumulative(info_times)

        eta_fixed = norm.ppf(1 - alpha) + norm.ppf(1 - beta)
        drift = eta_fixed * np.sqrt(ros)

        # Scaling Z statistics to drift
        z_h1 = z_h0 + drift * np.sqrt(info_times)
        num_sims = z_h0.shape[0]

        a = np.zeros(K)
        b = np.full(K, -10.0)

        # Stoppage state masks
        stopped_h0 = np.zeros(num_sims, dtype=bool)
        stopped_efficacy_h1 = np.zeros(num_sims, dtype=bool)
        stopped_futility_h1 = np.zeros(num_sims, dtype=bool)

        # Track counts to avoid repeated np.sum
        total_rej_h0 = 0
        total_fut_h1 = 0

        for i in range(K - 1):
            # 1. Solve a[i] under H0 (Binding for futility)
            rem_mask_h0 = ~stopped_h0
            rem_h0 = z_h0[rem_mask_h0, i]

            needed_a_abs = alpha_cum[i] * num_sims - total_rej_h0

            if needed_a_abs <= 0 or len(rem_h0) == 0:
                a[i] = 8.0  # High value
            else:
                target_a_frac = max(0, min(1, needed_a_abs / len(rem_h0)))
                a[i] = np.percentile(rem_h0, 100 * (1.0 - target_a_frac))

            # Update h0 stoppage for a[i]
            just_stopped_efficacy_h0 = (z_h0[:, i] > a[i]) & rem_mask_h0
            stopped_h0 |= just_stopped_efficacy_h0
            total_rej_h0 += int(np.count_nonzero(just_stopped_efficacy_h0))

            # 2. Solve b[i] under H1 (Binding for efficacy)
            # a[i] also stops trials under H1
            rem_mask_h1 = ~(stopped_efficacy_h1 | stopped_futility_h1)
            # Trials that cross a[i] stop for efficacy under H1
            just_stopped_efficacy_h1 = (z_h1[:, i] > a[i]) & rem_mask_h1
            stopped_efficacy_h1 |= just_stopped_efficacy_h1

            # Update remainder and solve b[i]
            rem_mask_h1 &= ~just_stopped_efficacy_h1
            rem_h1 = z_h1[rem_mask_h1, i]

            needed_b_abs = beta_cum[i] * num_sims - total_fut_h1

            if len(rem_h1) < 50 or needed_b_abs <= 0:
                b[i] = -8.0
            else:
                target_b_frac = max(0, min(1, needed_b_abs / len(rem_h1)))
                b[i] = np.percentile(rem_h1, 100 * target_b_frac)
                if b[i] > a[i]:
                    b[i] = a[i]

            # Update futility stoppage under H1
            just_stopped_futility_h1 = (z_h1[:, i] < b[i]) & rem_mask_h1
            stopped_futility_h1 |= just_stopped_futility_h1
            total_fut_h1 += int(np.count_nonzero(just_stopped_futility_h1))

            # Update h0 stoppage for b[i] (Binding futility)
            stopped_h0 |= (z_h0[:, i] < b[i]) & ~stopped_h0

        # 3. Terminal condition
        rem_mask_h0 = ~stopped_h0
        rem_h0_f = z_h0[rem_mask_h0, K - 1]
        needed_af_abs = alpha * num_sims - total_rej_h0

        if len(rem_h0_f) < 10:
            c_val = a[K - 2] if K > 1 else 2.0
        else:
            target_af_frac = max(0, min(1, needed_af_abs / len(rem_h0_f)))
            c_val = np.percentile(rem_h0_f, 100 * (1.0 - target_af_frac))

        a[K - 1] = c_val
        b[K - 1] = c_val

        # Calculate Power (Under H1)
        # Total acceptance is early futility + final acceptance
        final_rem_mask_h1 = ~(stopped_efficacy_h1 | stopped_futility_h1)
        accept_final_h1 = final_rem_mask_h1 & (z_h1[:, K - 1] <= c_val)
        tot_acc = (total_fut_h1 + np.count_nonzero(accept_final_h1)) / num_sims

        return a, b, tot_acc

    def solve_ros_inflation_factor(
        self,
        K: int,
        alpha: float,
        beta: float,
        rho: float,
    ) -> float:
        """Solve for R_OS inflation factor such that a_K = b_K in Table 7.6."""
        # Ultra-low simulation count for maximum speed
        n_sims_original = self.n_sims
        rng_original = self._rng
        self.n_sims = 10000
        self._rng = np.random.default_rng(42)

        info_times = np.linspace(1 / K, 1.0, K)
        z_h0 = self._generate_joint_z(info_times)

        from earlysign.methods.group_sequential.spending import RhoFamilySpending

        alpha_cum = RhoFamilySpending(alpha=alpha, rho=rho).cumulative(info_times)
        beta_cum = RhoFamilySpending(alpha=beta, rho=rho).cumulative(info_times)

        def f(ros: float) -> float:
            _, _, tot_acc = self._compute_ros_boundaries(
                K,
                alpha,
                beta,
                rho,
                ros,
                z_h0=z_h0,
                alpha_cum=alpha_cum,
                beta_cum=beta_cum,
            )
            return float(tot_acc - beta)

        try:
            # xtol=2e-3 provides sufficient precision for R_OS in Table 7.6
            res = root_scalar(f, bracket=[0.7, 3.0], xtol=2e-3)
            result = float(res.root)
        finally:
            self.n_sims = n_sims_original
            self._rng = rng_original
        return result

    def evaluate_asn_dual_boundaries(
        self,
        info_times: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        drift: float,
        z_base: Optional[np.ndarray] = None,
    ) -> float:
        """Evaluate the expected look number for a design with both efficacy and futility boundaries."""
        k = len(info_times)
        if z_base is None:
            z_base = self._generate_joint_z(info_times)
        z_sims = z_base + drift * np.sqrt(info_times)

        # stoppped[i] is true if trial stops at look i or earlier
        stopped = np.zeros(self.n_sims, dtype=bool)
        stop_looks = np.full(self.n_sims, k, dtype=int)

        for i in range(k):
            crossing = (z_sims[:, i] > a[i]) | (z_sims[:, i] < b[i])
            just_stopped = crossing & ~stopped
            stop_looks[just_stopped] = i + 1
            stopped = stopped | crossing

        return float(np.mean(stop_looks))

    def evaluate_ros_design_characteristics(
        self,
        K: int,
        alpha: float,
        beta: float,
        rho: float,
    ) -> Dict[str, float]:
        """Compute R_OS and ASN for one-sided asymmetric maximum information tests."""
        # Balanced simulation count for characterization
        n_sims_original = self.n_sims
        rng_original = self._rng
        self.n_sims = 20000
        self._rng = np.random.default_rng(42)

        try:
            ros = self.solve_ros_inflation_factor(K, alpha, beta, rho)

            info_times = np.linspace(1 / K, 1.0, K)
            z_h0 = self._generate_joint_z(info_times)
            a, b, _ = self._compute_ros_boundaries(K, alpha, beta, rho, ros, z_h0=z_h0)

            eta_fixed = norm.ppf(1 - alpha) + norm.ppf(1 - beta)
            delta_target = eta_fixed * np.sqrt(ros)

            # ASN calculation with reuse of samples
            asn_0 = self.evaluate_asn_dual_boundaries(
                info_times, a, b, drift=0.0, z_base=z_h0
            )
            asn_05delta = self.evaluate_asn_dual_boundaries(
                info_times, a, b, drift=0.5 * delta_target, z_base=z_h0
            )
            asn_delta = self.evaluate_asn_dual_boundaries(
                info_times, a, b, drift=delta_target, z_base=z_h0
            )
        finally:
            self.n_sims = n_sims_original
            self._rng = rng_original

        return {
            "r_os": ros,
            "asn_0": asn_0,
            "asn_05delta": asn_05delta,
            "asn_delta": asn_delta,
        }
