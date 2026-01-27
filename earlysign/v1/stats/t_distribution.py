"""Statistics and simulation helpers for T-distributions in group sequential tests."""

from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class TDistributionResolver:
    """Helper to transform between Normal and T distribution boundaries."""

    @staticmethod
    def significance_level_transform(
        z_boundaries: NDArray[np.float64], dofs: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Maps standard normal boundaries to T-boundaries with equal significance levels.

        t_k = qt(nu_k, 1 - Phi(z_k))
        """
        p_values = stats.norm.sf(z_boundaries)
        return cast(NDArray[np.float64], stats.t.isf(p_values, dofs))


class CanonicalTProcess:
    """Simulator for the sequence of T-statistics in a group sequential trial."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self._rng = rng or np.random.default_rng()

    def sample(
        self,
        m_counts: NDArray[np.int64],
        n_sims: int = 20000,
        drift: float = 0.0,
        p: int = 2,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Samples the joint sequence of (Z_k, T_k) statistics.

        Arguments:
            m_counts: Cumulative sample size per treatment group at each look.
            n_sims: Number of simulations.
            drift: Drift parameter for the Z-process.
            p: Parameters estimated (e.g., p=2 for 2-sample comparison).

        Returns:
            z_stats: (n_sims, k) array of standardized statistics.
            t_stats: (n_sims, k) array of T-statistics.
        """
        k = len(m_counts)
        m_inc = np.diff(m_counts, prepend=0)

        # 1. Simulate the Z-process (canonical Gaussian process)
        # Z_k = (B_A(m_k) - B_B(m_k)) / sqrt(2*m_k)
        # Actually we can just simulate the increments of sum-of-normals
        # increments of (X_Ai - X_Bi) ~ N(delta * 1, 2)
        # We assume sigma=1 for canonical simulation.

        # Delta contribution to sum: sum(inc) = drift * sqrt(m_K / 2) * (m_k / m_K)
        # J&T definition: drift theta' = mu_delta * sqrt(I_max).
        # For 2-sample, I = m / (2*sigma^2).

        # Simplified: B_k ~ N(drift * m_k / m_K * sqrt(I_max), 2*m_k)
        # Let's use the error increments directly.
        # eps_sum_inc ~ N(0, 2 * m_inc)
        # Total sum at look k = cumsum(eps_sum_inc) + delta * m_k

        delta = drift / np.sqrt(m_counts[-1] / 2.0)  # drift = delta * sqrt(I_max)

        raw_diff_inc = self._rng.normal(0, np.sqrt(2 * m_inc), size=(n_sims, k))
        cum_diff = np.cumsum(raw_diff_inc, axis=1) + delta * m_counts

        z_stats = cum_diff / np.sqrt(2 * m_counts)

        # 2. Simulate the Variance process
        # Q_k = (n_k - p) * S_k^2 / sigma^2 has independent chi-square increments
        dofs = p * m_counts - p  # e.g. 2*m - 2
        dof_inc = np.diff(dofs, prepend=0)

        # We need to handle the case where the first dof is very small.
        # But for simulation, chisquare(nu) requires nu > 0.
        # If dofs[0] is 1, it's fine.
        q_stats = np.cumsum(
            self._rng.chisquare(np.maximum(dof_inc, 1e-9), size=(n_sims, k)), axis=1
        )
        s2_ratio = q_stats / dofs

        t_stats = z_stats / np.sqrt(s2_ratio)

        return z_stats, t_stats

    def compute_stopping_probabilities(
        self,
        m_counts: NDArray[np.int64],
        upper: NDArray[np.float64],
        lower: NDArray[np.float64],
        n_sims: int = 20000,
        drift: float = 0.0,
        seed: Optional[int] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute stopping probabilities for efficacy and futility.

        Args:
            m_counts: Cumulative sample sizes.
            upper: Upper boundaries (efficacy).
            lower: Lower boundaries (futility).
            n_sims: Number of simulations.
            drift: Standardized drift.
            seed: RNG seed.

        Returns:
            Tuple of (prob_stop_efficacy, prob_stop_futility).
        """
        # Use localized RNG sequence if seed provided
        gen = np.random.default_rng(seed) if seed is not None else self._rng

        # CanonicalTProcess uses self._rng in sample, so we instantiate a new one with the specific seed.
        proc = CanonicalTProcess(rng=gen)

        _, t_samples = proc.sample(m_counts, n_sims=n_sims, drift=drift)

        k = t_samples.shape[1]
        stopped = np.zeros(n_sims, dtype=bool)
        prob_eff = np.zeros(k, dtype=float)
        prob_fut = np.zeros(k, dtype=float)

        for i in range(k):
            # Check upper crossing
            cross_u = (t_samples[:, i] > upper[i]) & ~stopped
            prob_eff[i] = np.mean(cross_u)
            stopped |= cross_u

            # Check lower crossing
            cross_l = (t_samples[:, i] < lower[i]) & ~stopped
            prob_fut[i] = np.mean(cross_l)
            stopped |= cross_l

        return prob_eff, prob_fut
