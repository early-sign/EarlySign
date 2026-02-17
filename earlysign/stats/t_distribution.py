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
        t: NDArray[np.float64],
        n_sims: int = 20000,
        drift: float = 0.0,
        m_counts: Optional[NDArray[np.int64]] = None,
        p: int = 2,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Samples the joint sequence of (Z_k, T_k) statistics.

        Arguments:
            t: Information times (0 to 1).
            n_sims: Number of simulations.
            drift: Drift parameter for the Z-process.
            m_counts: Optional raw cumulative sample size (if provided, t is derived).
            p: Parameters estimated (e.g., p=2 for 2-sample comparison).
            rng: Optional RNG override.

        Returns:
            z_stats: (n_sims, k) array of standardized statistics.
            t_stats: (n_sims, k) array of T-statistics.
        """
        gen = rng or self._rng
        k = len(t)

        # If m_counts not provided, derive from t vs max_n
        # However, T-distribution dofs depend on actual N, so m_counts is preferred.
        if m_counts is None:
            # Fallback assuming t = m / max_m (unit info)
            # This is an approximation if the actual max_m is unknown.
            # Usually for T-dist we need actual n.
            raise ValueError(
                "m_counts (raw sample sizes) is required for CanonicalTProcess as dofs depends on n."
            )

        m_inc = np.diff(m_counts, prepend=0)
        delta = drift / np.sqrt(m_counts[-1] / 2.0)  # drift = delta * sqrt(I_max)

        raw_diff_inc = gen.normal(0, np.sqrt(2 * m_inc), size=(n_sims, k))
        cum_diff = np.cumsum(raw_diff_inc, axis=1) + delta * m_counts

        z_stats = cum_diff / np.sqrt(2 * m_counts)

        # 2. Simulate the Variance process
        # Q_k = (n_k - p) * S_k^2 / sigma^2 has independent chi-square increments
        dofs = p * m_counts - p  # e.g. 2*m - 2
        dof_inc = np.diff(dofs, prepend=0)

        q_stats = np.cumsum(
            gen.chisquare(np.maximum(dof_inc, 1e-9), size=(n_sims, k)), axis=1
        )
        s2_ratio = q_stats / dofs

        t_stats = z_stats / np.sqrt(s2_ratio)

        return z_stats, t_stats

    def compute_stopping_probabilities(
        self,
        t: NDArray[np.float64],
        upper: Optional[NDArray[np.float64]] = None,
        lower: Optional[NDArray[np.float64]] = None,
        n_sims: int = 20000,
        drift: float = 0.0,
        seed: Optional[int] = None,
        m_counts: Optional[NDArray[np.int64]] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute stopping probabilities for efficacy and futility.

        Args:
            t: Information times.
            upper: Upper boundaries (efficacy).
            lower: Lower boundaries (futility).
            n_sims: Number of simulations.
            drift: Standardized drift.
            seed: RNG seed for reproducibility.
            m_counts: Raw sample sizes (required for T-distribution).

        Returns:
            Tuple of (prob_stop_efficacy, prob_stop_futility).
        """
        # Use localized RNG sequence if seed provided
        gen = np.random.default_rng(seed) if seed is not None else self._rng

        u_arr = upper if upper is not None else np.full(len(t), np.inf)
        l_arr = lower if lower is not None else np.full(len(t), -np.inf)

        # CanonicalTProcess uses self._rng in sample, so we instantiate a new one with the specific seed.
        proc = CanonicalTProcess(rng=gen)

        _, t_samples = proc.sample(t, n_sims=n_sims, drift=drift, m_counts=m_counts)

        k = t_samples.shape[1]
        stopped = np.zeros(n_sims, dtype=bool)
        prob_eff = np.zeros(k, dtype=float)
        prob_fut = np.zeros(k, dtype=float)

        for i in range(k):
            # Check upper crossing
            cross_u = (t_samples[:, i] > u_arr[i]) & ~stopped
            prob_eff[i] = np.mean(cross_u)
            stopped |= cross_u

            # Check lower crossing
            cross_l = (t_samples[:, i] < l_arr[i]) & ~stopped
            prob_fut[i] = np.mean(cross_l)
            stopped |= cross_l

        return prob_eff, prob_fut
