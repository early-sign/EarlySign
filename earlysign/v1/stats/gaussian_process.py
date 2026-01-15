"""Gaussian Process simulation for statistical applications.

This module provides general-purpose Gaussian Process simulation tools,
with specialized support for the canonical joint distribution used in
group sequential tests.
"""

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray


class GaussianProcess:
    """General purpose Gaussian Process simulation.

    Supports arbitrary mean and covariance functions.
    In the multivariate case, the process returns a vector of dimension D for each t.
    """

    def __init__(
        self,
        mean_func: Optional[Callable[[NDArray[Any]], NDArray[Any]]] = None,
        cov_func: Optional[Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]]] = None,
        rng: Optional[np.random.Generator] = None,
        dims: int = 1,
    ):
        """Initialize the Gaussian Process.

        Args:
            mean_func: Function that takes time points (k,) and returns means (k, D).
                Defaults to zero mean.
            cov_func: Function that takes two sets of time points (k1, k2) and returns
                the covariance matrix (k1, k2, D, D) or (k1, k2) if D=1.
            rng: Random number generator.
            dims: Dimension D of the process at each time point.
        """
        self.dims = dims
        self.mean_func = mean_func or (lambda t: np.zeros((len(t), dims)))
        self.cov_func = cov_func or (
            lambda t1, t2: np.where(t1 == t2, 1.0, 0.0)
            if dims == 1
            else (np.where(t1 == t2, 1.0, 0.0)[..., np.newaxis, np.newaxis] * np.eye(dims))
        )
        self._rng = rng or np.random.default_rng()

    def sample(self, t: NDArray[Any], n_sims: int) -> NDArray[Any]:
        """Sample multiple paths from the Gaussian Process at given time points.

        Args:
            t: Time points to sample at, shape (k,).
            n_sims: Number of simulations (paths) to generate.

        Returns:
            Array of shape (n_sims, k, D) containing sampled values.
            If D=1, returns (n_sims, k).
        """
        t_arr = np.asarray(t)
        k = len(t_arr)
        mean = self.mean_func(t_arr)  # (k, D)
        mean_flat = mean.flatten()  # (k*D,)

        if self.dims == 1:
            cov = self.cov_func(t_arr[:, np.newaxis], t_arr[np.newaxis, :])  # (k, k)
        else:
            cov_blocks = self.cov_func(
                t_arr[:, np.newaxis], t_arr[np.newaxis, :]
            )  # (k, k, D, D)
            cov = cov_blocks.transpose(0, 2, 1, 3).reshape(k * self.dims, k * self.dims)

        cov = (cov + cov.T) / 2.0
        samples_flat = self._rng.multivariate_normal(
            mean_flat, cov, size=n_sims
        )  # (n_sims, kD)

        if self.dims == 1:
            return samples_flat  # (n_sims, k)
        return samples_flat.reshape(n_sims, k, self.dims)

    def apply_stopping_rule(
        self,
        samples: NDArray[Any],
        upper: Optional[NDArray[Any]] = None,
        lower: Optional[NDArray[Any]] = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Determine where each path crosses boundaries.

        Args:
            samples: Sampled paths, shape (n_sims, k) (D=1 assumed for now).
            upper: Upper boundaries at each look, shape (k,).
            lower: Lower boundaries at each look, shape (k,).

        Returns:
            Tuple of (ever_stopped, stop_looks).
            ever_stopped: Boolean mask of shape (n_sims,).
            stop_looks: Look index at which path first stopped (1 to k).
        """
        n_sims, k = samples.shape
        stopped = np.zeros(n_sims, dtype=bool)
        stop_looks = np.full(n_sims, k, dtype=int)

        for i in range(k):
            crossing = np.zeros(n_sims, dtype=bool)
            if upper is not None:
                crossing |= samples[:, i] > upper[i]
            if lower is not None:
                crossing |= samples[:, i] < lower[i]

            just_stopped = crossing & ~stopped
            stop_looks[just_stopped] = i + 1
            stopped |= crossing

        return stopped, stop_looks

    def solve_boundary_step(
        self,
        look_idx: int,
        samples: NDArray[Any],
        ever_stopped_prev: NDArray[Any],
        target_cum_prob: float,
        side: str = "upper",
        bracket: tuple[float, float] = (-10.0, 10.0),
    ) -> float:
        """Solve for a boundary value at a specific look to match target cumulative probability.

        Args:
            look_idx: Current look index (0 to k-1).
            samples: Sampled paths.
            ever_stopped_prev: Boolean mask of paths already stopped before this look.
            target_cum_prob: Target total probability of having stopped by this look.
            side: 'upper' or 'lower' boundary to solve for.
            bracket: Root search bracket.

        Returns:
            The boundary value.
        """
        from scipy.optimize import root_scalar

        def f(val: float) -> float:
            if side == "upper":
                crossing = samples[:, look_idx] > val
            else:
                crossing = samples[:, look_idx] < val

            current_stopped = ever_stopped_prev | crossing
            return float(np.mean(current_stopped)) - target_cum_prob

        # Verify bracket
        if f(bracket[0]) * f(bracket[1]) > 0:
            # If target is not in bracket, return extreme
            return bracket[0] if f(bracket[0]) > 0 else bracket[1]

        res = root_scalar(f, bracket=bracket, xtol=1e-5)
        return float(res.root)


class CanonicalGaussianProcess(GaussianProcess):
    """Univariate Canonical Gaussian Process on [0, 1].

    Commonly used in group sequential tests.
    """

    def __init__(
        self,
        drift: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the Canonical Gaussian Process.

        Args:
            drift: Standardized drift delta = theta * sqrt(I_max).
            rng: Random number generator.
        """
        self.drift = drift
        super().__init__(
            mean_func=lambda t: (self.drift * np.sqrt(t))[:, np.newaxis],
            cov_func=self._canonical_cov,
            rng=rng,
            dims=1,
        )

    @staticmethod
    def _canonical_cov(t1: NDArray[Any], t2: NDArray[Any]) -> NDArray[Any]:
        """Compute the canonical covariance: sqrt(min(t1, t2) / max(t1, t2))."""
        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)

        res = np.zeros_like(t_min, dtype=float)
        mask = t_max > 0
        res[mask] = np.sqrt(t_min[mask] / t_max[mask])
        return res
