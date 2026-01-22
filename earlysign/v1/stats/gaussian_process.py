"""Gaussian Process simulation for statistical applications.

This module provides general-purpose Gaussian Process simulation tools,
with specialized support for the canonical joint distribution used in
group sequential tests.
"""

from typing import Any, Callable, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal, norm


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
            lambda t1, t2: (
                np.where(t1 == t2, 1.0, 0.0)
                if dims == 1
                else (
                    np.where(t1 == t2, 1.0, 0.0)[..., np.newaxis, np.newaxis]
                    * np.eye(dims)
                )
            )
        )
        self._rng = rng or np.random.default_rng()

    def sample(
        self, t: NDArray[Any], n_sims: int, rng: Optional[np.random.Generator] = None
    ) -> NDArray[Any]:
        """Sample multiple paths from the Gaussian Process at given time points.

        Args:
            t: Time points to sample at, shape (k,).
            n_sims: Number of simulations (paths) to generate.
            rng: Optional RNG override (useful for CRN).

        Returns:
            Array of shape (n_sims, k, D) containing sampled values.
            If D=1, returns (n_sims, k).
        """
        gen = rng or self._rng
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
        samples_flat = gen.multivariate_normal(
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
        return float(res)


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

    def sample(
        self, t: NDArray[Any], n_sims: int, rng: Optional[np.random.Generator] = None
    ) -> NDArray[Any]:
        """Sample multiple paths using Brownian motion increments."""
        gen = rng or self._rng
        t_arr = np.asarray(t)
        k = len(t_arr)
        dt = np.diff(np.insert(t_arr, 0, 0))

        # B(t) has drift self.drift and unit variance per unit time
        db = gen.normal(self.drift * dt, np.sqrt(dt), (n_sims, k))
        b = np.cumsum(db, axis=1)

        # Z(t) = B(t) / sqrt(t)
        z = b / np.sqrt(t_arr)
        return cast(NDArray[Any], z)

    @staticmethod
    def _canonical_cov(t1: NDArray[Any], t2: NDArray[Any]) -> NDArray[Any]:
        """Compute the canonical covariance: sqrt(min(t1, t2) / max(t1, t2))."""
        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)

        res = np.zeros_like(t_min, dtype=float)
        mask = t_max > 0
        res[mask] = np.sqrt(t_min[mask] / t_max[mask])
        return res

    def compute_crossing_probability(
        self,
        t: Sequence[float] | NDArray[Any],
        upper: Optional[Sequence[float] | NDArray[Any]] = None,
        lower: Optional[Sequence[float] | NDArray[Any]] = None,
        n_sims: int = 20000,
        method: str = "simulation",
        seed: Optional[int] = None,
    ) -> float:
        """Compute the probability of crossing the specified boundaries.

        This serves as a high-level interface for numerical integration or simulation
        engines to determine stopping probabilities for a given boundary shape.

        Args:
            t: Information times.
            upper: Upper boundary vector.
            lower: Lower boundary vector.
            n_sims: Number of simulations to use (if method="simulation").
            method: "simulation" (Monte Carlo) or "numerical_integration" (Jennison-Turnbull).
            seed: Optional seed for reproducibility (CRN) in simulation mode.

        Returns:
            Probability of crossing either upper or lower boundary at any look.
        """
        t_arr = np.asarray(t)
        u_arr = np.asarray(upper) if upper is not None else None
        l_arr = np.asarray(lower) if lower is not None else None

        if method == "numerical_integration":
            return self._compute_crossing_numerical(t_arr, u_arr, l_arr)
        elif method == "simulation":
            return self._compute_crossing_simulation(t_arr, u_arr, l_arr, n_sims, seed)
        else:
            raise ValueError(f"Method '{method}' is not implemented.")

    def _compute_crossing_numerical(
        self,
        t_arr: NDArray[Any],
        u_arr: Optional[NDArray[Any]],
        l_arr: Optional[NDArray[Any]],
    ) -> float:
        """Numerical integration engine for crossing probability."""
        k = len(t_arr)
        if k == 1:
            # Efficient 1D case
            mu = self.drift * np.sqrt(t_arr[0])
            u = u_arr[0] if u_arr is not None else np.inf
            lower_bound = l_arr[0] if l_arr is not None else -np.inf
            return 1.0 - float(norm.cdf(u, loc=mu) - norm.cdf(lower_bound, loc=mu))

        # Canonical covariance: Cov(Z_i, Z_j) = sqrt(t_i/t_j) for i <= j
        cov = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if t_arr[i] > 0 and t_arr[j] > 0:
                    cov[i, j] = np.sqrt(
                        min(t_arr[i], t_arr[j]) / max(t_arr[i], t_arr[j])
                    )
                elif i == j:
                    cov[i, j] = 1.0

        mean = self.drift * np.sqrt(t_arr)

        upper_bound = u_arr if u_arr is not None else np.full(k, np.inf)
        lower_bound = l_arr if l_arr is not None else np.full(k, -np.inf)

        try:
            # P(no crossing) = P(l < Z < u)
            # P(crossing) = 1 - P(no crossing)
            prob_within = multivariate_normal.cdf(
                upper_bound,
                mean=mean,
                cov=cov,
                lower_limit=lower_bound,
                allow_singular=True,
                abseps=1e-5,
            )
            return 1.0 - float(prob_within)
        except Exception as e:
            raise RuntimeError(f"Numerical integration failed: {e}") from e

    def _compute_crossing_simulation(
        self,
        t_arr: NDArray[Any],
        u_arr: Optional[NDArray[Any]],
        l_arr: Optional[NDArray[Any]],
        n_sims: int,
        seed: Optional[int] = None,
    ) -> float:
        """Simulation (Monte Carlo) engine for crossing probability."""
        # Use localized RNG sequence if seed provided (CRN)
        gen = np.random.default_rng(seed) if seed is not None else self._rng
        samples = self.sample(t_arr, n_sims, rng=gen)

        stopped, _ = self.apply_stopping_rule(samples, upper=u_arr, lower=l_arr)
        return float(np.mean(stopped))
