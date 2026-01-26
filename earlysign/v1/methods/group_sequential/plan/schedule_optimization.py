"""Optimal Scheduling for Group Sequential Designs.

This module provides tools to optimize the information time schedule (t_1, ..., t_K)
for group sequential designs to minimize the Average Sample Number (ASN) or
Expected ASN (E[ASN]) under a prior.

ALGORITHM DETAILS:
    - Optimization Strategy: Hyperspherical Decomposition + Multi-start Nelder-Mead.
    - Parameterization: The schedule increments x_k = t_k - t_{k-1} subject to
      sum(x) = 1, x > 0 are mapped to k-1 hyperspherical angles in [0, pi/2].
      This transforms the constrained simplex problem into a box-constrained
      problem suitable for derivative-free optimization.
    - Global Search: Uses Dirichlet sampling to find high-quality starting points
      (seeds) for the local optimizer, avoiding local minima.
    - Local Search: Uses Nelder-Mead (simplex method) on the top seeds to refine
      the schedule.
    - Statistical Engine: Uses `CanonicalJointModel` for precise boundary solving
      and ASN calculation. Supports both numerical integration (exact) and
      simulation (fast, consistent) via Common Random Numbers.

EXAMPLES:
    Optimization results for O'Brien-Fleming type spending (Power Family rho=3).
    Note: "Integral" uses exact numerical integration; "Sim(100k)" uses Monte Carlo
    simulation with 100,000 paths (Common Random Numbers).

    >>> # K=3, 100k simulations
    >>> from earlysign.v1.methods.group_sequential.plan.schedule_optimization import optimize_schedule
    >>> from earlysign.v1.methods.group_sequential.shared.spending import PowerFamilySpending
    >>> spending = PowerFamilySpending(budget=0.025, rho=3.0)
    >>> res = optimize_schedule(k_looks=3, efficacy_spending=spending, drift=3.0,
    ...                         method="simulation", n_sims=100000, seed=42)
    >>> print(f"Opt Schedule: {res.schedule}, ASN: {res.asn:.4f}")  # doctest: +SKIP
    Opt Schedule: [0.517 0.753 1.   ], ASN: 0.7804

    Full Comparison Table (Drift=3.0, alpha=0.025, rho=3):

    K   | Method     | Uniform ASN | Opt ASN   | Optimal Schedule
    ---------------------------------------------------------------------------
    2   | Integral   | 0.8650      | 0.8313    | [0.653, 1.000]
    3   | Integral   | 0.8059      | 0.7806    | [0.531, 0.758, 1.000]
    2   | Sim(100k)  | 0.8679      | 0.8344    | [0.668, 1.000]
    3   | Sim(100k)  | 0.8042      | 0.7804    | [0.517, 0.753, 1.000]
    4   | Sim(100k)  | 0.7813      | 0.7586    | [0.484, 0.659, 0.823, 1.000]
    5   | Sim(100k)  | 0.7604      | 0.7425    | [0.408, 0.584, 0.725, 0.854, 1.000]

    This shows that for strong efficacy signals (Drift=3.0), accelerating the
    early looks (front-loading) compared to uniform spacing reduces ASN.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from earlysign.v1.methods.group_sequential.plan.simulator import (
    OptimizationConfig,
    SequentialASNEstimator,
)
from earlysign.v1.methods.group_sequential.shared.spending import SpendingFunction


class HypersphericalGSDOptimizer:
    """Optimizes GSD schedules using Hyperspherical Decomposition."""

    def __init__(self, estimator: SequentialASNEstimator) -> None:
        self.estimator = estimator
        self.k = estimator.k_looks
        # Use the same seed as the estimator for consistency
        self._rng = np.random.default_rng(estimator._rng_seed)

    def _angles_to_simplex(self, angles: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform K-1 angles into K-dimensional increment vector x."""
        u = np.ones(self.k)
        sin_prod = 1.0
        for i in range(self.k - 1):
            u[i] = sin_prod * np.cos(angles[i])
            sin_prod *= np.sin(angles[i])
        u[-1] = sin_prod
        return np.square(u)

    def _simplex_to_angles(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse transform: Map increment vector x to K-1 angles."""
        u = np.sqrt(np.clip(x, 1e-10, 1.0))
        angles = []
        sin_prod = 1.0
        for i in range(self.k - 1):
            val = np.clip(u[i] / (sin_prod + 1e-14), -1.0, 1.0)
            phi = np.arccos(val)
            angles.append(phi)
            sin_prod *= np.sin(phi)
        return np.array(angles)

    def run(self) -> Tuple[NDArray[np.float64], float]:
        """Execute the optimization pipeline."""
        n_global = self.estimator.config.n_global_samples
        n_top = self.estimator.config.n_top_seeds

        # 1. Global Search (Mixture Dirichlet Sampling)
        # We use a mixture of Dirichlet distributions to ensure diversity:
        # - Alpha=1.0: Uniform on simplex (explores the center).
        # - Alpha=0.3: Sparse/Edge-heavy (explores schedules with some large/small steps).
        # This helps find optima that may be far from the uniform schedule (e.g. large initial step).
        candidates = []
        # Split budget into 3 strategies for diversity
        n_uniform = n_global // 3
        n_sparse = n_global // 3
        n_late_start = n_global - n_uniform - n_sparse

        # 1. Uniform samples (Center exploration)
        for _ in range(n_uniform):
            x_sample = self._rng.dirichlet(np.ones(self.k))
            val = self.estimator.calculate(x_sample)
            candidates.append((val, x_sample))

        # 2. Sparse samples (Edge exploration, alpha=0.3)
        for _ in range(n_sparse):
            x_sample = self._rng.dirichlet(np.full(self.k, 0.3))
            val = self.estimator.calculate(x_sample)
            candidates.append((val, x_sample))

        # 3. Late-Start samples (Focus on t_1 > 0.4)
        # Optimal schedules for conservative spending often have large t_1.
        # We sample t_1 ~ Uniform(0.4, 0.8) and distribute the rest.
        if self.k > 1:
            for _ in range(n_late_start):
                t1 = self._rng.uniform(0.4, 0.8)
                remainder = 1.0 - t1
                # Distribute remainder with sparse Dirichlet to keep "uneven" nature
                rest_sw = self._rng.dirichlet(np.full(self.k - 1, 0.5))
                x_rest = rest_sw * remainder
                x_sample = np.insert(x_rest, 0, t1)

                val = self.estimator.calculate(x_sample)
                candidates.append((val, x_sample))

        candidates.sort(key=lambda item: item[0])
        seeds = candidates[:n_top]

        # 2. Local Search (Nelder-Mead)
        best_asn = float("inf")
        best_x = np.ones(self.k) / self.k

        # Box constraints for angles [0, pi/2]
        bounds = [(0.0, np.pi / 2)] * (self.k - 1)

        for init_val, init_x in seeds:
            init_angles = self._simplex_to_angles(init_x)

            res = minimize(
                lambda a: self.estimator.calculate(self._angles_to_simplex(a)),
                init_angles,
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "xatol": self.estimator.config.tolerance,
                    "fatol": self.estimator.config.tolerance,
                    "maxfev": 200,  # Limit function evaluations per seed
                },
            )

            opt_x = self._angles_to_simplex(res.x)
            opt_asn = res.fun if res.success else self.estimator.calculate(opt_x)

            if opt_asn < best_asn:
                best_asn = opt_asn
                best_x = opt_x

        return best_x, best_asn


@dataclass
class OptimizationResult:
    """Result of schedule optimization."""

    schedule: NDArray[np.float64]
    asn: float
    power: float


def optimize_schedule(
    k_looks: int,
    efficacy_spending: Optional[SpendingFunction] = None,
    futility_spending: Optional[SpendingFunction] = None,
    alpha: float = 0.025,
    drift: float = 0.0,
    tails: int = 1,
    method: Literal["simulation", "numerical_integration"] = "simulation",
    n_sims: int = 20000,
    seed: Optional[int] = None,
) -> OptimizationResult:
    """Find the optimal information time schedule to minimize ASN.

    Derives the optimal schedule (t_1, ..., t_K) that minimizes the Average
    Sample Number (ASN) under the specified hypothesis (drift).

    Args:
        k_looks: Number of analyses (K).
        efficacy_spending: Spending function for efficacy boundaries.
        futility_spending: Spending function for futility boundaries (optional).
        alpha: Type I error rate (total).
        drift: The true effect size (theta * sqrt(I_max)) at which to minimize ASN.
        tails: 1 or 2 sided.
        method: "simulation" or "numerical_integration".
        n_sims: Number of simulations (if method="simulation").
        seed: Random seed.

    Returns:
        OptimizationResult object containing schedule, asn, and power.
    """
    config = OptimizationConfig(
        method=method, n_sims=n_sims, rng_seed=seed, n_global_samples=50, n_top_seeds=3
    )

    estimator = SequentialASNEstimator(
        k_looks=k_looks,
        alpha=alpha,
        efficacy_spending=efficacy_spending,
        futility_spending=futility_spending,
        drift=drift,
        tails=tails,
        config=config,
    )

    optimizer = HypersphericalGSDOptimizer(estimator)
    best_x, best_asn = optimizer.run()

    # Calculate final power
    _, best_power = estimator.evaluate(best_x)

    return OptimizationResult(
        schedule=np.cumsum(best_x), asn=best_asn, power=best_power
    )
