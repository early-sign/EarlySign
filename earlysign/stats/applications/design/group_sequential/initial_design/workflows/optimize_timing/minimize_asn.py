"""
Workflow helpers for minimizing expected sample size (ASN) via timing optimization.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm

from earlysign.stats.applications.design.group_sequential.initial_design.workflows.optimize_timing.spending import (
    SpendingStrategy,
)
from earlysign.stats.essentials.schemes.two_means.group_sequential import (
    NormalMeansASNCalculator,
)


def _project_to_min_gap(
    inner: NDArray[np.float64], min_gap: float
) -> NDArray[np.float64]:
    clipped = np.sort(np.clip(inner, 1e-6, 1 - 1e-6))
    t = np.concatenate(([0.0], clipped, [1.0]))
    for idx in range(1, len(t)):
        t[idx] = max(t[idx], t[idx - 1] + min_gap)
    if t[-1] > 1.0:
        span = max(t[-2] - t[1], 1e-9)
        scale = (1.0 - 2 * min_gap) / span
        t[1:-1] = (t[1:-1] - t[1]) * scale + min_gap
    return t[1:-1]


class MinimizeASNOptimizer:
    """Optimizer for choosing information times by minimizing ASN."""

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        sided: int,
        alternative: float,
        st_dev: float,
        allocation_ratio_planned: float,
        spending: SpendingStrategy,
        k_max: int,
        min_gap: float = 0.02,
        seed: Optional[int] = None,
        n_jobs: Optional[int] = 1,
        n_restarts: int = 12,
        restart_scale: float = 0.2,
    ) -> None:
        self.k_max = int(k_max)
        self.min_gap = float(min_gap)
        self.rng = np.random.default_rng(21 if seed is None else seed)
        self.n_jobs = n_jobs
        self.n_restarts = int(n_restarts)
        self.restart_scale = float(restart_scale)

        self.calculator = NormalMeansASNCalculator(
            alpha=alpha,
            beta=beta,
            sided=sided,
            alternative=alternative,
            st_dev=st_dev,
            allocation_ratio=allocation_ratio_planned,
            spending=spending,
        )

    def evaluate(self, info: Sequence[float]) -> float:
        return self.calculator.evaluate(list(info))

    def summarize_schedule_internal(
        self, info: Sequence[float]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        """Compute internal schedule artifacts: (rates, stagewise, expected, n_max)."""
        rates = self.calculator._validate_information_rates(info)
        per_stage = self.calculator._per_stage_alpha(rates)
        boundaries = self.calculator._z_boundaries(per_stage)
        n_max = self.calculator._n_max_from_power()
        cumulative_n = np.maximum(2.0, rates * n_max)

        sigma_eff = self.calculator.st_dev * np.sqrt(
            1.0 + 1.0 / self.calculator.allocation_ratio
        )
        kappa = (self.calculator.alternative / sigma_eff) * np.sqrt(n_max)
        mu = kappa * np.sqrt(rates)

        survival = 1.0
        stagewise = np.zeros_like(rates)
        for idx, (boundary, mean) in enumerate(zip(boundaries, mu)):
            reject_prob = float(np.clip(1.0 - norm.cdf(boundary - mean), 0.0, 1.0))
            stagewise[idx] = survival * reject_prob
            survival *= 1.0 - reject_prob

        expected = float(np.dot(stagewise, cumulative_n) + survival * n_max)
        return rates, stagewise, expected, n_max

    def objective(self, x_logit: NDArray[np.float64]) -> float:
        t_inner = _project_to_min_gap(expit(x_logit), self.min_gap)
        info = np.concatenate((t_inner, [1.0]))
        try:
            return float(self.evaluate(info.tolist()))
        except Exception:
            return 1e12

    def guards_ok(self, x_logit: NDArray[np.float64]) -> bool:
        t_inner = _project_to_min_gap(expit(x_logit), self.min_gap)
        info = np.concatenate((t_inner, [1.0]))
        try:
            rates, stagewise, _, _ = self.summarize_schedule_internal(info.tolist())
        except Exception:
            return False
        spacing_ok = bool(
            np.diff(np.concatenate(([0.0], rates))).min() >= self.min_gap - 1e-12
        )
        early = stagewise[:-1]
        early_ok = bool(len(early) == 0 or (early >= 0.01).all())
        return spacing_ok and early_ok

    def run_nm_from_start(
        self, x_start: NDArray[np.float64], maxiter: int
    ) -> Tuple[NDArray[np.float64], float]:
        res = minimize(
            self.objective,
            x_start,
            method="Nelder-Mead",
            options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-6, disp=False),
        )
        return np.asarray(res.x, dtype=float), float(res.fun)

    def minimize(self) -> List[float]:
        if self.k_max <= 1:
            return [1.0]

        init = np.linspace(0, 1, self.k_max + 1)[1:-1].clip(1e-6, 1 - 1e-6)
        x0 = logit(init)

        res0 = minimize(
            self.objective,
            x0,
            method="Nelder-Mead",
            options=dict(maxiter=600, xatol=1e-6, fatol=1e-6, disp=False),
        )
        x_best = np.asarray(res0.x, dtype=float)

        starts: List[NDArray[np.float64]] = [x_best]
        for _ in range(max(0, self.n_restarts - 1)):
            starts.append(
                x_best + self.rng.normal(0.0, self.restart_scale, size=x_best.size)
            )

        candidates: List[Tuple[NDArray[np.float64], float]] = []

        if self.n_jobs is None or self.n_jobs <= 1:
            for x_start in starts:
                x_try, f_try = self.run_nm_from_start(x_start, 300)
                candidates.append((x_try, f_try))
        else:
            with ThreadPoolExecutor(max_workers=max(1, self.n_jobs)) as executor:
                futures = [
                    executor.submit(self.run_nm_from_start, x_start, 300)
                    for x_start in starts
                ]
                for fut in as_completed(futures):
                    try:
                        candidates.append(fut.result())
                    except Exception:
                        continue

        feasible = [(x, f) for (x, f) in candidates if self.guards_ok(x)]
        if feasible:
            x_best, _ = min(feasible, key=lambda item: item[1])
        elif candidates:
            x_best, _ = min(candidates, key=lambda item: item[1])

        info = np.concatenate((_project_to_min_gap(expit(x_best), self.min_gap), [1.0]))
        # Reuse guards_ok to avoid duplicating the spacing/early checks.
        if not self.guards_ok(x_best):
            return []

        try:
            rates, _, _, _ = self.summarize_schedule_internal(info.tolist())
        except Exception:
            return []

        return [float(np.round(x, 10)) for x in rates]
