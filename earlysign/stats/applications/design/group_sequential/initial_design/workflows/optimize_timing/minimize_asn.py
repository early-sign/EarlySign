"""
Workflow helpers for minimizing expected sample size (ASN) via timing optimization.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit

from earlysign.stats.applications.design.group_sequential.initial_design.workflows.optimize_timing.spending import (
    SpendingStrategy,
)
from earlysign.stats.essentials.primitives.group_sequential import ASNDesignSummary
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


def _build_calculator(
    *,
    alpha: float,
    beta: float,
    sided: int,
    alternative: float,
    st_dev: float,
    allocation_ratio_planned: float,
    spending: SpendingStrategy,
) -> NormalMeansASNCalculator:
    return NormalMeansASNCalculator(
        alpha=alpha,
        beta=beta,
        sided=sided,
        alternative=alternative,
        st_dev=st_dev,
        allocation_ratio=allocation_ratio_planned,
        spending=spending,
    )


def minimize_asn_schedule(
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
) -> List[float]:
    """Return information fractions that minimize expected sample size."""
    if k_max <= 1:
        return [1.0]

    rng = np.random.default_rng(21 if seed is None else seed)
    calculator = _build_calculator(
        alpha=alpha,
        beta=beta,
        sided=sided,
        alternative=alternative,
        st_dev=st_dev,
        allocation_ratio_planned=allocation_ratio_planned,
        spending=spending,
    )

    def evaluate(info: Sequence[float]) -> ASNDesignSummary:
        return calculator.evaluate(list(info))

    def objective(x_logit: NDArray[np.float64]) -> float:
        t_inner = _project_to_min_gap(expit(x_logit), min_gap)
        info = np.concatenate((t_inner, [1.0]))
        try:
            summary = evaluate(info.tolist())
            return summary.expected_sample_size
        except Exception:
            return 1e12

    def guards_ok(x_logit: NDArray[np.float64]) -> bool:
        t_inner = _project_to_min_gap(expit(x_logit), min_gap)
        info = np.concatenate((t_inner, [1.0]))
        try:
            summary = evaluate(info.tolist())
        except Exception:
            return False
        spacing_ok = bool(
            np.diff(np.concatenate(([0.0], summary.information_rates))).min()
            >= min_gap - 1e-12
        )
        early = summary.stagewise_rejection[:-1]
        early_ok = bool(len(early) == 0 or (early >= 0.01).all())
        return spacing_ok and early_ok

    def run_nm_from_start(
        x_start: NDArray[np.float64], maxiter: int
    ) -> Tuple[NDArray[np.float64], float]:
        res = minimize(
            objective,
            x_start,
            method="Nelder-Mead",
            options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-6, disp=False),
        )
        return np.asarray(res.x, dtype=float), float(res.fun)

    init = np.linspace(0, 1, k_max + 1)[1:-1].clip(1e-6, 1 - 1e-6)
    x0 = logit(init)

    res0 = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options=dict(maxiter=600, xatol=1e-6, fatol=1e-6, disp=False),
    )
    x_best = np.asarray(res0.x, dtype=float)

    starts: List[NDArray[np.float64]] = [x_best]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(x_best + rng.normal(0.0, restart_scale, size=x_best.size))

    candidates: List[Tuple[NDArray[np.float64], float]] = []

    if n_jobs is None or n_jobs <= 1:
        for x_start in starts:
            x_try, f_try = run_nm_from_start(x_start, 300)
            candidates.append((x_try, f_try))
    else:
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as executor:
            futures = [
                executor.submit(run_nm_from_start, x_start, 300) for x_start in starts
            ]
            for fut in as_completed(futures):
                try:
                    candidates.append(fut.result())
                except Exception:
                    continue

    feasible = [(x, f) for (x, f) in candidates if guards_ok(x)]
    if feasible:
        x_best, _ = min(feasible, key=lambda item: item[1])
    elif candidates:
        x_best, _ = min(candidates, key=lambda item: item[1])

    info = np.concatenate((_project_to_min_gap(expit(x_best), min_gap), [1.0]))
    try:
        summary = evaluate(info.tolist())
    except Exception:
        return []

    spacing_ok = bool(
        np.diff(np.concatenate(([0.0], summary.information_rates))).min()
        >= min_gap - 1e-12
    )
    early = summary.stagewise_rejection[:-1]
    early_ok = bool(len(early) == 0 or (early >= 0.01).all())
    if not (spacing_ok and early_ok):
        return []

    return [float(np.round(x, 10)) for x in summary.information_rates]


def summarize_schedule(
    *,
    information_rates: Iterable[float],
    alpha: float,
    beta: float,
    sided: int,
    alternative: float,
    st_dev: float,
    allocation_ratio_planned: float,
    spending: SpendingStrategy,
) -> dict[str, Any]:
    """Return summary metrics for a candidate information schedule."""
    calculator = _build_calculator(
        alpha=alpha,
        beta=beta,
        sided=sided,
        alternative=alternative,
        st_dev=st_dev,
        allocation_ratio_planned=allocation_ratio_planned,
        spending=spending,
    )
    summary = calculator.evaluate(information_rates)
    return {
        "k": int(summary.information_rates.size),
        "information_rates": summary.information_rates.tolist(),
        "reject_per_stage": summary.stagewise_rejection.tolist(),
        "expected_subjects_H1": summary.expected_sample_size,
        "max_subjects": summary.max_sample_size,
    }


# Backwards compatibility helpers for existing call sites


def get_optimal_information_rates(**kwargs: Any) -> List[float]:
    return minimize_asn_schedule(**kwargs)


def get_design_characteristics(**kwargs: Any) -> dict[str, Any]:
    return summarize_schedule(**kwargs)
