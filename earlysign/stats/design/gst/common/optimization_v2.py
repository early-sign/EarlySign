"""
Group sequential design optimizer with strategy-injected spending and parallelizable
random restarts.

This module provides an "optimal information rate" finder for group-sequential designs,
minimizing expected sample size under H1.

Key ideas
---------
1) Separation of concerns:
   • Optimizer searches over information rates t = [t1, ..., tk=1]
   • Engine evaluates a given t for a chosen endpoint model (normal means here)
   • Spending strategies are injected (no branching in the Engine)

2) Extensibility:
   • Swap endpoint model by replacing the Engine
   • Swap α-spending rule by passing a different SpendingStrategy
   • Add β-spending/futility later by extending the strategy interface

3) Parallelizable random restarts:
   • Optional threaded restarts (controlled by `n_jobs`) to escape shallow minima
   • If `n_jobs <= 1`, restarts run serially

Example
-------
Basic usage with an O'Brien–Fleming α-spending strategy:

>>> spending = OBFSpending(alpha=0.05, sided=2)
>>> info = get_optimal_information_rates(
...     alpha=0.05, beta=0.10, sided=2,
...     alternative=0.2, st_dev=1.0, allocation_ratio_planned=1.0,
...     spending=spending,
...     k_max=4, min_gap=0.02, seed=21,
...     n_jobs=1, n_restarts=6, restart_scale=0.2,
... )
>>> bool(len(info) == 4 and abs(info[-1] - 1.0) < 1e-12)
True
>>> summary = get_design_characteristics(
...     information_rates=info,
...     alpha=0.05, beta=0.10, sided=2,
...     alternative=0.2, st_dev=1.0, allocation_ratio_planned=1.0,
...     spending=spending,
... )
>>> "expected_subjects_H1" in summary
True
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm

# ======================================================================================
#  Spending Strategy Interfaces
# ======================================================================================


class SpendingStrategy(Protocol):
    """
    Strategy interface for α-spending and boundary construction.

    This protocol allows injecting different α-spending families and boundary
    rules (e.g., O'Brien–Fleming, Pocock, Haybittle–Peto, user-defined).

    Required methods
    ----------------
    cumulative_alpha(t: np.ndarray) -> np.ndarray
        Returns cumulative α(t) for each information fraction in t (monotone).
        t must be in (0, 1], typically strictly increasing and ending at 1.

    boundaries_from_stage_alpha(stage_alpha: np.ndarray) -> np.ndarray
        Given per-stage α_i (increments of cumulative α), returns a vector of
        one-sided z-critical values (efficacy boundaries) for each stage.
        For two-sided tests, sidedness should already be reflected when
        computing the cumulative α(t) (e.g., using α/2 inside OBF).
    """

    def cumulative_alpha(self, t: np.ndarray) -> np.ndarray: ...
    def boundaries_from_stage_alpha(self, stage_alpha: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------------------
#  Concrete Spending Strategies
# --------------------------------------------------------------------------------------


@dataclass
class OBFSpending(SpendingStrategy):
    """
    Lan–DeMets O'Brien–Fleming spending strategy.

    For two-sided tests we split α at the spending level as:
        α(t) = 2 - 2 Φ(z_{α/2} / √t)
    For one-sided tests:
        α(t) = 1 - Φ(z_{α} / √t)

    The "sided" parameter determines the quantile used in the above formula.
    Boundary construction uses per-stage α_i via z = Φ^{-1}(1 - α_i).
    """

    alpha: float
    sided: int  # 1 or 2

    def cumulative_alpha(self, t: np.ndarray) -> np.ndarray:
        t = np.maximum(t, 1e-12)
        if self.sided == 2:
            z = norm.ppf(1 - self.alpha / 2.0)
            result: np.ndarray = 2 - 2 * norm.cdf(z / np.sqrt(t))
            return result
        else:
            z = norm.ppf(1 - self.alpha)
            result_one: np.ndarray = 1 - norm.cdf(z / np.sqrt(t))
            return result_one

    def boundaries_from_stage_alpha(self, stage_alpha: np.ndarray) -> np.ndarray:
        # One-sided z-boundaries; "sided" already handled in cumulative spending.
        a = np.clip(stage_alpha, 1e-16, 1 - 1e-16)
        result: np.ndarray = norm.ppf(1 - a)
        return result


@dataclass
class PocockSpending(SpendingStrategy):
    """
    Lan–DeMets Pocock-like spending strategy:

        α(t) = α * log(1 + (e − 1) t)

    Boundary construction uses per-stage α_i via z = Φ^{-1}(1 − α_i).
    """

    alpha: float

    def cumulative_alpha(self, t: np.ndarray) -> np.ndarray:
        result: np.ndarray = self.alpha * np.log1p((np.e - 1.0) * np.maximum(t, 0.0))
        return result

    def boundaries_from_stage_alpha(self, stage_alpha: np.ndarray) -> np.ndarray:
        a = np.clip(stage_alpha, 1e-16, 1 - 1e-16)
        result: np.ndarray = norm.ppf(1 - a)
        return result


@dataclass
class HPSpending(SpendingStrategy):
    """
    Haybittle–Peto-style strategy by composition:

    - Allocate α across looks using a base cumulative-spending strategy
      (e.g., OBFSpending) to obtain per-stage α_i.
    - Then apply a minimum interim |Z| floor (e.g., 3.0) when constructing
      boundaries. The final look typically remains governed by the spending.

    This reproduces the spirit of Haybittle–Peto: very stringent early looks,
    while allowing conventional spending-based control overall.
    """

    base: SpendingStrategy
    z_floor: float = 3.0  # minimum |Z| at interim (applied to all looks here)

    def cumulative_alpha(self, t: np.ndarray) -> np.ndarray:
        result: np.ndarray = self.base.cumulative_alpha(t)
        return result

    def boundaries_from_stage_alpha(self, stage_alpha: np.ndarray) -> np.ndarray:
        z = self.base.boundaries_from_stage_alpha(stage_alpha)
        result: np.ndarray = np.maximum(z, self.z_floor)
        return result


# ======================================================================================
#  Endpoint Engine
# ======================================================================================

TypeOfDesignName = str  # For user labeling only; no branching on this.


@dataclass
class NormalMeansTwoArmEngine:
    """
    NormalMeansTwoArmEngine
    =======================

    Design philosophy (why this Engine exists)
    ------------------------------------------
    The Engine encapsulates *endpoint-specific* evaluation while keeping the
    *optimization* generic. Given a set of information rates
        t = [t1, ..., tk = 1],
    the optimizer asks this Engine:

        (i) What are the stagewise rejection probabilities under H1?
       (ii) What is the expected sample size EN(H1)?

    Everything about how to answer that question—distributional assumptions,
    spending/boundaries, information→sample-size mapping—lives here.

    Separation of concerns
    ----------------------
    - Optimizer (outside this class):
        * Searches over admissible t (spacing constraints, restarts, Nelder–Mead).
        * Treats the Engine as a black-box objective EN(t).
    - Engine (this class):
        * Defines the statistical model (two-arm normal means).
        * Turns cumulative information rates into per-stage α via the injected
          spending strategy, derives one-sided Z boundaries, and evaluates
          stagewise rejection probabilities under H1.
        * Maps information fractions to cumulative sample size and computes EN.

    Inputs (immutable configuration)
    --------------------------------
    - alpha, beta, sided:
        Frequentist error targets and one-/two-sidedness (used by spending).
    - alternative, st_dev, allocation_ratio:
        Effect and scale under H1, and randomization ratio n_T / n_C.
    - spending:
        A SpendingStrategy defining α(t) and how to convert stage α_i into Z
        boundaries (e.g., OBF, Pocock, Haybittle–Peto-style composition).
    - min_stage_alpha:
        Numerical floor for per-stage α to avoid degenerate quantiles.

    Public contract
    ---------------
    design_and_plan(info_rates: Iterable[float]) -> (EN_H1: float, reject_per_stage: np.ndarray)

    Required preconditions:
        - information rates strictly increasing in (0, 1], last element == 1.0
    Returned values:
        - EN_H1: expected total subjects under H1 (approximate)
        - reject_per_stage: length-k vector of stagewise rejection probabilities
          (independent-looks approximation for speed/smoothness)

    Modeling choices & approximations
    ---------------------------------
    - Z-statistics under H1 are modeled with mean drift μ_i = κ sqrt(t_i),
      where κ = (δ / σ_eff) * sqrt(N_max). This ties signal growth to information.
    - N_max is calibrated via fixed-sample power sizing to (α, β) with a mild
      inflation (~1.05) to reflect group-sequential penalty; it provides a
      common scale for EN comparisons across t.
    - Stagewise rejection probabilities assume independence across looks
      (ignoring correlation). This yields a fast, smooth objective for
      optimization. If you need higher fidelity, replace the stagewise block
      with canonical joint normal probabilities or multivariate integration.
    - Information→sample size: n_i = t_i * N_max (linear with information),
      consistent with constant variance and fixed allocation ratio.

    Extensibility
    -------------
    - New endpoints (e.g., Rates, Survival) should implement the *same public
      contract*:
          design_and_plan(info_rates) -> (EN_H1, reject_per_stage)
      and are free to redefine:
        * the spending/boundary policy,
        * the mapping from information to sample size/events,
        * the Z/LR distribution under H1 (and correlation if modeled).
    - The optimizer then works unchanged.

    Determinism & state
    -------------------
    - The Engine is stateless and deterministic: given fixed inputs, it returns
      the same outputs. Randomness (if any) must be confined to the optimizer.

    Invariants & guards
    -------------------
    - Per-stage α_i are lower-bounded by `min_stage_alpha` to avoid NaN quantiles.
    - Boundaries are one-sided; two-sided tests are handled within spending.
    - design_and_plan validates that info_rates includes 1.0 and lies in (0, 1].

    Caveats
    -------
    - The independent-looks approximation can slightly misrepresent early-stage
      rejection when boundaries are extreme or when k is large. If this matters,
      consider plugging in a correlation-aware evaluator.

    """

    alpha: float
    beta: float
    sided: int
    alternative: float
    st_dev: float
    allocation_ratio: float
    spending: SpendingStrategy
    min_stage_alpha: float = 1e-9

    # ---- internal helpers ----------------------------------------------------

    def _per_stage_alpha(self, t: np.ndarray) -> np.ndarray:
        """Compute per-stage α_i by differencing cumulative α(t) from the injected strategy."""
        cum = self.spending.cumulative_alpha(t)
        inc = np.diff(np.concatenate([[0.0], cum]))
        inc = np.maximum(inc, self.min_stage_alpha)
        # Re-normalize to sum to α for numerical robustness (tiny rounding issues).
        result: np.ndarray = inc * (self.alpha / inc.sum())
        return result

    def _z_boundaries(self, t: np.ndarray) -> np.ndarray:
        """Map per-stage α_i into one-sided Z boundaries via the strategy."""
        a_i = self._per_stage_alpha(t)
        result: np.ndarray = self.spending.boundaries_from_stage_alpha(a_i)
        return result

    def _n_max_from_power(self) -> float:
        """Calibrate notional N_max to achieve target fixed-sample power."""
        z_alpha = norm.ppf(1 - (self.alpha / 2.0 if self.sided == 2 else self.alpha))
        z_beta = norm.ppf(1 - self.beta)
        r = self.allocation_ratio
        n_per_group = ((z_alpha + z_beta) * self.st_dev / abs(self.alternative)) ** 2
        n_total_fixed = n_per_group * (1 + r)
        return float(n_total_fixed * 1.05)  # mild inflation for GSD penalty

    # ---- public method -------------------------------------------------------

    def design_and_plan(self, info_rates: Iterable[float]) -> Tuple[float, np.ndarray]:
        """
        Evaluate a candidate set of information rates.

        Parameters
        ----------
        info_rates : Iterable[float]
            Strictly increasing values in (0, 1] including 1.0 as the last value.

        Returns
        -------
        (EN_H1, reject_per_stage)
            EN_H1 : float
                Expected number of subjects under H1 (approximate).
            reject_per_stage : np.ndarray
                Stagewise rejection probabilities under H1 (length k).
        """
        t = np.array(sorted({float(x) for x in info_rates if 0.0 < x <= 1.0}))
        if len(t) == 0 or t[-1] != 1.0:
            raise ValueError("information rates must lie in (0, 1] and include 1.0")

        # Boundaries and notional sample sizes
        z_eff = self._z_boundaries(t)
        n_max = self._n_max_from_power()
        n_i = np.maximum(2.0, t * n_max)

        # Stagewise approximation: Z_i ~ N(μ_i, 1), with μ_i = κ sqrt(t_i)
        r = self.allocation_ratio
        sigma_eff = self.st_dev * np.sqrt(1 + 1.0 / r)
        kappa = (self.alternative / sigma_eff) * np.sqrt(n_max)
        mu = kappa * np.sqrt(t)

        surv = 1.0
        rej = np.zeros(len(t))
        for i in range(len(t)):
            p_i = float(np.clip(1.0 - norm.cdf(z_eff[i] - mu[i]), 0.0, 1.0))
            rej[i] = surv * p_i
            surv *= 1.0 - p_i

        en = float(np.dot(rej, n_i) + surv * n_max)
        return en, rej


# ======================================================================================
#  Optimizer (search over information rates) with optional parallel restarts
# ======================================================================================


def _project_to_min_gap(inner: np.ndarray, min_gap: float) -> np.ndarray:
    """
    Project t1..t_{k-1} to satisfy the spacing constraint:
      0 < t1 < ... < t_{k-1} < 1  and  min(diff([0, t1..t_{k-1}, 1])) >= min_gap.

    The projection is order-preserving and uses a simple forward pass and
    proportional rescaling near the upper bound if needed.
    """
    x = np.sort(np.clip(inner, 1e-6, 1 - 1e-6))
    t = np.concatenate([[0.0], x, [1.0]])
    for i in range(1, len(t)):
        t[i] = max(t[i], t[i - 1] + min_gap)
    if t[-1] > 1.0:
        span = max(t[-2] - t[1], 1e-9)
        scale = (1.0 - 2 * min_gap) / span
        t[1:-1] = (t[1:-1] - t[1]) * scale + min_gap
    return t[1:-1]


def get_optimal_information_rates(
    *,
    # Endpoint & design parameters
    alpha: float,
    beta: float,
    sided: int,
    alternative: float,
    st_dev: float,
    allocation_ratio_planned: float,
    # Spending strategy (injected)
    spending: SpendingStrategy,
    # Optimizer config
    k_max: int,
    min_gap: float = 0.02,
    seed: Optional[int] = None,
    # Parallel restarts config
    n_jobs: Optional[int] = 1,
    n_restarts: int = 12,
    restart_scale: float = 0.2,
) -> List[float]:
    """
    Optimize t_1, ..., t_{k-1} to minimize EN(H1).

    Parameters
    ----------
    alpha, beta, sided, alternative, st_dev, allocation_ratio_planned :
        Endpoint configuration (see Engine docstring).
    spending : SpendingStrategy
        Injected α-spending strategy used to construct boundaries.
    k_max : int
        Total number of looks (including final). If k_max <= 1, returns [1.0].
    min_gap : float
        Minimum spacing between successive information points (including edges 0 and 1).
    seed : Optional[int]
        RNG seed for perturbations in random restarts.
    n_jobs : Optional[int]
        Number of parallel threads for random restarts. If <= 1, restarts run serially.
    n_restarts : int
        Total number of restart attempts (including the incumbent).
    restart_scale : float
        Standard deviation of Gaussian perturbations applied to the incumbent in logit space.

    Returns
    -------
    List[float]
        Optimized information rates [t1, ..., t_{k-1}, 1.0].
        Returns [] if guards fail (spacing or tiny early-stage rejection).
    """
    if k_max <= 1:
        return [1.0]

    rng = np.random.default_rng(21 if seed is None else seed)

    engine = NormalMeansTwoArmEngine(
        alpha=alpha,
        beta=beta,
        sided=sided,
        alternative=alternative,
        st_dev=st_dev,
        allocation_ratio=allocation_ratio_planned,
        spending=spending,
    )

    def objective(x_logit: np.ndarray) -> float:
        """Objective = EN(H1) evaluated at projected information rates."""
        t_inner = _project_to_min_gap(expit(x_logit), min_gap)
        try:
            en, _ = engine.design_and_plan(np.r_[t_inner, 1.0])
            return en
        except Exception:
            return 1e12

    def _guards_ok(x_logit: np.ndarray) -> bool:
        """Check spacing constraint and early-stage rejection guard."""
        t_inner = _project_to_min_gap(expit(x_logit), min_gap)
        info = np.r_[t_inner, 1.0]
        try:
            _, rej = engine.design_and_plan(info)
            spacing_ok = bool(np.diff(np.r_[0.0, info]).min() >= min_gap - 1e-12)
            early_ok = bool((len(rej) <= 1) or (rej[:-1] >= 0.01).all())
            return spacing_ok and early_ok
        except Exception:
            return False

    def _run_nm_from_start(
        x_start: np.ndarray, maxiter: int
    ) -> Tuple[np.ndarray, float]:
        """Run a single Nelder–Mead optimization from a given starting point."""
        res_local = minimize(
            objective,
            x_start,
            method="Nelder-Mead",
            options=dict(maxiter=maxiter, xatol=1e-6, fatol=1e-6, disp=False),
        )
        return res_local.x, float(res_local.fun)

    # Initial guess: equally spaced points (excluding 0 and 1), in logit space.
    init = np.linspace(0, 1, k_max + 1)[1:-1].clip(1e-6, 1 - 1e-6)
    x0 = logit(init)

    # First NM run (serial, high-iteration budget).
    res0 = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options=dict(maxiter=600, xatol=1e-6, fatol=1e-6, disp=False),
    )
    x_best, f_best = res0.x, float(res0.fun)

    # Prepare a batch of perturbed starts (including the incumbent).
    batch = [x_best]
    for _ in range(max(0, n_restarts - 1)):
        batch.append(x_best + rng.normal(0.0, restart_scale, size=x_best.size))

    candidates: List[Tuple[np.ndarray, float]] = []

    if n_jobs is None or n_jobs <= 1:
        # Serial restarts.
        for x_start in batch:
            x_try, f_try = _run_nm_from_start(x_start, 300)
            candidates.append((x_try, f_try))
    else:
        # Parallel restarts using threads (SciPy/NumPy release the GIL in heavy kernels).
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
            futures = [ex.submit(_run_nm_from_start, x_start, 300) for x_start in batch]
            for fut in as_completed(futures):
                try:
                    x_try, f_try = fut.result()
                    candidates.append((x_try, f_try))
                except Exception:
                    # Robust to any single worker failure
                    continue

    # Choose the best feasible candidate by guards; otherwise the best objective.
    feasible = [(x, f) for (x, f) in candidates if _guards_ok(x)]
    if feasible:
        x_best, f_best = min(feasible, key=lambda xf: xf[1])
    elif candidates:
        x_best, f_best = min(candidates, key=lambda xf: xf[1])

    # Finalize: project, evaluate, and validate guards one last time.
    info = np.r_[_project_to_min_gap(expit(x_best), min_gap), 1.0]
    try:
        _, rej = engine.design_and_plan(info)
        spacing_ok = bool(np.diff(np.r_[0.0, info]).min() >= min_gap - 1e-12)
        early_ok = bool((len(rej) <= 1) or (rej[:-1] >= 0.01).all())
        if not (spacing_ok and early_ok):
            return []
    except Exception:
        return []
    # Convert to list of floats, rounding each element
    return [float(np.round(x, 10)) for x in info]


# ======================================================================================
#  Convenience: Summaries
# ======================================================================================


def get_design_characteristics(
    *,
    information_rates: Iterable[float],
    alpha: float,
    beta: float,
    sided: int,
    alternative: float,
    st_dev: float,
    allocation_ratio_planned: float,
    spending: SpendingStrategy,
) -> Dict[str, Any]:
    """
    Produce a compact summary for quick inspection / debugging.
    """
    engine = NormalMeansTwoArmEngine(
        alpha=alpha,
        beta=beta,
        sided=sided,
        alternative=alternative,
        st_dev=st_dev,
        allocation_ratio=allocation_ratio_planned,
        spending=spending,
    )
    t = np.array(list(information_rates))
    en, rej = engine.design_and_plan(t)
    return {
        "k": len(t),
        "information_rates": t.tolist(),
        "reject_per_stage": rej.tolist(),
        "expected_subjects_H1": en,
    }
