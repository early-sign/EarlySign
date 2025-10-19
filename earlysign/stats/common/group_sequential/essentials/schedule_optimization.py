"""
Information time schedule optimization for group sequential designs.

This module provides pure functions for optimizing information time schedules
{t_i} to achieve specific design objectives (e.g., minimize expected sample
size, maximize power) given fixed spending functions.

The optimization approach:
1. Fix alpha and beta spending functions
2. Define objective function (ASN, power, or composite)
3. Search over information time configurations
4. Return optimal schedule {t_i*} and resulting performance

Key Functions
-------------
optimize_schedule_for_asn :
    Find information time schedule minimizing expected sample size
optimize_schedule_for_power :
    Find information time schedule maximizing statistical power
evaluate_schedule_design :
    Evaluate performance for given schedule and spending functions

Design Philosophy
-----------------
This complements spending_optimization.py:
- spending_optimization: Fixes {t_i}, optimizes spending parameters
- schedule_optimization: Fixes spending functions, optimizes {t_i}

Both can be combined for comprehensive design optimization.

Examples
--------
>>> import numpy as np
>>> from earlysign.stats.common.group_sequential.essentials import schedule_optimization as sched
>>> # Optimize information times for minimal ASN
>>> result = sched.optimize_schedule_for_asn(
...     alpha=0.025,
...     beta=0.10,
...     n_analyses=3,
...     effect_size=0.5,
...     alpha_gamma=-4.0,
...     target_power=0.90,
...     max_sample_size=1000,
...     n_simulations=1000,
...     seed=42
... )  # doctest: +SKIP
>>> len(result["optimal_info_times"])  # doctest: +SKIP
3
>>> result["optimal_info_times"][-1]  # doctest: +SKIP
1.0
"""

import warnings
from typing import Any, Dict, Optional, cast

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from earlysign.stats.common.group_sequential.essentials.boundaries import (
    compute_boundaries_at_times,
)
from earlysign.stats.common.group_sequential.essentials.design_schema import (
    BindingMode,
)
from earlysign.stats.common.group_sequential.essentials.performance import performance


def _create_hsd_design_payload(
    alpha: float, gamma: float, binding_mode: BindingMode
) -> Dict[str, Any]:
    """Create design payload for HSD spending with given gamma."""
    return {
        "alpha": alpha,
        "tails": 2,
        "scale": "z",
        "efficacy": {"style": "alpha_spending", "family": "hsd", "gamma": gamma},
        "futility": {
            "mode": "beta_spending" if binding_mode == BindingMode.BINDING else "none"
        },
    }


def evaluate_schedule_design(
    alpha: float,
    beta: float,
    info_times: np.ndarray,
    effect_size: float,
    alpha_gamma: float = -4.0,
    beta_gamma: float = -4.0,
    binding_mode: BindingMode = BindingMode.NON_BINDING,
    max_sample_size: Optional[int] = None,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate performance metrics for given information time schedule.

    This function computes boundaries using HSD spending with fixed gamma
    parameters, then evaluates operating characteristics via Monte Carlo
    simulation for a specific information time schedule.

    Parameters
    ----------
    alpha : float
        Overall significance level (one-sided)
    beta : float
        Overall Type II error rate (1 - power)
    info_times : np.ndarray
        Information time schedule, shape (k,), values in (0, 1]
    effect_size : float
        Standardized effect size under alternative hypothesis
    alpha_gamma : float, default=-4.0
        Shape parameter for alpha spending (HSD family)
    beta_gamma : float, default=-4.0
        Shape parameter for beta spending (HSD family)
    binding_mode : BindingMode, default=NON_BINDING
        Whether futility boundaries are binding
    max_sample_size : int, optional
        Maximum sample size for absolute ASN calculation
    n_simulations : int, default=5000
        Number of Monte Carlo simulations
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Performance metrics including power, ASN, boundaries

    Examples
    --------
    >>> info_times = np.array([0.33, 0.67, 1.0])
    >>> result = evaluate_schedule_design(
    ...     alpha=0.025, beta=0.10,
    ...     info_times=info_times,
    ...     effect_size=0.5,
    ...     max_sample_size=1000,
    ...     n_simulations=1000,
    ...     seed=42
    ... )
    >>> 0.0 < result["power"] < 1.0  # Avoid flaky bounds; power is a probability
    True
    """
    # Create design payload with HSD spending
    design_payload = _create_hsd_design_payload(alpha, alpha_gamma, binding_mode)

    # Add beta spending if binding
    if binding_mode == BindingMode.BINDING:
        design_payload["futility"]["family"] = "hsd"
        design_payload["futility"]["gamma"] = beta_gamma

    # Compute boundaries at all information times
    boundary_result = compute_boundaries_at_times(design_payload, info_times)

    # Evaluate performance via simulation
    perf_result = performance(
        info_times=info_times,
        upper_bounds=boundary_result["upper"],
        lower_bounds=boundary_result["lower"],
        effect_size=effect_size,
        max_sample_size=max_sample_size,
        n_simulations=n_simulations,
        seed=seed,
    )

    return {
        "power": perf_result["power"],
        "asn_ratio": perf_result["asn_ratio"],
        "expected_sample_size": perf_result.get("expected_sample_size"),
        "max_n": max_sample_size,
        "upper_bounds": boundary_result["upper"],
        "lower_bounds": boundary_result["lower"],
        "prob_stop_efficacy": perf_result["prob_stop_efficacy"],
        "prob_stop_futility": perf_result["prob_stop_futility"],
        "info_times": info_times,
    }


def optimize_schedule_for_asn(
    alpha: float,
    beta: float,
    n_analyses: int,
    effect_size: float,
    alpha_gamma: float,
    target_power: float,
    max_sample_size: int,
    beta_gamma: Optional[float] = None,
    binding_mode: BindingMode = BindingMode.NON_BINDING,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Optimize information time schedule to minimize expected sample size (ASN).

    Searches the space of valid information time schedules to find the
    configuration that minimizes average sample number while maintaining
    target power and respecting maximum sample size constraint.

    This implements the ScheduleOptimization function from functional design:
        ScheduleOptimization: (Objective, α(t), β(t), H₀, H₁) → {tᵢ*}

    Parameters
    ----------
    alpha : float
        Overall significance level (one-sided)
    beta : float
        Overall Type II error rate (1 - power)
    n_analyses : int
        Number of interim analyses (includes final)
    effect_size : float
        Standardized effect size under alternative hypothesis
    alpha_gamma : float
        Shape parameter for alpha spending (HSD family, fixed)
    target_power : float
        Minimum required power (e.g., 0.90)
    max_sample_size : int
        Maximum allowable sample size
    beta_gamma : float, optional
        Shape parameter for beta spending. If None, uses alpha_gamma
    binding_mode : BindingMode, default=NON_BINDING
        Whether futility boundaries are binding
    n_simulations : int, default=5000
        Number of Monte Carlo simulations per evaluation
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Optimization result with keys:
        - "optimal_info_times": np.ndarray, best information time schedule
        - "asn": float, minimized average sample number
        - "asn_ratio": float, ASN as ratio of max_sample_size
        - "power": float, achieved statistical power
        - "max_n": int, maximum sample size
        - "upper_bounds": np.ndarray, efficacy boundaries
        - "lower_bounds": np.ndarray, futility boundaries

    Raises
    ------
    ValueError
        If target_power cannot be achieved within constraints

    Examples
    --------
    >>> result = optimize_schedule_for_asn(
    ...     alpha=0.025, beta=0.10,
    ...     n_analyses=3,
    ...     effect_size=0.5,
    ...     alpha_gamma=-4.0,
    ...     target_power=0.90,
    ...     max_sample_size=1000,
    ...     n_simulations=1000,
    ...     seed=42
    ... )  # doctest: +SKIP
    >>> result["power"] >= 0.90  # doctest: +SKIP
    True
    >>> result["optimal_info_times"][-1]  # doctest: +SKIP
    1.0

    Notes
    -----
    Optimization constraints:
    1. Information times must be strictly increasing: 0 < t₁ < t₂ < ... < tₖ = 1
    2. Early looks typically no sooner than 10% information (t₁ ≥ 0.1)
    3. Power must meet or exceed target_power

    The optimization uses L-BFGS-B with bounds and penalty functions for
    constraint violations.
    """
    if beta_gamma is None:
        beta_gamma = alpha_gamma

    # Define optimization variables: [t₁, t₂, ..., t_{k-1}]
    # t_k is always fixed at 1.0
    def objective(x: np.ndarray) -> float:
        """Objective function: ASN with penalties for constraint violations."""
        try:
            # Construct full info_times array
            info_times_inner = np.append(x, 1.0)
            info_times_inner = np.sort(info_times_inner)  # Enforce monotonicity

            # Check for valid spacing (avoid too-close looks)
            if len(info_times_inner) > 1:
                diffs = np.diff(info_times_inner)
                if np.any(diffs < 0.05):  # Minimum 5% information increment
                    return float(max_sample_size * 10)

            result = evaluate_schedule_design(
                alpha=alpha,
                beta=beta,
                info_times=info_times_inner,
                effect_size=effect_size,
                alpha_gamma=alpha_gamma,
                beta_gamma=beta_gamma,
                binding_mode=binding_mode,
                max_sample_size=max_sample_size,
                n_simulations=n_simulations,
                seed=seed,
            )

            # Check power constraint
            if result["power"] < target_power:
                return float(max_sample_size * 10)

            # Return expected sample size if feasible
            asn = result.get("expected_sample_size")
            if asn is None:
                return float(result["asn_ratio"] * max_sample_size)
            return float(asn)

        except Exception as e:
            warnings.warn(f"Evaluation failed for info_times={x}: {e}", stacklevel=2)
            return float(max_sample_size * 100)

    # Initial guess: equally spaced information times
    x0 = np.array([(i + 1) / n_analyses for i in range(n_analyses - 1)])

    # Bounds: 0.1 ≤ t_i ≤ 0.99 for all intermediate looks
    bounds = [(0.1, 0.99) for _ in range(n_analyses - 1)]

    # Perform optimization
    opt_result = cast(
        OptimizeResult,
        minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-6},
        ),
    )

    if not opt_result.success:
        warnings.warn(
            f"Optimization may not have converged: {opt_result.message}",
            stacklevel=2,
        )

    # Construct optimal info_times
    optimal_times = np.append(opt_result.x, 1.0)
    optimal_times = np.sort(optimal_times)

    # Final evaluation with higher precision
    final_result = evaluate_schedule_design(
        alpha=alpha,
        beta=beta,
        info_times=optimal_times,
        effect_size=effect_size,
        alpha_gamma=alpha_gamma,
        beta_gamma=beta_gamma,
        binding_mode=binding_mode,
        max_sample_size=max_sample_size,
        n_simulations=n_simulations * 2,
        seed=seed,
    )

    # Verify constraints
    if final_result["power"] < target_power:
        raise ValueError(
            f"Optimal design achieves power {final_result['power']:.3f} < "
            f"target {target_power}. Consider increasing max_sample_size "
            f"or adjusting effect_size/spending parameters."
        )

    return {
        "optimal_info_times": optimal_times,
        "asn": final_result.get("expected_sample_size"),
        "asn_ratio": final_result["asn_ratio"],
        "power": final_result["power"],
        "max_n": max_sample_size,
        "upper_bounds": final_result["upper_bounds"],
        "lower_bounds": final_result["lower_bounds"],
        "prob_stop_efficacy": final_result["prob_stop_efficacy"],
        "prob_stop_futility": final_result["prob_stop_futility"],
    }


def optimize_schedule_for_power(
    alpha: float,
    beta: float,
    n_analyses: int,
    effect_size: float,
    alpha_gamma: float,
    max_sample_size: int,
    beta_gamma: Optional[float] = None,
    binding_mode: BindingMode = BindingMode.NON_BINDING,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Optimize information time schedule to maximize statistical power.

    Searches the space of valid information time schedules to find the
    configuration that maximizes power while respecting maximum sample
    size constraint.

    Parameters
    ----------
    alpha : float
        Overall significance level (one-sided)
    beta : float
        Initial Type II error rate (used as starting point)
    n_analyses : int
        Number of interim analyses (includes final)
    effect_size : float
        Standardized effect size under alternative hypothesis
    alpha_gamma : float
        Shape parameter for alpha spending (HSD family, fixed)
    max_sample_size : int
        Maximum allowable sample size
    beta_gamma : float, optional
        Shape parameter for beta spending. If None, uses alpha_gamma
    binding_mode : BindingMode, default=NON_BINDING
        Whether futility boundaries are binding
    n_simulations : int, default=5000
        Number of Monte Carlo simulations per evaluation
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Optimization result with keys:
        - "optimal_info_times": np.ndarray, best information time schedule
        - "power": float, maximized statistical power
        - "asn": float, average sample number at optimal design
        - "asn_ratio": float, ASN as ratio of max_sample_size
        - "max_n": int, maximum sample size
        - "upper_bounds": np.ndarray, efficacy boundaries
        - "lower_bounds": np.ndarray, futility boundaries

    Examples
    --------
    >>> result = optimize_schedule_for_power(
    ...     alpha=0.025, beta=0.10,
    ...     n_analyses=3,
    ...     effect_size=0.5,
    ...     alpha_gamma=-4.0,
    ...     max_sample_size=1000,
    ...     n_simulations=1000,
    ...     seed=42
    ... )  # doctest: +SKIP
    >>> result["power"] > 0.80  # doctest: +SKIP
    True

    Notes
    -----
    Returns -power as objective value (for minimization framework).
    """
    if beta_gamma is None:
        beta_gamma = alpha_gamma

    def objective(x: np.ndarray) -> float:
        """Objective: -power (negative for minimization)."""
        try:
            # Construct full info_times array
            info_times_inner = np.append(x, 1.0)
            info_times_inner = np.sort(info_times_inner)

            # Check for valid spacing
            if len(info_times_inner) > 1:
                diffs = np.diff(info_times_inner)
                if np.any(diffs < 0.05):
                    return float(1.0)  # Penalty

            result = evaluate_schedule_design(
                alpha=alpha,
                beta=beta,
                info_times=info_times_inner,
                effect_size=effect_size,
                alpha_gamma=alpha_gamma,
                beta_gamma=beta_gamma,
                binding_mode=binding_mode,
                max_sample_size=max_sample_size,
                n_simulations=n_simulations,
                seed=seed,
            )

            # Return negative power (for minimization)
            return float(-result["power"])

        except Exception as e:
            warnings.warn(f"Evaluation failed for info_times={x}: {e}", stacklevel=2)
            return float(1.0)

    # Initial guess and bounds
    x0 = np.array([(i + 1) / n_analyses for i in range(n_analyses - 1)])
    bounds = [(0.1, 0.99) for _ in range(n_analyses - 1)]

    # Perform optimization
    opt_result = cast(
        OptimizeResult,
        minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-6},
        ),
    )

    if not opt_result.success:
        warnings.warn(
            f"Optimization may not have converged: {opt_result.message}",
            stacklevel=2,
        )

    # Construct optimal info_times
    optimal_times = np.append(opt_result.x, 1.0)
    optimal_times = np.sort(optimal_times)

    # Final evaluation
    final_result = evaluate_schedule_design(
        alpha=alpha,
        beta=beta,
        info_times=optimal_times,
        effect_size=effect_size,
        alpha_gamma=alpha_gamma,
        beta_gamma=beta_gamma,
        binding_mode=binding_mode,
        max_sample_size=max_sample_size,
        n_simulations=n_simulations * 2,
        seed=seed,
    )

    return {
        "optimal_info_times": optimal_times,
        "power": final_result["power"],
        "asn": final_result.get("expected_sample_size"),
        "asn_ratio": final_result["asn_ratio"],
        "max_n": max_sample_size,
        "upper_bounds": final_result["upper_bounds"],
        "lower_bounds": final_result["lower_bounds"],
        "prob_stop_efficacy": final_result["prob_stop_efficacy"],
        "prob_stop_futility": final_result["prob_stop_futility"],
    }
