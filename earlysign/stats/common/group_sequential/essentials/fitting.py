"""
Fit spending functions to observed boundaries.

This module implements functions for reverse-engineering spending functions
from observed or designed boundaries. This is useful for:
- Analyzing existing designs
- Understanding spending patterns
- Documenting trial designs

Functions
---------
fit_spending:
    Fit alpha and beta spending functions to boundaries ({cᵢ}, {tᵢ}) → (α̂(t), β̂(t))

Examples
--------
>>> import numpy as np
>>> from earlysign.stats.common.group_sequential.essentials import fitting
>>>
>>> # Example: OBF boundaries at standard times
>>> info_times = np.array([0.33, 0.67, 1.0])
>>> upper_bounds = np.array([2.963, 2.359, 2.014])  # OBF-like
>>> lower_bounds = np.array([-np.inf, -np.inf, -np.inf])
>>>
>>> # Fit spending function to these boundaries
>>> result = fitting.fit_spending(
...     info_times=info_times,
...     upper_bounds=upper_bounds,
...     lower_bounds=lower_bounds,
...     alpha=0.05,
...     tails=2
... )
>>> 'alpha_spent' in result and 'beta_spent' in result
True
>>> len(result['alpha_spent']) == len(info_times)
True
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult, minimize_scalar

# =============================================================================
# Type Aliases
# =============================================================================

ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]

# =============================================================================
# Spending Function Fitting
# =============================================================================


def _compute_alpha_spent_from_boundary(
    *, z_upper: float, t: float, tails: int = 2
) -> float:
    """
    Compute cumulative alpha spent given an efficacy boundary.

    For a boundary z_upper at information time t, the cumulative alpha
    spent is the probability under H0 of crossing the boundary.

    Parameters
    ----------
    z_upper : float
        Upper boundary on Z-scale
    t : float
        Information time
    tails : int
        Number of tails (1 or 2)

    Returns
    -------
    alpha_spent : float
        Cumulative alpha spent up to time t
    """
    if np.isinf(z_upper):
        return 0.0

    # Standardized boundary (Z-scale already assumes unit variance at each look)
    # P(Z_t > z_upper | H0) where Z_t ~ N(0, 1) at each look
    p_cross = stats.norm.sf(z_upper)

    if tails == 2:
        alpha_spent = 2 * p_cross
    else:
        alpha_spent = p_cross

    return float(alpha_spent)


def _compute_beta_spent_from_boundary(
    *, z_lower: float, t: float, effect_size: float = 0.0
) -> float:
    """
    Compute cumulative beta spent given a futility boundary.

    For a boundary z_lower at information time t, the cumulative beta
    spent is the probability under H1 of crossing the boundary (futility stop).

    Parameters
    ----------
    z_lower : float
        Lower boundary on Z-scale
    t : float
        Information time
    effect_size : float
        Standardized effect size under H1

    Returns
    -------
    beta_spent : float
        Cumulative beta spent up to time t
    """
    if np.isinf(z_lower) and z_lower < 0:
        return 0.0

    # Under H1 with effect delta, Z_t ~ N(delta * sqrt(I_t), 1)
    # For standardized Z-scale, the mean is delta (effect size)
    mean_under_h1 = effect_size

    # P(Z_t < z_lower | H1)
    p_cross = stats.norm.cdf(z_lower, loc=mean_under_h1)

    return float(p_cross)


def fit_spending(
    *,
    info_times: ArrayLike,
    upper_bounds: ArrayLike,
    lower_bounds: ArrayLike,
    alpha: float,
    beta: Optional[float] = None,
    tails: int = 2,
    effect_size: float = 0.0,
) -> Dict[str, Any]:
    """
    Fit spending functions to observed boundaries.

    Implements the FitSpending function from functional design:
        FitSpending: ({cᵢ}, {tᵢ}) → (α̂(t), β̂(t))

    Given a set of boundaries and information times, computes the implied
    cumulative alpha and beta spending at each look. This reverse-engineers
    the spending pattern from the boundaries.

    Parameters
    ----------
    info_times : array_like
        Information times in (0, 1], length K
    upper_bounds : array_like
        Upper (efficacy) boundaries on Z-scale, length K
    lower_bounds : array_like
        Lower (futility) boundaries on Z-scale, length K
    alpha : float
        Total alpha (significance level)
    beta : float, optional
        Total beta (Type II error rate) for futility spending
    tails : int, default=2
        Number of tails (1 or 2)
    effect_size : float, default=0.0
        Standardized effect size for beta spending calculation

    Returns
    -------
    dict with keys:
        info_times : np.ndarray
            Information times (reference)
        alpha_spent : np.ndarray
            Cumulative alpha spent at each look
        alpha_incremental : np.ndarray
            Incremental alpha spent at each look
        beta_spent : np.ndarray or None
            Cumulative beta spent at each look (if beta provided)
        beta_incremental : np.ndarray or None
            Incremental beta spent at each look (if beta provided)
        upper_bounds : np.ndarray
            Upper boundaries (reference)
        lower_bounds : np.ndarray
            Lower boundaries (reference)

    Examples
    --------
    >>> import numpy as np
    >>> # OBF-like boundaries
    >>> info_times = np.array([0.5, 1.0])
    >>> upper_bounds = np.array([2.8, 2.0])
    >>> lower_bounds = np.array([-np.inf, -np.inf])
    >>> result = fit_spending(
    ...     info_times=info_times,
    ...     upper_bounds=upper_bounds,
    ...     lower_bounds=lower_bounds,
    ...     alpha=0.05,
    ...     tails=2
    ... )
    >>> len(result['alpha_spent']) == 2
    True
    >>> bool(0 < result['alpha_spent'][0] < result['alpha_spent'][1] <= 0.05)
    True
    """
    info_times = np.asarray(info_times, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)

    n_looks = len(info_times)

    # Compute cumulative alpha spending from upper boundaries
    alpha_spent = np.zeros(n_looks)
    for i, (t, z_upper) in enumerate(zip(info_times, upper_bounds)):
        alpha_spent[i] = _compute_alpha_spent_from_boundary(
            z_upper=z_upper, t=t, tails=tails
        )

    # Compute incremental alpha spending
    alpha_incremental = np.diff(alpha_spent, prepend=0.0)

    # Compute beta spending if requested
    beta_spent = None
    beta_incremental = None
    if beta is not None:
        beta_spent = np.zeros(n_looks)
        for i, (t, z_lower) in enumerate(zip(info_times, lower_bounds)):
            beta_spent[i] = _compute_beta_spent_from_boundary(
                z_lower=z_lower, t=t, effect_size=effect_size
            )
        beta_incremental = np.diff(beta_spent, prepend=0.0)

    return {
        "info_times": info_times,
        "alpha_spent": alpha_spent,
        "alpha_incremental": alpha_incremental,
        "beta_spent": beta_spent,
        "beta_incremental": beta_incremental,
        "upper_bounds": upper_bounds,
        "lower_bounds": lower_bounds,
    }


def fit_spending_function(
    *,
    info_times: ArrayLike,
    cumulative_spending: ArrayLike,
    total_budget: float,
    family: str = "power",
) -> Callable[[float], float]:
    """
    Fit a parametric spending function to observed spending pattern.

    Given cumulative spending at each look, fits a smooth parametric function
    that can be evaluated at any information time t ∈ (0, 1].

    Parameters
    ----------
    info_times : array_like
        Information times where spending is observed
    cumulative_spending : array_like
        Cumulative spending at each time point
    total_budget : float
        Total spending budget (e.g., alpha=0.05)
    family : str, default="power"
        Spending function family to fit:
        - "power": α(t) = budget * t^γ
        - "linear": α(t) = budget * t
        - "obf": O'Brien-Fleming-like

    Returns
    -------
    spending_func : callable
        Function t → α(t) that interpolates the spending pattern

    Examples
    --------
    >>> import numpy as np
    >>> info_times = np.array([0.33, 0.67, 1.0])
    >>> cumulative = np.array([0.002, 0.015, 0.05])
    >>> func = fit_spending_function(
    ...     info_times=info_times,
    ...     cumulative_spending=cumulative,
    ...     total_budget=0.05,
    ...     family="power"
    ... )
    >>> 0 < func(0.5) < 0.05
    True
    """
    info_times = np.asarray(info_times, dtype=float)
    cumulative_spending = np.asarray(cumulative_spending, dtype=float)

    if family == "linear":
        # Simple linear spending: α(t) = budget * t
        return lambda t: total_budget * t

    elif family == "power":
        # Fit power function: α(t) = budget * t^γ
        # Find γ that minimizes squared error

        def objective(gamma: float) -> float:
            predicted = total_budget * (info_times**gamma)
            return float(np.sum((predicted - cumulative_spending) ** 2))

        result = cast(
            OptimizeResult,
            minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded"),
        )
        gamma_opt = float(result.x)

        return lambda t: total_budget * (t**gamma_opt)

    else:
        # Default: piecewise linear interpolation
        # Ensure endpoint matches budget
        times_with_end = (
            np.append(info_times, 1.0) if info_times[-1] < 1.0 else info_times
        )
        spending_with_end = (
            np.append(cumulative_spending, total_budget)
            if info_times[-1] < 1.0
            else cumulative_spending
        )

        interpolator = interp1d(
            times_with_end,
            spending_with_end,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return cast(Callable[[float], float], interpolator)
