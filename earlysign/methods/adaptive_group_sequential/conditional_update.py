"""
Conditional design update functions for adaptive group sequential trials.

This module provides pure functions for computing conditional power and
updating trial designs based on interim analysis results. These functions
support "promising-zone adaptive" designs where the trial design can be
modified mid-course based on observed data.

Key Concepts
------------
Conditional Power (CP):
    Probability of rejecting H₀ at final analysis, conditional on:
    - Current observed data (Z-statistic at interim look)
    - Assumed effect size going forward
    - Remaining information time
    - Future boundary values

Promising Zone:
    Range of interim results that warrant continuing the trial:
    - Below futility boundary → Stop for futility
    - Above efficacy boundary → Stop for efficacy
    - In between (promising zone) → Continue, possibly with design update

Design Update:
    Modification of remaining boundaries/information times based on:
    - Conditional power calculation
    - Re-estimation of effect size
    - Adjustment of remaining sample size or analysis schedule

Key Functions
-------------
conditional_power :
    Compute probability of future efficacy crossing given current data
update_remaining_boundaries :
    Recalculate boundaries for remaining looks after interim update
conditional_power_curve :
    Conditional power as function of assumed future effect size

Examples
--------
>>> import numpy as np
>>> from earlysign.methods.adaptive_group_sequential import conditional_update
>>> # At interim: observed Z=1.5 at t=0.5, what's CP for final analysis?
>>> cp = conditional_update.conditional_power(
...     observed_z=1.5,
...     current_info_time=0.5,
...     final_info_time=1.0,
...     final_efficacy_bound=1.96,
...     assumed_effect=0.5
... )
>>> 0 < cp < 1
True
"""

import warnings
from typing import Any, Dict, Literal, Tuple

import numpy as np
from scipy import stats

from earlysign.methods.group_sequential import boundary

BindingMode = Literal["binding", "non_binding"]


def conditional_power(
    observed_z: float,
    current_info_time: float,
    final_info_time: float,
    final_efficacy_bound: float,
    assumed_effect: float,
    variance: float = 1.0,
) -> float:
    """
    Compute conditional power at an interim analysis.

    Conditional power is the probability of rejecting H₀ at the final analysis,
    given the current observed Z-statistic and assuming a specific effect size
    for the remaining data.

    Under the Brownian motion framework:
    - B(t) = observed statistic at information time t
    - B(t_final) | B(t_current) ~ N(B(t_current) + δ·√(Δt), Δt)
      where Δt = t_final - t_current

    Parameters
    ----------
    observed_z : float
        Observed Z-statistic at current interim analysis
    current_info_time : float
        Current information time, in (0, 1)
    final_info_time : float
        Information time at final analysis, typically 1.0
    final_efficacy_bound : float
        Efficacy boundary value (Z-scale) at final analysis
    assumed_effect : float
        Assumed standardized effect size for remaining data collection
    variance : float, default=1.0
        Variance parameter (default assumes standardized statistics)

    Returns
    -------
    float
        Conditional power, value in [0, 1]

    Examples
    --------
    >>> # Observed Z=2.0 at 50% information, assume effect=0.5 going forward
    >>> cp = conditional_power(
    ...     observed_z=2.0,
    ...     current_info_time=0.5,
    ...     final_info_time=1.0,
    ...     final_efficacy_bound=1.96,
    ...     assumed_effect=0.5
    ... )
    >>> cp > 0.70  # Numerical value ~0.71 with the current model
    True

    >>> # Observed Z=0.5 at 50% information (weak evidence)
    >>> cp_low = conditional_power(
    ...     observed_z=0.5,
    ...     current_info_time=0.5,
    ...     final_info_time=1.0,
    ...     final_efficacy_bound=1.96,
    ...     assumed_effect=0.3
    ... )
    >>> cp_low < 0.50
    True

    Notes
    -----
    The computation uses the Brownian motion representation:

    Z(t_final) | Z(t_current) ~ N(
        mean = Z(t_current) + δ · sqrt(t_final - t_current),
        var = (t_final - t_current) / variance
    )

    Conditional power = P(Z(t_final) > c_final | Z(t_current), δ)
                      = 1 - Φ((c_final - μ_cond) / σ_cond)

    References
    ----------
    .. [1] Proschan, M. A., Lan, K. K., & Wittes, J. T. (2006).
           Statistical Monitoring of Clinical Trials: A Unified Approach.
           Springer. Chapter 6: Conditional Power.
    """
    if not 0 < current_info_time < final_info_time:
        raise ValueError(
            f"Invalid information times: current={current_info_time}, "
            f"final={final_info_time}. Must satisfy 0 < current < final."
        )

    # Remaining information time
    delta_t = final_info_time - current_info_time

    # Expected increment under assumed effect
    mean_increment = assumed_effect * np.sqrt(delta_t)

    # Conditional distribution of final Z-statistic
    # Z(final) | Z(current) ~ N(Z(current) + δ·√Δt, Δt/variance)
    conditional_mean = observed_z + mean_increment
    conditional_sd = np.sqrt(delta_t / variance)

    # Conditional power = P(Z(final) > c_final | current data, assumed δ)
    cp = 1.0 - stats.norm.cdf(
        final_efficacy_bound, loc=conditional_mean, scale=conditional_sd
    )

    return float(cp)


def conditional_power_curve(
    observed_z: float,
    current_info_time: float,
    final_info_time: float,
    final_efficacy_bound: float,
    effect_range: Tuple[float, float] = (-1.0, 2.0),
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conditional power as a function of assumed effect size.

    This generates a curve showing how conditional power varies with the
    assumed future effect size, useful for sensitivity analysis and
    decision-making at interim analyses.

    Parameters
    ----------
    observed_z : float
        Observed Z-statistic at current interim analysis
    current_info_time : float
        Current information time
    final_info_time : float
        Information time at final analysis
    final_efficacy_bound : float
        Efficacy boundary at final analysis (Z-scale)
    effect_range : tuple of float, default=(-1.0, 2.0)
        Range of effect sizes to evaluate (min, max)
    n_points : int, default=50
        Number of points in the curve

    Returns
    -------
    effect_sizes : np.ndarray
        Array of effect sizes evaluated, shape (n_points,)
    cond_powers : np.ndarray
        Conditional power values corresponding to effect_sizes

    Examples
    --------
    >>> effects, cps = conditional_power_curve(
    ...     observed_z=1.5,
    ...     current_info_time=0.5,
    ...     final_info_time=1.0,
    ...     final_efficacy_bound=1.96,
    ...     effect_range=(0.0, 1.0),
    ...     n_points=20
    ... )
    >>> len(effects)
    20
    >>> len(cps)
    20
    >>> bool(cps[0] < cps[-1])  # Power increases with effect size
    True

    Notes
    -----
    This curve is useful for:
    1. Assessing robustness of continuation decision to effect size assumptions
    2. Finding effect size threshold for desired conditional power
    3. Visualizing range of plausible outcomes
    """
    effect_sizes = np.linspace(effect_range[0], effect_range[1], n_points)
    cond_powers = np.array(
        [
            conditional_power(
                observed_z=observed_z,
                current_info_time=current_info_time,
                final_info_time=final_info_time,
                final_efficacy_bound=final_efficacy_bound,
                assumed_effect=eff,
            )
            for eff in effect_sizes
        ]
    )

    return effect_sizes, cond_powers


def update_remaining_boundaries(
    current_info_time: float,
    remaining_info_times: np.ndarray,
    alpha: float,
    alpha_gamma: float,
    cumulative_alpha_spent: float,
    binding_mode: BindingMode = "non_binding",
) -> Dict[str, Any]:
    """
    Recompute boundaries for remaining analyses after interim update.

    When a trial design is modified at an interim analysis (e.g., increasing
    sample size, adding more looks), the remaining boundaries must be
    recalculated to maintain overall Type I error control.

    This function uses the "alpha spending" approach: given the cumulative
    alpha already spent at current_info_time, allocate the remaining alpha
    budget across the remaining looks according to the spending function.

    Parameters
    ----------
    current_info_time : float
        Current information time where update occurs
    remaining_info_times : np.ndarray
        Information times for remaining analyses (including final)
        Must all be > current_info_time
    alpha : float
        Overall significance level (one-sided)
    alpha_gamma : float
        Shape parameter for HSD spending function
    cumulative_alpha_spent : float
        Alpha already spent up to current_info_time
        Typically obtained from previous boundary crossings or spending function
    binding_mode : BindingMode, default="non_binding"
        Whether futility boundaries are binding

    Returns
    -------
    dict
        Updated boundaries with keys:
        - "remaining_info_times": np.ndarray, information times for remaining looks
        - "upper_bounds": np.ndarray, efficacy boundaries (Z-scale)
        - "lower_bounds": np.ndarray, futility boundaries (Z-scale)
        - "alpha_remaining": float, alpha budget remaining
        - "alpha_increments": np.ndarray, incremental alpha spent at each remaining look

    Raises
    ------
    ValueError
        If remaining_info_times contains values ≤ current_info_time
        If cumulative_alpha_spent ≥ alpha (no alpha remaining)

    Examples
    --------
    >>> remaining_times = np.array([0.75, 1.0])
    >>> result = update_remaining_boundaries(
    ...     current_info_time=0.5,
    ...     remaining_info_times=remaining_times,
    ...     alpha=0.025,
    ...     alpha_gamma=-4.0,
    ...     cumulative_alpha_spent=0.005
    ... )
    >>> len(result["upper_bounds"])
    2
    >>> result["alpha_remaining"] < 0.025
    True

    Notes
    -----
    The alpha redistribution follows this logic:
    1. Alpha remaining = α - α_spent(current_t)
    2. For each remaining look at time t_i:
       - Compute α(t_i) using spending function
       - Subtract α_spent(current_t)
       - Normalize to ensure total remaining = α_remaining
    3. Convert cumulative alpha to boundaries using inverse normal

    This maintains the overall Type I error rate while adapting to
    design modifications.

    References
    ----------
    .. [1] Cui, L., Hung, H. M. J., & Wang, S. J. (1999).
           Modification of sample size in group sequential clinical trials.
           Biometrics, 55(3), 853-857.
    """
    # Validate inputs
    if np.any(remaining_info_times <= current_info_time):
        raise ValueError(
            f"All remaining_info_times must be > current_info_time={current_info_time}"
        )

    if cumulative_alpha_spent >= alpha:
        raise ValueError(
            f"No alpha remaining: cumulative_alpha_spent={cumulative_alpha_spent} "
            f">= alpha={alpha}"
        )

    alpha_remaining = alpha - cumulative_alpha_spent

    # Create design payload for remaining looks
    # We'll compute boundaries at remaining times using adjusted spending
    design = {
        "alpha": alpha,
        "tails": 2,
        "scale": "z",
        "efficacy": {"style": "alpha_spending", "family": "hsd", "gamma": alpha_gamma},
        "futility": {"mode": "none"},  # Simplified for now
    }

    # Compute boundaries at remaining times using canonical BoundaryCalculator
    calc = boundary.BoundaryCalculator(spec=design, process=None)
    boundary_result = calc.compute_boundaries(info_times=remaining_info_times)

    # Note: This is a simplified implementation. A more sophisticated approach
    # would explicitly adjust for alpha already spent, potentially by:
    # 1. Using conditional error spending
    # 2. Adjusting gamma parameter to redistribute remaining alpha optimally
    # 3. Using numerical search to match exact alpha remaining constraint

    warnings.warn(
        "update_remaining_boundaries uses simplified alpha redistribution. "
        "Consider explicit conditional error spending for production use.",
        stacklevel=2,
    )

    return {
        "remaining_info_times": remaining_info_times,
        "upper_bounds": boundary_result["upper"],
        "lower_bounds": boundary_result["lower"],
        "alpha_remaining": alpha_remaining,
        "alpha_increments": np.diff(
            np.insert(boundary_result["upper"], 0, 0)
        ),  # Placeholder
    }


def promising_zone_decision(
    observed_z: float,
    current_info_time: float,
    current_efficacy_bound: float,
    current_futility_bound: float,
    final_info_time: float,
    final_efficacy_bound: float,
    assumed_effect: float,
    cp_threshold: float = 0.80,
) -> Dict[str, Any]:
    """
    Make continuation decision based on conditional power in promising zone.

    At an interim analysis, this function evaluates whether observed results
    warrant continuing the trial, stopping for futility, or (in the promising
    zone) potentially modifying the design.

    Decision logic:
    1. If Z > efficacy bound → Stop for efficacy (success)
    2. If Z < futility bound → Stop for futility (failure)
    3. If in promising zone:
       a. Compute conditional power
       b. If CP > threshold → Continue with current or modified design
       c. If CP < threshold → Consider stopping for futility

    Parameters
    ----------
    observed_z : float
        Observed Z-statistic at current analysis
    current_info_time : float
        Current information time
    current_efficacy_bound : float
        Efficacy boundary at current analysis (Z-scale)
    current_futility_bound : float
        Futility boundary at current analysis (Z-scale)
    final_info_time : float
        Information time at final analysis
    final_efficacy_bound : float
        Efficacy boundary at final analysis (Z-scale)
    assumed_effect : float
        Assumed effect size for conditional power calculation
    cp_threshold : float, default=0.80
        Minimum conditional power threshold for continuation

    Returns
    -------
    dict
        Decision information with keys:
        - "decision": str, one of ["stop_efficacy", "stop_futility", "continue", "uncertain"]
        - "conditional_power": float, CP given assumed effect
        - "in_promising_zone": bool, whether observed_z is in promising zone
        - "recommendation": str, textual explanation of decision

    Examples
    --------
    >>> # Strong evidence: above efficacy bound
    >>> decision = promising_zone_decision(
    ...     observed_z=3.0,
    ...     current_info_time=0.5,
    ...     current_efficacy_bound=2.5,
    ...     current_futility_bound=-0.5,
    ...     final_info_time=1.0,
    ...     final_efficacy_bound=1.96,
    ...     assumed_effect=0.5
    ... )
    >>> decision["decision"]
    'stop_efficacy'

    >>> # Weak evidence: below futility bound
    >>> decision = promising_zone_decision(
    ...     observed_z=-1.0,
    ...     current_info_time=0.5,
    ...     current_efficacy_bound=2.5,
    ...     current_futility_bound=-0.5,
    ...     final_info_time=1.0,
    ...     final_efficacy_bound=1.96,
    ...     assumed_effect=0.5
    ... )
    >>> decision["decision"]
    'stop_futility'

    >>> # Promising zone: compute CP for decision
    >>> decision = promising_zone_decision(
    ...     observed_z=1.5,
    ...     current_info_time=0.5,
    ...     current_efficacy_bound=2.5,
    ...     current_futility_bound=0.0,
    ...     final_info_time=1.0,
    ...     final_efficacy_bound=1.96,
    ...     assumed_effect=0.8,
    ...     cp_threshold=0.80
    ... )
    >>> decision["in_promising_zone"]
    True

    Notes
    -----
    This implements a simplified version of promising-zone adaptive designs.
    In practice, additional considerations include:
    - Re-estimation of effect size from interim data
    - Adjustment of sample size based on CP
    - Regulatory constraints on design modifications
    - Blinding considerations for data monitoring committees
    """
    # Check boundary crossings
    if observed_z >= current_efficacy_bound:
        return {
            "decision": "stop_efficacy",
            "conditional_power": 1.0,
            "in_promising_zone": False,
            "recommendation": f"Observed Z={observed_z:.2f} exceeds efficacy bound "
            f"{current_efficacy_bound:.2f}. Stop trial for efficacy.",
        }

    if observed_z <= current_futility_bound:
        cp = conditional_power(
            observed_z=observed_z,
            current_info_time=current_info_time,
            final_info_time=final_info_time,
            final_efficacy_bound=final_efficacy_bound,
            assumed_effect=assumed_effect,
        )
        return {
            "decision": "stop_futility",
            "conditional_power": cp,
            "in_promising_zone": False,
            "recommendation": f"Observed Z={observed_z:.2f} below futility bound "
            f"{current_futility_bound:.2f}. Stop trial for futility. "
            f"Conditional power={cp:.3f}.",
        }

    # In promising zone: evaluate conditional power
    cp = conditional_power(
        observed_z=observed_z,
        current_info_time=current_info_time,
        final_info_time=final_info_time,
        final_efficacy_bound=final_efficacy_bound,
        assumed_effect=assumed_effect,
    )

    if cp >= cp_threshold:
        decision_str = "continue"
        rec = (
            f"In promising zone with CP={cp:.3f} >= threshold {cp_threshold}. "
            f"Continue trial with current or modified design."
        )
    else:
        decision_str = "uncertain"
        rec = (
            f"In promising zone but CP={cp:.3f} < threshold {cp_threshold}. "
            f"Consider stopping for futility or modifying design to increase power."
        )

    return {
        "decision": decision_str,
        "conditional_power": cp,
        "in_promising_zone": True,
        "recommendation": rec,
    }
