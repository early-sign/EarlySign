"""
Performance and operating characteristics for group sequential designs.

This module implements functions for evaluating the performance of a group
sequential design, including:
- Power curves across effect sizes (PowerCurve_GST)
- Operating characteristics: power, ASN, expected sample size (Performance)
- Minimum detectable effect at each look (Sensitivity)

Functions
---------
PowerCurve_GST:
    power_curve(info_times, boundaries, effect_sizes, alternative) -> dict

Performance:
    performance(info_times, boundaries, effect_size, alternative) -> dict

Sensitivity:
    sensitivity(info_times, boundaries, alpha, null_hypothesis) -> dict

Examples
--------
>>> import numpy as np
>>> from earlysign.stats.common.group_sequential.essentials import performance
>>>
>>> # Define design parameters
>>> info_times = np.array([0.33, 0.67, 1.0])
>>> upper_bounds = np.array([2.963, 2.359, 2.014])  # OBF efficacy
>>> lower_bounds = np.array([-np.inf, -np.inf, -np.inf])  # No futility
>>>
>>> # Compute performance at specific effect size
>>> perf = performance.performance(
...     info_times=info_times,
...     upper_bounds=upper_bounds,
...     lower_bounds=lower_bounds,
...     effect_size=0.5,
...     variance=1.0,
...     max_sample_size=1000
... )
>>> 'power' in perf and 'asn_ratio' in perf and 'expected_sample_size' in perf
True
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# =============================================================================
# Type Aliases
# =============================================================================

ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]

# =============================================================================
# Power Computation via Simulation
# =============================================================================


def _simulate_trial(
    *,
    info_times: np.ndarray,
    upper_bounds: np.ndarray,
    lower_bounds: np.ndarray,
    effect_size: float,
    variance: float,
    rng: np.random.Generator,
) -> Tuple[int, str]:
    """
    Simulate a single trial through the group sequential design.

    Returns
    -------
    look : int
        Look number where decision was made (1-indexed), or len(info_times) if final
    decision : str
        "efficacy", "futility", or "continue"
    """
    n_looks = len(info_times)
    prev_t = 0.0
    cumulative_z = 0.0

    for i, t in enumerate(info_times):
        # Increment in information time
        dt = t - prev_t

        # Generate increment in Z-statistic (Brownian motion with drift)
        # Under alternative: Z ~ N(effect * sqrt(I), 1)
        # Increment: dZ ~ N(effect * sqrt(dI), dI)
        drift = effect_size * np.sqrt(dt)
        noise = np.sqrt(dt) * rng.standard_normal()
        cumulative_z += drift + noise

        # Scale to information time for boundary comparison
        scaled_z = cumulative_z / np.sqrt(t)

        # Check boundaries
        if scaled_z >= upper_bounds[i]:
            return i + 1, "efficacy"
        if scaled_z <= lower_bounds[i]:
            return i + 1, "futility"

        prev_t = t

    return n_looks, "continue"


def _power_by_simulation(
    *,
    info_times: np.ndarray,
    upper_bounds: np.ndarray,
    lower_bounds: np.ndarray,
    effect_size: float,
    variance: float,
    n_simulations: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate power and stopping probabilities via Monte Carlo simulation.

    Returns
    -------
    dict with keys:
        power : float
            Probability of rejecting H0
        asn_ratio : float
            Average sample number as ratio of max sample size
        prob_stop_efficacy : np.ndarray
            Probability of stopping for efficacy at each look
        prob_stop_futility : np.ndarray
            Probability of stopping for futility at each look
    """
    rng = np.random.default_rng(seed)
    n_looks = len(info_times)

    # Counters
    efficacy_stops = np.zeros(n_looks)
    futility_stops = np.zeros(n_looks)

    for _ in range(n_simulations):
        look, decision = _simulate_trial(
            info_times=info_times,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            effect_size=effect_size,
            variance=variance,
            rng=rng,
        )
        if decision == "efficacy":
            efficacy_stops[look - 1] += 1
        elif decision == "futility":
            futility_stops[look - 1] += 1

    # Compute statistics
    prob_efficacy = efficacy_stops / n_simulations
    prob_futility = futility_stops / n_simulations
    power = prob_efficacy.sum()

    # Average sample number (as ratio of max)
    # ASN = sum over all looks of (info_time * prob_stop_at_that_look)
    asn_ratio = 0.0
    for i, t in enumerate(info_times):
        prob_stop = prob_efficacy[i] + prob_futility[i]
        asn_ratio += t * prob_stop

    return {
        "power": float(power),
        "asn_ratio": float(asn_ratio),
        "prob_stop_efficacy": prob_efficacy,
        "prob_stop_futility": prob_futility,
    }


# =============================================================================
# Public API Functions
# =============================================================================


def power_curve(
    *,
    info_times: ArrayLike,
    upper_bounds: ArrayLike,
    lower_bounds: ArrayLike,
    effect_sizes: ArrayLike,
    variance: float = 1.0,
    n_simulations: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute power curve across multiple effect sizes.

    Implements the PowerCurve_GST function from functional design:
        PowerCurve_GST: ({tᵢ}, {cᵢ}, H₁) → [δ ↦ Power(δ)]

    Parameters
    ----------
    info_times : array_like
        Information times in [0, 1], length K
    upper_bounds : array_like
        Upper (efficacy) boundaries on Z-scale, length K
    lower_bounds : array_like
        Lower (futility) boundaries on Z-scale, length K
    effect_sizes : array_like
        Array of effect sizes to evaluate (standardized)
    variance : float, default=1.0
        Variance parameter for effect size scaling
    n_simulations : int, default=10000
        Number of Monte Carlo simulations per effect size
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        effect_sizes : np.ndarray
            Input effect sizes
        power : np.ndarray
            Power at each effect size
        asn_ratio : np.ndarray
            Average sample number ratio at each effect size

    Examples
    --------
    >>> import numpy as np
    >>> info_times = np.array([0.5, 1.0])
    >>> upper_bounds = np.array([2.8, 2.0])
    >>> lower_bounds = np.array([-np.inf, -np.inf])
    >>> effect_sizes = np.array([0.0, 0.2, 0.5])
    >>> result = power_curve(
    ...     info_times=info_times,
    ...     upper_bounds=upper_bounds,
    ...     lower_bounds=lower_bounds,
    ...     effect_sizes=effect_sizes,
    ...     n_simulations=1000,
    ...     seed=42
    ... )
    >>> len(result['power']) == len(effect_sizes)
    True
    >>> bool(result['power'][0] < result['power'][2])  # Power increases with effect
    True
    """
    info_times = np.asarray(info_times, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    effect_sizes = np.asarray(effect_sizes, dtype=float)

    powers = []
    asn_ratios = []

    for delta in effect_sizes:
        result = _power_by_simulation(
            info_times=info_times,
            upper_bounds=upper_bounds,
            lower_bounds=lower_bounds,
            effect_size=delta,
            variance=variance,
            n_simulations=n_simulations,
            seed=seed,
        )
        powers.append(result["power"])
        asn_ratios.append(result["asn_ratio"])

    return {
        "effect_sizes": effect_sizes,
        "power": np.array(powers),
        "asn_ratio": np.array(asn_ratios),
    }


def performance(
    *,
    info_times: ArrayLike,
    upper_bounds: ArrayLike,
    lower_bounds: ArrayLike,
    effect_size: float,
    variance: float = 1.0,
    max_sample_size: Optional[int] = None,
    n_simulations: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute operating characteristics for a group sequential design.

    Implements the Performance function from functional design:
        Performance: ({tᵢ}, {cᵢ}, H₁) → ({Powerᵢ(δ₀)}, ASN, TotalSN)

    Parameters
    ----------
    info_times : array_like
        Information times in [0, 1], length K
    upper_bounds : array_like
        Upper (efficacy) boundaries on Z-scale, length K
    lower_bounds : array_like
        Lower (futility) boundaries on Z-scale, length K
    effect_size : float
        Standardized effect size under alternative hypothesis
    variance : float, default=1.0
        Variance parameter
    max_sample_size : int, optional
        Maximum sample size for absolute ASN calculation
    n_simulations : int, default=10000
        Number of Monte Carlo simulations
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        power : float
            Overall power (probability of rejecting H0)
        asn_ratio : float
            Average sample number as ratio of max
        expected_sample_size : float or None
            Absolute expected sample size (if max_sample_size provided)
        prob_stop_efficacy : np.ndarray
            Probability of stopping for efficacy at each look
        prob_stop_futility : np.ndarray
            Probability of stopping for futility at each look
        info_times : np.ndarray
            Information times (for reference)

    Examples
    --------
    >>> import numpy as np
    >>> info_times = np.array([0.33, 0.67, 1.0])
    >>> upper_bounds = np.array([2.963, 2.359, 2.014])
    >>> lower_bounds = np.array([-np.inf, -np.inf, -np.inf])
    >>> result = performance(
    ...     info_times=info_times,
    ...     upper_bounds=upper_bounds,
    ...     lower_bounds=lower_bounds,
    ...     effect_size=0.5,
    ...     max_sample_size=1000,
    ...     n_simulations=1000,
    ...     seed=42
    ... )
    >>> 0.0 <= result['power'] <= 1.0
    True
    >>> 0.0 <= result['asn_ratio'] <= 1.0
    True
    """
    info_times = np.asarray(info_times, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)

    result = _power_by_simulation(
        info_times=info_times,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        effect_size=effect_size,
        variance=variance,
        n_simulations=n_simulations,
        seed=seed,
    )

    output = {
        "power": result["power"],
        "asn_ratio": result["asn_ratio"],
        "prob_stop_efficacy": result["prob_stop_efficacy"],
        "prob_stop_futility": result["prob_stop_futility"],
        "info_times": info_times,
    }

    if max_sample_size is not None:
        output["expected_sample_size"] = result["asn_ratio"] * max_sample_size
    else:
        output["expected_sample_size"] = None

    return output


def sensitivity(
    *,
    info_times: ArrayLike,
    upper_bounds: ArrayLike,
    lower_bounds: ArrayLike,
    alpha: float,
    variance: float = 1.0,
    power: float = 0.8,
    n_simulations: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute minimum detectable effect (MDE) at each look.

    Implements the Sensitivity function from functional design:
        Sensitivity: ({tᵢ}, {cᵢ}, H₀) → ({MDEᵢ})

    For each look i, computes the effect size that would be just detectable
    (i.e., achieves target power) if the trial stopped at that look.

    Parameters
    ----------
    info_times : array_like
        Information times in [0, 1], length K
    upper_bounds : array_like
        Upper (efficacy) boundaries on Z-scale, length K
    lower_bounds : array_like
        Lower (futility) boundaries on Z-scale, length K
    alpha : float
        Nominal significance level
    variance : float, default=1.0
        Variance parameter
    power : float, default=0.8
        Target power for MDE calculation
    n_simulations : int, default=10000
        Number of Monte Carlo simulations for power estimation
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        info_times : np.ndarray
            Information times
        mde : np.ndarray
            Minimum detectable effect at each look
        power_at_mde : np.ndarray
            Actual power achieved at each MDE (should ≈ target power)

    Examples
    --------
    >>> import numpy as np
    >>> info_times = np.array([0.5, 1.0])
    >>> upper_bounds = np.array([2.8, 2.0])
    >>> lower_bounds = np.array([-np.inf, -np.inf])
    >>> result = sensitivity(
    ...     info_times=info_times,
    ...     upper_bounds=upper_bounds,
    ...     lower_bounds=lower_bounds,
    ...     alpha=0.05,
    ...     power=0.8,
    ...     n_simulations=500,
    ...     seed=42
    ... )
    >>> len(result['mde']) == len(info_times)
    True
    >>> bool(result['mde'][0] > result['mde'][1])  # MDE decreases with more data
    True
    """
    info_times = np.asarray(info_times, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)

    n_looks = len(info_times)
    mdes = np.zeros(n_looks)
    powers_at_mde = np.zeros(n_looks)

    # For each look, find the effect size that gives target power
    # using bisection search
    for i in range(n_looks):
        # Create single-look design truncated at look i
        truncated_times = info_times[: i + 1]
        truncated_upper = upper_bounds[: i + 1]
        truncated_lower = lower_bounds[: i + 1]

        # Bisection search for MDE
        # Initial bracket: [0, very large effect]
        delta_low = 0.0
        delta_high = 10.0

        # Check if even large effect achieves target power
        power_high = _power_by_simulation(
            info_times=truncated_times,
            upper_bounds=truncated_upper,
            lower_bounds=truncated_lower,
            effect_size=delta_high,
            variance=variance,
            n_simulations=n_simulations,
            seed=seed,
        )["power"]

        if power_high < power:
            # Even maximum effect doesn't achieve target power
            mdes[i] = np.inf
            powers_at_mde[i] = power_high
            continue

        # Bisection
        delta_mid = delta_high  # Initialize with upper bound
        power_mid = power_high
        for _ in range(20):  # Max 20 iterations
            delta_mid = (delta_low + delta_high) / 2
            power_mid = _power_by_simulation(
                info_times=truncated_times,
                upper_bounds=truncated_upper,
                lower_bounds=truncated_lower,
                effect_size=delta_mid,
                variance=variance,
                n_simulations=n_simulations,
                seed=seed,
            )["power"]

            if abs(power_mid - power) < 0.01:  # Tolerance
                break

            if power_mid < power:
                delta_low = delta_mid
            else:
                delta_high = delta_mid

        mdes[i] = delta_mid
        powers_at_mde[i] = power_mid

    return {
        "info_times": info_times,
        "mde": mdes,
        "power_at_mde": powers_at_mde,
    }
