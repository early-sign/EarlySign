"""
Visualization and surface analysis for group sequential tests.

This module implements functions for creating 2D/3D visualizations of
group sequential test characteristics, particularly power surfaces that
show how power varies across both time and effect size dimensions.

Functions
---------
power_surface:
    Compute 2D power surface: (time, effect_size) → power

power_surface_grid:
    Generate grid of power values for plotting

mde_profile:
    Compute MDE evolution over time: time → MDE(time)

Examples
--------
>>> import numpy as np
>>> from earlysign.stats_old.common.group_sequential.essentials import visualization
>>>
>>> # Create power surface for visualization
>>> surface = visualization.power_surface(
...     info_times=[0.33, 0.67, 1.0],
...     upper_bounds=[2.5, 2.2, 2.0],
...     lower_bounds=[-np.inf, -np.inf, -np.inf],
...     effect_sizes=[0.1, 0.3, 0.5],
...     n_simulations=500,
...     seed=42
... )
>>> 'power_matrix' in surface
True
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from earlysign.stats_old.common.group_sequential.essentials import performance

# =============================================================================
# Type Aliases
# =============================================================================

ArrayLike = Union[np.ndarray, List[float]]

# =============================================================================
# Power Surface Functions
# =============================================================================


def power_surface(
    info_times: ArrayLike,
    upper_bounds: ArrayLike,
    lower_bounds: ArrayLike,
    effect_sizes: ArrayLike,
    time_points: Optional[ArrayLike] = None,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute 2D power surface showing power across time and effect size.

    Implements PowerSurface from functional design:
        PowerSurface: ({tᵢ}, {cᵢ}, H₁) → [(t, δ) ↦ Power(t, δ)]

    Creates a 2D surface showing how power varies with both:
    - Information time (x-axis): When analysis occurs
    - Effect size (y-axis): What effect is being detected

    This is useful for:
    - Understanding power evolution during the trial
    - Identifying when different effect sizes become detectable
    - Planning interim analyses based on power requirements

    Parameters
    ----------
    info_times : array-like
        Information times for the design
    upper_bounds : array-like
        Efficacy boundaries (Z-scale)
    lower_bounds : array-like
        Futility boundaries (Z-scale)
    effect_sizes : array-like
        Effect sizes to evaluate (standardized)
    time_points : array-like, optional
        Specific time points to evaluate (default: use info_times)
    n_simulations : int, default=5000
        Monte Carlo simulations for power estimation
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        power_matrix : np.ndarray
            Power values, shape (n_effect_sizes, n_time_points)
        effect_sizes : np.ndarray
            Effect sizes evaluated (y-axis)
        time_points : np.ndarray
            Time points evaluated (x-axis)
        info_times : np.ndarray
            Information times (for reference)
        upper_bounds : np.ndarray
            Efficacy boundaries (for reference)
        lower_bounds : np.ndarray
            Futility boundaries (for reference)

    Examples
    --------
    >>> import numpy as np
    >>> # Simple 3-look design
    >>> surface = power_surface(
    ...     info_times=[0.33, 0.67, 1.0],
    ...     upper_bounds=[2.5, 2.2, 2.0],
    ...     lower_bounds=[-np.inf, -np.inf, -np.inf],
    ...     effect_sizes=[0.2, 0.5],
    ...     n_simulations=500,
    ...     seed=42
    ... )
    >>> surface['power_matrix'].shape
    (2, 3)
    >>> bool(surface['power_matrix'][1, -1] > surface['power_matrix'][0, -1])
    True

    Notes
    -----
    The power at each (time, effect_size) point is computed using
    Monte Carlo simulation. For smoother surfaces, increase n_simulations,
    but this will increase computation time.

    For plotting, you can use:
    - Heatmap: plt.imshow(power_matrix, extent=[...])
    - Contour: plt.contour(time_points, effect_sizes, power_matrix)
    - 3D surface: ax.plot_surface(T, E, power_matrix)
    """
    info_times_arr = np.asarray(info_times)
    upper_bounds_arr = np.asarray(upper_bounds)
    lower_bounds_arr = np.asarray(lower_bounds)
    effect_sizes_arr = np.asarray(effect_sizes)

    # Use info_times as time_points if not specified
    if time_points is None:
        time_points_arr = info_times_arr
    else:
        time_points_arr = np.asarray(time_points)

    n_effects = len(effect_sizes_arr)
    n_times = len(time_points_arr)

    # Initialize power matrix
    power_matrix = np.zeros((n_effects, n_times))

    # Compute power for each (effect_size, time_point) combination
    for i, effect_size in enumerate(effect_sizes_arr):
        # Get performance for this effect size
        perf = performance.performance(
            info_times=info_times_arr,
            upper_bounds=upper_bounds_arr,
            lower_bounds=lower_bounds_arr,
            effect_size=effect_size,
            n_simulations=n_simulations,
            seed=seed,
        )

        # Extract power at requested time points
        # Power at each look is cumulative probability of stopping for efficacy
        # For time points matching info_times, use direct values
        # For intermediate points, interpolate
        prob_stop_efficacy = np.asarray(perf["prob_stop_efficacy"])
        power_by_look = prob_stop_efficacy.cumsum()

        for j, t in enumerate(time_points_arr):
            if t in info_times_arr:
                # Direct lookup
                idx = np.where(info_times_arr == t)[0][0]
                power_matrix[i, j] = power_by_look[idx]
            else:
                # Interpolate
                power_matrix[i, j] = np.interp(t, info_times_arr, power_by_look)

    return {
        "power_matrix": power_matrix,
        "effect_sizes": effect_sizes_arr,
        "time_points": time_points_arr,
        "info_times": info_times_arr,
        "upper_bounds": upper_bounds_arr,
        "lower_bounds": lower_bounds_arr,
    }


def power_surface_grid(
    info_times: ArrayLike,
    upper_bounds: ArrayLike,
    lower_bounds: ArrayLike,
    effect_size_range: Tuple[float, float] = (0.0, 1.0),
    n_effect_points: int = 20,
    time_range: Optional[Tuple[float, float]] = None,
    n_time_points: Optional[int] = None,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a fine grid of power values for smooth surface plotting.

    Convenience function that creates evenly-spaced grids for both
    effect size and time dimensions, then computes the power surface.

    Parameters
    ----------
    info_times : array-like
        Information times for the design
    upper_bounds : array-like
        Efficacy boundaries
    lower_bounds : array-like
        Futility boundaries
    effect_size_range : tuple, default=(0.0, 1.0)
        (min, max) effect sizes to evaluate
    n_effect_points : int, default=20
        Number of effect size points in grid
    time_range : tuple, optional
        (min, max) time points (default: min/max of info_times)
    n_time_points : int, optional
        Number of time points (default: use info_times)
    n_simulations : int, default=5000
        Monte Carlo simulations
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Same as power_surface(), with evenly-spaced grids

    Examples
    --------
    >>> import numpy as np
    >>> # Generate fine grid for plotting
    >>> grid = power_surface_grid(
    ...     info_times=[0.5, 1.0],
    ...     upper_bounds=[2.5, 2.0],
    ...     lower_bounds=[-np.inf, -np.inf],
    ...     effect_size_range=(0.1, 0.8),
    ...     n_effect_points=5,
    ...     n_simulations=500,
    ...     seed=42
    ... )
    >>> grid['power_matrix'].shape[0] == 5  # n_effect_points
    True
    """
    info_times_arr = np.asarray(info_times)

    # Create effect size grid
    effect_sizes = np.linspace(
        effect_size_range[0], effect_size_range[1], n_effect_points
    )

    # Create time grid
    if time_range is None:
        time_range = (info_times_arr.min(), info_times_arr.max())

    if n_time_points is None:
        time_points = info_times_arr
    else:
        time_points = np.linspace(time_range[0], time_range[1], n_time_points)

    # Compute power surface
    return power_surface(
        info_times=info_times_arr,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        effect_sizes=effect_sizes,
        time_points=time_points,
        n_simulations=n_simulations,
        seed=seed,
    )


def mde_profile(
    info_times: ArrayLike,
    upper_bounds: ArrayLike,
    lower_bounds: ArrayLike,
    alpha: float = 0.05,
    power: float = 0.8,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute MDE (minimum detectable effect) evolution over time.

    Implements MDEProfile from functional design:
        MDEProfile: ({tᵢ}, {cᵢ}, H₀) → [t ↦ MDE(t)]

    Shows how the minimum detectable effect changes as the trial progresses
    and more information accumulates.

    Parameters
    ----------
    info_times : array-like
        Information times
    upper_bounds : array-like
        Efficacy boundaries
    lower_bounds : array-like
        Futility boundaries
    alpha : float, default=0.05
        Significance level
    power : float, default=0.8
        Target power for MDE calculation
    n_simulations : int, default=5000
        Monte Carlo simulations
    seed : int, optional
        Random seed

    Returns
    -------
    dict with keys:
        mde : np.ndarray
            MDE at each look
        info_times : np.ndarray
            Information times
        alpha : float
            Significance level
        power : float
            Target power

    Examples
    --------
    >>> import numpy as np
    >>> # Compute MDE profile
    >>> profile = mde_profile(
    ...     info_times=[0.5, 1.0],
    ...     upper_bounds=[2.5, 2.0],
    ...     lower_bounds=[-np.inf, -np.inf],
    ...     alpha=0.05,
    ...     power=0.8,
    ...     n_simulations=500,
    ...     seed=42
    ... )
    >>> len(profile['mde']) == 2
    True
    >>> bool(profile['mde'][1] < profile['mde'][0])  # MDE decreases over time
    True

    Notes
    -----
    MDE typically decreases over time as more information accumulates,
    meaning smaller effects become detectable. This profile is useful
    for understanding when specific effect sizes become reliably detectable.
    """
    # Use sensitivity function to compute MDE at each look
    sens = performance.sensitivity(
        info_times=info_times,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        alpha=alpha,
        power=power,
        n_simulations=n_simulations,
        seed=seed,
    )

    return {
        "mde": sens["mde"],
        "info_times": np.asarray(info_times),
        "alpha": alpha,
        "power": power,
    }


def power_contours(
    info_times: ArrayLike,
    upper_bounds: ArrayLike,
    lower_bounds: ArrayLike,
    target_powers: ArrayLike = [0.5, 0.8, 0.9],
    effect_size_range: Tuple[float, float] = (0.0, 1.0),
    n_effect_points: int = 50,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Find contour lines for specific power levels.

    For each target power level, finds the effect size required at each
    time point to achieve that power. Useful for visualization.

    Parameters
    ----------
    info_times : array-like
        Information times
    upper_bounds : array-like
        Efficacy boundaries
    lower_bounds : array-like
        Futility boundaries
    target_powers : array-like, default=[0.5, 0.8, 0.9]
        Power levels to find contours for
    effect_size_range : tuple, default=(0.0, 1.0)
        Range of effect sizes to search
    n_effect_points : int, default=50
        Grid resolution for searching
    n_simulations : int, default=5000
        Monte Carlo simulations
    seed : int, optional
        Random seed

    Returns
    -------
    dict with keys:
        contours : dict
            {power_level: effect_sizes} for each target power
        info_times : np.ndarray
            Information times

    Examples
    --------
    >>> import numpy as np
    >>> # Find 80% power contour
    >>> contours = power_contours(
    ...     info_times=[0.5, 1.0],
    ...     upper_bounds=[2.5, 2.0],
    ...     lower_bounds=[-np.inf, -np.inf],
    ...     target_powers=[0.8],
    ...     n_effect_points=10,
    ...     n_simulations=500,
    ...     seed=42
    ... )
    >>> 0.8 in contours['contours']
    True
    """
    info_times_arr = np.asarray(info_times)
    target_powers_arr = np.asarray(target_powers)

    # Compute full power surface
    surface = power_surface_grid(
        info_times=info_times_arr,
        upper_bounds=upper_bounds,
        lower_bounds=lower_bounds,
        effect_size_range=effect_size_range,
        n_effect_points=n_effect_points,
        n_time_points=len(info_times_arr),
        n_simulations=n_simulations,
        seed=seed,
    )

    power_matrix = surface["power_matrix"]
    effect_sizes = surface["effect_sizes"]

    # Find contours by interpolation
    contours = {}
    for target_power in target_powers_arr:
        # For each time point, find effect size achieving target power
        contour_effect_sizes = []
        for j in range(len(info_times_arr)):
            power_at_time = power_matrix[:, j]
            # Interpolate to find effect size for target power
            if power_at_time.max() >= target_power >= power_at_time.min():
                effect_size = np.interp(target_power, power_at_time, effect_sizes)
            else:
                # Target power not achievable or always exceeded
                effect_size = np.nan
            contour_effect_sizes.append(effect_size)

        contours[target_power] = np.array(contour_effect_sizes)

    return {
        "contours": contours,
        "info_times": info_times_arr,
    }
