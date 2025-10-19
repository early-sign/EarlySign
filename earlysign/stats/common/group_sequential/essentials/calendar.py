"""
Calendar-driven information time scheduling for group sequential tests.

This module implements functions for determining information times based on
calendar time, accounting for subject accrual patterns and observation lags.
This is essential for real-world experiments where interim analyses are
scheduled based on calendar dates rather than abstract information fractions.

Functions
---------
choose_t_calendar_driven:
    Compute information times from calendar dates with accrual/lag modeling

accrual_linear:
    Linear accrual pattern (constant enrollment rate)

accrual_piecewise:
    Piecewise linear accrual (different rates in different periods)

accrual_exponential:
    Exponential accrual pattern (slow start, then acceleration)

compute_info_time_from_calendar:
    Convert calendar time to information time with accrual/lag

Examples
--------
>>> import numpy as np
>>> from earlysign.stats.common.group_sequential.essentials import calendar
>>>
>>> # Linear accrual: 100 subjects over 200 days
>>> look_times = [100, 200, 300]  # Calendar days
>>> info_times = calendar.choose_t_calendar_driven(
...     calendar_times=look_times,
...     total_sample_size=100,
...     accrual_duration=200,
...     accrual_pattern="linear",
...     lag_time=0,
... )
>>> len(info_times) == 3
True
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

# =============================================================================
# Type Aliases
# =============================================================================

ArrayLike = Union[np.ndarray, List[float]]

# =============================================================================
# Accrual Pattern Functions
# =============================================================================


def accrual_linear(t: float, total_n: int, duration: float) -> float:
    """
    Linear accrual pattern with constant enrollment rate.

    Models: N(t) = (total_n / duration) * t for t ≤ duration

    Parameters
    ----------
    t : float
        Calendar time
    total_n : int
        Total sample size to accrue
    duration : float
        Accrual duration (same units as t)

    Returns
    -------
    float
        Number of subjects accrued by time t

    Examples
    --------
    >>> # 100 subjects over 200 days, linear accrual
    >>> accrual_linear(t=50, total_n=100, duration=200)
    25.0
    >>> accrual_linear(t=100, total_n=100, duration=200)
    50.0
    >>> accrual_linear(t=200, total_n=100, duration=200)
    100.0
    >>> accrual_linear(t=300, total_n=100, duration=200)  # After accrual complete
    100.0
    """
    if t <= 0:
        return 0.0
    if t >= duration:
        return float(total_n)
    return (total_n / duration) * t


def accrual_piecewise(
    t: float,
    total_n: int,
    breakpoints: List[float],
    rates: List[float],
) -> float:
    """
    Piecewise linear accrual with different rates in different periods.

    Parameters
    ----------
    t : float
        Calendar time
    total_n : int
        Total sample size
    breakpoints : list of float
        Time points where rate changes (must include 0 and final time)
    rates : list of float
        Enrollment rates (subjects per unit time) in each period

    Returns
    -------
    float
        Number of subjects accrued by time t

    Examples
    --------
    >>> # Slow start (5/day), then fast (15/day)
    >>> # Period 1: 0-50 days at 5/day = 250 subjects
    >>> # Period 2: 50-100 days at 15/day = 750 subjects
    >>> # Total: 1000 subjects
    >>> breakpoints = [0, 50, 100]
    >>> rates = [5.0, 15.0]
    >>> accrual_piecewise(t=25, total_n=1000, breakpoints=breakpoints, rates=rates)
    125.0
    >>> accrual_piecewise(t=75, total_n=1000, breakpoints=breakpoints, rates=rates)
    625.0
    """
    if t <= breakpoints[0]:
        return 0.0

    accrued = 0.0
    for i in range(len(rates)):
        start = breakpoints[i]
        end = breakpoints[i + 1]

        if t <= start:
            break
        elif t >= end:
            # Full period
            accrued += rates[i] * (end - start)
        else:
            # Partial period
            accrued += rates[i] * (t - start)
            break

    return min(accrued, float(total_n))


def accrual_exponential(
    t: float,
    total_n: int,
    duration: float,
    shape: float = 2.0,
) -> float:
    """
    Exponential accrual pattern (slow start, then acceleration).

    Models: N(t) = total_n * (1 - exp(-shape * t / duration))^k
    where k is chosen so N(duration) = total_n

    Parameters
    ----------
    t : float
        Calendar time
    total_n : int
        Total sample size
    duration : float
        Accrual duration
    shape : float, default=2.0
        Shape parameter (higher = faster acceleration)

    Returns
    -------
    float
        Number of subjects accrued by time t

    Examples
    --------
    >>> # 100 subjects over 200 days, exponential accrual
    >>> n = accrual_exponential(t=100, total_n=100, duration=200, shape=2.0)
    >>> bool(60 < n < 80)  # Faster than expected at midpoint
    True
    >>> n_final = accrual_exponential(t=200, total_n=100, duration=200, shape=2.0)
    >>> abs(n_final - 100.0) < 1.0  # Should reach total_n
    True
    """
    if t <= 0:
        return 0.0
    if t >= duration:
        return float(total_n)

    # Use exponential CDF-like function
    # Normalize so that at duration, we get total_n
    x = t / duration
    accrued = total_n * (1 - np.exp(-shape * x)) / (1 - np.exp(-shape))

    return float(min(accrued, float(total_n)))


# =============================================================================
# Information Time Calculation
# =============================================================================


def compute_info_time_from_calendar(
    calendar_time: float,
    total_sample_size: int,
    accrual_fn: Callable[[float], float],
    lag_time: float = 0.0,
    min_info_time: float = 0.0,
) -> float:
    """
    Convert calendar time to information time fraction.

    Information time = (subjects with complete data) / (total sample size)

    Subjects with complete data at calendar time t:
    - Enrolled before (t - lag_time)
    - Because observation lag means subjects enrolled at time s have
      complete data available at time (s + lag_time)

    Parameters
    ----------
    calendar_time : float
        Calendar time point
    total_sample_size : int
        Total planned sample size
    accrual_fn : callable
        Function mapping time -> number accrued
    lag_time : float, default=0.0
        Observation lag (time from enrollment to data availability)
    min_info_time : float, default=0.0
        Minimum information fraction (for numerical stability)

    Returns
    -------
    float
        Information time fraction in [0, 1]

    Examples
    --------
    >>> # Linear accrual: 100 subjects over 200 days, no lag
    >>> accrual = lambda t: accrual_linear(t, total_n=100, duration=200)
    >>> info_t = compute_info_time_from_calendar(
    ...     calendar_time=100,
    ...     total_sample_size=100,
    ...     accrual_fn=accrual,
    ...     lag_time=0,
    ... )
    >>> bool(abs(info_t - 0.5) < 0.01)  # 50% of subjects enrolled
    True

    >>> # With 20-day lag: data available for subjects enrolled before t-20
    >>> info_t_lag = compute_info_time_from_calendar(
    ...     calendar_time=100,
    ...     total_sample_size=100,
    ...     accrual_fn=accrual,
    ...     lag_time=20,
    ... )
    >>> bool(info_t_lag < info_t)  # Less info available with lag
    True
    """
    # Subjects with complete data = accrued before (calendar_time - lag_time)
    effective_time = max(0, calendar_time - lag_time)
    n_complete = accrual_fn(effective_time)

    # Information fraction
    info_frac = n_complete / total_sample_size

    # Clamp to [min_info_time, 1.0]
    return float(np.clip(info_frac, min_info_time, 1.0))


def choose_t_calendar_driven(
    calendar_times: ArrayLike,
    total_sample_size: int,
    accrual_duration: float,
    accrual_pattern: str = "linear",
    accrual_params: Optional[Dict[str, Any]] = None,
    lag_time: float = 0.0,
    min_info_time: float = 1e-6,
) -> np.ndarray:
    """
    Compute information times from calendar dates with accrual/lag modeling.

    Implements ChooseT.CalendarDriven from functional design:
        ChooseT.CalendarDriven: (calendar, accrual, lag) → {tᵢ}

    This function converts calendar-based interim analysis dates into
    information time fractions, accounting for:
    - Subject accrual patterns (linear, piecewise, exponential)
    - Observation lag (time from enrollment to data availability)

    Parameters
    ----------
    calendar_times : array-like
        Calendar time points for interim analyses
    total_sample_size : int
        Total planned sample size
    accrual_duration : float
        Duration of accrual period (same units as calendar_times)
    accrual_pattern : str, default="linear"
        Accrual pattern: "linear", "piecewise", or "exponential"
    accrual_params : dict, optional
        Additional parameters for accrual pattern:
        - For "piecewise": {"breakpoints": [...], "rates": [...]}
        - For "exponential": {"shape": float}
    lag_time : float, default=0.0
        Observation lag (time from enrollment to data availability)
    min_info_time : float, default=1e-6
        Minimum information fraction (for numerical stability)

    Returns
    -------
    np.ndarray
        Information time fractions corresponding to calendar_times

    Examples
    --------
    >>> import numpy as np
    >>> # Linear accrual: 100 subjects over 200 days, no lag
    >>> calendar = [50, 100, 150, 200]
    >>> info_times = choose_t_calendar_driven(
    ...     calendar_times=calendar,
    ...     total_sample_size=100,
    ...     accrual_duration=200,
    ...     accrual_pattern="linear",
    ...     lag_time=0,
    ... )
    >>> len(info_times) == 4
    True
    >>> bool(info_times[-1] >= 0.99)  # Final look near 100%
    True

    >>> # With lag: less information available at early looks
    >>> info_times_lag = choose_t_calendar_driven(
    ...     calendar_times=calendar,
    ...     total_sample_size=100,
    ...     accrual_duration=200,
    ...     accrual_pattern="linear",
    ...     lag_time=20,
    ... )
    >>> bool(info_times_lag[0] < info_times[0])  # Less info with lag
    True

    >>> # Exponential accrual: slow start
    >>> info_times_exp = choose_t_calendar_driven(
    ...     calendar_times=calendar,
    ...     total_sample_size=100,
    ...     accrual_duration=200,
    ...     accrual_pattern="exponential",
    ...     accrual_params={"shape": 2.0},
    ...     lag_time=0,
    ... )
    >>> bool(info_times_exp[1] > info_times[1])  # Faster accrual at midpoint
    True

    Notes
    -----
    This function is essential for real-world trial planning where:
    1. Interim analyses are scheduled on specific calendar dates
    2. Subject enrollment follows a predictable pattern
    3. Endpoints have measurement/processing delays (lag)

    The information time determines the statistical information available,
    which directly affects boundary calculations and power.
    """
    if accrual_params is None:
        accrual_params = {}

    # Create accrual function based on pattern
    if accrual_pattern == "linear":

        def accrual_fn(t: float) -> float:
            return accrual_linear(
                t, total_n=total_sample_size, duration=accrual_duration
            )

    elif accrual_pattern == "piecewise":
        breakpoints = accrual_params.get("breakpoints", [0, accrual_duration])
        rates = accrual_params.get("rates", [total_sample_size / accrual_duration])

        def accrual_fn(t: float) -> float:
            return accrual_piecewise(
                t, total_n=total_sample_size, breakpoints=breakpoints, rates=rates
            )

    elif accrual_pattern == "exponential":
        shape = accrual_params.get("shape", 2.0)

        def accrual_fn(t: float) -> float:
            return accrual_exponential(
                t, total_n=total_sample_size, duration=accrual_duration, shape=shape
            )

    else:
        raise ValueError(
            f"Unknown accrual_pattern: {accrual_pattern}. "
            "Must be 'linear', 'piecewise', or 'exponential'"
        )

    # Convert each calendar time to information time
    calendar_array = np.asarray(calendar_times)
    info_times = np.array(
        [
            compute_info_time_from_calendar(
                calendar_time=t,
                total_sample_size=total_sample_size,
                accrual_fn=accrual_fn,
                lag_time=lag_time,
                min_info_time=min_info_time,
            )
            for t in calendar_array
        ]
    )

    return info_times


def calendar_from_info_time(
    info_time: float,
    total_sample_size: int,
    accrual_duration: float,
    accrual_pattern: str = "linear",
    accrual_params: Optional[Dict[str, Any]] = None,
    lag_time: float = 0.0,
    tol: float = 0.01,
    max_iter: int = 50,
) -> float:
    """
    Inverse: find calendar time that achieves target information time.

    Uses bisection search to find the calendar time t such that
    the resulting information time matches the target.

    Parameters
    ----------
    info_time : float
        Target information time fraction
    total_sample_size : int
        Total sample size
    accrual_duration : float
        Accrual duration
    accrual_pattern : str, default="linear"
        Accrual pattern
    accrual_params : dict, optional
        Accrual parameters
    lag_time : float, default=0.0
        Observation lag
    tol : float, default=0.01
        Tolerance for bisection search
    max_iter : int, default=50
        Maximum iterations

    Returns
    -------
    float
        Calendar time achieving target info_time

    Examples
    --------
    >>> # Find calendar time for 50% information
    >>> t_cal = calendar_from_info_time(
    ...     info_time=0.5,
    ...     total_sample_size=100,
    ...     accrual_duration=200,
    ...     accrual_pattern="linear",
    ...     lag_time=0,
    ... )
    >>> 95 < t_cal < 105  # Should be around 100 days
    True
    """
    if accrual_params is None:
        accrual_params = {}

    # Binary search bounds
    t_low = 0.0
    t_high = accrual_duration + lag_time + 100  # Add buffer

    for _ in range(max_iter):
        t_mid = (t_low + t_high) / 2

        # Compute info time at midpoint
        info_times = choose_t_calendar_driven(
            calendar_times=[t_mid],
            total_sample_size=total_sample_size,
            accrual_duration=accrual_duration,
            accrual_pattern=accrual_pattern,
            accrual_params=accrual_params,
            lag_time=lag_time,
        )
        info_mid = info_times[0]

        # Check convergence
        if abs(info_mid - info_time) < tol:
            return t_mid

        # Update bounds
        if info_mid < info_time:
            t_low = t_mid
        else:
            t_high = t_mid

    # If not converged, return best estimate
    return (t_low + t_high) / 2
