"""
Boundary calculation functions for group sequential designs.

This module implements the core Design function from the functional design system:
    Design: (α(t), β(t), H₀, BindingMode, {tᵢ}) → {cᵢ}

Functions compute critical boundary values (efficacy and futility) at each
information time based on spending functions and design parameters.

Functions
---------
Core design function:
    resolve_boundary_from_design(design_payload, info_time, look) -> (upper, lower, scale)
    compute_boundaries_at_times(design_payload, info_times) -> dict

Spending-based boundaries:
    efficacy_boundary_from_spending(alpha_fn, info_time, tails) -> float
    futility_boundary_from_spending(beta_fn, info_time) -> float

Examples
--------
>>> from earlysign.stats.common.group_sequential.essentials import boundaries
>>> import numpy as np
>>>
>>> # Define design
>>> design = {
...     "alpha": 0.05,
...     "tails": 2,
...     "scale": "z",
...     "efficacy": {"style": "alpha_spending", "family": "obf"},
...     "futility": {"mode": "none"},
... }
>>>
>>> # Compute boundary at 50% information
>>> upper, lower, scale = boundaries.resolve_boundary_from_design(
...     design_payload=design,
...     info_time=0.5,
...     look=2
... )
>>> scale
'z'
>>> round(upper, 3)
2.772
>>> lower
-inf

>>> # Alternatively, use the class-based spending helpers directly
>>> from earlysign.stats.essentials.methods.group_sequential.spending import OBFSpending
>>> import numpy as np
>>> s = OBFSpending(alpha=0.05, sided=2)
>>> # alpha at 50% information (two-sided uses internal halving)
>>> round(float(s.cumulative(np.array([0.5]))[0]), 6)
0.005575
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from earlysign.stats.common.group_sequential.essentials import conversions
from earlysign.stats.common.group_sequential.essentials.design_schema import (
    validate_design_payload,
)
from earlysign.stats.essentials.methods.group_sequential.spending import (
    HSDSpending,
    OBFSpending,
    PocockSpending,
    SpendingFunction,
)

# =============================================================================
# Main Boundary Resolution Function
# =============================================================================


def resolve_boundary_from_design(
    *,
    design_payload: Mapping[str, Any],
    info_time: float,
    look: Optional[int] = None,
) -> Tuple[float, float, str]:
    """
    Compute (upper, lower, scale) boundaries given a design and info_time.

    This is the main interface for boundary calculation, matching the
    functional design signature (simplified):
        Design: (design_spec, t) → (upper, lower)

    Parameters
    ----------
    design_payload : dict
        Design specification with keys: alpha, tails, scale, efficacy, futility.
    info_time : float
        Information time in (0, 1].
    look : int, optional
        Look number (1-indexed). Required for significance_level style.

    Returns
    -------
    upper : float
        Upper (efficacy) boundary on the design's scale.
    lower : float
        Lower (futility) boundary on the design's scale.
    scale : str
        Scale of returned boundaries ("z" or "bm").

    Examples
    --------
    >>> design = {
    ...     "alpha": 0.05,
    ...     "tails": 2,
    ...     "scale": "z",
    ...     "efficacy": {"style": "alpha_spending", "family": "obf"},
    ...     "futility": {"mode": "none"},
    ... }
    >>> upper, lower, scale = resolve_boundary_from_design(
    ...     design_payload=design, info_time=0.5, look=2
    ... )
    >>> scale
    'z'
    >>> round(upper, 3)
    2.772
    >>> lower
    -inf

    Notes
    -----
    The function handles:
    - Alpha spending (obf, pocock, hsd) for efficacy boundaries
    - Per-look significance levels for efficacy boundaries
    - Futility boundaries (none, symmetric, fixed_z, beta_spending)
    - Scale conversions (z ↔ bm)
    """
    t = float(info_time)
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"info_time must be in [0, 1], got {t}")

    # Validate payload
    design = dict(design_payload)
    validate_design_payload(design)

    float(design["alpha"])
    int(design["tails"])
    scale = str(design["scale"])

    # Compute efficacy (upper) boundary
    upper_z = _resolve_efficacy_upper_z(design, t, look)

    # Compute futility (lower) boundary
    lower_z = _resolve_futility_lower_z(design, upper_z, look)

    # Convert to target scale if needed
    if scale == "bm":
        upper = conversions.z_to_brownian(upper_z, t)
        lower = (
            conversions.z_to_brownian(lower_z, t) if np.isfinite(lower_z) else lower_z
        )
    else:  # scale == "z"
        upper = upper_z
        lower = lower_z

    return upper, lower, scale


def compute_boundaries_at_times(
    design_payload: Mapping[str, Any], info_times: np.ndarray
) -> Dict[str, Any]:
    """
    Compute boundaries at multiple information times.

    Convenience function for batch boundary calculation.

    Parameters
    ----------
    design_payload : dict
        Design specification.
    info_times : np.ndarray
        Array of information times.

    Returns
    -------
    dict
        Dictionary with keys:
        - info_times: np.ndarray
        - upper: np.ndarray (efficacy boundaries)
        - lower: np.ndarray (futility boundaries)
        - scale: str

    Examples
    --------
    >>> import numpy as np
    >>> design = {
    ...     "alpha": 0.05,
    ...     "tails": 2,
    ...     "scale": "z",
    ...     "efficacy": {"style": "alpha_spending", "family": "obf"},
    ...     "futility": {"mode": "none"},
    ... }
    >>> times = np.array([0.33, 0.67, 1.0])
    >>> result = compute_boundaries_at_times(design, times)
    >>> result["scale"]
    'z'
    >>> len(result["upper"])
    3
    """
    n_looks = len(info_times)
    upper_vals = np.zeros(n_looks)
    lower_vals = np.zeros(n_looks)
    scale = None

    for i, t in enumerate(info_times):
        upper, lower, sc = resolve_boundary_from_design(
            design_payload=design_payload, info_time=t, look=i + 1
        )
        upper_vals[i] = upper
        lower_vals[i] = lower
        if scale is None:
            scale = sc

    return {
        "info_times": info_times,
        "upper": upper_vals,
        "lower": lower_vals,
        "scale": scale,
    }


# =============================================================================
# Spending-based Boundary Functions
# =============================================================================


def efficacy_boundary_from_spending(
    alpha_fn: Callable[[float], float], info_time: float, tails: int = 2
) -> float:
    """
    Compute efficacy boundary from alpha spending function.

    Parameters
    ----------
    alpha_fn : callable
        Alpha spending function α(t) taking info_time and returning
        cumulative alpha spent.
    info_time : float
        Information time in (0, 1].
    tails : int, default=2
        Number of tails.

    Returns
    -------
    float
        Upper boundary on Z scale.

    Examples
    --------
    >>> from earlysign.stats.essentials.methods.group_sequential.spending import OBFSpending
    >>> import numpy as np
    >>> s = OBFSpending(alpha=0.05, sided=2)
    >>> alpha_fn = lambda t: float(s.cumulative(np.array([t]))[0])
    >>> z = efficacy_boundary_from_spending(alpha_fn, info_time=0.5, tails=2)
    >>> round(z, 3)
    2.772
    """
    alpha_spent = alpha_fn(info_time)
    upper_z, _ = conversions.cumulative_to_nominal_z(alpha_spent, tails=tails)
    return upper_z


def futility_boundary_from_spending(
    beta_fn: Callable[[float], float], info_time: float
) -> float:
    """
    Compute futility boundary from beta spending function.

    Parameters
    ----------
    beta_fn : callable
        Beta spending function β(t) taking info_time and returning
        cumulative beta spent.
    info_time : float
        Information time in (0, 1].

    Returns
    -------
    float
        Lower boundary on Z scale (negative value).

    Examples
    --------
    >>> from earlysign.stats.essentials.methods.group_sequential.spending import OBFSpending
    >>> import numpy as np
    >>> s_beta = OBFSpending(alpha=0.10, sided=1)
    >>> beta_fn = lambda t: float(s_beta.cumulative(np.array([t]))[0])
    >>> z = futility_boundary_from_spending(beta_fn, info_time=0.5)
    >>> round(z, 3)
    -1.812
    """
    beta_spent = beta_fn(info_time)
    # Beta boundaries are one-sided (lower)
    z, _ = conversions.cumulative_to_nominal_z(beta_spent, tails=1)
    return -z  # Negative for futility


# =============================================================================
# Internal Helpers
# =============================================================================


def _resolve_efficacy_upper_z(
    design: Mapping[str, Any], t: float, look: Optional[int]
) -> float:
    """
    Resolve efficacy (upper) boundary on Z-scale.

    Internal helper implementing efficacy boundary logic.
    """
    alpha = float(design["alpha"])
    tails = int(design["tails"])
    efficacy = design["efficacy"]
    style = efficacy["style"]

    if style == "alpha_spending":
        # Spending function approach
        family = efficacy.get("family", "obf")
        gamma = efficacy.get("gamma", -4.0)

        # Instantiate class-based spending and compute cumulative
        # Annotate `s` with the SpendingFunction protocol so different
        # concrete spending implementations can be assigned without
        # causing type-checker errors.
        s: SpendingFunction
        key = str(family).lower()
        if key in ("obf", "obrien_fleming", "o'brien-fleming"):
            s = OBFSpending(alpha=alpha, sided=tails)
        elif key == "pocock":
            s = PocockSpending(alpha=alpha)
        elif key == "hsd":
            s = HSDSpending(alpha=alpha, gamma=gamma)
        else:
            raise ValueError(f"Unknown spending family: {family}")

        alpha_spent = float(s.cumulative(np.array([t]))[0])

        # Convert to Z boundary
        upper_z, _ = conversions.cumulative_to_nominal_z(alpha_spent, tails=tails)
        return upper_z

    elif style == "significance_level":
        # Per-look significance levels
        if look is None:
            raise ValueError("look number required for significance_level style")

        alpha_levels = efficacy.get("alpha_levels")
        if alpha_levels is None:
            raise ValueError("alpha_levels required for significance_level style")

        # Get level for this look
        level = _get_alpha_level_for_look(alpha_levels, look)

        # Convert to Z boundary
        upper_z, _ = conversions.level_to_nominal_z(level, tails=tails)
        return upper_z

    else:
        raise ValueError(f"Unknown efficacy style: {style}")


def _resolve_futility_lower_z(
    design: Mapping[str, Any], upper_z: float, look: Optional[int]
) -> float:
    """
    Resolve futility (lower) boundary on Z-scale.

    Internal helper implementing futility boundary logic.
    """
    tails = int(design["tails"])
    futility = design["futility"]
    mode = futility["mode"]

    if mode == "none":
        # No futility boundary
        return float("-inf")

    elif mode == "symmetric":
        # Symmetric: lower = -upper
        if tails != 2:
            raise ValueError("Symmetric futility only valid for two-sided tests")
        return -upper_z

    elif mode == "fixed_z":
        # Fixed Z value(s)
        z_val = futility.get("z")
        if z_val is None:
            raise ValueError("z value required for fixed_z futility mode")

        # Handle dict (per-look) or scalar
        if isinstance(z_val, dict):
            if look is None:
                raise ValueError("look number required for per-look fixed_z")
            if look not in z_val:
                raise KeyError(f"Look {look} not in futility z values")
            return float(z_val[look])
        else:
            # Scalar z value
            return float(z_val)

    elif mode == "beta_spending":
        # Beta spending function
        futility.get("family", "obf")
        futility.get("gamma", -4.0)
        futility.get("beta", 0.10)  # Default power = 0.90

        # Beta spending requires the information time to compute cumulative
        # beta; this helper does not receive the info_time. Preserve the
        # previous behavior and note that refactoring is needed to support
        # beta_spending here (caller must supply info_time-aware logic).
        raise NotImplementedError("beta_spending futility mode needs refactoring")

    else:
        raise ValueError(f"Unknown futility mode: {mode}")


def _get_alpha_level_for_look(
    alpha_levels: Union[Mapping[int, float], List[float]], look: int
) -> float:
    """
    Get alpha level for a specific look.

    Internal helper for parsing alpha_levels.
    """
    if isinstance(alpha_levels, dict):
        if look not in alpha_levels:
            raise KeyError(f"Look {look} not found in alpha_levels")
        return float(alpha_levels[look])

    # List/sequence
    idx = look - 1
    if idx < 0 or idx >= len(alpha_levels):
        raise IndexError(f"Look {look} out of range for alpha_levels")
    return float(alpha_levels[idx])
