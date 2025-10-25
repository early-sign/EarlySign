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

# Delegate primary boundary computation to the new centralized implementation
from earlysign.stats.essentials.methods.group_sequential import boundary
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
    design: Optional[Mapping[str, Any]] = None,
    design_payload: Optional[Mapping[str, Any]] = None,
    info_time: float,
    look: Optional[int] = None,
) -> Tuple[float, float, str]:
    """Compatibility wrapper that delegates to the canonical
    `earlysign.stats.essentials.methods.group_sequential.boundary.resolve_boundary`.

    This wrapper preserves the original function signature so callers in the
    codebase can be migrated incrementally while using the new implementation.
    """
    # Support legacy callers that pass the payload as `design_payload=` as
    # well as newer callers that use `design=`. Prefer `design` when both
    # are present.
    cfg = design if design is not None else design_payload
    if cfg is None:
        raise TypeError("Either 'design' or 'design_payload' must be provided")

    # Delegate to the canonical BoundaryCalculator implementation by
    # constructing a calculator from the validated config and calling the
    # instance method. This avoids module-level helper indirection.
    calc = boundary.BoundaryCalculator(spec=cfg, process=None)
    return calc.compute_boundary(info_time=info_time, look=look)


def compute_boundaries_at_times(
    design: Mapping[str, Any], info_times: np.ndarray
) -> Dict[str, Any]:
    """Compatibility wrapper that delegates to the canonical
    `earlysign.stats.essentials.methods.group_sequential.boundary.compute_boundaries`.

    Preserves the original return shape expected by callers.
    """
    # Construct a calculator and compute boundaries directly.
    calc = boundary.BoundaryCalculator(spec=design, process=None)
    return calc.compute_boundaries(info_times=info_times)


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
