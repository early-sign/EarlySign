"""
Type definitions and schema validation for group sequential designs.

This module defines enums, dataclasses, and validation schemas for design
specifications. These types support both the functional design system and
the existing Ledger-based architecture.

Types
-----
BindingMode : Enum
    Whether futility boundaries are binding (contribute to alpha) or non-binding.
SpendingFamily : Enum
    Spending function families (OBF, Pocock, HSD).
BoundaryScale : Enum
    Scale for boundary representation (Z or Brownian motion).
DesignPayload : TypedDict
    Schema for design payload dictionaries.

Examples
--------
>>> from earlysign.stats.common.group_sequential.essentials import design_schema
>>> mode = design_schema.BindingMode.NON_BINDING
>>> mode.value
'non_binding'

>>> family = design_schema.SpendingFamily.OBF
>>> family.value
'obf'
"""

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

# =============================================================================
# Enums
# =============================================================================


class BindingMode(str, Enum):
    """
    Whether futility boundaries contribute to Type I error spending.

    Attributes
    ----------
    BINDING : str
        Futility boundaries are binding: early stopping for futility
        contributes to alpha spending. Efficacy boundaries can be less
        conservative as a result.
    NON_BINDING : str
        Futility boundaries are non-binding: early stopping for futility
        does not contribute to alpha. Efficacy boundaries remain conservative.
        This is the typical choice in most applications.

    Examples
    --------
    >>> BindingMode.BINDING.value
    'binding'
    >>> BindingMode.NON_BINDING.value
    'non_binding'

    Notes
    -----
    In clinical trials, non-binding futility is more common to preserve
    Type I error control even if the trial continues after crossing a
    futility boundary (e.g., due to external reasons).
    """

    BINDING = "binding"
    NON_BINDING = "non_binding"


class SpendingFamily(str, Enum):
    """
    Spending function families for group sequential boundaries.

    Attributes
    ----------
    OBF : str
        O'Brien-Fleming: Conservative early, liberal late.
    POCOCK : str
        Pocock: More uniform spending across looks.
    HSD : str
        Hwang-Shih-DeCani: Flexible family with gamma parameter.

    Examples
    --------
    >>> SpendingFamily.OBF.value
    'obf'
    >>> SpendingFamily.POCOCK.value
    'pocock'
    """

    OBF = "obf"
    POCOCK = "pocock"
    HSD = "hsd"


class BoundaryScale(str, Enum):
    """
    Scale for representing boundaries and statistics.

    Attributes
    ----------
    Z : str
        Standardized Z-statistic scale (most common).
    BROWNIAN : str
        Brownian motion scale B(t) = Z·√t.

    Examples
    --------
    >>> BoundaryScale.Z.value
    'z'
    >>> BoundaryScale.BROWNIAN.value
    'bm'
    """

    Z = "z"
    BROWNIAN = "bm"


class FutilityMode(str, Enum):
    """
    Mode for futility boundary specification.

    Attributes
    ----------
    NONE : str
        No futility boundary (lower bound = -∞).
    SYMMETRIC : str
        Symmetric futility boundary (lower = -upper). Only valid for two-sided tests.
    FIXED_Z : str
        Fixed Z-value futility boundary (user-specified).
    BETA_SPENDING : str
        Beta spending function determines futility boundary.

    Examples
    --------
    >>> FutilityMode.NONE.value
    'none'
    >>> FutilityMode.SYMMETRIC.value
    'symmetric'
    """

    NONE = "none"
    SYMMETRIC = "symmetric"
    FIXED_Z = "fixed_z"
    BETA_SPENDING = "beta_spending"


# =============================================================================
# TypedDict Schemas (for payload validation)
# =============================================================================


class EfficacySpec(TypedDict, total=False):
    """
    Specification for efficacy (upper) boundary.

    Keys
    ----
    style : {"alpha_spending", "significance_level"}
        How to determine the boundary.
    family : str, optional
        Spending function family (if style="alpha_spending").
    gamma : float, optional
        Shape parameter for HSD family.
    alpha_levels : List[float] or Dict[int, float], optional
        Per-look significance levels (if style="significance_level").
    """

    style: str
    family: Optional[str]
    gamma: Optional[float]
    alpha_levels: Optional[Union[List[float], Dict[int, float]]]


class FutilitySpec(TypedDict, total=False):
    """
    Specification for futility (lower) boundary.

    Keys
    ----
    mode : {"none", "symmetric", "fixed_z", "beta_spending"}
        How to determine the futility boundary.
    binding : bool, optional
        Whether futility is binding (default: False).
    z : float or Dict[int, float], optional
        Fixed Z value(s) for futility (if mode="fixed_z").
    family : str, optional
        Beta spending function family (if mode="beta_spending").
    gamma : float, optional
        Shape parameter for HSD family.
    """

    mode: str
    binding: Optional[bool]
    z: Optional[Union[float, Dict[int, float]]]
    family: Optional[str]
    gamma: Optional[float]


class DesignPayload(TypedDict, total=False):
    """
    Complete design payload schema.

    This matches the schema used in the existing codebase for backward
    compatibility while supporting new features.

    Keys
    ----
    alpha : float
        Overall Type I error rate.
    tails : int
        Number of tails (1 or 2).
    scale : str
        Boundary scale ("z" or "bm").
    efficacy : EfficacySpec
        Efficacy boundary specification.
    futility : FutilitySpec
        Futility boundary specification.
    binding_mode : str, optional
        Binding mode (binding or non_binding). Default: non_binding.
    planned_max_n : int, optional
        Planned maximum sample size.

    Examples
    --------
    >>> design: DesignPayload = {
    ...     "alpha": 0.05,
    ...     "tails": 2,
    ...     "scale": "z",
    ...     "efficacy": {"style": "alpha_spending", "family": "obf"},
    ...     "futility": {"mode": "none"},
    ... }
    """

    alpha: float
    tails: int
    scale: str
    efficacy: EfficacySpec
    futility: FutilitySpec
    binding_mode: Optional[str]
    planned_max_n: Optional[int]


# =============================================================================
# Validation
# =============================================================================


def validate_design_payload(payload: Dict[str, Any]) -> None:
    """
    Validate a design payload dictionary.

    Checks that required fields are present and have valid values.
    Raises ValueError with descriptive message if validation fails.

    Parameters
    ----------
    payload : dict
        Design payload to validate.

    Raises
    ------
    ValueError
        If payload is invalid.

    Examples
    --------
    >>> payload = {
    ...     "alpha": 0.05,
    ...     "tails": 2,
    ...     "scale": "z",
    ...     "efficacy": {"style": "alpha_spending", "family": "obf"},
    ...     "futility": {"mode": "none"},
    ... }
    >>> validate_design_payload(payload)  # No error

    >>> bad_payload = {"alpha": 0.05}  # Missing required fields
    >>> validate_design_payload(bad_payload)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Missing required field: tails
    """
    # Required fields
    required_fields = ["alpha", "tails", "scale", "efficacy", "futility"]
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")

    # Validate alpha
    alpha = payload["alpha"]
    if not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Validate tails
    tails = payload["tails"]
    if tails not in (1, 2):
        raise ValueError(f"tails must be 1 or 2, got {tails}")

    # Validate scale
    scale = payload["scale"]
    if scale not in ("z", "bm"):
        raise ValueError(f"scale must be 'z' or 'bm', got {scale}")

    # Validate efficacy
    efficacy = payload["efficacy"]
    if not isinstance(efficacy, dict):
        raise ValueError(f"efficacy must be a dict, got {type(efficacy)}")
    if "style" not in efficacy:
        raise ValueError("efficacy must have 'style' field")
    if efficacy["style"] not in ("alpha_spending", "significance_level"):
        raise ValueError(
            f"efficacy.style must be 'alpha_spending' or 'significance_level', "
            f"got {efficacy['style']}"
        )

    # Validate futility
    futility = payload["futility"]
    if not isinstance(futility, dict):
        raise ValueError(f"futility must be a dict, got {type(futility)}")
    if "mode" not in futility:
        raise ValueError("futility must have 'mode' field")
    if futility["mode"] not in ("none", "symmetric", "fixed_z", "beta_spending"):
        raise ValueError(
            f"futility.mode must be one of: none, symmetric, fixed_z, beta_spending, "
            f"got {futility['mode']}"
        )

    # Validate binding_mode if present
    if "binding_mode" in payload:
        binding_mode = payload["binding_mode"]
        if binding_mode not in ("binding", "non_binding"):
            raise ValueError(
                f"binding_mode must be 'binding' or 'non_binding', got {binding_mode}"
            )


# =============================================================================
# Default payload constructors
# =============================================================================


def default_obf_design(
    alpha: float = 0.05, *, tails: int = 2, binding: bool = False
) -> DesignPayload:
    """
    Create a default O'Brien-Fleming design payload.

    Parameters
    ----------
    alpha : float, default=0.05
        Overall Type I error rate.
    tails : int, default=2
        Number of tails.
    binding : bool, default=False
        Whether futility is binding.

    Returns
    -------
    DesignPayload
        Default OBF design specification.

    Examples
    --------
    >>> design = default_obf_design(alpha=0.025)
    >>> design["alpha"]
    0.025
    >>> design["efficacy"]["family"]
    'obf'
    """
    return DesignPayload(
        alpha=alpha,
        tails=tails,
        scale="z",
        efficacy=EfficacySpec(style="alpha_spending", family="obf"),
        futility=FutilitySpec(mode="none", binding=binding),
        binding_mode="binding" if binding else "non_binding",
    )


def default_pocock_design(
    alpha: float = 0.05, *, tails: int = 2, binding: bool = False
) -> DesignPayload:
    """
    Create a default Pocock design payload.

    Parameters
    ----------
    alpha : float, default=0.05
        Overall Type I error rate.
    tails : int, default=2
        Number of tails.
    binding : bool, default=False
        Whether futility is binding.

    Returns
    -------
    DesignPayload
        Default Pocock design specification.

    Examples
    --------
    >>> design = default_pocock_design()
    >>> design["efficacy"]["family"]
    'pocock'
    """
    return DesignPayload(
        alpha=alpha,
        tails=tails,
        scale="z",
        efficacy=EfficacySpec(style="alpha_spending", family="pocock"),
        futility=FutilitySpec(mode="none", binding=binding),
        binding_mode="binding" if binding else "non_binding",
    )
