"""
Functional-style declaration of boundary roles (maps to gst.md).

This module provides the small, focused machinery that maps a design
specification to per-look efficacy and futility boundaries. The top-level
concepts below are expressed in a functional-programming-like signature
style used in docs/source/methods/gst.md so it's easy to trace how this
file fits the larger pipeline.

Core notation used here
 - spec := BoundaryCalculatorSpec (immutable design descriptor)
 - t := information time in [0, 1]
 - look := integer look index (1-based)
 - process := StochasticProcess primitive (e.g. Brownian motion)

Core functions / roles (signatures)
 - BoundaryDesign: (alpha, tails, scale, efficacySpec, futilitySpec) -> spec
         * Construct the immutable `BoundaryCalculatorSpec` describing the
             design. This is the translation point from higher-level `Design`
             payloads to the local spec object.

 - InstantiateProcess: (process_name | process_instance) -> process
         * Return a `StochasticProcess` primitive that knows how to convert
             between statistic scale (Z) and process scale (e.g. BM).

 - ComputeBoundary: (spec, t, look?) -> (upper: float, lower: float, scale: str)
         * Core single-look computation. Mirrors `Design -> boundary` in
             gst.md and is implemented by `BoundaryCalculator.compute_boundary`.

 - ComputeBoundaries: (spec, info_times: np.ndarray) ->
         { info_times, upper: np.ndarray, lower: np.ndarray, scale }
         * Vectorized convenience to compute boundaries for all planned looks.

 - EfficacyUpper: (spec, t, look?) -> upper_z
         * Compute the nominal upper Z-level for a given information time.
             Implemented as `_resolve_efficacy_upper_z` and uses spending
             functions (OBF, Pocock, HSD) or explicit per-look alpha levels.

 - FutilityLower: (spec, upper_z, t, look?) -> lower_z
         * Compute futility (lower) bound on Z-scale. Implemented as
             `_resolve_futility_lower_z` and supports symmetric, fixed-z, and
             beta-spending modes.

Composability intent
 - The file intentionally separates the small, serializable spec objects
     (`BoundaryCalculatorSpec`, `EfficacySpec`, `FutilitySpec`) from the
     computational class `BoundaryCalculator`. That keeps ledger/storage
     interactions simple and makes the calculator a pure computational
     operator that can be constructed, reused, or replaced.

Protocol vs concrete implementation (brief)
 - Preferred pattern: define a lightweight `BoundaryCalculatorProtocol`
     with the `ResolveBoundary` and `ComputeBoundaries` signatures and keep
     `BoundaryCalculator` as the canonical implementation. This enables
     future alternative implementations (e.g. Monte-Carlo approximators
     or highly vectorized engines) without breaking callers.

Usage mapping to gst.md pipeline
 - `Design` -> produce `spec` (BoundaryDesign)
 - `Design` -> `ComputeBoundary(spec, t)` (per-look boundary resolution)
 - `Design` (batch) -> `ComputeBoundaries(spec, info_times)` (batch)

All content here follows the repository convention: code and comments
are in English and the module is designed to be small and explicit.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from earlysign.stats.common.group_sequential.essentials import conversions
from earlysign.stats.common.group_sequential.essentials.design_schema import (
    validate_design_payload,
)
from earlysign.stats.essentials.methods.group_sequential import spending as spending_mod
from earlysign.stats.essentials.primitives.stochastic_processes import (
    BrownianMotionProcess,
    StochasticProcess,
)

# Essentials-level dataclasses for boundary configuration. Defined here per
# request so the canonical BoundaryCalculator exposes its configuration type
# directly from the module.


@dataclass(frozen=True)
class EfficacySpec:
    """Specification of the efficacy (upper) boundary policy."""

    style: str  # e.g. 'alpha_spending' | 'significance_level'
    family: Optional[str] = None  # e.g. 'obf' | 'pocock' | 'hsd'
    gamma: Optional[float] = None
    # per-look alpha levels for 'significance_level' style
    alpha_levels: Optional[Union[Mapping[int, float], List[float]]] = None


@dataclass(frozen=True)
class FutilitySpec:
    """Specification of the futility (lower) boundary policy."""

    mode: str  # 'none' | 'symmetric' | 'fixed_z' | 'beta_spending'
    # scalar z or per-look mapping
    z: Optional[Union[float, Mapping[int, float]]] = None
    family: Optional[str] = None
    gamma: Optional[float] = None
    beta: Optional[float] = None  # target beta (1-power) when using beta_spending


@dataclass(frozen=True)
class BoundaryCalculatorSpec:
    """Configuration for BoundaryCalculator initialization.

    This object contains the information that is stable for a given design
    and will typically be created once and passed to the calculator.
    """

    alpha: float
    tails: int
    scale: str = "z"
    efficacy: EfficacySpec = EfficacySpec(style="alpha_spending")
    futility: FutilitySpec = FutilitySpec(mode="none")
    # optional: name of preferred stochastic process primitive (eg 'bm')
    process: Optional[str] = None


class BoundaryCalculator:
    """Compute boundaries from design payloads and information times.

    Parameters
    ----------
    process:
        Stochastic process abstraction. Defaults to Brownian motion which
        matches the existing codebase behavior.
    """

    def __init__(
        self,
        spec: Union[Mapping[str, Any], "BoundaryCalculatorSpec"],
        process: Optional[StochasticProcess] = None,
    ) -> None:
        # Normalize and validate spec at construction time so the
        # calculator instance is bound to a specific spec.
        if isinstance(spec, BoundaryCalculatorSpec):
            spec_dict = asdict(spec)
        else:
            spec_dict = dict(spec)
        validate_design_payload(spec_dict)

        # store normalized spec on the instance
        self.spec: Mapping[str, Any] = spec_dict
        # The process can still be overridden; if not provided default to BM.
        self.process = process if process is not None else BrownianMotionProcess()

    # ---- Core single-time computation ---------------------------------
    def compute_boundary(
        self, info_time: float, look: Optional[int] = None
    ) -> Tuple[float, float, str]:
        """Compute (upper, lower, scale) for a single information time.

        The function mirrors the previous `resolve_boundary_from_design`
        semantics but is organized around the `BoundaryCalculator`'s
        process abstraction. This method is the implementation of the
        functional `ComputeBoundary` role described in the module docstring.
        """

        t = float(info_time)
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"info_time must be in [0, 1], got {t}")
        if not self.spec:
            raise ValueError(
                "BoundaryCalculator requires a spec payload at initialization"
            )

        # Use the instance-bound spec
        spec = self.spec
        float(spec["alpha"])
        int(spec["tails"])
        scale = str(spec["scale"]).lower()

        # Efficacy (upper) boundary on Z scale (helpers use self.spec)
        upper_z = self._resolve_efficacy_upper_z(t, look)

        # Futility (lower) boundary on Z scale
        lower_z = self._resolve_futility_lower_z(upper_z, t, look)

        # Convert to requested scale using the process
        if scale == "bm":
            upper = self.process.stat_to_process(upper_z, t)
            lower = (
                self.process.stat_to_process(lower_z, t)
                if np.isfinite(lower_z)
                else lower_z
            )
        else:
            upper = upper_z
            lower = lower_z

        return float(upper), float(lower), scale

    def compute_boundaries(
        self,
        info_times: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute boundaries at multiple information times.

        Returns a dict with keys: info_times, upper, lower, scale.
        """
        n = len(info_times)
        upper = np.zeros(n, dtype=float)
        lower = np.zeros(n, dtype=float)
        scale: Optional[str] = None

        for i, t in enumerate(info_times):
            up, lo, sc = self.compute_boundary(info_time=float(t), look=i + 1)
            upper[i] = up
            lower[i] = lo
            if scale is None:
                scale = sc

        return {
            "info_times": np.asarray(info_times, dtype=float),
            "upper": upper,
            "lower": lower,
            "scale": scale,
        }

    # ---- Helpers ------------------------------------------------------
    def _resolve_efficacy_upper_z(self, t: float, look: Optional[int]) -> float:
        """Resolve efficacy (upper) Z-boundary using the instance spec."""
        spec = self.spec
        efficacy = spec["efficacy"]
        style = efficacy["style"]

        if style == "alpha_spending":
            family = efficacy.get("family", "obf")
            gamma = efficacy.get("gamma", -4.0)

            # Instantiate spending function (existing implementations)
            key = str(family).lower()
            s: spending_mod.SpendingFunction  # Type hinting
            if key in ("obf", "obrien_fleming", "o'brien-fleming"):
                # For two-sided tests, internal spending expects half-alpha
                s = spending_mod.OBFSpending(
                    alpha=float(spec["alpha"]), sided=int(spec["tails"])
                )
            elif key == "pocock":
                s = spending_mod.PocockSpending(alpha=float(spec["alpha"]))
            elif key == "hsd":
                s = spending_mod.HSDSpending(
                    alpha=float(spec["alpha"]), gamma=float(gamma)
                )
            else:
                raise ValueError(f"Unknown spending family: {family}")

            alpha_spent = float(s.cumulative(np.array([t]))[0])
            upper_z, _ = conversions.cumulative_to_nominal_z(
                alpha_spent, tails=int(spec["tails"])
            )
            return upper_z

        elif style == "significance_level":
            if look is None:
                raise ValueError("look number required for significance_level style")
            alpha_levels = efficacy.get("alpha_levels")
            if alpha_levels is None:
                raise ValueError("alpha_levels required for significance_level style")
            level = self._get_alpha_level_for_look(alpha_levels, look)
            upper_z, _ = conversions.level_to_nominal_z(level, tails=int(spec["tails"]))
            return upper_z

        else:
            raise ValueError(f"Unknown efficacy style: {style}")

    def _resolve_futility_lower_z(
        self, upper_z: float, info_time: float, look: Optional[int]
    ) -> float:
        """Resolve futility (lower) Z-boundary using the instance spec."""
        spec = self.spec
        futility = spec["futility"]
        mode = futility["mode"]

        if mode == "none":
            return float("-inf")

        if mode == "symmetric":
            if int(spec["tails"]) != 2:
                raise ValueError("Symmetric futility only valid for two-sided tests")
            return -upper_z

        if mode == "fixed_z":
            z_val = futility.get("z")
            if z_val is None:
                raise ValueError("z value required for fixed_z futility mode")
            if isinstance(z_val, dict):
                if look is None:
                    raise ValueError("look number required for per-look fixed_z")
                if look not in z_val:
                    raise KeyError(f"Look {look} not in futility z values")
                return float(z_val[look])
            else:
                return float(z_val)

        if mode == "beta_spending":
            # Beta spending: convert beta(t) cumulative to lower Z (negative).
            family = futility.get("family", "obf")
            gamma = futility.get("gamma", -4.0)
            beta = futility.get("beta", 0.10)

            key = str(family).lower()
            s_beta: spending_mod.SpendingFunction  # Type hinting
            if key in ("obf", "obrien_fleming", "o'brien-fleming"):
                s_beta = spending_mod.OBFSpending(alpha=float(beta), sided=1)
            elif key == "pocock":
                s_beta = spending_mod.PocockSpending(alpha=float(beta))
            elif key == "hsd":
                s_beta = spending_mod.HSDSpending(alpha=float(beta), gamma=float(gamma))
            else:
                raise ValueError(f"Unknown futility spending family: {family}")

            beta_spent = float(s_beta.cumulative(np.array([info_time]))[0])
            z_one_sided, _ = conversions.cumulative_to_nominal_z(beta_spent, tails=1)
            return -z_one_sided

        raise ValueError(f"Unknown futility mode: {mode}")

    @staticmethod
    def _get_alpha_level_for_look(alpha_levels: Any, look: int) -> float:
        if isinstance(alpha_levels, dict):
            if look not in alpha_levels:
                raise KeyError(f"Look {look} not found in alpha_levels")
            return float(alpha_levels[look])
        idx = look - 1
        if idx < 0 or idx >= len(alpha_levels):
            raise IndexError(f"Look {look} out of range for alpha_levels")
        return float(alpha_levels[idx])
