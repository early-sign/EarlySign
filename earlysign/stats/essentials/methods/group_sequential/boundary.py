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
from math import sqrt
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm

from earlysign.stats.common.group_sequential.essentials.design_schema import (
    validate_design_payload,
)
from earlysign.stats.essentials.methods.group_sequential import spending as spending_mod

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
        Optional adapter providing custom scale conversions. Must expose a
        ``stat_to_process(value: float, t: float) -> float`` method if
        provided. When absent, canonical Brownian (Z → B(t)) conversions from
        ``asymptotic_processes`` are used.
    """

    def __init__(
        self,
        spec: Union[Mapping[str, Any], "BoundaryCalculatorSpec"],
        process: Optional[Any] = None,
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
        self.process = process

    # ---- Core single-time computation ---------------------------------
    def compute_boundary(
        self, info_time: float, look: Optional[int] = None
    ) -> Tuple[float, float, str]:
        """Compute (upper, lower, scale) for a single information time.

        This method is the primary entry point for callers that instantiate
        :class:`BoundaryCalculator`. It resolves efficacy and futility levels
        on the requested output scale and mirrors the conceptual
        ``Design → boundary`` step described in ``gst.md``.
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
            upper = self._stat_to_process_scale(upper_z, t)
            lower = (
                self._stat_to_process_scale(lower_z, t)
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
            upper_z, _ = nominal_z_from_spent_alpha(
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
            upper_z, _ = nominal_z_from_level(level, tails=int(spec["tails"]))
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
            z_one_sided, _ = nominal_z_from_spent_alpha(beta_spent, tails=1)
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

    def _stat_to_process_scale(self, value: float, info_time: float) -> float:
        if self.process is not None and hasattr(self.process, "stat_to_process"):
            return float(self.process.stat_to_process(value, info_time))
        return convert_statistic_scale(
            value,
            from_scale="z",
            to_scale="bm",
            info_time=info_time,
        )


def nominal_z_from_spent_alpha(
    spent_alpha: float, *, tails: int = 2
) -> Tuple[float, float]:
    """Return nominal Z boundaries corresponding to cumulative alpha spending."""

    if spent_alpha < 0.0:
        raise ValueError("spent_alpha must be non-negative.")
    if spent_alpha == 0.0:
        return float("inf"), float("-inf") if tails == 2 else float("-inf")
    if tails == 2:
        z = float(norm.isf(spent_alpha / 2.0))
        return z, -z
    if tails == 1:
        z = float(norm.isf(spent_alpha))
        return z, float("-inf")
    raise ValueError(f"tails must be 1 or 2, got {tails}")


def nominal_z_from_level(alpha_level: float, *, tails: int = 2) -> Tuple[float, float]:
    """Return nominal Z boundaries from a per-look significance level."""

    if not (0.0 < alpha_level < 1.0):
        raise ValueError("alpha_level must lie in (0, 1).")
    return nominal_z_from_spent_alpha(alpha_level, tails=tails)


def z_to_brownian(z: float, t: float) -> float:
    """Convert a Z-statistic to Brownian-motion scale."""

    if not (0.0 <= t <= 1.0):
        raise ValueError("t must be in [0, 1].")
    return float(z) * sqrt(max(t, 0.0))


def brownian_to_z(b: float, t: float) -> float:
    """Convert a Brownian-motion value to Z-scale."""

    if not (0.0 < t <= 1.0):
        raise ValueError("t must be in (0, 1].")
    return float(b) / sqrt(max(t, 1e-12))


def convert_statistic_scale(
    value: float, *, from_scale: str, to_scale: str, info_time: float
) -> float:
    """Convert between supported statistic scales."""

    fs = from_scale.lower()
    ts = to_scale.lower()
    if fs not in {"z", "bm"}:
        raise ValueError(f"Unsupported from_scale '{from_scale}'.")
    if ts not in {"z", "bm"}:
        raise ValueError(f"Unsupported to_scale '{to_scale}'.")
    if not (0.0 <= info_time <= 1.0):
        raise ValueError("info_time must be in [0, 1].")
    if fs == ts:
        return float(value)
    if fs == "z" and ts == "bm":
        return z_to_brownian(value, info_time)
    if fs == "bm" and ts == "z":
        return brownian_to_z(value, info_time)
    return float(value)
