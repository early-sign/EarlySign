"""
Boundary calculator for group-sequential designs using primitive processes.

This module provides a clean, well-documented `BoundaryCalculator` class
that centralizes the business logic for computing efficacy and futility
boundaries given a design payload (validated with `design_schema`). It is
designed to be process-agnostic by depending on the primitives in
`earlysign.stats.essentials.primitives.stochastic_processes`.

The API emphasizes clarity: callers provide a design payload and an
information time (or an array thereof) and receive explicit upper/lower
boundaries together with the scale.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

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


class BoundaryCalculator:
    """Compute boundaries from design payloads and information times.

    Parameters
    ----------
    process:
        Stochastic process abstraction. Defaults to Brownian motion which
        matches the existing codebase behavior.
    """

    def __init__(self, process: Optional[StochasticProcess] = None) -> None:
        self.process = process if process is not None else BrownianMotionProcess()

    # ---- Core single-time computation ---------------------------------
    def resolve_boundary(
        self,
        *,
        design_payload: Mapping[str, Any],
        info_time: float,
        look: Optional[int] = None,
    ) -> Tuple[float, float, str]:
        """Resolve (upper, lower, scale) for a single information time.

        The function mirrors the previous `resolve_boundary_from_design`
        semantics but is organized around the `BoundaryCalculator`'s
        process abstraction.
        """

        t = float(info_time)
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"info_time must be in [0, 1], got {t}")

        # Validate and normalise design
        design = dict(design_payload)
        validate_design_payload(design)

        float(design["alpha"])
        int(design["tails"])
        scale = str(design["scale"]).lower()

        # Efficacy (upper) boundary on Z scale
        upper_z = self._resolve_efficacy_upper_z(design, t, look)

        # Futility (lower) boundary on Z scale
        lower_z = self._resolve_futility_lower_z(design, upper_z, t, look)

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
        self, design_payload: Mapping[str, Any], info_times: np.ndarray
    ) -> Dict[str, Any]:
        """Compute boundaries at multiple information times.

        Returns a dict with keys: info_times, upper, lower, scale.
        """
        n = len(info_times)
        upper = np.zeros(n, dtype=float)
        lower = np.zeros(n, dtype=float)
        scale: Optional[str] = None

        for i, t in enumerate(info_times):
            up, lo, sc = self.resolve_boundary(
                design_payload=design_payload, info_time=float(t), look=i + 1
            )
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
    def _resolve_efficacy_upper_z(
        self, design: Mapping[str, Any], t: float, look: Optional[int]
    ) -> float:
        efficacy = design["efficacy"]
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
                    alpha=float(design["alpha"]), sided=int(design["tails"])
                )
            elif key == "pocock":
                s = spending_mod.PocockSpending(alpha=float(design["alpha"]))
            elif key == "hsd":
                s = spending_mod.HSDSpending(
                    alpha=float(design["alpha"]), gamma=float(gamma)
                )
            else:
                raise ValueError(f"Unknown spending family: {family}")

            alpha_spent = float(s.cumulative(np.array([t]))[0])
            upper_z, _ = conversions.cumulative_to_nominal_z(
                alpha_spent, tails=int(design["tails"])
            )
            return upper_z

        elif style == "significance_level":
            if look is None:
                raise ValueError("look number required for significance_level style")
            alpha_levels = efficacy.get("alpha_levels")
            if alpha_levels is None:
                raise ValueError("alpha_levels required for significance_level style")
            level = self._get_alpha_level_for_look(alpha_levels, look)
            upper_z, _ = conversions.level_to_nominal_z(
                level, tails=int(design["tails"])
            )
            return upper_z

        else:
            raise ValueError(f"Unknown efficacy style: {style}")

    def _resolve_futility_lower_z(
        self,
        design: Mapping[str, Any],
        upper_z: float,
        info_time: float,
        look: Optional[int],
    ) -> float:
        futility = design["futility"]
        mode = futility["mode"]

        if mode == "none":
            return float("-inf")

        if mode == "symmetric":
            if int(design["tails"]) != 2:
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


# Convenience module-level functions using a default calculator
_default_calc = BoundaryCalculator()


def resolve_boundary(
    *, design_payload: Mapping[str, Any], info_time: float, look: Optional[int] = None
) -> Tuple[float, float, str]:
    return _default_calc.resolve_boundary(
        design_payload=design_payload, info_time=info_time, look=look
    )


def compute_boundaries(
    design_payload: Mapping[str, Any], info_times: np.ndarray
) -> Dict[str, Any]:
    return _default_calc.compute_boundaries(
        design_payload=design_payload, info_times=info_times
    )
