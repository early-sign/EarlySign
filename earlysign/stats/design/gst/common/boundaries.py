"""Boundary value calculation for group sequential designs.

This design-level adapter returns the legacy ``critical_values(spec)`` shape
while delegating the canonical per-look boundary computation to
``earlysign.stats.essentials.methods.group_sequential.boundary``.

The goal is to preserve backward compatibility for ``DesignLab`` and the
simulation code while allowing the new central implementation to be the
single source of truth.
"""

from typing import Any, Dict

import numpy as np

import earlysign.stats.essentials.methods.group_sequential.spending as spending
from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.common.types import SpendingFunction


class BoundaryCalculator:
    """Calculate critical boundary values for group sequential trial designs.

    This adapter preserves the original ``critical_values(spec)`` API and
    return shape (``info_times``, ``cumulative_alpha``, ``z_efficacy``,
    ``z_futility``) while using the new canonical calculator under the hood.
    """

    @staticmethod
    def critical_values(spec: DesignSpec) -> Dict[str, Any]:
        """Calculate critical values for efficacy and futility bounds.

        Returns the legacy dictionary shape expected by ``DesignLab`` and the
        simulation code.
        """
        # Resolved information times
        t = spec.resolved_info_times()
        len(t)

        # Compute cumulative alpha using existing spending logic so callers
        # receive the same ``cumulative_alpha`` values as before.
        alpha_total = (
            spec.test.alpha if spec.test.sided == "one" else spec.test.alpha / 2.0
        )

        s: spending.SpendingFunction
        if spec.boundary.spending_function == SpendingFunction.OBRIEN_FLEMING:
            sided = 1 if spec.test.sided == "one" else 2
            s = spending.OBFSpending(alpha=alpha_total, sided=sided)
            cumulative_alpha = s.cumulative(t)
        elif spec.boundary.spending_function == SpendingFunction.POCOCK:
            s = spending.PocockSpending(alpha=alpha_total)
            cumulative_alpha = s.cumulative(t)
        elif spec.boundary.spending_function == SpendingFunction.HSD:
            s = spending.HSDSpending(alpha=alpha_total, gamma=spec.boundary.hsd_gamma)
            cumulative_alpha = s.cumulative(t)
        else:
            raise ValueError(
                f"Unknown spending function: {spec.boundary.spending_function}"
            )

        # Use the canonical calculator to get per-look upper/lower boundaries.
        from earlysign.stats.essentials.methods.group_sequential.boundary import (
            compute_boundaries as _compute_boundaries,
        )

        design_payload: dict[str, Any] = {
            "alpha": float(spec.test.alpha),
            "tails": 1 if spec.test.sided == "one" else 2,
            "scale": "z",
            "efficacy": {
                "style": "alpha_spending",
                "family": spec.boundary.spending_function.value,
                "gamma": float(spec.boundary.hsd_gamma),
            },
            "futility": {"mode": "none"},
        }

        if spec.boundary.futility_enabled:
            if spec.boundary.futility_z is not None:
                design_payload["futility"] = {
                    "mode": "fixed_z",
                    "z": float(spec.boundary.futility_z),
                }
            elif spec.boundary.futility_spending is not None:
                design_payload["futility"] = {
                    "mode": "beta_spending",
                    "family": spec.boundary.futility_spending.value,
                    "gamma": float(spec.boundary.hsd_gamma),
                    "beta": float(1 - spec.test.power),
                }
            else:
                design_payload["futility"] = {"mode": "symmetric"}

        boundaries = _compute_boundaries(design_payload=design_payload, info_times=t)

        # Map to legacy output names. The canonical calculator returns
        # 'upper' and 'lower' arrays; expose them as z_efficacy and z_futility.
        z_efficacy = np.asarray(boundaries["upper"], dtype=float)
        z_futility = (
            np.asarray(boundaries["lower"], dtype=float)
            if boundaries.get("lower") is not None
            else None
        )

        return {
            "info_times": t,
            "cumulative_alpha": cumulative_alpha,
            "z_efficacy": z_efficacy,
            "z_futility": z_futility,
        }
