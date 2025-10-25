"""Design-layer adapter: convert DesignSpec -> BoundaryCalculatorSpec

This adapter maps the app-level DesignSpec to the essentials-level
BoundaryCalculatorSpec and calls the canonical boundary calculator.
"""

from typing import Any, Optional, Type, cast

import numpy as np

import earlysign.stats.essentials.methods.group_sequential.boundary as boundary_mod
from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.common.types import SpendingFunction
from earlysign.stats.essentials.methods.group_sequential import spending as spending_mod
from earlysign.stats.essentials.methods.group_sequential.boundary import (
    BoundaryCalculatorSpec,
    EfficacySpec,
    FutilitySpec,
)


class BoundaryCalculator:
    """Adapter that builds a BoundaryCalculatorSpec from a DesignSpec.

    This thin adapter converts the application-level `DesignSpec` into the
    essentials-level `BoundaryCalculatorSpec`, calls the canonical
    `compute_boundaries(...)`, and returns the legacy-shaped dictionary of
    critical values expected by older callers.
    """

    def __init__(self, spec: DesignSpec) -> None:
        self.spec = spec

    def _critical_values(self) -> Any:
        # Resolve information times
        t = np.asarray(self.spec.resolved_info_times(), dtype=float)

        # sided -> tails mapping ('one'|'two') -> 1|2
        tails = 1 if getattr(self.spec.test, "sided", "one") == "one" else 2

        # Choose spending family
        sf = getattr(self.spec.boundary, "spending_function", None)
        s: spending_mod.SpendingFunction
        if sf == SpendingFunction.POCOCK:
            s = spending_mod.PocockSpending(alpha=float(self.spec.test.alpha))
        elif sf == SpendingFunction.HSD:
            s = spending_mod.HSDSpending(
                alpha=float(self.spec.test.alpha),
                gamma=float(self.spec.boundary.hsd_gamma),
            )
        else:
            # default: O'Brien-Fleming style
            s = spending_mod.OBFSpending(alpha=float(self.spec.test.alpha), sided=tails)

        cumulative_alpha = np.asarray(s.cumulative(t), dtype=float)

        hsd_gamma = (
            self.spec.boundary.hsd_gamma
            if getattr(self.spec.boundary, "hsd_gamma", None) is not None
            else -4.0
        )

        eff = EfficacySpec(
            style="alpha_spending",
            family=str(
                getattr(self.spec.boundary.spending_function, "value", "obrien_fleming")
            ),
            gamma=float(hsd_gamma),
        )

        fut = FutilitySpec(mode="none")
        if getattr(self.spec.boundary, "futility_enabled", False):
            z_val = getattr(self.spec.boundary, "futility_z", None)
            if z_val is not None:
                fut = FutilitySpec(mode="fixed_z", z=float(z_val))
            elif getattr(self.spec.boundary, "futility_spending", None) is not None:
                fut = FutilitySpec(
                    mode="beta_spending",
                    family=str(
                        getattr(
                            self.spec.boundary.futility_spending,
                            "value",
                            "obrien_fleming",
                        )
                    ),
                    gamma=float(hsd_gamma),
                    beta=float(1 - float(self.spec.test.power)),
                )
            else:
                fut = FutilitySpec(mode="symmetric")

        spec_cfg = BoundaryCalculatorSpec(
            alpha=float(self.spec.test.alpha),
            tails=tails,
            scale="z",
            efficacy=eff,
            futility=fut,
            process=None,
        )

        calc = boundary_mod.BoundaryCalculator(spec=spec_cfg, process=None)
        boundaries = calc.compute_boundaries(info_times=t)

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

    # Flexible compatibility entry-point
    def critical_values(self, spec: Optional[DesignSpec] = None) -> Any:
        """Compatibility entry point supporting both instance and class-style calls.

        - Instance usage: adapter = BoundaryCalculator(spec); adapter.critical_values()
        - Class-style usage: BoundaryCalculator.critical_values(spec)

        When called on the class, ``self`` will be the class object (a type), and
        ``spec`` must be provided. When called on an instance, ``self`` is the
        instance and ``spec`` is ignored.
        """

        # If called like `BoundaryCalculator.critical_values(spec)` then the
        # function receives the `spec` as the first (self) argument. Rely on
        # duck-typing as well as the explicit DesignSpec type check to be
        # robust against situations where the DesignSpec class identity may
        # differ (e.g., re-exports or shadowed imports in tests/notebooks).
        def _looks_like_design_spec(obj: Any) -> bool:
            if isinstance(obj, DesignSpec):
                return True
            # Duck-typing: design spec should expose resolved_info_times and test/boundary
            return (
                hasattr(obj, "resolved_info_times")
                and hasattr(obj, "test")
                and hasattr(obj, "boundary")
            )

        if _looks_like_design_spec(self):
            return BoundaryCalculator(cast(DesignSpec, self))._critical_values()

        # Called as class-method: `self` is the class object
        if isinstance(self, type):
            if spec is None:
                raise TypeError(
                    "critical_values called on class requires a spec argument"
                )
            cls = cast(Type["BoundaryCalculator"], self)
            inst = cls(spec)
            return inst._critical_values()

        # Called as instance method: return adapter computed values
        return self._critical_values()
