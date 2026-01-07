from typing import Optional

from typing_extensions import Self

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.boundary import (
    BoundaryCalculator,
    BoundaryCalculatorSpec,
    EfficacySpec,
    FutilitySpec,
)


class GSTStoppingRuleEngine:
    """
    Engine for evaluating a single StoppingResult (Efficacy or Futility).
    Encapsulates logic for Schedule (when to look) and Boundary (critical values).
    """

    @classmethod
    def from_spec(
        cls,
        rule: GST.StoppingRule,
        rule_type: str,
        total_budget: float = 0.05,
        side: int = 1,
    ) -> Self:
        """
        Factory method to create an engine from a GST.StoppingRule spec.
        """
        return cls(rule, rule_type, total_budget, side)

    def __init__(
        self,
        rule: GST.StoppingRule,
        rule_type: str,  # "efficacy" or "futility"
        total_budget: float = 0.05,  # alpha or beta
        side: int = 1,
    ) -> None:
        self.rule = rule
        self.rule_type = rule_type
        self.total_budget = total_budget
        self.side = side

        self._calculator: Optional[BoundaryCalculator] = None
        self._setup_calculator()

    def _setup_calculator(self) -> None:
        """Maps ES3 StoppingRule to internal BoundaryCalculatorSpec."""
        boundary = self.rule.boundary

        # If Fixed, we don't need a calculator, we just return the value.
        if isinstance(boundary, GST.FixedBoundary):
            self._calculator = None
            return

        if isinstance(boundary, GST.SpendingBoundary):
            # We map to BoundaryCalculator
            # BoundaryCalculator requires BOTH Efficacy and Futility specs usually for initialization
            # to be valid, but we can set the "Other" to "none".

            if self.rule_type == "efficacy":
                # J&T Power Family: rho. HSD: gamma.
                input_params = boundary.spending_function.params or {}
                gamma = input_params.get("gamma") or input_params.get("rho")

                eff_spec = EfficacySpec(
                    style="alpha_spending",
                    family=boundary.spending_function.type,  # "obrien_fleming" etc
                    gamma=float(gamma) if gamma is not None else None,
                )
                fut_spec = FutilitySpec(mode="none")

                # Spec for BC
                spec = BoundaryCalculatorSpec(
                    alpha=self.total_budget,
                    tails=self.side,
                    scale="z",
                    efficacy=eff_spec,
                    futility=fut_spec,
                )
                self._calculator = BoundaryCalculator(spec)

            elif self.rule_type == "futility":
                # Use Beta Spending for Futility side
                # We use a dummy efficacy spec as BoundaryCalculator requires one.
                eff_spec = EfficacySpec(
                    style="alpha_spending", family="obrien_fleming", alpha_levels=[]
                )

                # Futility using Beta Spending
                input_params = boundary.spending_function.params or {}
                gamma = input_params.get("gamma") or input_params.get("rho")

                # Treat engines as independent (non-binding logic)
                fut_spec = FutilitySpec(
                    mode="beta_spending",
                    family=boundary.spending_function.type,
                    gamma=float(gamma) if gamma is not None else None,
                    beta=self.total_budget,  # Target Beta
                )

                spec = BoundaryCalculatorSpec(
                    alpha=0.025,  # Dummy alpha
                    tails=self.side,
                    scale="z",
                    efficacy=eff_spec,
                    futility=fut_spec,
                )
                self._calculator = BoundaryCalculator(spec)

    def get_boundary_at_look(
        self, look_index: int, info_frac: float
    ) -> Optional[float]:
        """
        Returns the critical value (Z-scale) for the given look.
        Returns None if no boundary (e.g. invalid look).
        """
        boundary_spec = self.rule.boundary

        # 1. Fixed Boundary
        if isinstance(boundary_spec, GST.FixedBoundary):
            return boundary_spec.value

        # 2. Spending Boundary
        if isinstance(boundary_spec, GST.SpendingBoundary):
            if self._calculator:
                # BC.compute_boundary returns (upper, lower, scale)
                upper, lower, _ = self._calculator.compute_boundary(
                    info_time=info_frac, look=look_index
                )

                if self.rule_type == "efficacy":
                    return upper
                elif self.rule_type == "futility":
                    return lower

        return None

    def get_max_sample_size(self) -> int:
        """Returns implied max sample size if defined in Schedule."""
        if self.rule.schedule.unit == "sample_size":
            pts = self.rule.schedule.interim_points
            if pts:
                return int(max(pts))
        return 0

    def get_schedule_sample_size(self, look_index: int) -> Optional[int]:
        """Returns the scheduled sample size for the look index (if available)."""
        if self.rule.schedule.unit == "sample_size":
            pts = self.rule.schedule.interim_points
            if pts and look_index >= 0 and look_index < len(pts):
                return int(pts[look_index])
        return None
