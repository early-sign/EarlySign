"""
Group-sequential stopping policy (EarlySign v1).

This module defines the execution-layer StoppingPolicy and its factory,
which translate abstract ES3 policy specifications into concrete
spending strategies and boundary properties.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.shared.spending import (
    SpendingFunction,
    SpendingFunctionFactory,
)


@dataclass(frozen=True)
class StoppingPolicy:
    """Abstract base class for execution-layer Stopping Logic."""

    sided: str = "two"
    alpha_binding: bool = True
    beta_binding: bool = False


@dataclass(frozen=True)
class SpendingFunctionStoppingPolicy(StoppingPolicy):
    """Spending-function based stopping policy (Lan-DeMets)."""

    efficacy_spending: Optional[SpendingFunction] = None
    futility_spending: Optional[SpendingFunction] = None


@dataclass(frozen=True, kw_only=True)
class BoundaryFunctionStoppingPolicy(StoppingPolicy):
    """Shape-based boundary policy (O'Brien-Fleming, Pocock, Whitehead)."""

    function_name: str  # "obrien_fleming", "pocock", "whitehead"
    parameters: dict[str, Any] = field(default_factory=dict)


class StoppingPolicyFactory:
    """Factory for creating StoppingPolicy instances from ES3 specs."""

    @staticmethod
    def build_from_spec(spec: GST.StoppingPolicySpec) -> StoppingPolicy:
        """Translates a StoppingPolicySpec into a concrete StoppingPolicy."""
        policy = spec.root

        if isinstance(policy, GST.AlphaSpendingPolicy):
            factory = SpendingFunctionFactory(budget=policy.budget)
            return SpendingFunctionStoppingPolicy(
                efficacy_spending=factory.build_from_spec(policy.spending_fn),
                sided=str(policy.sided) if policy.sided else "two",
            )

        if isinstance(policy, GST.BetaSpendingPolicy):
            factory = SpendingFunctionFactory(budget=policy.budget)
            return SpendingFunctionStoppingPolicy(
                futility_spending=factory.build_from_spec(policy.spending_fn),
                sided="one",
            )

        if isinstance(policy, GST.AlphaBetaSpendingPolicy):
            eff_factory = SpendingFunctionFactory(budget=policy.alpha_budget)
            fut_factory = SpendingFunctionFactory(budget=policy.beta_budget)
            return SpendingFunctionStoppingPolicy(
                efficacy_spending=eff_factory.build_from_spec(policy.alpha_spending_fn),
                futility_spending=fut_factory.build_from_spec(policy.beta_spending_fn),
                alpha_binding=(
                    policy.alpha_binding if policy.alpha_binding is not None else True
                ),
                beta_binding=(
                    policy.beta_binding if policy.beta_binding is not None else False
                ),
                sided="one",
            )

        if isinstance(policy, GST.OBrienFlemingBoundaryPolicy):
            # OBF Shape
            return BoundaryFunctionStoppingPolicy(
                function_name="obrien_fleming",
                parameters={"alpha": policy.alpha},
                sided=str(policy.sided) if policy.sided else "two",
            )

        if isinstance(policy, GST.WhiteheadBoundaryPolicy):
            # Whitehead Shape
            return BoundaryFunctionStoppingPolicy(
                function_name="whitehead",
                parameters={
                    "alpha": policy.alpha,
                    "beta": policy.beta,
                },
                sided="one",
                # Triangular design defaults
                alpha_binding=True,
                beta_binding=False,
            )

        if isinstance(policy, GST.PocockBoundaryPolicy):
            # Pocock Shape
            return BoundaryFunctionStoppingPolicy(
                function_name="pocock",
                parameters={"alpha": policy.alpha},
                sided=str(policy.sided) if policy.sided else "two",
            )

        if isinstance(policy, GST.WangTsiatisBoundaryPolicy):
            # Wang-Tsiatis Shape
            return BoundaryFunctionStoppingPolicy(
                function_name="wang_tsiatis",
                parameters={
                    "delta": policy.delta,
                    "alpha": policy.alpha,
                },
                sided=str(policy.sided) if policy.sided else "two",
            )

        raise ValueError(f"Unknown stopping policy type: {type(policy)}")
