"""
Group-sequential stopping policy (EarlySign v1).

This module defines the execution-layer StoppingPolicy and its factory,
which translate abstract ES3 policy specifications into concrete
spending strategies and boundary properties.
"""

from dataclasses import dataclass
from typing import Optional

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.shared.spending import (
    OBFSpending,
    PocockSpending,
    SpendingFunction,
    SpendingFunctionFactory,
)


@dataclass(frozen=True)
class StoppingPolicy:
    """Unified execution-layer Stopping Policy.

    This class encapsulates all information needed by a statistical engine
    to calculate boundaries: spending functions for efficacy and futility,
    binding status, and sidedness.
    """

    efficacy_spending: Optional[SpendingFunction] = None
    futility_spending: Optional[SpendingFunction] = None
    alpha_binding: bool = True
    beta_binding: bool = False
    sided: str = "two"


class StoppingPolicyFactory:
    """Factory for creating StoppingPolicy instances from ES3 specs."""

    @staticmethod
    def build_from_spec(spec: GST.StoppingPolicySpec) -> StoppingPolicy:
        """Translates a StoppingPolicySpec into a concrete StoppingPolicy.

        Args:
            spec: The ES3 StoppingPolicySpec (discriminated union).

        Returns:
            A StoppingPolicy instance configured for execution.
        """
        policy = spec.root

        if isinstance(policy, GST.AlphaSpendingPolicy):
            factory = SpendingFunctionFactory(budget=policy.budget)
            return StoppingPolicy(
                efficacy_spending=factory.build_from_spec(policy.spending_fn),
                sided=str(policy.sided) if policy.sided else "two",
            )

        if isinstance(policy, GST.BetaSpendingPolicy):
            factory = SpendingFunctionFactory(budget=policy.budget)
            return StoppingPolicy(
                futility_spending=factory.build_from_spec(policy.spending_fn),
                sided="one",  # Futility is typically 1-sided
            )

        if isinstance(policy, GST.AlphaBetaSpendingPolicy):
            eff_factory = SpendingFunctionFactory(budget=policy.alpha_budget)
            fut_factory = SpendingFunctionFactory(budget=policy.beta_budget)
            return StoppingPolicy(
                efficacy_spending=eff_factory.build_from_spec(policy.alpha_spending_fn),
                futility_spending=fut_factory.build_from_spec(policy.beta_spending_fn),
                alpha_binding=(
                    policy.alpha_binding if policy.alpha_binding is not None else True
                ),
                beta_binding=(
                    policy.beta_binding if policy.beta_binding is not None else False
                ),
                sided="one",  # Typically 1-sided for A/B
            )

        if isinstance(policy, GST.OBrienFlemingPolicy):
            # Shortcut for Lan-DeMets OBF
            sided_val = 2 if policy.sided == "two" else 1
            return StoppingPolicy(
                efficacy_spending=OBFSpending(budget=policy.budget, sided=sided_val),
                sided=str(policy.sided) if policy.sided else "two",
            )

        if isinstance(policy, GST.WhiteheadPolicy):
            # Whitehead (Triangular) approx: OBF for efficacy, Pocock for futility
            return StoppingPolicy(
                efficacy_spending=OBFSpending(budget=policy.alpha_budget, sided=1),
                futility_spending=PocockSpending(budget=policy.beta_budget),
                alpha_binding=True,
                beta_binding=False,
                sided="one",
            )

        raise ValueError(f"Unknown stopping policy type: {type(policy)}")
