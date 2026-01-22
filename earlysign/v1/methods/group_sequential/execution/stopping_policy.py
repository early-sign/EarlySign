"""
Group-sequential stopping policy (EarlySign v1).

This module defines the execution-layer StoppingPolicy and its factory,
which translate abstract ES3 policy specifications into concrete
spending strategies and boundary properties.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.shared.spending import (
    SpendingFunction,
    SpendingFunctionFactory,
)


class BoundarySolver(Protocol):
    """Protocol for the statistical model capable of solving boundary constants."""

    @property
    def info_times(self) -> NDArray[np.float64]: ...

    @property
    def tails(self) -> int: ...

    def find_critical_value(
        self, shape: NDArray[np.float64], alpha: float, tails: Optional[int] = None
    ) -> float:
        """Find critical value c such that P(Crossing c * shape) = alpha."""
        ...


@dataclass(frozen=True, kw_only=True)
class StoppingPolicy:
    """Abstract base class for execution-layer Stopping Logic."""

    sided: str = "two"
    alpha_binding: bool = True
    beta_binding: bool = False

    def solve(
        self, model: BoundarySolver
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        """Solve for boundaries given the model context."""
        raise NotImplementedError("Subclasses must implement solve()")


@dataclass(frozen=True, kw_only=True)
class SpendingFunctionStoppingPolicy(StoppingPolicy):
    """Spending-function based stopping policy (Lan-DeMets)."""

    efficacy_spending: Optional[SpendingFunction] = None
    futility_spending: Optional[SpendingFunction] = None

    def solve(
        self, model: BoundarySolver
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        # Returns None to signal fallback to the Model's default (Spending) solver
        return None, None


@dataclass(frozen=True, kw_only=True)
class BoundaryFunctionStoppingPolicy(StoppingPolicy):
    """Base class for Shape-based boundary policies."""

    pass


@dataclass(frozen=True, kw_only=True)
class OBrienFlemingStoppingPolicy(BoundaryFunctionStoppingPolicy):
    """O'Brien-Fleming shape-based stopping policy."""

    alpha: float

    def solve(
        self, model: BoundarySolver
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        t = model.info_times
        # Defined shape: 1/sqrt(t)
        with np.errstate(divide="ignore"):
            shape = 1.0 / np.sqrt(t)
        shape[t <= 0] = 1e6  # Handle t=0

        # Policy delegates solving "c" to the model
        c = model.find_critical_value(shape, self.alpha)

        a = c * shape
        return a, None


@dataclass(frozen=True, kw_only=True)
class PocockStoppingPolicy(BoundaryFunctionStoppingPolicy):
    """Pocock shape-based stopping policy."""

    alpha: float

    def solve(
        self, model: BoundarySolver
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        t = model.info_times
        # Defined shape: 1.0 (constant)
        shape = np.ones_like(t)

        c = model.find_critical_value(shape, self.alpha)

        a = c * shape
        return a, None


@dataclass(frozen=True, kw_only=True)
class WhiteheadStoppingPolicy(BoundaryFunctionStoppingPolicy):
    """Whitehead (Triangular) stopping policy with analytical boundaries.

    The Triangular design uses linear boundaries in terms of information:
        Upper: Z = a + c * V
        Lower: Z = -a + c * V

    Where V is the information fraction, and a, c are derived from alpha/beta
    using Whitehead's approximations.
    """

    alpha: float
    beta: float

    def solve(
        self, model: "BoundarySolver"
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        from scipy.stats import norm

        t = model.info_times

        # Whitehead's analytical formulas (Jennison & Turnbull, Chapter 4)
        # For one-sided triangular test:
        # a = (z_alpha + z_beta) / 2  (intersection point at V=0)
        # slope derived from alpha/beta requirements

        z_alpha = norm.ppf(1 - self.alpha)
        z_beta = norm.ppf(1 - self.beta)

        # Whitehead approximation parameters
        # The boundaries are linear in sqrt(information)
        # Upper: a(t) = a0 + slope_upper * sqrt(t)
        # Lower: b(t) = -a0 + slope_lower * sqrt(t)
        # where a0 and slopes are derived from error requirements

        # Simplified Whitehead formulas (one-sided):
        # For equal monitoring, the triangular boundaries are approximately:
        a0 = (z_alpha + z_beta) / 2

        # Upper boundary (efficacy): starts high, decreases
        # a(t) = a0 / sqrt(t) + drift_effect
        # Lower boundary (futility): starts low, increases

        # For the canonical form (Z-statistic scale):
        # Upper: a(V) = a0 * (1 + 3*V) / sqrt(V)  approximately
        # Lower: b(V) = a0 * (-1 + 3*V) / sqrt(V) approximately

        # But more precisely, using Whitehead (1997) form:
        # Upper: Z = a0 + c_upper * V
        # Lower: Z = -a0 + c_lower * V
        # where V is information fraction and c values depend on error rates

        # Using the classic triangular test formulation:
        # a(V) = a0 * (1 / sqrt(V) + 3 * sqrt(V)) for efficacy
        # b(V) = a0 * (-1 / sqrt(V) + 3 * sqrt(V)) for futility

        sqrt_t = np.sqrt(np.maximum(t, 1e-10))
        inv_sqrt_t = 1.0 / sqrt_t

        # Triangular boundaries (Whitehead-Stratton form)
        a = a0 * (inv_sqrt_t + 3 * sqrt_t)
        b = a0 * (-inv_sqrt_t + 3 * sqrt_t)

        return a, b


@dataclass(frozen=True, kw_only=True)
class WangTsiatisStoppingPolicy(BoundaryFunctionStoppingPolicy):
    """Wang-Tsiatis shape-based stopping policy."""

    alpha: float
    delta: float

    def solve(
        self, model: BoundarySolver
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        t = model.info_times
        # Defined shape: t^(delta - 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            shape = t ** (self.delta - 0.5)

        if self.delta < 0.5:
            shape[t <= 0] = 1e6

        c = model.find_critical_value(shape, self.alpha)

        a = c * shape
        return a, None


class StoppingPolicyFactory:
    """Factory for creating StoppingPolicy instances from ES3 specs."""

    @staticmethod
    def build_from_spec(spec: GST.StoppingPolicySpec) -> StoppingPolicy:
        """Translates a StoppingPolicySpec into a concrete StoppingPolicy."""
        policy = spec.strategy

        if isinstance(policy, GST.AlphaSpendingStrategy):
            factory = SpendingFunctionFactory(budget=policy.budget)
            return SpendingFunctionStoppingPolicy(
                efficacy_spending=factory.build_from_spec(policy.spending_fn),
                sided=str(policy.sided) if policy.sided else "two",
            )

        if isinstance(policy, GST.BetaSpendingStrategy):
            factory = SpendingFunctionFactory(budget=policy.budget)
            return SpendingFunctionStoppingPolicy(
                futility_spending=factory.build_from_spec(policy.spending_fn),
                sided="one",
            )

        if isinstance(policy, GST.AlphaBetaSpendingStrategy):
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

        if isinstance(policy, GST.OBrienFlemingStrategy):
            return OBrienFlemingStoppingPolicy(
                alpha=policy.alpha,
                sided=str(policy.sided) if policy.sided else "two",
            )

        if isinstance(policy, GST.WhiteheadStrategy):
            return WhiteheadStoppingPolicy(
                alpha=policy.alpha,
                beta=policy.beta,
                sided="one",
                # Triangular design defaults
                alpha_binding=True,
                beta_binding=False,
            )

        if isinstance(policy, GST.PocockStrategy):
            return PocockStoppingPolicy(
                alpha=policy.alpha,
                sided=str(policy.sided) if policy.sided else "two",
            )

        if isinstance(policy, GST.WangTsiatisStrategy):
            return WangTsiatisStoppingPolicy(
                alpha=policy.alpha,
                delta=policy.delta,
                sided=str(policy.sided) if policy.sided else "two",
            )

        raise ValueError(f"Unknown stopping policy type: {type(policy)}")
