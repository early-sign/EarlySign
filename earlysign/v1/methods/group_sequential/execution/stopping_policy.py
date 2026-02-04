from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, cast

import numpy as np
from numpy.typing import NDArray

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.shared.spending import (
    SpendingFunction,
    SpendingFunctionFactory,
)


class BoundarySolver(Protocol):
    """Protocol for the statistical model capable of solving boundary constants.

    Attributes:
        info_times (NDArray[np.float64]): The information times for each look.
        tails (int): The number of tails for the test (1 or 2).
    """

    @property
    def info_times(self) -> NDArray[np.float64]: ...

    @property
    def tails(self) -> int: ...

    def find_critical_value(
        self, shape: NDArray[np.float64], alpha: float, tails: Optional[int] = None
    ) -> float:
        """Find critical value c such that P(Crossing c * shape) = alpha.

        Args:
            shape (NDArray[np.float64]): The shape of the boundary.
            alpha (float): The significance level.
            tails (Optional[int]): The number of tails for the test.
                If None, uses the `tails` attribute of the solver.

        Returns:
            float: The critical value c.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class StoppingPolicy(ABC):
    """Abstract base class for execution-layer Stopping Logic.

    Attributes:
        sided (str): "one" or "two" for one-sided or two-sided tests.
        alpha_binding (bool): True if the efficacy boundary is binding, False otherwise.
        beta_binding (bool): True if the futility boundary is binding, False otherwise.
    """

    sided: str = "two"
    alpha_binding: bool = True
    beta_binding: bool = False

    @abstractmethod
    def solve(
        self, model: BoundarySolver
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        """Solve for boundaries given the model context.

        Args:
            model (BoundarySolver): The statistical model providing information times
                and critical value finding capabilities.

        Returns:
            Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]: A tuple containing
                the efficacy boundaries and futility boundaries (if applicable).
                Each can be None if not solved or not applicable.
        """
        ...

    @abstractmethod
    def get_boundary(
        self,
        model: Any,
        look_index: int,
        info_time: float,
        rule_type: str = "efficacy",
    ) -> Optional[float]:
        """Compute boundary at a specific look and information time.

        Args:
            model (Any): The statistical model or engine providing context.
            look_index (int): The index of the current look.
            info_time (float): The current information time.
            rule_type (str): The type of boundary to retrieve ("efficacy" or "futility").

        Returns:
            Optional[float]: The boundary value, or None if no boundary is applicable
                or found for the given parameters.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class SpendingFunctionStoppingPolicy(StoppingPolicy):
    """Spending-function based stopping policy (Lan-DeMets).

    Attributes:
        efficacy_spending (Optional[SpendingFunction]): Optional efficacy spending function.
        futility_spending (Optional[SpendingFunction]): Optional futility spending function.
    """

    efficacy_spending: Optional[SpendingFunction] = None
    futility_spending: Optional[SpendingFunction] = None

    def solve(
        self, model: BoundarySolver
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        """Solve for boundaries.

        Note:
            For spending functions, the actual boundary solving is typically
            handled by the `Model/Engine` during initialization, as it requires
            iterative calculations based on the spending function and the
            underlying statistical model. This method returns None, None
            as a placeholder.

        Args:
            model (BoundarySolver): The statistical model.

        Returns:
            Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]: Always returns
                (None, None) as boundaries are solved dynamically.
        """
        # Implementation in Model/Engine for spending functions
        return None, None

    def get_boundary(
        self,
        model: Any,
        look_index: int,
        info_time: float,
        rule_type: str = "efficacy",
    ) -> Optional[float]:
        """Compute boundary at a specific look and information time using a spending function.

        This method implements a time-based (Spending) strategy: it follows the
        realized information time via numerical search. If `info_time` deviates
        significantly from the planned information time, the boundary is re-solved
        stage-by-stage using the spending function and historical data.

        Args:
            model (Any): The statistical model or engine providing context,
                including planned information times and historical boundaries.
            look_index (int): The index of the current look.
            info_time (float): The current information time.
            rule_type (str): The type of boundary to retrieve ("efficacy" or "futility").

        Returns:
            Optional[float]: The calculated boundary value, or None if no spending
                function is defined for the given `rule_type`.
        """
        engine = model
        planned_t = engine._points[look_index]

        # If very close to planned, use pre-calculated for speed
        if abs(info_time - planned_t) < 1e-5:
            if rule_type == "efficacy":
                if engine.efficacy_boundaries is not None and look_index < len(
                    engine.efficacy_boundaries
                ):
                    return float(engine.efficacy_boundaries[look_index])
            else:
                if engine.futility_boundaries is not None and look_index < len(
                    engine.futility_boundaries
                ):
                    return float(engine.futility_boundaries[look_index])

        # Otherwise, re-solve using history and spending function
        sf = (
            self.efficacy_spending
            if rule_type == "efficacy"
            else self.futility_spending
        )
        if sf is None:
            return None

        target = float(sf.cumulative(np.asarray([info_time]))[0])
        history_t = engine._points[:look_index]
        eff_hist = (
            engine.efficacy_boundaries[:look_index]
            if engine.efficacy_boundaries is not None
            else None
        )
        fut_hist = (
            engine.futility_boundaries[:look_index]
            if engine.futility_boundaries is not None
            else None
        )

        return cast(
            Optional[float],
            engine.canonical_model.solve_next_boundary(
                previous_times=history_t,
                current_t=info_time,
                target_cumulative_prob=target,
                previous_efficacy=eff_hist,
                previous_futility=fut_hist,
                rule_type=rule_type,
                drift=1.0,  # Standard drift used for futility solving (usually solved at init)
            ),
        )


@dataclass(frozen=True, kw_only=True)
class BoundaryFunctionStoppingPolicy(StoppingPolicy):
    """
    Index-based boundary policies.
    These designs assume equidistant looks and are defined by the look index number.
    """

    def get_boundary(
        self,
        model: Any,
        look_index: int,
        info_time: float,
        rule_type: str = "efficacy",
    ) -> Optional[float]:
        """
        Index-based lookup: anchors to the pre-calculated boundary for the current look_index.
        """
        engine = model
        if rule_type == "efficacy":
            if engine.efficacy_boundaries is not None and look_index < len(
                engine.efficacy_boundaries
            ):
                return float(engine.efficacy_boundaries[look_index])
        elif rule_type == "futility":
            if engine.futility_boundaries is not None and look_index < len(
                engine.futility_boundaries
            ):
                return float(engine.futility_boundaries[look_index])
        return None


@dataclass(frozen=True, kw_only=True)
class OBrienFlemingStoppingPolicy(BoundaryFunctionStoppingPolicy):
    """O'Brien-Fleming shape-based stopping policy."""

    alpha: float
    _c: Optional[float] = None  # Internal cache for the solved constant

    def solve(
        self, model: BoundarySolver
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        t = model.info_times
        with np.errstate(divide="ignore"):
            shape = 1.0 / np.sqrt(t)
        shape[t <= 0] = 1e6
        c = model.find_critical_value(shape, self.alpha)
        # Store c for internal reference
        object.__setattr__(self, "_c", c)
        a = c * shape
        return a, None

    # Inherits get_boundary (index-based lookup) from BoundaryFunctionStoppingPolicy


@dataclass(frozen=True, kw_only=True)
class PocockStoppingPolicy(BoundaryFunctionStoppingPolicy):
    """Pocock shape-based stopping policy."""

    alpha: float
    _c: Optional[float] = None

    def solve(
        self, model: BoundarySolver
    ) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
        t = model.info_times
        shape = np.ones_like(t)
        c = model.find_critical_value(shape, self.alpha)
        object.__setattr__(self, "_c", c)
        a = c * shape
        return a, None

    # Inherits get_boundary (index-based lookup) from BoundaryFunctionStoppingPolicy


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
        t = model.info_times
        a, b = self._calculate(t)
        return a, b

    def _calculate(
        self, t: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - self.alpha)
        z_beta = norm.ppf(1 - self.beta)
        a0 = (z_alpha + z_beta) / 2
        sqrt_t = np.sqrt(np.maximum(t, 1e-10))
        inv_sqrt_t = 1.0 / sqrt_t
        a = a0 * (inv_sqrt_t + 3 * sqrt_t)
        b = a0 * (-inv_sqrt_t + 3 * sqrt_t)
        return a, b

    def get_boundary(
        self,
        model: Any,
        look_index: int,
        info_time: float,
        rule_type: str = "efficacy",
    ) -> Optional[float]:
        """
        Time-based (Analytical) Strategy: follow realized info_time.
        Whitehead Triangular boundaries are analytical and follow realized info_time
        even without a spending search.
        """
        t_arr = np.asarray([info_time], dtype=float)
        a_arr, b_arr = self._calculate(t_arr)
        if rule_type == "efficacy":
            return float(a_arr[0])
        elif rule_type == "futility":
            return float(b_arr[0])
        return None


@dataclass(frozen=True, kw_only=True)
class WangTsiatisStoppingPolicy(BoundaryFunctionStoppingPolicy):
    """Wang-Tsiatis shape-based stopping policy."""

    alpha: float
    delta: float
    _c: Optional[float] = None

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
        object.__setattr__(self, "_c", c)

        a = c * shape
        return a, None

    # Inherits get_boundary (index-based lookup) from BoundaryFunctionStoppingPolicy


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
