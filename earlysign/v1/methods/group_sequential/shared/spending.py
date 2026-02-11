"""Group-sequential spending classes (EarlySign v1).

Provides small, focused spending strategies used to allocate error over
information fractions in group-sequential designs.
"""

import math
from typing import Any, ClassVar, Dict, List, Protocol, Type

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


class SpendingFunction(Protocol):
    """Protocol describing a spending function implementation."""

    name: ClassVar[str]

    budget: float
    params: Dict[str, Any]

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]: ...
    def boundaries_from_stage_alpha(
        self, stage_alpha: NDArray[Any]
    ) -> NDArray[Any]: ...


class OBrienFlemingSpending(SpendingFunction):
    """Lan–DeMets O'Brien–Fleming style spending (1-sided).

    For two-sided tests, provide budget=alpha/2.
    """

    def __init__(self, budget: float, **params: Any) -> None:
        if not (0.0 < budget < 1.0):
            raise ValueError("budget must be in (0, 1)")
        self.budget = float(budget)
        self.params = params

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]:
        t_arr = np.asarray(t, dtype=float)
        t_arr = np.clip(t_arr, 0.0, 1.0)
        t_arr = np.maximum(t_arr, 1e-12)
        z = float(norm.isf(self.budget))
        # 1-sided formula: α(t) = P(Z > z_α / √t) = norm.sf(z_α / √t)
        # Numerical Stability: We use norm.sf instead of 1-norm.cdf to avoid precision
        # loss (underflow to 0.0) at very early information fractions (large z_alpha/sqrt(t)).
        return np.asarray(norm.sf(z / np.sqrt(t_arr)), dtype=float)

    def boundaries_from_stage_alpha(self, stage_alpha: NDArray[Any]) -> NDArray[Any]:
        a = np.clip(np.asarray(stage_alpha, dtype=float), 1e-16, 1.0 - 1e-16)
        return np.asarray(norm.ppf(1.0 - a), dtype=float)

    name = "obrien_fleming"


class PocockSpending(SpendingFunction):
    """Pocock-like spending (approximate continuous form)."""

    def __init__(self, budget: float, **params: Any) -> None:
        if not (0.0 < budget < 1.0):
            raise ValueError("budget must be in (0, 1)")
        self.budget = float(budget)
        self.params = params

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]:
        t_arr = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        return np.asarray(self.budget * np.log1p((math.e - 1.0) * t_arr), dtype=float)

    def boundaries_from_stage_alpha(self, stage_alpha: NDArray[Any]) -> NDArray[Any]:
        a = np.clip(np.asarray(stage_alpha, dtype=float), 1e-16, 1.0 - 1e-16)
        return np.asarray(norm.ppf(1.0 - a), dtype=float)

    name = "pocock"


class HwangShihDeCaniSpending(SpendingFunction):
    """Hwang–Shih–DeCani family."""

    def __init__(self, budget: float, gamma: float = -4.0, **params: Any) -> None:
        if not (0.0 < budget < 1.0):
            raise ValueError("budget must be in (0, 1)")
        self.budget = float(budget)
        self.gamma = float(gamma)
        self.params = {"gamma": self.gamma, **params}

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]:
        t_arr = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        if abs(self.gamma) < 1e-12:
            return np.asarray(self.budget * t_arr, dtype=float)
        num = 1.0 - np.exp(-self.gamma * t_arr)
        den = 1.0 - math.exp(-self.gamma)
        return np.asarray(self.budget * (num / den), dtype=float)

    def boundaries_from_stage_alpha(self, stage_alpha: NDArray[Any]) -> NDArray[Any]:
        a = np.clip(np.asarray(stage_alpha, dtype=float), 1e-16, 1.0 - 1e-16)
        return np.asarray(norm.ppf(1.0 - a), dtype=float)

    name = "hwang_shih_decani"


class PowerFamilySpending(SpendingFunction):
    """Power-family spending function (Kim-DeMets)."""

    def __init__(self, budget: float, rho: float = 2.0, **params: Any) -> None:
        if not (0.0 < budget < 1.0):
            raise ValueError("budget must be in (0, 1)")
        if rho <= 0:
            raise ValueError("rho must be positive")
        self.budget = float(budget)
        self.rho = float(rho)
        self.params = {"rho": self.rho, **params}

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]:
        t_arr = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        return np.asarray(self.budget * (t_arr**self.rho), dtype=float)

    def boundaries_from_stage_alpha(self, stage_alpha: NDArray[Any]) -> NDArray[Any]:
        a = np.clip(np.asarray(stage_alpha, dtype=float), 1e-16, 1.0 - 1e-16)
        return np.asarray(norm.ppf(1.0 - a), dtype=float)

    name = "power_family"


_SPENDING_CLASSES: List[Type[SpendingFunction]] = [
    OBrienFlemingSpending,
    PocockSpending,
    HwangShihDeCaniSpending,
    PowerFamilySpending,
]

_SPENDING_REGISTRY: Dict[str, Type[SpendingFunction]] = {
    cls.name: cls for cls in _SPENDING_CLASSES
}


def get_spending_class(name: str) -> Type[SpendingFunction]:
    key = str(name).strip().lower()
    if key not in _SPENDING_REGISTRY:
        raise KeyError(f"Unknown spending family '{name}'")
    return _SPENDING_REGISTRY[key]


class SpendingFunctionFactory:
    """Factory for creating SpendingFunction instances from ES3 specs.

    Example::

        >>> from earlysign.schema.ES3.GST import SpendingFunction
        >>> factory = SpendingFunctionFactory(budget=0.025)
        >>> spec = SpendingFunction(family="obrien_fleming")
        >>> sf = factory.build_from_spec(spec)
        >>> sf.name
        'obrien_fleming'
    """

    def __init__(self, budget: float) -> None:
        """Initialize factory with error budget.

        Args:
            budget: The alpha or beta budget for spending functions.
        """
        self.budget = budget

    def build_from_spec(self, spec: Any) -> SpendingFunction:
        """Create a SpendingFunction from an ES3 SpendingFunctionSpec.

        Args:
            spec: SpendingFunctionSpec with family and optional params.

        Returns:
            Instantiated SpendingFunction.
        """
        family = str(spec.family).strip().lower()
        spending_cls = get_spending_class(family)
        params = dict(spec.params) if spec.params else {}
        return spending_cls(budget=self.budget, **params)  # type: ignore[call-arg]
