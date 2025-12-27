"""Group-sequential spending classes.

Provides small, focused spending strategies used to allocate error over
information fractions in group-sequential designs. Each concrete class
implements a `cumulative(t)` method and `boundaries_from_stage_alpha(..)`.

Background & families
---------------------
This module replaces the older functional helpers and exposes three common
families as configurable classes:

- O'Brien-Fleming (OBF): conservative early, liberal late
- Pocock: more uniform spending across looks
- Hwang–Shih–DeCani (HSD): flexible one-parameter family (gamma)

Each class is vectorized: pass a numpy array of information fractions to
``cumulative(t)`` to obtain cumulative spending at those fractions. To get
per-stage increments, use ``np.diff(np.concatenate([[0.0], cum]))``.

Examples (numeric sanity checks)
--------------------------------
>>> import numpy as np
>>> s = OBFSpending(alpha=0.05, sided=2)
>>> round(float(s.cumulative(np.array([0.5]))[0]), 6)
0.005575

>>> s2 = PocockSpending(alpha=0.05)
>>> round(float(s2.cumulative(np.array([0.5]))[0]), 6)
0.031006

References
----------
Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for
clinical trials. Biometrika, 70(3), 659-663.

Hwang, I. K., Shih, W. J., & De Cani, J. S. (1990). Group sequential designs
using a family of type I error probability spending functions. Statistics in
Medicine, 9(12), 1439-1445.
"""

import math
from typing import Any, Dict, Protocol, Type

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


class SpendingFunction(Protocol):
    """Protocol describing a spending function implementation.

    Methods
    -------
    cumulative(t) -> np.ndarray
        Return cumulative spending at information fractions `t`.

    boundaries_from_stage_alpha(stage_alpha) -> np.ndarray
        Convert per-stage α increments into one-sided z-critical values.
    """

    @property
    def name(self) -> str: ...

    alpha: float

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]: ...

    def boundaries_from_stage_alpha(
        self, stage_alpha: NDArray[Any]
    ) -> NDArray[Any]: ...


class OBFSpending(SpendingFunction):
    """Lan–DeMets O'Brien–Fleming style spending.

    Examples
    --------
    >>> import numpy as np
    >>> s = OBFSpending(alpha=0.05, sided=2)
    >>> round(float(s.cumulative(np.array([0.5]))[0]), 6)
    0.005575

    >>> # Boundaries from stagewise increments at a two-look design
    >>> t = np.array([0.5, 1.0])
    >>> cum = s.cumulative(t)
    >>> stage_alpha = np.diff(np.concatenate([[0.0], cum]))
    >>> np.round(s.boundaries_from_stage_alpha(stage_alpha), 3)
    array([2.538, 1.701])

    Parameters
    ----------
    alpha
        overall error level (e.g., 0.05)
    sided
        1 or 2 (number of tails). For sided==2 the spending uses α/2
        internally in the OBF formula and cumulative() returns the total
        (not halved) two-sided level.
    """

    def __init__(self, alpha: float, sided: int = 2) -> None:
        if sided not in (1, 2):
            raise ValueError("sided must be 1 or 2")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.sided = int(sided)

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]:
        t_arr = np.asarray(t, dtype=float)
        t_arr = np.clip(t_arr, 0.0, 1.0)
        # avoid divide-by-zero
        t_arr = np.maximum(t_arr, 1e-12)
        if self.sided == 2:
            z = float(norm.isf(self.alpha / 2.0))
            return np.asarray(2.0 - 2.0 * norm.cdf(z / np.sqrt(t_arr)), dtype=float)
        else:
            z = float(norm.isf(self.alpha))
            return np.asarray(1.0 - norm.cdf(z / np.sqrt(t_arr)), dtype=float)

    def boundaries_from_stage_alpha(self, stage_alpha: NDArray[Any]) -> NDArray[Any]:
        a = np.clip(np.asarray(stage_alpha, dtype=float), 1e-16, 1.0 - 1e-16)
        return np.asarray(norm.ppf(1.0 - a), dtype=float)

    @property
    def name(self) -> str:
        return "obrien_fleming"


class PocockSpending(SpendingFunction):
    """Pocock-like spending (approximate continuous form).

    Examples
    --------
    >>> import numpy as np
    >>> s = PocockSpending(alpha=0.05)
    >>> round(float(s.cumulative(np.array([0.5]))[0]), 6)
    0.031006

    Uses α(t) = α * log(1 + (e - 1) t)
    """

    def __init__(self, alpha: float) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]:
        t_arr = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        return np.asarray(self.alpha * np.log1p((math.e - 1.0) * t_arr), dtype=float)

    def boundaries_from_stage_alpha(self, stage_alpha: NDArray[Any]) -> NDArray[Any]:
        a = np.clip(np.asarray(stage_alpha, dtype=float), 1e-16, 1.0 - 1e-16)
        return np.asarray(norm.ppf(1.0 - a), dtype=float)

    @property
    def name(self) -> str:
        return "pocock"


class HSDSpending(SpendingFunction):
    """Hwang–Shih–DeCani family.

    Examples
    --------
    >>> import numpy as np
    >>> s = HSDSpending(alpha=0.05, gamma=0.0)
    >>> round(float(s.cumulative(np.array([0.5]))[0]), 6)
    0.025

    gamma controls shape: gamma=0 linear, gamma<0 OBF-like, gamma>0 Pocock-like.
    """

    def __init__(self, alpha: float, gamma: float = -4.0) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]:
        t_arr = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        if abs(self.gamma) < 1e-12:
            return np.asarray(self.alpha * t_arr, dtype=float)
        num = 1.0 - np.exp(-self.gamma * t_arr)
        den = 1.0 - math.exp(-self.gamma)
        return np.asarray(self.alpha * (num / den), dtype=float)

    def boundaries_from_stage_alpha(self, stage_alpha: NDArray[Any]) -> NDArray[Any]:
        a = np.clip(np.asarray(stage_alpha, dtype=float), 1e-16, 1.0 - 1e-16)
        return np.asarray(norm.ppf(1.0 - a), dtype=float)

    @property
    def name(self) -> str:
        return "hsd"


class RhoFamilySpending(SpendingFunction):
    """Rho-family power spending function.

    Formula: alpha(t) = alpha * t^rho, for rho > 0.
    Identical to Kim-DeMets (1987) / Jennison & Turnbull (2000) rho family.
    rho=1 is linear spending (HSD gamma=0).
    rho=2, 3 provide OBF-like conservative early spending.
    """

    def __init__(self, alpha: float, rho: float = 2.0) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if rho <= 0:
            raise ValueError("rho must be positive")
        self.alpha = float(alpha)
        self.rho = float(rho)

    def cumulative(self, t: NDArray[Any]) -> NDArray[Any]:
        t_arr = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        return np.asarray(self.alpha * (t_arr**self.rho), dtype=float)

    def boundaries_from_stage_alpha(self, stage_alpha: NDArray[Any]) -> NDArray[Any]:
        a = np.clip(np.asarray(stage_alpha, dtype=float), 1e-16, 1.0 - 1e-16)
        return np.asarray(norm.ppf(1.0 - a), dtype=float)

    @property
    def name(self) -> str:
        return "rho"


_SPENDING_REGISTRY: Dict[str, Type[SpendingFunction]] = {
    "obrien_fleming": OBFSpending,
    "pocock": PocockSpending,
    "hsd": HSDSpending,
    "rho": RhoFamilySpending,
}


def get_spending_class(name: str) -> Type[SpendingFunction]:
    """Return the spending class registered under ``name``."""

    key = str(name).strip().lower()
    if key not in _SPENDING_REGISTRY:
        raise KeyError(
            f"Unknown spending family '{name}'. "
            f"Available: {sorted(_SPENDING_REGISTRY)}"
        )
    return _SPENDING_REGISTRY[key]
