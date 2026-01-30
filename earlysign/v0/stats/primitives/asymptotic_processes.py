"""Gaussian process primitives for sequential boundary calculations.

This module provides lightweight abstractions for discrete-time Gaussian
processes that underpin group-sequential designs. The focus is on
conditionally Gaussian models with information-time parametrisation.

Key components include:

* :class:`ConditionedNormal` – a tiny container for univariate Gaussian laws.
* :class:`ProcessHistory` – immutable storage for realised looks.
* :class:`BoxPathConstraint` – helper to encode simple path constraints such as
  bounds on realised statistics.
* :class:`GaussianSequentialProcess` – protocol describing the expected API.
* :class:`BrownianMotion` – drifted Brownian motion on information time.
* :class:`AsymptoticZProcess` – Brownian motion scaled to match the
  asymptotic distribution of sequential Z-statistics.

Downstream code (boundary calculators, spending functions, etc.) should rely
on these primitives rather than rolling bespoke Gaussian maths.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Self, Sequence, Tuple, Union, cast

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Basic Gaussian containers
# ---------------------------------------------------------------------------


@dataclass
class ConditionedNormal:
    """Lightweight representation of a univariate normal distribution."""

    mean: float
    var: float

    def __post_init__(self) -> None:
        if self.var < 0:
            raise ValueError("Variance must be non-negative.")
        self.mean = float(self.mean)
        self.var = float(self.var)

    def cdf(self, x: float) -> float:
        r"""Return :math:`\mathbb{P}[X \le x]`."""

        if self.var == 0.0:
            return float(x >= self.mean)
        z = (x - self.mean) / math.sqrt(self.var)
        return float(norm.cdf(z))

    def survival(self, x: float) -> float:
        r"""Return :math:`\mathbb{P}[X > x]`."""

        return 1.0 - self.cdf(x)

    def ppf(self, q: float) -> float:
        """Percent point function (inverse CDF)."""

        if not (0.0 < q < 1.0):
            raise ValueError("q must lie in (0, 1).")
        if self.var == 0.0:
            return float(self.mean)
        return float(self.mean + math.sqrt(self.var) * norm.ppf(q))

    def isf(self, q: float) -> float:
        """Inverse survival function."""

        if not (0.0 < q < 1.0):
            raise ValueError("q must lie in (0, 1).")
        if self.var == 0.0:
            return float(self.mean)
        return float(self.mean + math.sqrt(self.var) * norm.isf(q))


# ---------------------------------------------------------------------------
# History and path-constraint helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessHistory:
    """Immutable container of information times and realised statistics."""

    times: np.ndarray
    values: np.ndarray

    @classmethod
    def from_pairs(cls, pairs: "ProcessHistoryLike") -> Self:
        if isinstance(pairs, cls):
            return pairs
        times: list[float] = []
        values: list[float] = []
        for t, v in cast(Sequence[Tuple[float, float]], pairs):
            times.append(float(t))
            values.append(float(v))
        if not times:
            return cls.empty()
        arr_times = np.asarray(times, dtype=float)
        arr_values = np.asarray(values, dtype=float)
        order = np.argsort(arr_times, kind="mergesort")
        return cls(times=arr_times[order], values=arr_values[order])

    @classmethod
    def empty(cls) -> Self:
        return cls(times=np.empty(0, dtype=float), values=np.empty(0, dtype=float))


ProcessHistoryLike = Union[ProcessHistory, Sequence[Tuple[float, float]]]

PathConstraint = Callable[[np.ndarray, np.ndarray], None]


@dataclass(frozen=True)
class BoxPathConstraint:
    """Simple path constraint described by optional lower/upper envelopes."""

    lower: Optional[Callable[[float], float]] = None
    upper: Optional[Callable[[float], float]] = None

    def __call__(self, times: np.ndarray, values: np.ndarray) -> None:
        if times.size == 0:
            return
        if self.lower is not None:
            lower_vals = np.fromiter((self.lower(t) for t in times), dtype=float)
            if np.any(values < lower_vals):
                raise ValueError("History violates lower path constraint.")
        if self.upper is not None:
            upper_vals = np.fromiter((self.upper(t) for t in times), dtype=float)
            if np.any(values > upper_vals):
                raise ValueError("History violates upper path constraint.")


def _coerce_history(history: Optional[ProcessHistoryLike]) -> ProcessHistory:
    if history is None:
        return ProcessHistory.empty()
    return ProcessHistory.from_pairs(history)


def _solve_psd(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve linear system accounting for positive-semidefinite covariance."""

    if matrix.size == 0:
        return np.zeros_like(rhs, dtype=float)
    try:
        return cast(np.ndarray, np.linalg.solve(matrix, rhs))
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(matrix, hermitian=True)
        return cast(np.ndarray, pinv @ rhs)


def _gaussian_step(
    mu_hist: np.ndarray,
    cov_hist: np.ndarray,
    cov_cross: np.ndarray,
    mu_next: float,
    var_next: float,
    history_values: np.ndarray,
) -> ConditionedNormal:
    if cov_hist.size == 0:
        return ConditionedNormal(mean=mu_next, var=var_next)
    delta = history_values - mu_hist
    solve_delta = _solve_psd(cov_hist, delta)
    solve_cross = _solve_psd(cov_hist, cov_cross)
    mean = mu_next + cov_cross @ solve_delta
    var = var_next - cov_cross @ solve_cross
    var = float(max(var, 0.0))
    return ConditionedNormal(mean=float(mean), var=var)


# ---------------------------------------------------------------------------
# Protocol definition
# ---------------------------------------------------------------------------


class GaussianSequentialProcess(Protocol):
    """Protocol describing a discrete-time Gaussian sequential process."""

    @property
    def t_grid(self) -> np.ndarray: ...

    @property
    def drift(self) -> float: ...

    def joint(self) -> Tuple[np.ndarray, np.ndarray]: ...

    def next_conditional(
        self,
        *,
        next_time: float,
        history: Optional[ProcessHistoryLike] = None,
        path_constraint: Optional[PathConstraint] = None,
    ) -> ConditionedNormal: ...


# ---------------------------------------------------------------------------
# Brownian motion primitive
# ---------------------------------------------------------------------------


class BrownianMotion(GaussianSequentialProcess):
    r"""Drifted Brownian motion on information time.

    Examples
    --------
    >>> bm = BrownianMotion(t_grid=[0.5, 1.0], drift=0.2)
    >>> # Unconditional first look at t=0.5
    >>> d0 = bm.next_conditional(next_time=0.5)
    >>> round(d0.mean, 4), round(d0.var, 4)
    (0.1, 0.5)
    >>> # Conditional second look given history at 0.5
    >>> history = [(0.5, 0.3)]
    >>> d1 = bm.next_conditional(next_time=1.0, history=history)
    >>> round(d1.mean, 4), round(d1.var, 4)
    (0.4, 0.5)
    """

    def __init__(
        self,
        t_grid: Sequence[float],
        drift: float = 0.0,
        *,
        path_constraint: Optional[PathConstraint] = None,
    ) -> None:
        grid = np.asarray(t_grid, dtype=float)
        if grid.size == 0 or not np.all(np.diff(grid) > 0):
            raise ValueError(
                "t_grid must be strictly increasing with at least one point."
            )
        self._t_grid = grid
        self._drift = float(drift)
        self._path_constraint = path_constraint

    @property
    def t_grid(self) -> np.ndarray:
        return self._t_grid

    @property
    def drift(self) -> float:
        return self._drift

    def joint(self) -> Tuple[np.ndarray, np.ndarray]:
        ts = self._t_grid
        mean = self._drift * ts
        cov = np.minimum(ts[:, None], ts[None, :])
        return mean, cov

    def next_conditional(
        self,
        *,
        next_time: float,
        history: Optional[ProcessHistoryLike] = None,
        path_constraint: Optional[PathConstraint] = None,
    ) -> ConditionedNormal:
        hist = _coerce_history(history)
        cont = path_constraint or self._path_constraint
        if cont is not None:
            cont(hist.times, hist.values)

        t_next = float(next_time)
        if t_next <= 0.0:
            raise ValueError("next_time must be positive.")

        if hist.times.size == 0:
            mean = self._drift * t_next
            var = t_next
            return ConditionedNormal(mean=mean, var=var)

        mu_hist = self._drift * hist.times
        mu_next = self._drift * t_next
        cov_hist = np.minimum(hist.times[:, None], hist.times[None, :])
        cov_cross = np.minimum(hist.times, t_next)
        var_next = t_next
        return _gaussian_step(
            mu_hist, cov_hist, cov_cross, mu_next, var_next, hist.values
        )


# ---------------------------------------------------------------------------
# Asymptotic Z-process primitive
# ---------------------------------------------------------------------------


class AsymptoticZProcess(GaussianSequentialProcess):
    r"""Gaussian process matching the asymptotic law of sequential Z-statistics.

    Many score/Wald-style interim statistics converge in distribution to
    Brownian motion after suitable deterministic scaling of mean and variance.
    This class encodes that scaling and exposes the same predictive interface
    as :class:`BrownianMotion`.
    """

    def __init__(
        self,
        t_grid: Sequence[float],
        drift: float,
        mean_scale: Callable[[float], float],
        std_scale: Callable[[float], float],
        *,
        path_constraint: Optional[PathConstraint] = None,
    ) -> None:
        grid = np.asarray(t_grid, dtype=float)
        if grid.size == 0 or not np.all(np.diff(grid) > 0):
            raise ValueError(
                "t_grid must be strictly increasing with at least one point."
            )
        self._t_grid = grid
        self._drift = float(drift)
        self._mean_scale = mean_scale
        self._std_scale = std_scale
        self._path_constraint = path_constraint

    @property
    def t_grid(self) -> np.ndarray:
        return self._t_grid

    @property
    def drift(self) -> float:
        return self._drift

    def _mean_at(self, t: float) -> float:
        return float(self._mean_scale(t) * self._drift)

    def _var_at(self, t: float) -> float:
        s = self._std_scale(t)
        return float((s**2) * t)

    def _cov(self, s_t: float, t_t: float) -> float:
        return float(self._std_scale(s_t) * self._std_scale(t_t) * min(s_t, t_t))

    def joint(self) -> Tuple[np.ndarray, np.ndarray]:
        ts = self._t_grid
        mean = np.array([self._mean_at(t) for t in ts], dtype=float)
        cov = np.zeros((ts.size, ts.size), dtype=float)
        for i, ti in enumerate(ts):
            for j, tj in enumerate(ts):
                cov[i, j] = self._cov(ti, tj)
        return mean, cov

    def next_conditional(
        self,
        *,
        next_time: float,
        history: Optional[ProcessHistoryLike] = None,
        path_constraint: Optional[PathConstraint] = None,
    ) -> ConditionedNormal:
        hist = _coerce_history(history)
        cont = path_constraint or self._path_constraint
        if cont is not None:
            cont(hist.times, hist.values)

        t_next = float(next_time)
        if t_next <= 0.0:
            raise ValueError("next_time must be positive.")

        if hist.times.size == 0:
            mean = self._mean_at(t_next)
            var = self._var_at(t_next)
            return ConditionedNormal(mean=mean, var=var)

        mu_hist = np.array([self._mean_at(t) for t in hist.times], dtype=float)
        mu_next = self._mean_at(t_next)

        n = hist.times.size
        cov_hist = np.zeros((n, n), dtype=float)
        for i, ti in enumerate(hist.times):
            for j, tj in enumerate(hist.times):
                cov_hist[i, j] = self._cov(ti, tj)
        cov_cross = np.array([self._cov(t, t_next) for t in hist.times], dtype=float)
        var_next = self._var_at(t_next)
        return _gaussian_step(
            mu_hist, cov_hist, cov_cross, mu_next, var_next, hist.values
        )


# ---------------------------------------------------------------------------
# Scalar utilities used by boundary calculators
# ---------------------------------------------------------------------------
