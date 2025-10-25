"""
Primitive stochastic-process abstractions for boundary calculations.

This module defines a small, well-documented protocol for stochastic
processes used in group-sequential boundary calculations and provides
several concrete, ready-to-use implementations. The intent is that
downstream boundary logic depends only on the abstractions here so more
exotic processes (score processes, t-processes, ...) can be added later
and swapped in with minimal changes.

Design notes
------------
- Each process exposes conversions between the canonical Z-scale used for
  nominal boundary specification and the process-specific scale used for
  visualization or process-aware logic (e.g., Brownian motion B(t) = Z*√t).
- Each process can provide a correlation matrix for a vector of
  information times; this is useful for future improvements that take
  correlation between interim looks into account.

The module purposely implements simple, well-tested behavior for the
initial processes; more advanced processes can be added following the
same protocol.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from earlysign.stats.common.group_sequential.essentials import conversions


class StochasticProcess(Protocol):
    """Protocol for a stochastic process abstraction.

    Implementations provide conversions between the canonical Z-scale and
    a process-specific scale at an information time `t`, plus a correlation
    matrix factory for multiple information times.

    Alpha / spending semantics (cumulative alpha -> nominal Z) belong in
    the spending/conversions layer, not in the primitives.
    """

    name: str

    def stat_to_process(self, stat: float, t: float) -> float:
        """Map a statistic value (e.g. Z) to the process-specific scale at info time t.

        The name `stat` is intentionally generic: the primitive should not
        assume the statistic is a Z; callers may pass a Z, score, t-statistic,
        or any other scalar statistic as appropriate for the scheme.
        """
        raise NotImplementedError()

    def process_to_stat(self, value: float, t: float) -> float:
        """Map a process-scale value back to the statistic scale at info time t."""
        raise NotImplementedError()

    def mean_process(self, t: float, *, drift: float = 0.0) -> float:
        """Return E[process(t)] under a given process-scale drift.

        The `drift` parameter is in process scale per unit information; the
        precise mapping from an effect-size to `drift` is the responsibility
        of higher-level code (effect calculators).
        """
        raise NotImplementedError()

    def var_process(self, t: float) -> float:
        """Return Var(process(t))."""
        raise NotImplementedError()

    def correlation_matrix(self, info_times: np.ndarray) -> np.ndarray:
        """Return corr matrix for process values at the given info times.

        The returned matrix should be shape (k, k) for k = len(info_times).
        """
        raise NotImplementedError()


class BrownianMotionProcess:
    """Brownian motion process B(t) = Z * sqrt(t).

    Under the null, B(t) ~ N(0, t) and Cov(B(t_i), B(t_j)) = min(t_i, t_j).
    This implementation supplies exact conversions and the closed-form
    correlation matrix for Brownian motion.
    """

    name = "brownian"

    def stat_to_process(self, stat: float, t: float) -> float:
        # stat is typically a Z-statistic; map to Brownian B(t) = Z*sqrt(t)
        return conversions.z_to_brownian(stat, t)

    def process_to_stat(self, value: float, t: float) -> float:
        return conversions.brownian_to_z(value, t)

    def correlation_matrix(self, info_times: np.ndarray) -> np.ndarray:
        t = np.asarray(info_times, dtype=float)
        k = t.shape[0]
        # Guard against zeros
        t = np.maximum(t, 0.0)
        mat = np.empty((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                if t[i] == 0.0 or t[j] == 0.0:
                    # If any time is zero, correlation is defined as 0 when
                    # the other is non-zero and 1 when both zero.
                    mat[i, j] = 1.0 if (t[i] == 0.0 and t[j] == 0.0) else 0.0
                else:
                    # corr = min(ti, tj) / sqrt(ti * tj) = sqrt(min/max)
                    mat[i, j] = math.sqrt(min(t[i], t[j]) / max(t[i], t[j]))
        return mat


class ScoreProcess(BrownianMotionProcess):
    """Score process approximation.

    Asymptotically equivalent to Brownian motion in many settings. For
    pragmatic use we inherit Brownian behavior; override if/when a more
    precise score-specific mapping is required.
    """

    name = "score"


class TProcess:
    """t-process approximation.

    Placeholder that currently uses Brownian-style conversions. Intended
    as an extension point where small-sample t-process behaviors can be
    implemented later.
    """

    name = "t_process"

    def stat_to_process(self, stat: float, t: float) -> float:
        return conversions.z_to_brownian(stat, t)

    def process_to_stat(self, value: float, t: float) -> float:
        return conversions.brownian_to_z(value, t)

    def correlation_matrix(self, info_times: np.ndarray) -> np.ndarray:
        # Use the Brownian correlation as a conservative default
        return BrownianMotionProcess().correlation_matrix(info_times)
