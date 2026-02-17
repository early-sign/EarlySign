"""High-level GSD Solver Interface.

Provides a clean API for solving GSD boundaries during the planning phase
without exposing the internal configuration and model mechanics.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from earlysign.methods.group_sequential.execution.stopping_policy import (
    BoundarySolver,
    StoppingPolicy,
)
from earlysign.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
    Config,
)


class DesignSolver(BoundarySolver):
    """Bridge for Solving GSD Boundaries during planning."""

    def __init__(
        self,
        info_times: NDArray[np.float64],
        tails: int = 1,
        rng_seed: Optional[int] = None,
    ):
        self._info_times = info_times
        self._tails = tails
        self._model = CanonicalJointModel(
            Config(info_times=info_times, tails=tails, rng_seed=rng_seed)
        )

    @property
    def info_times(self) -> NDArray[np.float64]:
        return self._info_times

    @property
    def tails(self) -> int:
        return self._tails

    def find_critical_value(
        self, shape: NDArray[np.float64], alpha: float, tails: Optional[int] = None
    ) -> float:
        return self._model.find_boundary_constant(
            self._info_times, shape, alpha, tails=tails or self._tails
        )


def solve_boundaries(
    policy: StoppingPolicy,
    info_times: NDArray[np.float64] | List[float],
    tails: int = 1,
    rng_seed: Optional[int] = None,
) -> Tuple[Optional[NDArray[Any]], Optional[NDArray[Any]]]:
    """Lightweight convenience function for solving GSD boundaries.

    Args:
        policy: The stopping policy (e.g. OBrienFlemingStoppingPolicy).
        info_times: Realized or planned information times.
        tails: Number of tails for calculation.
        rng_seed: Random seed for simulation-based solving.

    Returns:
        (efficacy_boundaries, futility_boundaries)
    """
    t = np.asarray(info_times)
    solver = DesignSolver(t, tails=tails, rng_seed=rng_seed)
    return policy.solve(solver)
