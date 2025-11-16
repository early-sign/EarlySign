"""Group-sequential operating characteristics utilities.

This module provides a protocol for a Simulator and a Procedure and a
convenient compute_oc_curve() function that evaluates a Procedure across a
range of effect sizes by delegating the heavy lifting to the provided
Simulator implementation.

The Simulator implementation is intentionally left to the scheme-specific
modules under `earlysign.stats.essentials.schemes` because different schemes
require different data generation and simulation logic. Callers should pass
an object that implements the Simulator protocol defined below.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np


@dataclass
class OCPointResult:
    """Operating characteristics at a single effect size.

    Attributes:
        effect_size: effect size evaluated
        expected_sample_size: expected sample size (ESS)
        power: estimated power (probability of rejecting H0)
        max_sample_size: maximum sample size observed in simulations
        stop_distribution: mapping from actual total sample size (int) at stop time
            to counts. Legacy formats using 1-based analysis indices (1..n_looks)
            may be accepted by some visualizers, but simulators SHOULD emit
            sample-size keyed stop distributions to avoid ambiguity.
        metadata: optional additional info produced by simulator
    """

    effect_size: float
    expected_sample_size: float
    power: float
    max_sample_size: float
    stop_distribution: Dict[int, int]
    metadata: Optional[Dict[str, Any]] = None


@runtime_checkable
class Procedure(Protocol):
    """Protocol for a statistical procedure.

    This project uses a single, explicit API in which the Procedure is the
    authoritative source for analysis timing and stopping decisions. The
    simulator's responsibility is strictly to generate data batches and
    hand them to the Procedure via `ingest`.

    Required methods:
        - ingest(cumulative: Dict[str, Any]) -> None
            Accept generated cumulative data (one increment) and update
            internal state used for the stopping decision.
        - should_stop(look: int) -> Optional[Dict[str, Any]]
            After ingesting data, return None to continue or a dict with at
            least the key 'reject' (boolean) to indicate the trial stopped
            and whether H0 was rejected.
        - reset() -> None
            Reinitialize internal state for a new simulation/replication.

    NOTE:
        boundary computation and information-time resolution are
        internal to the Procedure implementation and are not part of the
        simulator-facing protocol. Procedures should perform any setup when
        constructed or on `reset()`.
    """

    def ingest(self, cumulative: Dict[str, Any]) -> None:
        """Accept cumulative (or incremental) data produced by the simulator.

        The Procedure updates internal state from `cumulative` so that
        subsequent calls to `should_stop` can make a decision.
        """
        ...

    def should_stop(self, look: int) -> Optional[Dict[str, Any]]:
        """Return None to continue, or a dict indicating stop info.

        Example return value when stopping: {'reject': True, 'metadata': {...}}
        """
        ...

    def reset(self) -> None:
        """Reset internal state to begin a new simulation/replication."""
        ...


@runtime_checkable
class BatchedProcedure(Procedure, Protocol):
    """Optional vectorized interface for simulators that operate on arrays."""

    def reset_batch(self, n_simulations: int) -> None:
        """Prepare internal state for a batch of `n_simulations` replications."""
        ...

    def ingest_batch(
        self,
        cumulative: Dict[str, np.ndarray],
        *,
        active_mask: np.ndarray,
    ) -> None:
        """Ingest batched cumulative data for all simulations at once."""
        ...

    def should_stop_batch(
        self,
        look: int,
        *,
        active_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (stop_mask, reject_mask) for the current batch."""
        ...


@runtime_checkable
class Simulator(Protocol):
    """Protocol for a simulator that drives data generation only.

    The simulator should not encode analysis timing or stopping logic.
    Instead it produces data batches which are passed to the Procedure's
    `ingest` method. The Procedure is authoritative about boundaries and
    stopping decisions.

    Null-hypothesis/baseline parameters are part of the simulation
    scenario and must be supplied by the caller to the simulator
    (for example `p_control`).
    """

    def simulate(
        self,
        procedure: Procedure,
        *,
        requests: Sequence[Any],
        rng_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Sequence[OCPointResult]:
        """Run simulations for the given procedure.

        Parameters
        - requests: scheme-specific request objects describing each scenario to run
        - rng_seed: optional RNG seed for reproducibility of this call
        - kwargs: additional scheme-specific parameters shared across requests
        """
        ...


def compute_oc_curve(
    simulator: Simulator,
    procedure: Procedure,
    *,
    requests: Sequence[Any],
    rng_seed: Optional[int] = None,
    simulator_kwargs: Optional[Dict[str, Any]] = None,
) -> List[OCPointResult]:
    """Delegate to ``simulator.simulate`` for a prepared list of requests."""

    base_kwargs = dict(simulator_kwargs or {})
    results = simulator.simulate(
        procedure,
        requests=requests,
        rng_seed=rng_seed,
        **base_kwargs,
    )
    return list(results)
