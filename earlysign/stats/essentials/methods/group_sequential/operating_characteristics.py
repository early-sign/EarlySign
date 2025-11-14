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
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

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
        effect_size: Optional[float] = None,
        n_simulations: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> OCPointResult:
        """Run simulations for the given procedure.

        Parameters
        - effect_size: effect size to simulate (if simulator has a default it may be overridden)
        - n_simulations: overrides simulator's default number of simulations
        - rng_seed: optional RNG seed for reproducibility of this call
        """
        ...


@runtime_checkable
class BatchSimulator(Simulator, Protocol):
    """Optional extension that can evaluate multiple effect sizes in one call."""

    def simulate_many(
        self,
        procedure: Procedure,
        effect_sizes: Sequence[float],
        n_simulations: Optional[int] = None,
        rng_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Sequence[OCPointResult]: ...


def compute_oc_curve(
    simulator: Simulator,
    procedure: Procedure,
    effect_sizes: Sequence[float],
    n_simulations: int = 2000,
    simulator_kwargs: Optional[Dict[str, Any]] = None,
) -> List[OCPointResult]:
    """Compute operating characteristics across effect sizes.

    This function simply iterates over `effect_sizes`, delegates the
    simulation work to the provided `simulator` by passing the same
    `procedure`, and collects the returned per-effect results into an
    `OCCurveResult`.

    Args:
        simulator: an object implementing the Simulator protocol
        procedure: statistical procedure object to evaluate
        effect_sizes: sequence of effect sizes to evaluate
        n_simulations: number of Monte Carlo replications per effect size

    Returns:
        OCCurveResult
    """

    # Defensive copy to numpy array for convenience
    effect_sizes_arr = np.asarray(effect_sizes, dtype=float)

    results: List[OCPointResult] = []

    base_kwargs = dict(simulator_kwargs or {})
    base_kwargs.setdefault("n_simulations", int(n_simulations))

    if isinstance(simulator, BatchSimulator):
        batch_points = simulator.simulate_many(
            procedure,
            [float(es) for es in effect_sizes_arr],
            **base_kwargs,
        )
        for es, point in zip(effect_sizes_arr, batch_points):
            if not np.isclose(point.effect_size, float(es)):
                point.effect_size = float(es)
            results.append(point)
        return results

    for es in effect_sizes_arr:
        call_kwargs = dict(base_kwargs)
        call_kwargs["effect_size"] = float(es)

        point = simulator.simulate(procedure, **call_kwargs)

        if not np.isclose(point.effect_size, float(es)):
            point.effect_size = float(es)

        results.append(point)

    return results
