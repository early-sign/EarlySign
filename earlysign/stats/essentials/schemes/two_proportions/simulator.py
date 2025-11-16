from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    BatchedProcedure,
    OCPointResult,
    Procedure,
)
from earlysign.stats.essentials.methods.group_sequential.simulation import (
    SamplingStrategy,
)


@dataclass(frozen=True)
class TwoProportionsSimulationRequest:
    """Specification for a single two-proportions simulation scenario."""

    p_control: float
    effect_size: float
    n_simulations: int
    max_total: int
    sampling: SamplingStrategy


@dataclass
class TwoProportionsSimulator:
    """Concise Monte-Carlo simulator for two-proportions group-sequential trials.

    This preserves the public contract: the passed-in ``procedure`` must
    implement ``reset()``, ``ingest(cumulative_dict)`` and
    ``should_stop(look_index)`` and the simulator only drives data to it.
    """

    effect_size: float
    n_simulations: int = 200
    allocation_ratio: float = 1.0
    strategy: Optional[SamplingStrategy] = None

    def __post_init__(self) -> None:
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if self.allocation_ratio <= 0.0:
            raise ValueError("allocation_ratio must be positive")
        if not np.isfinite(self.effect_size):
            raise ValueError("effect_size must be finite")

    def simulate(
        self,
        procedure: Procedure,
        *,
        requests: Optional[Sequence[TwoProportionsSimulationRequest]] = None,
        rng_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Run one or more simulation scenarios.

        When ``requests`` is provided the method returns a sequence of
        :class:`OCPointResult` objects in the same order. When legacy
        keyword arguments (``p_control``, ``effect_size`` …) are supplied,
        the behaviour mirrors the historical API and returns a single
        :class:`OCPointResult`.
        """

        if requests is not None:
            request_list = list(requests)
            legacy_mode = False
        else:
            try:
                p_control = float(kwargs["p_control"])
            except KeyError as exc:  # pragma: no cover - defensive legacy guard
                raise TypeError(
                    "p_control is required when requests are not provided"
                ) from exc

            effect = float(kwargs.get("effect_size", self.effect_size))
            n_sim = int(kwargs.get("n_simulations", self.n_simulations))
            strategy = kwargs.get("sampling") or self.strategy
            if strategy is None:
                raise ValueError("Provide a sampling strategy to the simulator")
            max_total = int(kwargs.get("max_total", strategy.max_total))
            legacy_request = TwoProportionsSimulationRequest(
                p_control=p_control,
                effect_size=effect,
                n_simulations=n_sim,
                max_total=max_total,
                sampling=strategy,
            )
            request_list = [legacy_request]
            legacy_mode = True

        if not request_list:
            return [] if not legacy_mode else []

        request_infos: List[
            Tuple[
                TwoProportionsSimulationRequest,
                SamplingStrategy,
                Tuple[Tuple[int, int], ...],
            ]
        ] = []
        for req in request_list:
            strategy = req.sampling
            schedule = tuple(strategy.schedule)
            request_infos.append((req, strategy, schedule))
        rng = np.random.default_rng(rng_seed)

        grouped: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], List[int]] = {}
        for idx, (req, _, schedule) in enumerate(request_infos):
            key = (req.n_simulations, schedule)
            grouped.setdefault(key, []).append(idx)

        results: List[Optional[OCPointResult]] = [None] * len(request_infos)

        for (n_sim, schedule), indices in grouped.items():
            schedule_arr = np.asarray(schedule, dtype=int)
            n_looks = schedule_arr.shape[0]
            inc_a = schedule_arr[:, 0] if n_looks else np.zeros(0, dtype=int)
            inc_b = schedule_arr[:, 1] if n_looks else np.zeros(0, dtype=int)

            reqs = [request_infos[i][0] for i in indices]
            control_probs = np.asarray([req.p_control for req in reqs], dtype=float)
            treatment_probs = np.asarray(
                [req.p_control + req.effect_size for req in reqs], dtype=float
            )

            control_samples = _sample_group_counts(
                rng=rng,
                n_simulations=n_sim,
                increments=inc_a,
                probabilities=control_probs,
            )
            treatment_samples = _sample_group_counts(
                rng=rng,
                n_simulations=n_sim,
                increments=inc_b,
                probabilities=treatment_probs,
            )

            for local_idx, req_idx in enumerate(indices):
                req, strategy, _ = request_infos[req_idx]
                if isinstance(procedure, BatchedProcedure):
                    control_block = control_samples[:, local_idx, :]
                    treatment_block = treatment_samples[:, local_idx, :]
                    results[req_idx] = self._simulate_with_batched_procedure(
                        procedure=procedure,
                        control_counts=control_block,
                        treatment_counts=treatment_block,
                        schedule=schedule,
                        strategy=strategy,
                        max_total=req.max_total,
                        n_sim=req.n_simulations,
                        effect=req.effect_size,
                    )
                else:
                    control_draws = tuple(
                        control_samples[:, local_idx, look_idx]
                        for look_idx in range(n_looks)
                    )
                    treatment_draws = tuple(
                        treatment_samples[:, local_idx, look_idx]
                        for look_idx in range(n_looks)
                    )
                    results[req_idx] = self._simulate_from_counts(
                        procedure=procedure,
                        control_draws=control_draws,
                        treatment_draws=treatment_draws,
                        schedule=schedule,
                        strategy=strategy,
                        max_total=req.max_total,
                        n_sim=req.n_simulations,
                        effect=req.effect_size,
                    )

        cleaned = [res for res in results if res is not None]
        if legacy_mode:
            if not cleaned:
                raise RuntimeError("Simulator returned no results for the legacy call")
            return cleaned[0]
        return cleaned

    def _simulate_from_counts(
        self,
        *,
        procedure: Procedure,
        control_draws: Sequence[np.ndarray],
        treatment_draws: Sequence[np.ndarray],
        schedule: Sequence[Tuple[int, int]],
        strategy: SamplingStrategy,
        max_total: int,
        n_sim: int,
        effect: float,
    ) -> OCPointResult:
        stop_counts: Dict[int, int] = {}
        rejections = 0
        total_sample_sizes: List[int] = []

        for sim_idx in range(n_sim):
            procedure.reset()
            totals: Dict[str, int] = {"cum_nA": 0, "cum_nB": 0, "look_idx": 0}
            stopped = False

            for look_idx, (inc_a, inc_b) in enumerate(schedule):
                data = {
                    "nA": int(inc_a),
                    "mA": int(control_draws[look_idx][sim_idx]),
                    "nB": int(inc_b),
                    "mB": int(treatment_draws[look_idx][sim_idx]),
                }
                procedure.ingest(data)
                totals["cum_nA"] += int(inc_a)
                totals["cum_nB"] += int(inc_b)
                totals["look_idx"] += 1
                decision = procedure.should_stop(totals["look_idx"])
                if decision is not None:
                    total_n = int(totals["cum_nA"] + totals["cum_nB"])
                    stop_counts.setdefault(total_n, 0)
                    stop_counts[total_n] += 1
                    if bool(decision.get("reject", False)):
                        rejections += 1
                    total_sample_sizes.append(total_n)
                    stopped = True
                    break

            if not stopped:
                total_n = int(totals["cum_nA"] + totals["cum_nB"])
                stop_counts.setdefault(total_n, 0)
                stop_counts[total_n] += 1
                total_sample_sizes.append(total_n)

        expected_sample_size = (
            float(np.mean(total_sample_sizes)) if total_sample_sizes else 0.0
        )
        power = float(rejections) / float(max(1, n_sim))

        metadata: Dict[str, Any] = {
            "n_simulations": int(n_sim),
            "allocation_ratio": float(self.allocation_ratio),
            "sampling_strategy": strategy.description,
        }
        if strategy.batch_size is not None:
            metadata["batch_size"] = int(strategy.batch_size)

        strategy_meta = dict(strategy.metadata())
        metadata.update(strategy_meta)
        if "schedule" not in metadata:
            metadata["schedule"] = [
                (int(inc_a), int(inc_b)) for inc_a, inc_b in schedule
            ]

        return OCPointResult(
            effect_size=effect,
            expected_sample_size=expected_sample_size,
            power=power,
            max_sample_size=float(int(max_total)),
            stop_distribution=stop_counts,
            metadata=metadata,
        )

    def _simulate_with_batched_procedure(
        self,
        *,
        procedure: BatchedProcedure,
        control_counts: np.ndarray,
        treatment_counts: np.ndarray,
        schedule: Sequence[Tuple[int, int]],
        strategy: SamplingStrategy,
        max_total: int,
        n_sim: int,
        effect: float,
    ) -> OCPointResult:
        """Vectorized helper for :class:`BatchedProcedure` implementations."""

        n_simulations, n_looks = control_counts.shape
        procedure.reset_batch(n_simulations)

        active = np.ones(n_simulations, dtype=bool)
        total_n = np.zeros(n_simulations, dtype=int)
        rejections = np.zeros(n_simulations, dtype=bool)
        total_sample_sizes: List[int] = []
        stop_counts: Dict[int, int] = {}

        for look_idx, (inc_a, inc_b) in enumerate(schedule):
            if not np.any(active):
                break

            payload = {
                "nA": np.full(n_simulations, int(inc_a), dtype=int),
                "mA": control_counts[:, look_idx],
                "nB": np.full(n_simulations, int(inc_b), dtype=int),
                "mB": treatment_counts[:, look_idx],
            }
            procedure.ingest_batch(payload, active_mask=active)
            total_n[active] += int(inc_a + inc_b)

            stop_mask, reject_mask = procedure.should_stop_batch(
                look_idx + 1, active_mask=active
            )
            stop_mask = np.asarray(stop_mask, dtype=bool) & active
            if not np.any(stop_mask):
                continue

            reject_mask = np.asarray(reject_mask, dtype=bool) & stop_mask
            stopped_totals = total_n[stop_mask]
            unique_totals, counts = np.unique(stopped_totals, return_counts=True)
            for value, count in zip(unique_totals.tolist(), counts.tolist()):
                stop_counts[value] = stop_counts.get(value, 0) + count
            total_sample_sizes.extend(stopped_totals.tolist())
            rejections |= reject_mask
            active[stop_mask] = False

        if np.any(active):
            remaining_totals = total_n[active]
            unique_totals, counts = np.unique(remaining_totals, return_counts=True)
            for value, count in zip(unique_totals.tolist(), counts.tolist()):
                stop_counts[value] = stop_counts.get(value, 0) + count
            total_sample_sizes.extend(remaining_totals.tolist())

        expected_sample_size = (
            float(np.mean(total_sample_sizes)) if total_sample_sizes else 0.0
        )
        power = float(np.count_nonzero(rejections)) / float(max(1, n_sim))

        metadata: Dict[str, Any] = {
            "n_simulations": int(n_sim),
            "allocation_ratio": float(self.allocation_ratio),
            "sampling_strategy": strategy.description,
            "batched": True,
        }
        if strategy.batch_size is not None:
            metadata["batch_size"] = int(strategy.batch_size)

        strategy_meta = dict(strategy.metadata())
        metadata.update(strategy_meta)
        if "schedule" not in metadata:
            metadata["schedule"] = [
                (int(inc_a), int(inc_b)) for inc_a, inc_b in schedule
            ]

        return OCPointResult(
            effect_size=effect,
            expected_sample_size=expected_sample_size,
            power=power,
            max_sample_size=float(int(max_total)),
            stop_distribution=stop_counts,
            metadata=metadata,
        )


def _sample_group_counts(
    *,
    rng: np.random.Generator,
    n_simulations: int,
    increments: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Draw binomial counts for ``n_simulations``×``len(probabilities)`` requests.

    The function relies on NumPy broadcasting to construct the per-look
    ``n`` and ``p`` matrices before drawing all replications in one call.

    >>> counts = _sample_group_counts(
    ...     rng=np.random.default_rng(0),
    ...     n_simulations=2,
    ...     increments=np.array([1, 2]),
    ...     probabilities=np.array([0.5, 0.25]),
    ... )
    >>> counts.shape
    (2, 2, 2)
    >>> counts
    array([[[1, 1],
            [0, 0]],
           [[1, 2],
            [0, 1]]])
    """

    num_requests = int(probabilities.shape[0])
    n_looks = int(increments.shape[0])
    n_matrix = np.broadcast_to(increments, (num_requests, n_looks))
    p_matrix = np.broadcast_to(probabilities[:, None], (num_requests, n_looks))
    n_full = np.broadcast_to(n_matrix, (n_simulations, num_requests, n_looks))
    p_full = np.broadcast_to(p_matrix, (n_simulations, num_requests, n_looks))
    samples = rng.binomial(n=n_full, p=p_full)
    return samples.astype(int)
