from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    OCPointResult,
    Procedure,
)
from earlysign.stats.essentials.methods.group_sequential.simulation import (
    SamplingStrategy,
)


def _sampler_gen(
    *,
    batch_size: Optional[int],
    allocation_ratio: float,
    pA: float,
    pB: float,
    rng: np.random.Generator,
    max_total: Optional[int],
    totals: Dict[str, int],
    schedule: Optional[Sequence[Tuple[int, int]]] = None,
) -> Iterator[Dict[str, int]]:
    """Simple generator yielding per-batch data and updating `totals`.

    totals is a mutable dict the caller can read at any time; it will
    contain keys 'cum_nA', 'cum_nB' and 'look_idx'. This keeps the
    sampler implementation minimal (yield + totals) as requested.
    """

    if schedule is not None:
        for inc_a, inc_b in schedule:
            if inc_a == 0 and inc_b == 0:
                totals["look_idx"] += 1
                yield {"nA": 0, "mA": 0, "nB": 0, "mB": 0}
                continue
            drawA = int(rng.binomial(int(inc_a), pA)) if inc_a else 0
            drawB = int(rng.binomial(int(inc_b), pB)) if inc_b else 0

            totals["cum_nA"] += int(inc_a)
            totals["cum_nB"] += int(inc_b)
            totals["look_idx"] += 1

            yield {
                "nA": int(inc_a),
                "mA": drawA,
                "nB": int(inc_b),
                "mB": drawB,
            }
        return

    if batch_size is None:
        raise ValueError("batch_size must be provided when schedule is absent")

    batch_total = int(batch_size + round(batch_size * allocation_ratio))
    alloc = float(allocation_ratio)

    while True:
        # respect max_total: compute remaining samples allowed (both groups)
        if max_total is not None:
            consumed = int(totals.get("cum_nA", 0) + totals.get("cum_nB", 0))
            remaining = int(max_total) - consumed
            if remaining <= 0:
                return
            cur_batch_total = min(batch_total, remaining)
        else:
            cur_batch_total = batch_total

        # decide batch sizes (fixed per-batch; will not overshoot max_total)
        incA = int(round(cur_batch_total / (1.0 + alloc)))
        incA = max(0, min(incA, cur_batch_total))
        incB = cur_batch_total - incA
        if incA == 0 and incB == 0:
            return

        drawA = int(rng.binomial(incA, pA)) if incA else 0
        drawB = int(rng.binomial(incB, pB)) if incB else 0

        totals["cum_nA"] += incA
        totals["cum_nB"] += incB
        totals["look_idx"] += 1
        yield {"nA": incA, "mA": drawA, "nB": incB, "mB": drawB}


@dataclass
class TwoProportionsSimulator:
    """Concise Monte-Carlo simulator for two-proportions group-sequential trials.

    This preserves the public contract: the passed-in ``procedure`` must
    implement ``reset()``, ``ingest(cumulative_dict)`` and
    ``should_stop(look_index)`` and the simulator only drives data to it.
    """

    effect_size: float
    n_simulations: int = 2000
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
        p_control: float,
        effect_size: Optional[float] = None,
        n_simulations: Optional[int] = None,
        rng_seed: Optional[int] = None,
        max_total: Optional[int] = None,
        sampling: Optional[SamplingStrategy] = None,
    ) -> OCPointResult:
        effect = (
            float(effect_size) if effect_size is not None else float(self.effect_size)
        )
        results = self.simulate_many(
            procedure,
            effect_sizes=[effect],
            p_control=p_control,
            n_simulations=n_simulations,
            rng_seed=rng_seed,
            max_total=max_total,
            sampling=sampling,
        )
        return results[0]

    def simulate_many(
        self,
        procedure: Procedure,
        *,
        p_control: float,
        effect_sizes: Sequence[float],
        n_simulations: Optional[int] = None,
        rng_seed: Optional[int] = None,
        max_total: Optional[int] = None,
        sampling: Optional[SamplingStrategy] = None,
    ) -> Sequence[OCPointResult]:
        effect_list = [float(es) for es in effect_sizes]
        if not effect_list:
            return []

        n_sim = (
            int(n_simulations) if n_simulations is not None else int(self.n_simulations)
        )
        rng = np.random.default_rng(rng_seed)

        strategy = sampling or self.strategy
        if strategy is None:
            raise ValueError("Provide a sampling strategy to the simulator")

        schedule = list(strategy.schedule)
        if not schedule:
            raise ValueError("Sampling strategy produced an empty schedule")

        strategy_total = int(strategy.max_total)
        if max_total is None:
            max_total = strategy_total
        elif int(max_total) != strategy_total:
            raise ValueError(
                "max_total must match the sampling strategy's maximum total"
            )

        control_draws: List[np.ndarray] = []
        treatment_uniforms: List[Optional[np.ndarray]] = []
        for inc_a, inc_b in schedule:
            if inc_a > 0:
                control_draws.append(
                    rng.binomial(int(inc_a), float(p_control), size=n_sim).astype(int)
                )
            else:
                control_draws.append(np.zeros(n_sim, dtype=int))
            if inc_b > 0:
                treatment_uniforms.append(rng.random((n_sim, int(inc_b)), dtype=float))
            else:
                treatment_uniforms.append(None)

        treatment_counts: List[List[np.ndarray]] = [
            [np.zeros(n_sim, dtype=int) for _ in schedule] for _ in effect_list
        ]
        for look_idx, uniforms in enumerate(treatment_uniforms):
            if uniforms is None:
                continue
            for eff_idx, effect in enumerate(effect_list):
                threshold = float(np.clip(p_control + effect, 0.0, 1.0))
                treatment_counts[eff_idx][look_idx] = (
                    (uniforms < threshold).sum(axis=1).astype(int)
                )

        results: List[OCPointResult] = []
        for eff_idx, effect in enumerate(effect_list):
            count_result = self._simulate_from_counts(
                procedure=procedure,
                control_draws=control_draws,
                treatment_draws=treatment_counts[eff_idx],
                schedule=schedule,
                strategy=strategy,
                max_total=int(max_total),
                n_sim=n_sim,
                effect=effect,
            )
            results.append(count_result)
        return results

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
