from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    OCPointResult,
    Procedure,
)


def compute_cumulative_sample_sizes(
    info_times: Sequence[float], planned_max_n: int
) -> List[int]:
    """Return cumulative sample-size targets using ceiling rounding.

    The schedule uses ``ceil(planned_max_n * info_time)`` for each information
    fraction and enforces monotonic growth with the final element equal to
    ``planned_max_n``. This avoids undershooting the planned maxima when the
    optimiser emits fractions that would otherwise round down.
    """

    max_total = int(planned_max_n)
    if max_total <= 0:
        raise ValueError("planned_max_n must be positive")

    rates = [float(rate) for rate in info_times]
    if not rates:
        raise ValueError("info_times must be non-empty")

    cumulative: List[int] = []
    previous = 0
    for idx, rate in enumerate(rates):
        if rate <= 0:
            target = previous
        else:
            target = int(ceil(max_total * rate))
        if idx == len(rates) - 1:
            target = max_total
        target = max(previous, min(target, max_total))
        cumulative.append(target)
        previous = target

    cumulative[-1] = max_total
    return cumulative


def _build_sampling_plan(
    cumulative_totals: Sequence[int], allocation_ratio: float
) -> List[Tuple[int, int]]:
    """Derive per-look sample increments from cumulative totals."""

    alloc = float(allocation_ratio)
    if alloc <= 0.0:
        raise ValueError("allocation_ratio must be positive")

    plan: List[Tuple[int, int]] = []
    prev_a = 0
    prev_b = 0
    for total in cumulative_totals:
        total_int = int(total)
        if total_int < 0:
            raise ValueError("cumulative totals must be non-negative")
        target_a = int(ceil(total_int / (1.0 + alloc)))
        target_b = total_int - target_a
        inc_a = target_a - prev_a
        inc_b = target_b - prev_b
        if inc_a < 0 or inc_b < 0:
            raise ValueError("cumulative totals must be non-decreasing")
        if inc_a > 0 or inc_b > 0:
            plan.append((inc_a, inc_b))
        prev_a = target_a
        prev_b = target_b

    return plan


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
    batch_size: Optional[int] = None
    allocation_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
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
        info_times: Optional[Sequence[float]] = None,
        cumulative_sizes: Optional[Sequence[int]] = None,
    ) -> OCPointResult:
        """Run Monte-Carlo replications and return operating characteristics.

        The implementation is intentionally compact: each replication runs
        incremental batches until ``max_total`` is reached or ``procedure``
        requests to stop. When ``batch_size`` is ``None`` the caller must
        provide either ``info_times`` or ``cumulative_sizes`` so that the
        simulator can derive an efficient sampling schedule aligned with the
        design information fractions.
        """

        n_sim = (
            int(n_simulations) if n_simulations is not None else int(self.n_simulations)
        )
        effect = (
            float(effect_size) if effect_size is not None else float(self.effect_size)
        )
        rng = np.random.default_rng(rng_seed)

        schedule: Optional[Sequence[Tuple[int, int]]] = None
        if self.batch_size is None:
            if cumulative_sizes is not None:
                schedule = _build_sampling_plan(cumulative_sizes, self.allocation_ratio)
            elif info_times is not None and max_total is not None:
                cumulative_sizes = compute_cumulative_sample_sizes(
                    info_times, max_total
                )
                schedule = _build_sampling_plan(cumulative_sizes, self.allocation_ratio)
            else:
                raise ValueError(
                    "Provide info_times or cumulative_sizes when batch_size is None"
                )

        if max_total is None:
            if self.batch_size is None:
                raise ValueError("max_total is required when batch_size is None")
            max_total = int(
                self.batch_size + round(self.batch_size * self.allocation_ratio)
            )
        else:
            max_total = int(max_total)

        # stop_counts: map actual total sample size at stopping -> count
        stop_counts: Dict[int, int] = {}
        rejections = 0
        total_sample_sizes: List[int] = []

        for _ in range(n_sim):
            procedure.reset()

            totals: Dict[str, int] = {"cum_nA": 0, "cum_nB": 0, "look_idx": 0}
            gen = _sampler_gen(
                batch_size=self.batch_size,
                allocation_ratio=float(self.allocation_ratio),
                pA=float(p_control),
                pB=float(p_control + effect),
                rng=rng,
                max_total=max_total,
                totals=totals,
                schedule=schedule,
            )

            stopped = False
            for data in gen:
                procedure.ingest(data)
                decision = procedure.should_stop(totals["look_idx"])
                if decision is not None:
                    # Use the actual cumulative total sample size as the key
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

        return OCPointResult(
            effect_size=effect,
            expected_sample_size=expected_sample_size,
            power=power,
            max_sample_size=float(int(max_total)),
            stop_distribution=stop_counts,
            metadata={
                "n_simulations": int(n_sim),
                "batch_size": (
                    int(self.batch_size) if self.batch_size is not None else None
                ),
                "allocation_ratio": float(self.allocation_ratio),
            },
        )
