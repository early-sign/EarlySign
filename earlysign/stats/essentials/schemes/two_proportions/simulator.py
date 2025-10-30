from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np

from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    OCPointResult,
    Procedure,
)


def _sampler_gen(
    *,
    batch_size: int,
    allocation_ratio: float,
    pA: float,
    pB: float,
    rng: np.random.Generator,
    max_total: Optional[int],
    totals: Dict[str, int],
) -> Iterator[Dict[str, int]]:
    """Simple generator yielding per-batch data and updating `totals`.

    totals is a mutable dict the caller can read at any time; it will
    contain keys 'cum_nA', 'cum_nB' and 'look_idx'. This keeps the
    sampler implementation minimal (yield + totals) as requested.
    """

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
    batch_size: int = 100
    allocation_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if self.batch_size <= 0:
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
    ) -> OCPointResult:
        """Run Monte-Carlo replications and return operating characteristics.

        The implementation is intentionally compact: each replication runs
        incremental batches until ``max_total`` is reached or ``procedure``
        requests to stop.
        """

        n_sim = (
            int(n_simulations) if n_simulations is not None else int(self.n_simulations)
        )
        effect = (
            float(effect_size) if effect_size is not None else float(self.effect_size)
        )
        rng = np.random.default_rng(rng_seed)

        if max_total is None:
            max_total = int(
                self.batch_size + round(self.batch_size * self.allocation_ratio)
            )

        stop_counts: Dict[int, int] = {}
        rejections = 0
        total_sample_sizes: List[int] = []

        for _ in range(n_sim):
            procedure.reset()

            totals: Dict[str, int] = {"cum_nA": 0, "cum_nB": 0, "look_idx": 0}
            gen = _sampler_gen(
                batch_size=int(self.batch_size),
                allocation_ratio=float(self.allocation_ratio),
                pA=float(p_control),
                pB=float(p_control + effect),
                rng=rng,
                max_total=max_total,
                totals=totals,
            )

            stopped = False
            for data in gen:
                procedure.ingest(data)
                decision = procedure.should_stop(totals["look_idx"])
                if decision is not None:
                    idx = totals["look_idx"] if totals["look_idx"] > 0 else 1
                    stop_counts.setdefault(idx, 0)
                    stop_counts[idx] += 1
                    if bool(decision.get("reject", False)):
                        rejections += 1
                    total_sample_sizes.append(int(totals["cum_nA"] + totals["cum_nB"]))
                    stopped = True
                    break

            if not stopped:
                idx = totals["look_idx"] if totals["look_idx"] > 0 else 1
                stop_counts.setdefault(idx, 0)
                stop_counts[idx] += 1
                total_sample_sizes.append(int(totals["cum_nA"] + totals["cum_nB"]))

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
                "batch_size": int(self.batch_size),
                "allocation_ratio": float(self.allocation_ratio),
            },
        )
