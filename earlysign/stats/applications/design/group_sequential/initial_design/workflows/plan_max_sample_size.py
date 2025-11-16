"""
Plan GST sample sizes that maintain power at the design alternative.

The workflow performs a bounded binary search between ``2 * k`` and
``max_multiplier * fsd_total``. At each iteration it calls an ``estimate_power``
callback (typically the Monte-Carlo simulator) and accepts the first candidate
whose power falls inside ``[target_power, target_power + tolerance]``.

Example
-------
>>> wf = PlanMaxSampleSizeWorkflow(
...     estimate_power=lambda info, n: min(0.5 + n / 200.0, 0.95),
...     target_power=0.8,
...     tolerance=0.01,
...     max_multiplier=4,
... )
>>> wf.search(info_times=[0.5, 1.0], k=2, fsd_total=40)
62
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Sequence

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


@dataclass
class PlanMaxSampleSizeWorkflow:
    """Compute the minimal GST budget that preserves power at the design H1."""

    estimate_power: Callable[[Sequence[float], int], float]
    target_power: float
    tolerance: float
    max_multiplier: int

    def search(self, *, info_times: Sequence[float], k: int, fsd_total: int) -> int:
        lo = 2 * int(k)
        hi = max(2 * int(k), int(self.max_multiplier * fsd_total))
        best_n = hi
        progress_active = tqdm is not None and logger.isEnabledFor(logging.INFO)
        search_bar = (
            tqdm(
                total=None,
                desc=f"Power search (k={k})",
                unit="candidate",
                leave=False,
            )
            if progress_active
            else None
        )
        log_context = (
            logging_redirect_tqdm(loggers=[logger])
            if progress_active
            else nullcontext()
        )
        try:
            with log_context:
                while lo <= hi:
                    if search_bar is not None:
                        search_bar.update()
                    mid = (lo + hi) // 2
                    achieved = self.estimate_power(info_times, int(mid))
                    if (
                        self.target_power
                        <= achieved
                        <= self.target_power + self.tolerance
                    ):
                        best_n = int(mid)
                        logger.info(
                            (
                                "Accepting planned_max_n=%s with achieved power=%s "
                                "within [%s, %s] (early-stop)"
                            ),
                            int(mid),
                            achieved,
                            self.target_power,
                            self.target_power + self.tolerance,
                        )
                        break
                    if achieved < self.target_power:
                        lo = mid + 1
                    else:
                        hi = mid - 1
        finally:
            if search_bar is not None:
                search_bar.close()

        return int(best_n)
