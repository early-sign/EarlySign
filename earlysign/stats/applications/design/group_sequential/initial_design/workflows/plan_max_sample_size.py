"""
Plan GST sample sizes that maintain power at the design alternative.

This module exposes two reusable helpers:

``PlanMaxSampleSizeWorkflow``
    Binary-searches the minimal planned maximum sample size whose Monte-Carlo
    power estimate falls inside ``[target_power, target_power + tolerance]``.

``MonteCarloPowerEstimator``
    Adapter that turns the usual design-building callbacks (build context,
    payload, procedure, simulator request) into an ``estimate_power`` callable
    suitable for the workflow.

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

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


@dataclass
class PlanMaxSampleSizeWorkflow:
    """
    Iteratively select the smallest planned maximum sample size that preserves power.

    Jennison & Turnbull (2000, Sec. 3) show that, under the canonical joint
    distribution and simple spending functions, one can express power as a smooth,
    monotone function of the final information level.  In practice we frequently
    couple the GST machinery with ledger-driven sampling strategies, covariate
    adjustments, or simulation-based test statistics that break those
    assumptions.  Because of that, we favor a generic monotone search over
    "closed-form" updates: it works for all the bespoke procedures we support and
    it produces the same result as the textbook shortcut whenever the shortcut
    is valid.
    """

    estimate_power: Callable[[Sequence[float], int], float]
    target_power: float
    tolerance: float
    max_multiplier: int

    def search(self, *, info_times: Sequence[float], k: int, fsd_total: int) -> int:
        """
        Binary-search the minimal ``planned_max_n`` that meets the power target.

        Parameters
        ----------
        info_times:
            Normalized information times for the design candidate.  They are
            forwarded to ``estimate_power`` so it can instantiate the right
            procedure.
        k:
            Number of analyses (interim + final).  We use it to set the smallest
            feasible ``planned_max_n``—the textbook lower bound is two patients
            per stage, i.e., ``2 * k`` observations.
        fsd_total:
            Reference sample size from the fixed-sample design.  We only use it
            to cap the initial bracketing interval; ``max_multiplier`` controls
            how aggressive that upper bound can be.

        Returns
        -------
        int
            The minimal ``planned_max_n`` whose estimated power falls within
            ``[target_power, target_power + tolerance]``.  If the interval is
            never hit we return the best feasible value explored, mirroring the
            usual GST planning heuristics.
        """
        # The lower bracket uses the "two participants per look" heuristic so we
        # never ask the estimator for degenerate designs.  The upper bracket
        # inflates the fixed-sample size by ``max_multiplier`` so that users can
        # widen the search when Monte-Carlo noise makes the power curve flatter.
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
                    # Accept the first sample size that lands inside the
                    # tolerance band—monotonicity ensures that further
                    # candidates will only overshoot power by a wider margin.
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


@dataclass
class MonteCarloPowerEstimator:
    """
    Estimate GST power via simulation for a candidate design.

    Parameters
    ----------
    build_design_context:
        Callable returning an object with ``info_times``, ``planned_max_n`` and
        ``sampling_plan`` attributes for the candidate design.
    build_design_payload:
        Callable that produces the payload consumed by the procedure factory.
    make_procedure:
        Callable that instantiates a simulator-compatible procedure.
    build_simulation_request:
        Callable that builds the simulator request for a given effect size.
    simulator:
        Simulator exposing ``simulate(procedure, requests, rng_seed)``.
    target_effect:
        Design effect size for which power should be estimated.

    Example
    -------
    >>> class _Context:
    ...     def __init__(self, n):
    ...         self.info_times = [0.5, 1.0]
    ...         self.planned_max_n = n
    ...         self.sampling_plan = type(
    ...             \"_Plan\", (), {\"strategy\": object(), \"metadata\": {}}
    ...         )()
    >>> estimator = MonteCarloPowerEstimator(
    ...     build_design_context=lambda info, n: _Context(n),
    ...     build_design_payload=lambda info, n: {},
    ...     make_procedure=lambda info, n, payload: type(
    ...         \"_Proc\", (), {\"reset\": lambda self: None}
    ...     )(),
    ...     build_simulation_request=lambda effect, n, strat: {},
    ...     simulator=type(
    ...         \"_Sim\", (), {\"simulate\": lambda self, proc, requests, rng_seed: [type(\"_Point\", (), {\"power\": 0.8, \"metadata\": {}})()]}
    ...     )(),
    ...     target_effect=0.2,
    ... )
    >>> round(estimator([0.5, 1.0], 80), 2)
    0.8
    """

    build_design_context: Callable[[Sequence[float], int], Any]
    build_design_payload: Callable[[Sequence[float], int], Mapping[str, Any]]
    make_procedure: Callable[[Sequence[float], int, Mapping[str, Any]], Any]
    build_simulation_request: Callable[[float, int, Any], Any]
    simulator: Any
    target_effect: float
    seed: Optional[int] = None
    logger: logging.Logger = logger

    def __call__(self, info_times: Sequence[float], planned_max_n: int) -> float:
        context = self.build_design_context(info_times, planned_max_n)
        payload = self.build_design_payload(context.info_times, context.planned_max_n)
        procedure = self.make_procedure(
            context.info_times, context.planned_max_n, payload
        )
        procedure.reset()

        sampling_plan = context.sampling_plan
        meta_before = getattr(sampling_plan, "metadata", None)
        self.logger.info(
            "Estimating power (planned_max_n=%s, info_times=%s, sampling_meta=%s)",
            context.planned_max_n,
            context.info_times,
            meta_before,
        )

        request = self.build_simulation_request(
            float(self.target_effect),
            context.planned_max_n,
            sampling_plan.strategy,
        )
        points = self.simulator.simulate(
            procedure,
            requests=[request],
            rng_seed=self.seed,
        )
        if not points:
            raise RuntimeError("Simulator returned no results during power estimation")

        point = points[0]
        power_value = float(point.power)
        self.logger.info(
            "Estimated power (planned_max_n=%s) -> %s",
            context.planned_max_n,
            power_value,
        )
        return power_value
