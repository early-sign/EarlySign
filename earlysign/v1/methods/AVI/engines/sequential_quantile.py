import math
from typing import List, Optional, Tuple

from earlysign.schema.ES3.AVI import (
    Protocol,
    SequentialQuantileMethodSpec,
)
from earlysign.schema.ES3.AVI.Log import DecisionStatus, SequentialQuantileLookResult
from earlysign.schema.ES3.SequentialQuantile import Scoreboard
from earlysign.v1.framework.trace import TraceId


class SequentialQuantileEngine:
    """
    Engine for Howard & Ramdas (2022) Sequential Quantile estimation.
    This engine is stateless with respect to raw observations. it expects
    pre-calculated order statistics (CI bounds) in its metrics.
    """

    def __init__(self, protocol: Protocol):
        self.protocol = protocol

    @staticmethod
    def get_confidence_interval_ranks(
        n: int, method: SequentialQuantileMethodSpec
    ) -> Tuple[int, int]:
        """
        Calculate the 1-indexed ranks (L_t, U_t) for the confidence interval.
        Based on Corollary 2 of Howard & Ramdas (2022).

        Args:
            n: sample size
            method: the method specification containing quantile (q) and alpha.

        Returns:
            Tuple of (Lower Rank, Upper Rank)
        """
        if n == 0:
            return 0, 0

        q = method.quantile
        alpha = method.alpha

        # Formula: epsilon_t = 0.85 * t^(-1) * (log log(e t) + C)
        # C = log(1612 / alpha) / 1.25
        c = math.log(1612 / alpha) / 1.25
        log_log_et = math.log(math.log(math.e * n))
        epsilon_t = 0.85 * (log_log_et + c) / n

        # Howard & Ramdas Corollary 2:
        # Confidence interval is [X_(L_t), X_(U_t)] where
        # L_t = floor(n(q - epsilon_t)) + 1  (Adjusted for 1-indexing and conservative bound)
        # U_t = ceil(n(q + epsilon_t))
        # Note: We clamp to [1, n].

        l_rank = max(1, math.floor(n * (q - epsilon_t)) + 1)
        u_rank = min(n, math.ceil(n * (q + epsilon_t)))

        return l_rank, u_rank

    def run(
        self, metrics: Scoreboard, trace: Optional[List[TraceId]] = None
    ) -> SequentialQuantileLookResult:
        """
        Evaluate the stopping condition based on injected metrics.
        Returns a result indicating STOP_EFFICACY if bounds are disjoint.
        """
        method = self.protocol.method
        if not isinstance(method, SequentialQuantileMethodSpec):
            raise TypeError("Expected SequentialQuantileMethodSpec")

        # In a typical A/B test setup in this framework:
        # Arm 0 is Control, Arm 1 is Treatment (or generic key based)
        arm_ids = sorted(metrics.arms.keys())
        if len(arm_ids) < 2:
            # Need at least two arms for A/B check
            return SequentialQuantileLookResult(
                sample_n=0,
                estimated_quantile=0.0,
                interval_lower=0.0,
                interval_upper=0.0,
                status=DecisionStatus.CONTINUE_,
            )

        # Basic logic: compare the first two arms
        ctrl_id = arm_ids[0]
        treat_id = arm_ids[1]

        ctrl = metrics.arms[ctrl_id].metrics
        treat = metrics.arms[treat_id].metrics

        # Stopping Condition: Disjoint confidence intervals
        # [L_c, U_c] and [L_t, U_t] do not overlap
        # i.e., U_c < L_t or U_t < L_c
        is_disjoint = (ctrl.ci_upper < treat.ci_lower) or (
            treat.ci_upper < ctrl.ci_lower
        )

        status = DecisionStatus.CONTINUE_
        if is_disjoint:
            status = DecisionStatus.STOP_EFFICACY

        # Check for max_n (horizon) if defined
        total_n = ctrl.n + treat.n
        if (
            method.max_n
            and total_n >= method.max_n
            and status == DecisionStatus.CONTINUE_
        ):
            status = DecisionStatus.STOP_PLAN_END_REACHED

        # For the result record, we provide the estimate from the 'Treatment' arm (convention)
        return SequentialQuantileLookResult(
            sample_n=total_n,
            estimated_quantile=treat.quantile_estimate,
            interval_lower=treat.ci_lower,
            interval_upper=treat.ci_upper,
            status=status,
        )
