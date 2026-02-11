import math
from typing import List, Optional, Tuple

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.schema.ES3.AVI import (
    Protocol,
    SequentialQuantileMethodSpec,
)
from earlysign.schema.ES3.AVI.Log import DecisionStatus, SequentialQuantileLookResult
from earlysign.schema.ES3.SequentialQuantile import Scoreboard
from earlysign.v1.framework.trace import TraceId


class SequentialQuantileEngine:
    """Engine for Howard & Ramdas (2022) Sequential Quantile estimation.

    This engine is stateless with respect to raw observations. It expects
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
        """Evaluate the stopping condition based on injected metrics.

        Args:
            metrics: The scoreboard containing pre-calculated CI bounds.
            trace: Optional parent traces.

        Returns:
            A result indicating whether the estimation has converged (disjoint bounds).
        """
        method = self.protocol.method
        if not isinstance(method, SequentialQuantileMethodSpec):
            raise TypeError("Expected SequentialQuantileMethodSpec")

        arms_struct = self.protocol.task.arms
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            ctrl_id = arms_struct.control_arm_name
            treat_id = arms_struct.treatment_arm_name
        else:
            raise ValueError(
                f"SequentialQuantileEngine requires a TwoArmComparison arm structure, but got {type(arms_struct).__name__}."
            )

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
