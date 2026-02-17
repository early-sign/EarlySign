from typing import Any

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.schema.ES3.Binomial import (
    ArmMetrics as BinomialArmMetrics,
    ArmStatus as BinomialArmStatus,
    Scoreboard as BinomialScoreboard,
)
from earlysign.schema.ES3.Continuous import (
    ArmMetrics as ContinuousArmMetrics,
    ArmStatus as ContinuousArmStatus,
    Scoreboard as ContinuousScoreboard,
)
from earlysign.schema.ES3.YEAST import Protocol
from earlysign.schema.ES3.YEAST.Log import (
    Boundary as BoundarySchema,
    DecisionStatus,
    LookResult,
)


class BinomialYEASTEngine:
    """
    Orchestrator for Binomial YEAST execution.

    Calculates the trajectory (difference in successes) and compares it against
    the pre-calculated fixed boundary.
    """

    def __init__(self, protocol: Protocol):
        self.protocol = protocol

    def run(
        self,
        metrics: BinomialScoreboard,
        boundary: BoundarySchema,
        **kwargs: Any,
    ) -> LookResult:
        """
        Computes the test result given current summary statistics.

        Args:
            metrics: Scoreboard containing the aggregated metrics for all arms.
            boundary: The current boundary schema containing the threshold value.

        Returns:
            LookResult containing the trajectory, boundaries, and status.
        """
        # Extract arm names
        arms_struct = self.protocol.task.arms
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            control_key = arms_struct.control_arm_name
            treatment_key = arms_struct.treatment_arm_name
        else:
            raise ValueError(
                f"YEAST engine requires TwoArmComparison, but got {type(arms_struct).__name__}."
            )

        default_arm = BinomialArmStatus(
            metrics=BinomialArmMetrics(total=0, successes=0, p_hat=0.0), is_active=True
        )
        summary_c = metrics.arms.get(control_key, default_arm).metrics
        summary_t = metrics.arms.get(treatment_key, default_arm).metrics

        cumulative_n = summary_c.total + summary_t.total
        raw_diff = float(summary_t.successes - summary_c.successes)

        # Standardized trajectory: S_n / sqrt(n_effective)
        # where n_effective = 2 / (1/n_c + 1/n_t)
        # For equal n, n_effective = n_per_arm.
        if summary_c.total > 0 and summary_t.total > 0:
            n_eff = 2.0 / (1.0 / summary_c.total + 1.0 / summary_t.total)
            trajectory = raw_diff / (n_eff**0.5)
        else:
            trajectory = 0.0

        # Use the passed boundary value
        boundary_val = boundary.value
        is_crossed = False
        if boundary_val is not None:
            is_crossed = trajectory > boundary_val

        status = DecisionStatus.CONTINUE_
        if is_crossed:
            status = DecisionStatus.STOP_EFFICACY

        # Check for max N
        if hasattr(self.protocol.method, "expected_num_observations"):
            if cumulative_n >= self.protocol.method.expected_num_observations:
                if status == DecisionStatus.CONTINUE_:
                    status = DecisionStatus.STOP_PLAN_END_REACHED

        return LookResult(
            sample_n=cumulative_n,
            info_frac=0.0,  # Placeholder
            trajectory=trajectory,
            raw_difference=raw_diff,
            efficacy_boundary=boundary_val,
            is_efficacy_crossed=is_crossed,
            status=status,
        )


class ContinuousYEASTEngine:
    """
    Orchestrator for Continuous YEAST execution.

    Calculates the trajectory (difference in sums) and compares it against
    the pre-calculated fixed boundary.
    """

    def __init__(self, protocol: Protocol):
        self.protocol = protocol

    def run(
        self,
        metrics: ContinuousScoreboard,
        boundary: BoundarySchema,
        **kwargs: Any,
    ) -> LookResult:
        """
        Computes the test result given current summary statistics.
        """
        arms_struct = self.protocol.task.arms
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            control_key = arms_struct.control_arm_name
            treatment_key = arms_struct.treatment_arm_name
        else:
            raise ValueError(
                f"YEAST engine requires TwoArmComparison, but got {type(arms_struct).__name__}."
            )

        default_arm = ContinuousArmStatus(
            metrics=ContinuousArmMetrics(n=0, mean=0.0, variance=0.0), is_active=True
        )
        summary_c = metrics.arms.get(control_key, default_arm).metrics
        summary_t = metrics.arms.get(treatment_key, default_arm).metrics

        cumulative_n = summary_c.n + summary_t.n
        raw_diff = (summary_t.mean * summary_t.n) - (summary_c.mean * summary_c.n)

        if summary_c.n > 0 and summary_t.n > 0:
            n_eff = 2.0 / (1.0 / summary_c.n + 1.0 / summary_t.n)
            trajectory = raw_diff / (n_eff**0.5)
        else:
            trajectory = 0.0

        boundary_val = boundary.value
        is_crossed = False
        if boundary_val is not None:
            is_crossed = trajectory > boundary_val

        status = DecisionStatus.CONTINUE_
        if is_crossed:
            status = DecisionStatus.STOP_EFFICACY

        if hasattr(self.protocol.method, "expected_num_observations"):
            if cumulative_n >= self.protocol.method.expected_num_observations:
                if status == DecisionStatus.CONTINUE_:
                    status = DecisionStatus.STOP_PLAN_END_REACHED

        return LookResult(
            sample_n=cumulative_n,
            info_frac=0.0,
            trajectory=float(trajectory),
            raw_difference=float(raw_diff),
            efficacy_boundary=boundary_val,
            is_efficacy_crossed=is_crossed,
            status=status,
        )
