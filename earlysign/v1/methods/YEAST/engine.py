from typing import Any

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
        arms = self.protocol.task.arms
        if len(arms) < 2:
            raise ValueError("Protocol must define at least 2 arms.")

        control_key = arms[0]
        treatment_key = arms[1]

        default_arm = BinomialArmStatus(
            metrics=BinomialArmMetrics(n=0, successes=0, p_hat=0.0), is_active=True
        )
        summary_c = metrics.arms.get(control_key, default_arm).metrics
        summary_t = metrics.arms.get(treatment_key, default_arm).metrics

        cumulative_n = summary_c.n + summary_t.n

        # Trajectory: Difference in successes (S_n)
        trajectory = float(summary_t.successes - summary_c.successes)

        # Use the passed boundary value
        boundary_val = boundary.value
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
        arms = self.protocol.task.arms
        if len(arms) < 2:
            raise ValueError("Protocol must define at least 2 arms.")

        control_key = arms[0]
        treatment_key = arms[1]

        default_arm = ContinuousArmStatus(
            metrics=ContinuousArmMetrics(n=0, mean=0.0, variance=0.0), is_active=True
        )
        summary_c = metrics.arms.get(control_key, default_arm).metrics
        summary_t = metrics.arms.get(treatment_key, default_arm).metrics

        cumulative_n = summary_c.n + summary_t.n

        # Trajectory: Difference in sums (S_n)
        # S_n = sum(X_t) - sum(X_c)
        trajectory = (summary_t.mean * summary_t.n) - (summary_c.mean * summary_c.n)

        boundary_val = boundary.value
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
            efficacy_boundary=boundary_val,
            is_efficacy_crossed=is_crossed,
            status=status,
        )
