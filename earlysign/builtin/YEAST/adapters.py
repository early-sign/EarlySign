from typing import Tuple

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


class BinomialAdapter:
    @staticmethod
    def extract_stats(
        metrics: BinomialScoreboard, control_key: str, treatment_key: str
    ) -> Tuple[int, int, int, int]:
        """
        Extracts (n_c, n_t, successes_c, successes_t) from BinomialScoreboard.
        """
        default_arm = BinomialArmStatus(
            arm_name="default",
            metrics=BinomialArmMetrics(total=0, successes=0, p_hat=0.0),
            is_active=True,
        )
        arms = metrics.arms or {}
        summary_c = arms.get(control_key, default_arm).metrics
        summary_t = arms.get(treatment_key, default_arm).metrics

        return (
            summary_c.total or 0,
            summary_t.total or 0,
            summary_c.successes or 0,
            summary_t.successes or 0,
        )


class ContinuousAdapter:
    @staticmethod
    def extract_stats(
        metrics: ContinuousScoreboard, control_key: str, treatment_key: str
    ) -> Tuple[int, int, float, float]:
        """
        Extracts (n_c, n_t, mean_c, mean_t) from ContinuousScoreboard.
        """
        default_arm = ContinuousArmStatus(
            arm_name="default",
            metrics=ContinuousArmMetrics(total=0, mean=0.0, variance=0.0),
            is_active=True,
        )
        arms = metrics.arms or {}
        summary_c = arms.get(control_key, default_arm).metrics
        summary_t = arms.get(treatment_key, default_arm).metrics

        c_tot = summary_c.total
        t_tot = summary_t.total
        c_mean = summary_c.mean
        t_mean = summary_t.mean

        return (
            int(c_tot or 0),
            int(t_tot or 0),
            float(c_mean or 0.0),
            float(t_mean or 0.0),
        )
