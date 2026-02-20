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
            metrics=BinomialArmMetrics(total=0, successes=0, p_hat=0.0), is_active=True
        )
        summary_c = metrics.arms.get(control_key, default_arm).metrics
        summary_t = metrics.arms.get(treatment_key, default_arm).metrics

        return (
            summary_c.total,
            summary_t.total,
            summary_c.successes,
            summary_t.successes,
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
            metrics=ContinuousArmMetrics(total=0, mean=0.0, variance=0.0),
            is_active=True,
        )
        summary_c = metrics.arms.get(control_key, default_arm).metrics
        summary_t = metrics.arms.get(treatment_key, default_arm).metrics

        return summary_c.total, summary_t.total, summary_c.mean, summary_t.mean
