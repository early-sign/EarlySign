from typing import Tuple

from earlysign.schema.ES3.Binomial import (
    Scoreboard as BinomialScoreboard,
)
from earlysign.schema.ES3.Continuous import (
    Scoreboard as ContinuousScoreboard,
)


class BinomialAdapter:
    @staticmethod
    def extract_stats(
        metrics: BinomialScoreboard, control_key: str, treatment_key: str
    ) -> Tuple[int, int, float, float]:
        """
        Extracts (n_c, n_t, p_hat_c, p_hat_t)
        """
        s_c = metrics.arms.get(control_key)
        s_t = metrics.arms.get(treatment_key)

        if not s_c or not s_t:
            return 0, 0, 0.0, 0.0

        return (
            s_c.metrics.total,
            s_t.metrics.total,
            s_c.metrics.p_hat,
            s_t.metrics.p_hat,
        )

    @staticmethod
    def extract_total_stats(metrics: BinomialScoreboard) -> Tuple[int, int]:
        """
        Extracts aggregated (total_n, total_successes) across all arms.
        """
        n_total = sum(a.metrics.total for a in metrics.arms.values())
        s_total = sum(a.metrics.successes for a in metrics.arms.values())
        return n_total, s_total


class ContinuousAdapter:
    @staticmethod
    def extract_stats(
        metrics: ContinuousScoreboard, control_key: str, treatment_key: str
    ) -> Tuple[int, int, float, float, float, float]:
        """
        Extracts (n_c, n_t, mean_c, mean_t, var_c, var_t)
        """
        s_c = metrics.arms.get(control_key)
        s_t = metrics.arms.get(treatment_key)

        if not s_c or not s_t:
            return 0, 0, 0.0, 0.0, 0.0, 0.0

        return (
            s_c.metrics.total,
            s_t.metrics.total,
            s_c.metrics.mean,
            s_t.metrics.mean,
            s_c.metrics.variance or 0.0,
            s_t.metrics.variance or 0.0,
        )
