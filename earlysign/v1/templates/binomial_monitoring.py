from typing import Any, Dict

from pydantic import BaseModel

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.AVI import MethodSpec, Protocol as AVIProtocol, TaskSpec
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.AVI import BinomialEValueEngine
from earlysign.v1.methods.AVI.engines.binomial_e_value import (
    EProcessProtocol,
    compute_binomial_e_value,
)
from earlysign.v1.methods.AVI.reporting import (
    BinomialEValueFinalProjector,
    BinomialEValueProgressProjector,
)
from earlysign.v1.templates.base import TemplateBase

# --- ES3 Protocol Manifest ---


class BinomialMonitoringProtocol(AVIProtocol):
    name: str = "Binomial Monitoring (AVI)"
    task: TaskSpec
    method: MethodSpec


class DecisionRecord(BaseModel):
    action: str
    e_value: float


class BinomialMonitoringTemplate(TemplateBase[EProcessProtocol]):
    """Template for real-time monitoring of a Binomial A/B test using e-processes.

    Examples:
        >>> import ibis, duckdb  # noqa: F401
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.schema.ES3.Binomial import ArmData
        >>> from earlysign.v1.methods.AVI.engines.binomial_e_value import EProcessProtocol
        >>>
        >>> # Setup
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="doctest_binom_mon")
        >>>
        >>> # 1. Design: H0: p=0.5, H1: p=0.7, Alpha=0.05
        >>> protocol = EProcessProtocol(null_p=0.5, alt_p=0.7, alpha=0.05)
        >>> template = BinomialMonitoringTemplate(ledger)
        >>> template.set_protocol(protocol)
        >>>
        >>> # 2. Update with H0-like data
        >>> batch1 = ArmData(n=100, success=50, arm="control")
        >>> template.update([batch1])
        >>> report1 = template.report_progress()
        >>> report1["status"]
        'continue'
        >>> report1["arms"]["control"]["successes"]
        50
        >>> round(report1["trajectory"], 2)
        0.0
        >>>
        >>> # 3. Update with H1-like data to cross threshold
        >>> batch2 = ArmData(n=900, success=650, arm="control")
        >>> template.update([batch2])
        >>> report2 = template.report_progress()
        >>> report2["status"]
        'stop_efficacy'
        >>> final_report = template.report_result()
        >>> final_report["is_rejected"]
        True
    """

    _protocol_class = EProcessProtocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def update(self, batch: list[Any]) -> None:
        """
        Run update cycle with E-value check.
        """
        from earlysign.v1.framework.projector import ProtocolProjector
        from earlysign.v1.methods.binomial import Scoreboard as BinomialScoreboard

        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.commit(item, trace=[])

        with Session(self.ledger) as sess:
            # 1. Read Protocol
            protocol = sess.read(ProtocolProjector(EProcessProtocol)).data

            # 2. Read Metrics
            metrics = sess.read(BinomialScoreboard(identity="metrics"))

            # 3. Run Engine
            engine = BinomialEValueEngine(protocol)

            # 4. Commit LookResult
            # Note: We can use sess.call_and_commit or just sess.commit(engine.run(metrics.data))
            look_result = engine.run(metrics.data)
            sess.commit(look_result)

    def report_progress(self) -> Dict[str, Any]:
        """
        Performs an e-check and returns the current progress report.
        Reconstructs state via MonitoringProgressProjector.
        """
        with Session(self.ledger) as sess:
            # 1. Read Report (Projector handles protocol and summary reconstruction internally)
            traced_report = sess.read(BinomialEValueProgressProjector())
            report = traced_report.data

            return report.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """Returns the final study report."""
        with Session(self.ledger) as sess:
            return sess.read(BinomialEValueFinalProjector()).data.model_dump(
                mode="json"
            )

    def run_backtest(self, batches: Any) -> Dict[str, Any]:
        """
        Historical Analysis: Replays data and stops immediately on a stopping decision.
        Returns a FinalReport.

        Data Requirements:
        - `batches`: Iterator yielding `BatchObservation` objects or lists of them.
        - Each `BatchObservation` must have `n`, `success`, and `arm`.
        """
        for i, batch in enumerate(batches):
            with Session(self.ledger) as sess:
                items = batch if isinstance(batch, list) else [batch]
                for item in items:
                    sess.commit(item, trace=[])

            res = self.report_progress()
            if res["is_rejected"]:
                return self.report_result()

        return self.report_result()

    def plot_result(self) -> Any:
        """
        Generates a summary plot of the E-process monitoring.

        Returns:
            matplotlib.figure.Figure: The generated plot figure.
        """
        import matplotlib.pyplot as plt

        from earlysign.v1.framework.projector import ProtocolProjector

        # 1. Get Final Result & Protocol
        final_res = self.report_result()
        with Session(self.ledger) as sess:
            protocol_res = sess.read(ProtocolProjector(EProcessProtocol))
            p = protocol_res.data

            # 2. Reconstruct History
            # We need E-values over time (n).
            t = sess.table
            df = t.execute()

        if "n" not in df.columns:
            # Fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Monitoring Result (History Unavailable)")
            threshold = 1.0 / p.alpha
            ax.axhline(threshold, color="r", linestyle="--", label="Threshold")
            ax.plot(final_res["n"], final_res["e_value"], "bo", label="Final E-value")
            return fig

        # Group/Sort
        # Assuming single stream of updates
        # Check columns. If 'arm' exists, sum them if 2-sample.
        # But 'Monitoring' usually implies we use 'n' and 'successes' from the summary accumulators.
        # The E-value function takes aggregated n, k.

        # Sort by arrival
        if "created_at" in df.columns:
            df = df.sort_values("created_at")

        # Calculate cumulative stats
        df["n_cum"] = df["n"].cumsum()
        df["s_cum"] = df["successes"].cumsum()

        # Compute E-value trajectory
        history_e = []
        history_n = []

        null_p = p.null_p
        alt_p = p.alt_p
        alpha = p.alpha
        threshold = 1.0 / alpha

        for idx, row in df.iterrows():
            n_val = row["n_cum"]
            s_val = row["s_cum"]

            res = compute_binomial_e_value(
                n=n_val, successes=s_val, null_p=null_p, alt_p=alt_p or 0.0, alpha=alpha
            )
            history_e.append(res.e_value)
            history_n.append(n_val)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Threshold
        ax.axhline(
            threshold,
            color="r",
            linestyle="--",
            label=f"Threshold (1/α = {threshold:.1f})",
        )

        # Trajectory
        ax.plot(history_n, history_e, "g-", label="E-value")

        # Scale
        # E-values can grow huge under H1. Log scale is often better.
        ax.set_yscale("log")

        # Final Point
        ax.scatter([history_n[-1]], [history_e[-1]], color="green", zorder=5)

        ax.set_xlabel("Sample Size (N)")
        ax.set_ylabel("E-value (Log Scale)")
        ax.set_title("Anytime-Valid Monitoring")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig
