from typing import Any, Dict

from pydantic import BaseModel

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.AVI import MethodSpec, Protocol as AVIProtocol, TaskSpec
from earlysign.v1.framework.controller import Controller
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

# --- ES3 Protocol Manifest ---


class BinomialMonitoringProtocol(AVIProtocol):
    name: str = "Binomial Monitoring (AVI)"
    task: TaskSpec
    method: MethodSpec


class DecisionRecord(BaseModel):
    action: str
    e_value: float


class BinomialMonitoringController(Controller[EProcessProtocol]):
    """Controller for real-time monitoring of a Binomial A/B test using e-processes.

    Examples:
        >>> import ibis, duckdb  # noqa: F401
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.schema.ES3.Binomial import BinomialArmData
        >>> from earlysign.v1.methods.AVI.engines.binomial_e_value import EProcessProtocol
        >>> from earlysign.v1.controllers.binomial_monitoring import BinomialMonitoringController
        >>>
        >>> # Setup
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="doctest_binom_mon")
        >>>
        >>> # 1. Design: H0: p=0.5, H1: p=0.7, Alpha=0.05
        >>> protocol = EProcessProtocol(null_p=0.5, alt_p=0.7, alpha=0.05)
        >>> controller = BinomialMonitoringController(ledger)
        >>> controller.set_protocol(protocol)
        >>>
        >>> # 2. Update with H0-like data (p=0.5, success=50/100)
        >>> batch1 = BinomialArmData(total=100, success=50, arm="control")
        >>> controller.update([batch1])
        >>> report1 = controller.report_progress()
        >>> print(f"E-value: {report1['trajectory']:.2f}, Status: {report1['status']}")
        E-value: 0.00, Status: continue
        >>>
        >>> # 3. Update with H1-like data (p=0.7, success=650/900 more samples)
        >>> batch2 = BinomialArmData(total=900, success=650, arm="control")
        >>> controller.update([batch2])
        >>> report2 = controller.report_progress()
        >>> print(f"E-value: {report2['trajectory']:.2f}, Status: {report2['status']}")
        E-value: 543250443896186605971754527833980928.00, Status: stop_efficacy
        >>> final_report = controller.report_result()
        >>> print(f"Is Rejected: {final_report['is_rejected']}")
        Is Rejected: True
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

        # 1. Unified Session
        with Session(self.ledger) as sess:
            if batch:
                for item in batch:
                    sess.commit(item, trace=[])

            # 2. Read Protocol & Metrics
            protocol = sess.read(ProtocolProjector(EProcessProtocol)).data
            metrics = sess.read(BinomialScoreboard(identity="metrics"))

            # 3. Run Engine & Commit
            engine = BinomialEValueEngine(protocol)
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
            t = sess.table
            df = t.execute()

        if "total" not in df.columns:
            # Fallback for simple display if no history
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Monitoring Result (History Unavailable)")
            threshold = 1.0 / p.alpha
            ax.axhline(threshold, color="r", linestyle="--", label="Threshold")
            ax.plot(
                final_res["sample_n"],
                final_res["trajectory"],
                "bo",
                label="Final E-value",
            )
            return fig

        # Sort by arrival
        if "created_at" in df.columns:
            df = df.sort_values("created_at")

        # Calculate cumulative stats for the e-process
        # For our specific E-process template, we assume we want to track 'total' and 'success'
        if "success" not in df.columns and "successes" in df.columns:
            df["success"] = df["successes"]

        df["total_cum"] = df["total"].cumsum()
        df["success_cum"] = df["success"].cumsum()

        # Compute E-value trajectory
        history_e = []
        history_total = []

        null_p = p.null_p
        alt_p = p.alt_p
        alpha = p.alpha
        threshold = 1.0 / alpha

        for idx, row in df.iterrows():
            total_val = row["total_cum"]
            success_val = row["success_cum"]

            res = compute_binomial_e_value(
                n=total_val,
                successes=success_val,
                null_p=null_p,
                alt_p=alt_p or 0.0,
                alpha=alpha,
            )
            history_e.append(res.e_value)
            history_total.append(total_val)

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
        ax.plot(history_total, history_e, "g-", label="E-value")

        # Scale
        ax.set_yscale("log")

        # Final Point
        ax.scatter([history_total[-1]], [history_e[-1]], color="green", zorder=5)

        ax.set_xlabel("Sample Size (Total)")
        ax.set_ylabel("E-value (Log Scale)")
        ax.set_title("Anytime-Valid Monitoring")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig
