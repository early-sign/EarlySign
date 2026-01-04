from typing import TYPE_CHECKING, Any, Dict

from pydantic import BaseModel

from earlysign.v1.framework.session import Session
from earlysign.v1.methods.actions import Decision, UpdateProtocol
from earlysign.v1.methods.anytime_valid.protocol import EProcessProtocol
from earlysign.v1.methods.anytime_valid.report import (
    MonitoringFinalProjector,
    MonitoringProgressProjector,
)

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger


class DecisionRecord(BaseModel):
    action: str
    e_value: float


class BinomialMonitoringTemplate:
    """
    Safe testing / Continuous monitoring template using e-processes.
    """

    def __init__(self, ledger: "Ledger"):
        self.ledger = ledger

    def set_protocol(self, protocol: EProcessProtocol):
        """
        Persists the monitoring protocol to the ledger.
        """
        with Session(self.ledger) as sess:
            UpdateProtocol(sess, protocol)

    def report_progress(self) -> Dict[str, Any]:
        """
        Performs an e-check and returns the current progress report.
        Reconstructs state via MonitoringProgressProjector.
        """
        with Session(self.ledger) as sess:
            # 1. Read Report (Projector handles protocol and summary reconstruction internally)
            traced_report = sess.Read(MonitoringProgressProjector())
            report = traced_report.data

            # 2. Record Decision if rejected
            if report.is_rejected:
                Decision(
                    sess,
                    DecisionRecord(action="Reject H0", e_value=float(report.e_value)),
                    trace=traced_report.trace,
                )

            return report.model_dump()

    def report_result(self) -> Dict[str, Any]:
        """Returns the final study report."""
        with Session(self.ledger) as sess:
            return sess.Read(MonitoringFinalProjector()).data.model_dump()

    def run_backtest(self, batches: Any) -> Dict[str, Any]:
        """
        Historical Analysis: Replays data and stops immediately on a stopping decision.
        Returns a FinalReport.

        Data Requirements:
        - `batches`: Iterator yielding `BatchObservation` objects or lists of them.
        - Each `BatchObservation` must have `n`, `success`, and `variant`.
        """
        from earlysign.v1.methods.actions import Ingest

        for i, batch in enumerate(batches):
            with Session(self.ledger) as sess:
                items = batch if isinstance(batch, list) else [batch]
                for item in items:
                    Ingest(sess, item)

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
        import pandas as pd
        import numpy as np
        from earlysign.v1.framework.projector import ProtocolProjector
        from earlysign.v1.methods.anytime_valid.protocol import EProcessProtocol
        from earlysign.v1.methods.anytime_valid.e_process import compute_binomial_e_value

        # 1. Get Final Result & Protocol
        final_res = self.report_result()
        with Session(self.ledger) as sess:
            protocol_res = sess.Read(ProtocolProjector(EProcessProtocol))
            p = protocol_res.data
            
            # 2. Reconstruct History
            # We need E-values over time (n).
            t = sess.table
            df = t.execute()

        if 'n' not in df.columns:
            # Fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Monitoring Result (History Unavailable)")
            threshold = 1.0 / p.alpha
            ax.axhline(threshold, color='r', linestyle='--', label='Threshold')
            ax.plot(final_res['n'], final_res['e_value'], 'bo', label='Final E-value')
            return fig

        # Group/Sort
        # Assuming single stream of updates
        # Check columns. If 'variant' exists, sum them if 2-sample.
        # But 'Monitoring' usually implies we use 'n' and 'successes' from the summary accumulators.
        # The E-value function takes aggregated n, k.
        
        # Sort by arrival
        if 'created_at' in df.columns:
            df = df.sort_values('created_at')
        
        # Calculate cumulative stats
        df['n_cum'] = df['n'].cumsum()
        df['s_cum'] = df['successes'].cumsum()
        
        # Compute E-value trajectory
        history_e = []
        history_n = []
        
        null_p = p.null_p
        alt_p = p.alt_p
        alpha = p.alpha
        threshold = 1.0 / alpha
        
        for idx, row in df.iterrows():
            n_val = row['n_cum']
            s_val = row['s_cum']
            
            res = compute_binomial_e_value(
                n=n_val,
                successes=s_val,
                null_p=null_p,
                alt_p=alt_p,
                alpha=alpha
            )
            history_e.append(res.e_value)
            history_n.append(n_val)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Threshold
        ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold (1/α = {threshold:.1f})')
        
        # Trajectory
        ax.plot(history_n, history_e, 'g-', label='E-value')
        
        # Scale
        # E-values can grow huge under H1. Log scale is often better.
        ax.set_yscale('log')
        
        # Final Point
        ax.scatter([history_n[-1]], [history_e[-1]], color='green', zorder=5)

        ax.set_xlabel("Sample Size (N)")
        ax.set_ylabel("E-value (Log Scale)")
        ax.set_title("Anytime-Valid Monitoring")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
