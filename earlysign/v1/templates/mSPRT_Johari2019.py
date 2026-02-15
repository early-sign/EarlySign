"""Always Valid Inference (mSPRT - Johari 2019).

This template implements the Mixture Sequential Probability Ratio Test (mSPRT)
as described in Johari et al. (2019).

Reference:
    Johari, R., Pekelis, L., & Walsh, J. (2019).
    Always Valid Inference: Continuous Monitoring of A/B Tests.
    https://arxiv.org/abs/1512.04922

Examples:
    >>> import ibis, duckdb  # noqa: F401
    >>> from earlysign.core.ledger import Ledger
    >>> import earlysign.schema.ES3.Base as ES3_BASE
    >>> from earlysign.v1.templates.mSPRT_Johari2019 import BinomialJohari2019Template
    >>> from earlysign.schema.ES3.Binomial import ArmData
    >>> from earlysign.schema.ES3.AVI.Log import DecisionStatus

    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="doctest_msprt")
    >>>
    >>> # Design mSPRT (Binomial)
    >>> template = BinomialJohari2019Template(ledger)
    >>> protocol = template.design(
    ...     arms=ES3_BASE.TwoArmComparison(control_arm_name="control", treatment_arm_name="treatment"),
    ...     alpha=0.05,
    ...     tau=0.1,  # Mixing parameter ~ MDE
    ...     sides="two"
    ... )
    >>> template.set_protocol(protocol)
    >>>
    >>> # Update with data
    >>> batch = [ArmData(n=100, success=20, arm="control"), ArmData(n=100, success=30, arm="treatment")]
    >>> template.update(batch)
    >>>
    >>> # Check Report (Batch 1)
    >>> res = template.report_progress()
    >>> print(f"Diff: {res['trajectory']:.3f}, Boundary: {res['boundary']:.3f}, Status: {res['status']}")
    Diff: 0.100, Boundary: 0.192, Status: continue
    >>>
    >>> # Update with more data (Batch 2)
    >>> batch2 = [ArmData(n=1000, success=200, arm="control"), ArmData(n=1000, success=300, arm="treatment")]
    >>> template.update(batch2)
    >>> res2 = template.report_progress()
    >>> print(f"Diff: {res2['trajectory']:.3f}, Boundary: {res2['boundary']:.3f}, Status: {res2['status']}")
    Diff: 0.100, Boundary: 0.057, Status: stop_efficacy
"""

from typing import Any, Dict, List, Literal, Optional

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.AVI import (
    MSPRTMethodSpec,
    Protocol,
    TaskSpec,
)
from earlysign.schema.ES3.AVI.Log import DecisionStatus, LookResult
from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
from earlysign.schema.ES3.Continuous import ArmData as ContinuousArmData
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.AVI import mSPRTEngine
from earlysign.v1.methods.AVI.reporting import FinalProjector, ProgressProjector
from earlysign.v1.methods.binomial import Scoreboard as BinomialScoreboard
from earlysign.v1.methods.continuous import Scoreboard as ContinuousScoreboard
from earlysign.v1.framework.template import TemplateBase


class BinomialJohari2019Template(TemplateBase[Protocol]):
    """Template for Binomial mSPRT (Johari 2019)."""

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        arms: ES3_BASE.ArmStructure,
        alpha: float,
        tau: float,
        sides: Literal["one", "two"] = "two",
    ) -> Protocol:
        """
        Design an mSPRT experiment for binomial data.

        Args:
            arms: List of arm names (exactly 2).
            alpha: Target false positive rate (at any time).
            tau: Mixing standard deviation (tuning parameter for the mixing distribution).
                 Commonly set to the expected effect size or MDE.
            sides: "one" or "two" sided testing.
        """
        method = MSPRTMethodSpec(
            alpha=alpha,
            variance=None,  # Binomial variance is implicit/estimated by engine
            sides=sides,
            mde=tau,
        )
        task = TaskSpec(arms=arms, response_type="binary")
        return Protocol(name="mSPRT (Johari 2019)", task=task, method=method)

    def update(self, batch: List[BinomialArmData]) -> None:
        """
        Update the experiment with a batch of data.
        """
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(Protocol)).data
            metrics = sess.read(BinomialScoreboard(identity="metrics"))

            if not isinstance(protocol.task.arms, ES3_BASE.TwoArmComparison):
                raise NotImplementedError(
                    f"mSPRT (Binomial) on {type(protocol.task.arms).__name__} is not yet supported in this template. "
                    "Currently, only TwoArmComparison is supported."
                )

            # Run mSPRT Engine
            engine = mSPRTEngine(protocol)
            sess.call_and_commit(LookResult, engine.run, metrics=metrics)

    def report_progress(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")

    def backtest(self, batches: Any) -> Dict[str, Any]:
        """
        Historical Analysis: Replays data and stops immediately on a stopping decision.

        Args:
           batches: Iterator yielding `ArmData` objects or lists of them.
        """
        for i, batch in enumerate(batches):
            self.update(batch if isinstance(batch, list) else [batch])
            prog = self.report_progress()
            if prog.get("status") != DecisionStatus.CONTINUE_:
                break

        return self.report_result()

    def backtest_from_table(
        self,
        table: Any,
        *,
        arm_col: str = "arm",
        n_col: str = "n",
        success_col: str = "success",
        order_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Historical Analysis from an Ibis table.
        Replays data from the table and stops immediately on a stopping decision.

        Args:
            table: Ibis table containing historical data.
            arm_col: Column name for the arm identifier.
            n_col: Column name for the number of trials.
            success_col: Column name for the number of successes.
            order_by: Column name to order the data by.
        """
        from earlysign.schema.ES3.Binomial import ArmData

        # 1. Project and order
        if order_by:
            # We must include order_by in our select if we want to order by it,
            # or just use the original table's column.
            data_table = table.select(
                arm=table[arm_col],
                n=table[n_col],
                success=table[success_col],
                _order=table[order_by],
            ).order_by("_order")
        else:
            data_table = table.select(
                arm=table[arm_col],
                n=table[n_col],
                success=table[success_col],
            )

        # 2. Replay & Stop
        df = data_table.execute()

        for _, row in df.iterrows():
            batch = [
                ArmData(
                    arm=str(row["arm"]),
                    n=int(row["n"]),
                    success=int(row["success"]),
                )
            ]
            self.update(batch)
            prog = self.report_progress()
            if prog.get("status") != DecisionStatus.CONTINUE_:
                break

        return self.report_result()


class ContinuousJohari2019Template(TemplateBase[Protocol]):
    """Template for Continuous mSPRT (Johari 2019)."""

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        arms: ES3_BASE.ArmStructure,
        alpha: float,
        tau: float,
        variance: float,
        sides: Literal["one", "two"] = "two",
    ) -> Protocol:
        """
        Design an mSPRT experiment for continuous data.

        Args:
            variance: Known variance of the outcome (assumed fixed).
        """
        method = MSPRTMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            mde=tau,
        )
        task = TaskSpec(arms=arms, response_type="continuous")
        return Protocol(name="Continuous mSPRT (Johari 2019)", task=task, method=method)

    def update(self, batch: List[ContinuousArmData]) -> None:
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(Protocol)).data
            metrics = sess.read(ContinuousScoreboard(identity="metrics"))

            if not isinstance(protocol.task.arms, ES3_BASE.TwoArmComparison):
                raise NotImplementedError(
                    f"mSPRT (Continuous) on {type(protocol.task.arms).__name__} is not yet supported in this template. "
                    "Currently, only TwoArmComparison is supported."
                )

            engine = mSPRTEngine(protocol)
            sess.call_and_commit(LookResult, engine.run, metrics=metrics)

    def report_progress(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")
