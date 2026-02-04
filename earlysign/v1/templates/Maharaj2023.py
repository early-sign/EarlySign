from typing import Any, Dict, List, Literal, Optional

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.AVI import (
    GAVIMethodSpec,
    MSPRTMethodSpec,
    Protocol,
    TaskSpec,
)
from earlysign.schema.ES3.AVI.Log import LookResult
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.AVI import GAVIEngine, mSPRTEngine
from earlysign.v1.methods.AVI.reporting import FinalProjector, ProgressProjector
from earlysign.v1.methods.binomial import Scoreboard as BinomialScoreboard
from earlysign.v1.methods.continuous import Scoreboard as ContinuousScoreboard
from earlysign.v1.templates.base import TemplateBase


class Maharaj2023Template(TemplateBase[Protocol]):
    """Template for Anytime-Valid Inference (AVI) following Maharaj et al. (2023).

    Supports "Unknown Variance" scenarios where variance is estimated from data.

    Reference:
        Maharaj, A., et al. (2023). Anytime-Valid Confidence Sequences in an
        Enterprise A/B Testing Platform. WWW '23 Companion.
        https://doi.org/10.1145/3543873.3584635

    Examples:
        >>> import ibis, duckdb  # noqa: F401
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.schema.ES3.AVI.Log import DecisionStatus
        >>> from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
        >>> from earlysign.schema.ES3.Continuous import ArmData as ContinuousArmData
        >>>
        >>> # Setup ledger
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="maharaj_test_001")
        >>>
        >>> # Initialize template
        >>> template = Maharaj2023Template(ledger)
        >>>
        >>> # 1. Design GAVI with UNKNOWN variance (variance=None)
        >>> protocol = Maharaj2023Template.design_gavi(
        ...     arms=["control", "treatment"],
        ...     alpha=0.05,
        ...     variance=None,
        ...     sides="two",
        ...     max_n=1000
        ... )
        >>> template.set_protocol(protocol)
        >>>
        >>> # 2. Simulate Data Update (Binomial)
        >>> batch1 = [
        ...     BinomialArmData(n=100, success=50, arm="control"),
        ...     BinomialArmData(n=100, success=52, arm="treatment"),
        ... ]
        >>> template.update(batch1)
        >>> report1 = template.report_progress()
        >>> report1["status"]
        'continue'
        >>> round(report1["trajectory"], 4)
        0.02
        >>>
        >>> # 3. Large difference update
        >>> batch2 = [
        ...     BinomialArmData(n=300, success=150, arm="control"),
        ...     BinomialArmData(n=300, success=248, arm="treatment"),
        ... ]
        >>> template.update(batch2)
        >>> report2 = template.report_progress()
        >>> report2["status"]
        'stop_efficacy'
        >>> report2["sample_n"]
        800
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design_gavi(
        cls,
        arms: List[str],
        alpha: float,
        variance: Optional[float],
        sides: Literal["one", "two"],
        max_n: int,
    ) -> Protocol:
        """
        Design a GAVI (General Asymptotic Confidence Sequence) test.
        If `variance` is None, it will be estimated from data (Maharaj et al., 2023).
        """
        method = GAVIMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            max_n=max_n,
        )
        task = TaskSpec(
            kind="AVI", arms=arms, response_type="binary"
        )  # Defaulting to binary for now, adjustable via overload if needed
        return Protocol(name="Maharaj2023 GAVI", task=task, method=method)

    @classmethod
    def design_m_sprt(
        cls,
        arms: List[str],
        alpha: float,
        variance: Optional[float],
        sides: Literal["one", "two"],
        mde: float,
    ) -> Protocol:
        """
        Design an mSPRT test.
        """
        method = MSPRTMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            mde=mde,
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="binary")
        return Protocol(name="Maharaj2023 mSPRT", task=task, method=method)

    def update(self, batch: List[Any]) -> None:
        """
        Update the ledger with new data and run the AVI engine.
        Accepts list of ArmData (Binomial or Continuous).
        """
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    # Determine trace if needed, empty for now
                    sess.Commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.Read(ProtocolProjector(Protocol)).data

            # Determine response type to pick correct scoreboard
            # This logic mimics the standard AVI template but is simplified here
            response_type = getattr(protocol.task, "response_type", "binary")

            if response_type == "binary":
                metrics = sess.Read(BinomialScoreboard(identity="metrics"))
            else:
                metrics = sess.Read(ContinuousScoreboard(identity="metrics"))

            # Select Engine
            if protocol.method.kind == "GAVI":
                engine = GAVIEngine(protocol)
            elif protocol.method.kind == "mSPRT":
                engine = mSPRTEngine(protocol)
            else:
                raise ValueError(f"Unknown AVI method kind: {protocol.method.kind}")

            sess.CallAndCommit(LookResult, engine.run, metrics=metrics)

    def report_progress(self) -> Dict[str, Any]:
        """Returns the current interim report."""
        with Session(self.ledger) as sess:
            return sess.Read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.Read(FinalProjector()).data.model_dump(mode="json")
