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


class Maharaj2023Template:
    """
    Template for Anytime-Valid Inference (AVI) following Maharaj et al. (2023).

    This template primarily supports "Unknown Variance" scenarios (GAVI / Asymptotic Confidence Sequences)
    where the variance is estimated from the data stream rather than being pre-specified.

    Reference:
        Maharaj, A., Sinha, R., Arbour, D., Waudby-Smith, I., Liu, S. Z., Sinha, M., Addanki, R., Ramdas, A., Garg, M., & Swaminathan, V. (2023).
        Anytime-valid confidence sequences in an enterprise A/B testing platform. 396–400.
        https://doi.org/10.1145/3543873.3584635

    Doctests:

        >>> import ibis
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.schema.ES3.AVI.Log import DecisionStatus
        >>> from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
        >>> from earlysign.schema.ES3.Continuous import ArmData as ContinuousArmData

        # Setup ledger
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="maharaj_test_001")

        # Initialize template
        >>> template = Maharaj2023Template(ledger)

        # 1. Design GAVI with UNKNOWN variance (variance=None)
        >>> protocol = Maharaj2023Template.design_gavi(
        ...     arms=["control", "treatment"],
        ...     alpha=0.05,
        ...     variance=None,  # Adaptive/Unknown variance
        ...     sides="two",
        ...     max_n=1000
        ... )
        >>> template.set_protocol(protocol)

        # 2. Simulate Data Update (Binomial)
        # Small difference, should continue.
        # Control: 50/100 (p=0.5), Treatment: 52/100 (p=0.52)
        >>> batch1 = [
        ...     BinomialArmData(n=100, success=50, arm="control"),
        ...     BinomialArmData(n=100, success=52, arm="treatment"),
        ... ]
        >>> template.update(batch1)
        >>> report1 = template.report_progress()
        >>> report1["status"]
        'continue'
        >>> abs(report1["trajectory"] - 0.02) < 1e-9
        True

        # Boundary should be calculated using estimated variance.
        # Var approx p(1-p) ~ 0.25 each. Sigma2_effective ~ 0.25.
        # n = 100.
        # Verify boundary is reasonable (e.g. > 0 and < 1)
        >>> 0 < report1["boundary"] < 1
        True

        # 3. Large difference update
        # Control: 200/400 (0.5), Treatment: 300/400 (0.75)
        # Diff = 0.25.
        >>> batch2 = [
        ...     BinomialArmData(n=300, success=150, arm="control"), # +300
        ...     BinomialArmData(n=300, success=248, arm="treatment"), # +300
        ... ]
        >>> template.update(batch2)
        >>> report2 = template.report_progress()
        >>> report2["sample_n"]
        800

        # With such large difference, it should stop for efficacy.
        >>> report2["status"]
        'stop_efficacy'
        >>> report2["is_rejected"] = (report2["status"] == 'stop_efficacy')
        >>> report2["is_rejected"]
        True

    """

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_protocol(self, protocol: Protocol) -> None:
        protocol = Protocol.model_validate(protocol)
        with Session(self.ledger) as sess:
            sess.Commit(protocol)

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
        with Session(self.ledger) as sess:
            return sess.Read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.Read(FinalProjector()).data.model_dump(mode="json")
