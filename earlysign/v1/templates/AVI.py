from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.AVI import (
    GAVIMethodSpec,
    MSPRTMethodSpec,
    Protocol,
    TaskSpec,
)
from earlysign.schema.ES3.AVI.Log import LookResult
from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
from earlysign.schema.ES3.Continuous import ArmData as ContinuousArmData
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.AVI import GAVIEngine, mSPRTEngine
from earlysign.v1.methods.AVI.reporting import FinalProjector, ProgressProjector
from earlysign.v1.methods.binomial import Scoreboard as BinomialScoreboard
from earlysign.v1.methods.continuous import Scoreboard as ContinuousScoreboard
from earlysign.v1.templates.base import TemplateBase


# Define simple task specs for manual design usage if needed, similar to YEAST template
class BinomialAVITaskSpec(BaseModel):
    kind: Literal["AVI"] = "AVI"
    arms: List[str]
    response_type: Literal["binary"] = "binary"


class ContinuousAVITaskSpec(BaseModel):
    kind: Literal["AVI"] = "AVI"
    arms: List[str]
    response_type: Literal["continuous"] = "continuous"


class BinomialAVITemplate(TemplateBase[Protocol]):
    """Template for AVI on Binomial data.

    Example:
        >>> import ibis, duckdb  # noqa: F401
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.schema.ES3.AVI.Log import DecisionStatus
        >>> from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
        >>>
        >>> # Setup
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="doctest_avi_binom")
        >>>
        >>> # 1. Design GAVI
        >>> template = BinomialAVITemplate(ledger)
        >>> protocol = BinomialAVITemplate.design_gavi(
        ...     arms=["control", "treatment"],
        ...     alpha=0.05,
        ...     variance=0.25,
        ...     sides="two",
        ...     max_n=1000,
        ... )
        >>> template.set_protocol(protocol)
        >>>
        >>> # 2. Update with small difference
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
        >>> round(report1["boundary"], 4)
        0.2726
        >>>
        >>> # 3. Update with large difference crossing boundary
        >>> batch2 = [
        ...     BinomialArmData(n=400, success=200, arm="control"),
        ...     BinomialArmData(n=400, success=298, arm="treatment"),
        ... ]
        >>> template.update(batch2)
        >>> report2 = template.report_progress()
        >>> report2["status"]
        'stop_efficacy'
        >>>
        >>> # 4. Design mSPRT
        >>> ledger_msprt = ledger.bind(experiment_id="doctest_msprt_binom")
        >>> template_msprt = BinomialAVITemplate(ledger_msprt)
        >>> protocol_msprt = BinomialAVITemplate.design_m_sprt(
        ...     arms=["control", "treatment"],
        ...     alpha=0.05,
        ...     variance=0.25,
        ...     sides="two",
        ...     mde=0.1,
        ... )
        >>> template_msprt.set_protocol(protocol_msprt)
        >>>
        >>> # 5. Update with large difference
        >>> batch_msprt = [
        ...     BinomialArmData(n=500, success=250, arm="control"),
        ...     BinomialArmData(n=500, success=325, arm="treatment"),
        ... ]
        >>> template_msprt.update(batch_msprt)
        >>> report_msprt = template_msprt.report_progress()
        >>> report_msprt["status"]
        'stop_efficacy'
        >>> round(report_msprt["trajectory"], 4)
        0.15
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design_gavi(
        cls,
        arms: List[str],
        alpha: float,
        variance: float,
        sides: Literal["one", "two"],
        max_n: int,
    ) -> Protocol:
        method = GAVIMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            max_n=max_n,
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="binary")
        return Protocol(name="Binomial GAVI", task=task, method=method)

    @classmethod
    def design_m_sprt(
        cls,
        arms: List[str],
        alpha: float,
        variance: float,
        sides: Literal["one", "two"],
        mde: float,
    ) -> Protocol:
        method = MSPRTMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            mde=mde,
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="binary")
        return Protocol(name="Binomial mSPRT", task=task, method=method)

    def update(self, batch: List[BinomialArmData]) -> None:
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.Commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.Read(ProtocolProjector(Protocol)).data
            metrics = sess.Read(BinomialScoreboard(identity="metrics"))

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


class ContinuousAVITemplate(TemplateBase[Protocol]):
    """
    Template for AVI on Continuous data.
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design_gavi(
        cls,
        arms: List[str],
        alpha: float,
        variance: float,
        sides: Literal["one", "two"],
        max_n: int,
    ) -> Protocol:
        method = GAVIMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            max_n=max_n,
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="continuous")
        return Protocol(name="Continuous GAVI", task=task, method=method)

    @classmethod
    def design_m_sprt(
        cls,
        arms: List[str],
        alpha: float,
        variance: float,
        sides: Literal["one", "two"],
        mde: float,
    ) -> Protocol:
        method = MSPRTMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            mde=mde,
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="continuous")
        return Protocol(name="Continuous mSPRT", task=task, method=method)

    def update(self, batch: List[ContinuousArmData]) -> None:
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.Commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.Read(ProtocolProjector(Protocol)).data
            metrics = sess.Read(ContinuousScoreboard(identity="metrics"))

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
