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


# Define simple task specs for manual design usage if needed, similar to YEAST template
class BinomialAVITaskSpec(BaseModel):
    kind: Literal["AVI"] = "AVI"
    arms: List[str]
    response_type: Literal["binary"] = "binary"


class ContinuousAVITaskSpec(BaseModel):
    kind: Literal["AVI"] = "AVI"
    arms: List[str]
    response_type: Literal["continuous"] = "continuous"


class BinomialAVITemplate:
    """
    Template for AVI on Binomial data.
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


class ContinuousAVITemplate:
    """
    Template for AVI on Continuous data.
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
