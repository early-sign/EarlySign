from typing import Any, Dict, List, Literal

from pydantic import BaseModel

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
from earlysign.schema.ES3.Continuous import ArmData as ContinuousArmData
from earlysign.schema.ES3.YEAST import MethodSpec, Protocol, TaskSpec
from earlysign.schema.ES3.YEAST.Log import Boundary as BoundarySchema, LookResult
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.binomial import Scoreboard as BinomialScoreboard
from earlysign.v1.methods.continuous import Scoreboard as ContinuousScoreboard
from earlysign.v1.methods.YEAST.boundary import Boundary
from earlysign.v1.methods.YEAST.engine import BinomialYEASTEngine, ContinuousYEASTEngine
from earlysign.v1.methods.YEAST.reporting import FinalProjector, ProgressProjector


class BinomialYeastTaskSpec(BaseModel):
    """
    User-facing Task Specification for Binomial YEAST.
    """

    kind: Literal["yeast"] = "yeast"
    arms: List[str]
    response_type: Literal["binary"] = "binary"
    hypotheses: Dict[str, Any]  # Simplified for template input compatibility


class ContinuousYeastTaskSpec(BaseModel):
    """
    User-facing Task Specification for Continuous YEAST.
    """

    kind: Literal["yeast"] = "yeast"
    arms: List[str]
    response_type: Literal["continuous"] = "continuous"
    hypotheses: Dict[str, Any]


class BinomialYeastTemplate:
    """
    Template for YEAST (Your Evidence Accumulation Sequential Test) on Binomial data.
    Standardized to use Session, Engine, and Projectors.
    """

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_protocol(self, protocol: Protocol) -> None:
        """
        Persists the trial protocol to the ledger.
        """
        protocol = Protocol.model_validate(protocol)
        with Session(self.ledger) as sess:
            sess.Commit(protocol)

            # Persist the initial boundary
            boundary_val = Boundary.calculate(protocol)
            sess.Commit(BoundarySchema(value=boundary_val))

    @classmethod
    def design(
        cls,
        task: BinomialYeastTaskSpec,
        significance_level: float,
        expected_num_observations: int,
        estimated_variance: float,
    ) -> Protocol:
        """
        Design a YEAST protocol from parameters.
        """
        method = MethodSpec(
            kind="yeast",
            significance_level=significance_level,
            expected_num_observations=expected_num_observations,
            estimated_variance=estimated_variance,
        )

        return Protocol(
            name="Binomial YEAST Protocol",
            task=TaskSpec(kind="yeast", arms=task.arms, response_type="binary"),
            method=method,
        )

    def update(self, batch: List[BinomialArmData]) -> None:
        """
        Update the experiment with a batch of data.
        """
        # 1. Ingest Data
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.Commit(item, trace=[])

        # 2. Analysis
        with Session(self.ledger) as sess:
            # Reconstruct Protocol from Ledger
            protocol = sess.Read(ProtocolProjector(Protocol))

            # Read Metrics
            metrics = sess.Read(BinomialScoreboard(identity="metrics"))

            # Read Boundary
            boundary = sess.Read(Boundary(identity="boundary"))

            # 3. Engine Execution
            engine = BinomialYEASTEngine(protocol.data)

            # 4. Commit Result via CallAndCommit
            sess.CallAndCommit(
                LookResult,
                engine.run,
                metrics=metrics,
                boundary=boundary.data,
            )

    def report_progress(self) -> Dict[str, Any]:
        """
        Report current status.
        """
        with Session(self.ledger) as sess:
            report = sess.Read(ProgressProjector()).data
            return report.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """
        Report final result.
        """
        with Session(self.ledger) as sess:
            return sess.Read(FinalProjector()).data.model_dump(mode="json")


class ContinuousYeastTemplate:
    """
    Template for YEAST (Your Evidence Accumulation Sequential Test) on Continuous data.
    """

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_protocol(self, protocol: Protocol) -> None:
        """
        Persists the trial protocol to the ledger.
        """
        protocol = Protocol.model_validate(protocol)
        with Session(self.ledger) as sess:
            sess.Commit(protocol)

            # Persist the initial boundary
            boundary_val = Boundary.calculate(protocol)
            sess.Commit(BoundarySchema(value=boundary_val))

    @classmethod
    def design(
        cls,
        task: ContinuousYeastTaskSpec,
        significance_level: float,
        expected_num_observations: int,
        estimated_variance: float,
    ) -> Protocol:
        """
        Design a YEAST protocol from parameters.
        """
        method = MethodSpec(
            kind="yeast",
            significance_level=significance_level,
            expected_num_observations=expected_num_observations,
            estimated_variance=estimated_variance,
        )

        return Protocol(
            name="Continuous YEAST Protocol",
            task=TaskSpec(kind="yeast", arms=task.arms, response_type="continuous"),
            method=method,
        )

    def update(self, batch: List[ContinuousArmData]) -> None:
        """
        Update the experiment with a batch of data.
        """
        # 1. Ingest Data
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.Commit(item, trace=[])

        # 2. Analysis
        with Session(self.ledger) as sess:
            # Reconstruct Protocol from Ledger
            protocol = sess.Read(ProtocolProjector(Protocol))

            # Read Metrics
            metrics = sess.Read(ContinuousScoreboard(identity="metrics"))

            # Read Boundary
            boundary = sess.Read(Boundary(identity="boundary"))

            # 3. Engine Execution
            engine = ContinuousYEASTEngine(protocol.data)

            # 4. Commit Result via CallAndCommit
            sess.CallAndCommit(
                LookResult,
                engine.run,
                metrics=metrics,
                boundary=boundary.data,
            )

    def report_progress(self) -> Dict[str, Any]:
        """
        Report current status.
        """
        with Session(self.ledger) as sess:
            report = sess.Read(ProgressProjector()).data
            return report.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """
        Report final result.
        """
        with Session(self.ledger) as sess:
            return sess.Read(FinalProjector()).data.model_dump(mode="json")
