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
from earlysign.v1.templates.base import TemplateBase


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


class BinomialYeastTemplate(TemplateBase[Protocol]):
    """Template for YEAST on Binomial data.

    Standardized to use Session, Engine, and Projectors.

    Examples:
        >>> import ibis, duckdb  # noqa: F401
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
        >>> from earlysign.schema.ES3.YEAST.Log import DecisionStatus
        >>>
        >>> # Setup
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="doctest_yeast_binomial")
        >>>
        >>> # 1. Design Protocol
        >>> task = BinomialYeastTaskSpec(
        ...     arms=["control", "treatment"],
        ...     response_type="binary",
        ...     hypotheses={
        ...         "h_null_description": "diff <= 0",
        ...         "h_alt_description": "diff > 0",
        ...         "test_logic": {"kind": "superiority"},
        ...         "target_effect": {
        ...             "type": "binary",
        ...             "proportions": {"control": 0.5, "treatment": 0.6},
        ...         },
        ...     },
        ... )
        >>> template = BinomialYeastTemplate(ledger)
        >>> protocol = BinomialYeastTemplate.design(
        ...     task=task,
        ...     significance_level=0.05,
        ...     expected_num_observations=200,
        ...     estimated_variance=1.0,
        ... )
        >>> template.set_protocol(protocol)
        >>>
        >>> # 2. Update with data (Batch 1: Below boundary)
        >>> batch1 = [
        ...     BinomialArmData(n=50, success=20, arm="control"),
        ...     BinomialArmData(n=50, success=30, arm="treatment"),
        ... ]
        >>> template.update(batch1)
        >>> report1 = template.report_progress()
        >>> report1["status"]
        'continue'
        >>> report1["trajectory"]
        10.0
        >>>
        >>> # 3. Update with more data (Batch 2: Crossing boundary)
        >>> batch2 = [
        ...     BinomialArmData(n=50, success=20, arm="control"),
        ...     BinomialArmData(n=50, success=45, arm="treatment"),
        ... ]
        >>> template.update(batch2)
        >>> report2 = template.report_progress()
        >>> report2["status"]
        'stop_efficacy'
        >>> result = template.report_result()
        >>> result["is_rejected"]
        True
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_protocol(self, protocol: Protocol) -> None:
        """
        Persists the trial protocol to the ledger and calculates initial boundary.
        """
        super().set_protocol(protocol)
        with Session(self.ledger) as sess:
            protocol = sess.Read(ProtocolProjector(Protocol)).data
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
        """Report current status.

        Returns:
            A dictionary containing the current progress report.
        """
        with Session(self.ledger) as sess:
            report = sess.Read(ProgressProjector()).data
            return report.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """Report final result.

        Returns:
            A dictionary containing the final result of the experiment.
        """
        with Session(self.ledger) as sess:
            return sess.Read(FinalProjector()).data.model_dump(mode="json")


class ContinuousYeastTemplate(TemplateBase[Protocol]):
    """Template for YEAST (Your Evidence Accumulation Sequential Test) on Continuous data.

    Examples:
        >>> import ibis
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.schema.ES3.Continuous import ArmData as ContinuousArmData
        >>> from earlysign.schema.ES3.YEAST.Log import DecisionStatus
        >>>
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="doctest_yeast_continuous")
        >>>
        >>> task = ContinuousYeastTaskSpec(
        ...     arms=["control", "treatment"],
        ...     response_type="continuous",
        ...     hypotheses={},
        ... )
        >>> template = ContinuousYeastTemplate(ledger)
        >>> protocol = ContinuousYeastTemplate.design(
        ...     task=task,
        ...     significance_level=0.05,
        ...     expected_num_observations=100,
        ...     estimated_variance=1.0,
        ... )
        >>> template.set_protocol(protocol)
        >>>
        >>> # Batch 1: Below boundary
        >>> batch1 = [
        ...     ContinuousArmData(n=10, sum_x=10.0, sum_x2=20.0, arm="control"),
        ...     ContinuousArmData(n=10, sum_x=25.0, sum_x2=70.0, arm="treatment"),
        ... ]
        >>> template.update(batch1)
        >>> report1 = template.report_progress()
        >>> report1["status"]
        'continue'
        >>> report1["trajectory"]
        15.0

        >>> # Batch 2: Above boundary
        >>> batch2 = [
        ...     ContinuousArmData(n=10, sum_x=10.0, sum_x2=20.0, arm="control"),
        ...     ContinuousArmData(n=10, sum_x=20.0, sum_x2=50.0, arm="treatment"),
        ... ]
        >>> template.update(batch2)
        >>> report2 = template.report_progress()
        >>> report2["status"]
        'stop_efficacy'
        >>> report2["trajectory"]
        25.0
        >>> report2["sample_n"]
        40
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_protocol(self, protocol: Protocol) -> None:
        """
        Persists the trial protocol to the ledger and calculates initial boundary.
        """
        super().set_protocol(protocol)
        with Session(self.ledger) as sess:
            protocol = sess.Read(ProtocolProjector(Protocol)).data
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
