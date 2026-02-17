from typing import Any, Dict, List, Literal

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.Binomial import BinomialArmData
from earlysign.schema.ES3.Continuous import ContinuousArmData
from earlysign.schema.ES3.YEAST import (
    MethodSpec,
    Protocol,
    ResponseType,
    TaskSpec as YeastTaskSpec,
)
from earlysign.schema.ES3.YEAST.Log import Boundary as BoundarySchema, LookResult
from earlysign.v1.framework.controller import Controller
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.binomial import Scoreboard as BinomialScoreboard
from earlysign.v1.methods.continuous import Scoreboard as ContinuousScoreboard
from earlysign.v1.methods.YEAST.engine import BinomialYEASTEngine, ContinuousYEASTEngine
from earlysign.v1.methods.YEAST.reporting import FinalProjector, ProgressProjector


class BinomialKurennoy2025TaskSpec(YeastTaskSpec):
    """User-facing Task Specification for Binomial YEAST."""

    kind: Literal["yeast"] = "yeast"
    arms: ES3_BASE.ArmStructure
    response_type: ResponseType = ResponseType.BINARY
    hypotheses: Dict[str, Any]


class ContinuousKurennoy2025TaskSpec(YeastTaskSpec):
    """User-facing Task Specification for Continuous YEAST."""

    kind: Literal["yeast"] = "yeast"
    arms: ES3_BASE.ArmStructure
    response_type: ResponseType = ResponseType.CONTINUOUS
    hypotheses: Dict[str, Any]


class BinomialKurennoy2025Controller(Controller[Protocol]):
    """Controller for YEAST (Your Evidence Accumulation Sequential Test) on Binomial data.

    Based on the method described in:
        Kurennoy, A., Dodin, M., Gurbanov, T., & Ramallo, A. P. (2025, October 29).
        YEAST: Yet another sequential test. The Thirty-Ninth Annual Conference on Neural
        Information Processing Systems. https://openreview.net/forum?id=aq3tgx5wcu

    Standardized to use Session, Engine, and Projectors.

    Examples:
        >>> import ibis, duckdb  # noqa: F401
        >>> from earlysign.core.ledger import Ledger
        >>> import earlysign.schema.ES3.Base as ES3_BASE
        >>> from earlysign.v1.controllers.YEAST_Kurennoy2025 import BinomialKurennoy2025Controller, BinomialKurennoy2025TaskSpec
        >>> from earlysign.schema.ES3.Binomial import BinomialArmData

        >>> # Setup
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="doctest_yeast_bin")
        >>> controller = BinomialKurennoy2025Controller(ledger)

        >>> # 1. Design
        >>> task = BinomialKurennoy2025TaskSpec(
        ...     arms=ES3_BASE.TwoArmComparison(control_arm_name="A", treatment_arm_name="B"),
        ...     hypotheses={}
        ... )
        >>> protocol = controller.design(task, significance_level=0.05, expected_num_observations=1000, estimated_variance=0.25)
        >>> controller.set_protocol(protocol)

        >>> # 2. Update
        >>> batch = [BinomialArmData(total=100, success=20, arm="A"), BinomialArmData(total=100, success=25, arm="B")]
        >>> controller.update(batch)

        >>> # 3. Report
        >>> res = controller.report_progress()
        >>> print(f"Evidence (Trajectory): {res['trajectory']:.4f}")
        Evidence (Trajectory): 0.5000
        >>> # Boundary is calculated on the fly by the engine if not in protocol
        >>> print(f"Boundary: {res['boundary']:.4f}, Status: {res['status']}")
        Boundary: 30.9898, Status: continue
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_protocol(self, protocol: Protocol) -> None:
        """
        Persists the trial protocol to the ledger and calculates initial boundary.
        """
        super().set_protocol(protocol)
        if not protocol.method.boundary_sequence:
            # Pre-calculate boundary if not present
            # For YEAST, boundary is usually static or calculated on fly?
            # The Engine implementation handles boundary generation.
            pass

        with Session(self.ledger) as sess:
            sess.commit(protocol)

    @classmethod
    def design(
        cls,
        task: BinomialKurennoy2025TaskSpec,
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
            boundary_sequence=[],  # Will be populated by Engine/Designer if needed
        )

        return Protocol(
            name="YEAST Binomial",
            task=task,
            method=method,
        )

    def update(self, batch: List[BinomialArmData]) -> None:
        """
        Update the experiment with a batch of data.
        """
        # 1. Ingest Data
        with Session(self.ledger) as sess:
            if batch:
                for item in batch:
                    sess.commit(item, trace=[])

            # Reconstruct Protocol from Ledger
            protocol = sess.read(ProtocolProjector(Protocol)).data

            # Read Metrics
            metrics = sess.read(BinomialScoreboard(identity="metrics"))

            # 3. Engine Execution
            engine = BinomialYEASTEngine(protocol)

            # Extract current boundary if available in protocol sequence, else calculate
            boundary_val = None
            if protocol.method.boundary_sequence:
                boundary_val = protocol.method.boundary_sequence[0]
            else:
                from earlysign.v1.methods.YEAST.boundary import Boundary

                boundary_val = Boundary.calculate(protocol)

            boundary = BoundarySchema(value=boundary_val)

            # 4. Commit Result via CallAndCommit
            sess.call_and_commit(
                LookResult,
                engine.run,
                metrics=metrics,
                boundary=boundary,
            )

    def report_progress(self) -> Dict[str, Any]:
        """Report current status.

        Returns:
            A dictionary containing the current progress report.
        """
        with Session(self.ledger) as sess:
            report = sess.read(ProgressProjector()).data
            return report.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """Report final result.

        Returns:
            A dictionary containing the final result of the experiment.
        """
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")


class ContinuousKurennoy2025Controller(Controller[Protocol]):
    """Controller for YEAST (Your Evidence Accumulation Sequential Test) on Continuous data.

    Based on:
        Kurennoy, A., Dodin, M., Gurbanov, T., & Ramallo, A. P. (2025). YEAST: Yet another sequential test.

    Examples:
        >>> import ibis
        >>> from earlysign.core.ledger import Ledger
        >>> import earlysign.schema.ES3.Base as ES3_BASE
        >>> from earlysign.v1.controllers.YEAST_Kurennoy2025 import ContinuousKurennoy2025Controller, ContinuousKurennoy2025TaskSpec
        >>> from earlysign.schema.ES3.Continuous import ContinuousArmData

        >>> con = ibis.duckdb.connect(":memory:")
        >>> ledger = Ledger(con, "events_cont")
        >>> ledger.ensure()
        >>> controller = ContinuousKurennoy2025Controller(ledger)

        >>> task = ContinuousKurennoy2025TaskSpec(
        ...     arms=ES3_BASE.TwoArmComparison(control_arm_name="A", treatment_arm_name="B"),
        ...     hypotheses={}
        ... )
        >>> protocol = controller.design(task, significance_level=0.05, expected_num_observations=1000, estimated_variance=1.0)
        >>> controller.set_protocol(protocol)

        >>> batch = [ContinuousArmData(total=10, sum_x=5.0, sum_x2=10.0, arm="A")]
        >>> controller.update(batch)
        >>> res = controller.report_progress()
        >>> res["status"]
        'continue'
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_protocol(self, protocol: Protocol) -> None:
        """
        Persists the trial protocol to the ledger and calculates initial boundary.
        """
        super().set_protocol(protocol)
        if not protocol.method.boundary_sequence:
            # Pre-calculate boundary if not present
            pass

        with Session(self.ledger) as sess:
            sess.commit(protocol)

    @classmethod
    def design(
        cls,
        task: ContinuousKurennoy2025TaskSpec,
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
            boundary_sequence=[],  # Will be populated by Engine/Designer if needed
        )
        return Protocol(
            name="YEAST Continuous",
            task=task,
            method=method,
        )

    def update(self, batch: List[ContinuousArmData]) -> None:
        """
        Update the experiment with a batch of data.
        """
        with Session(self.ledger) as sess:
            if batch:
                for item in batch:
                    sess.commit(item, trace=[])

            # 1. Read State
            protocol_traced = sess.read(ProtocolProjector(Protocol))
            metrics = sess.read(ContinuousScoreboard(identity="metrics"))

            # 2. Run Engine (YEAST Logic)
            engine = ContinuousYEASTEngine(protocol_traced.data)

            # Extract boundary
            boundary_val = None
            if protocol_traced.data.method.boundary_sequence:
                boundary_val = protocol_traced.data.method.boundary_sequence[0]
            else:
                from earlysign.v1.methods.YEAST.boundary import Boundary

                boundary_val = Boundary.calculate(protocol_traced.data)

            boundary = BoundarySchema(value=boundary_val)

            # 3. Commit Result
            # CallAndCommit ensures that 'LookResult' is causally linked to 'metrics'
            sess.call_and_commit(
                LookResult,
                engine.run,
                metrics=metrics,
                boundary=boundary,
            )

    def report_progress(self) -> Dict[str, Any]:
        """
        Report current status.
        """
        with Session(self.ledger) as sess:
            return sess.read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """
        Report final result.
        """
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")
