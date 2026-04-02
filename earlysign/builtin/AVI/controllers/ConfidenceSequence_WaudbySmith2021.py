"""Confidence Sequence Controller (Waudby-Smith 2021).

This controller implements Confidence Sequences (CS) derived using the
Generalized Anytime-Valid Inference (GAVI) framework as described in
Waudby-Smith et al. (2021).

It is used for scenarios where the variance is KNOWN or bounded.

Reference:
    Waudby-Smith, I., Ramdas, A., Wu, Z., & Karbaschi, M. (2021).
    Time-uniform confidence spheres for means of random vectors.
    https://arxiv.org/abs/2102.04631
    (Specifically the variance-adaptive or bounded case when variance is supplied)

Examples:
    >>> import ibis, duckdb  # noqa: F401
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.builtin.AVI.controllers.ConfidenceSequence_WaudbySmith2021 import BinomialConfidenceSequenceWaudbySmith2021Controller
    >>> import earlysign.schema.ES3.Base as ES3_BASE
    >>> from earlysign.schema.ES3.Binomial import BinomialArmData

    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="doctest_cs_ws2021")
    >>> controller = BinomialConfidenceSequenceWaudbySmith2021Controller(ledger)
    >>>
    >>> # Design CS
    >>> protocol = controller.design(
    ...     arms=ES3_BASE.TwoArmComparison(control_arm_name="control", treatment_arm_name="treatment"),
    ...     alpha=0.05,
    ...     variance=0.25, # Max variance for Bernoulli
    ...     sides="two",
    ...     max_n=1000,
    ...     burn_in=10
    ... )
    >>> controller.set_protocol(protocol)
    >>>
    >>> # Update (Batch 1: Low data)
    >>> batch = [BinomialArmData(total=50, success=10, arm="control"), BinomialArmData(total=50, success=15, arm="treatment")]
    >>> controller.update(batch)
    >>> res1 = controller.report_progress()
    >>> print(f"Diff: {res1['trajectory']:.3f}, Boundary: {res1['boundary']:.4f}, Status: {res1['status']}")
    Diff: 0.100, Boundary: 0.3337, Status: continue
    >>>
    >>> # Update (Batch 2: High data crossing threshold)
    >>> batch2 = [BinomialArmData(total=500, success=100, arm="control"), BinomialArmData(total=500, success=250, arm="treatment")]
    >>> controller.update(batch2)
    >>> res2 = controller.report_progress()
    >>> print(f"Diff: {res2['trajectory']:.3f}, Boundary: {res2['boundary']:.4f}, Status: {res2['status']}")
    Diff: 0.282, Boundary: 0.0655, Status: stop_detected
"""

from typing import Any, Dict, List, Literal

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.builtin.AVI import GAVIEngine
from earlysign.builtin.AVI.reporting import FinalProjector, ProgressProjector
from earlysign.builtin.AVI.schema import (
    GAVIMethodSpec,
    LookResult,
    Protocol,
    ResponseType,
    Sides,
    TaskSpec,
)
from earlysign.core.ledger import Ledger
from earlysign.framework.controller import Controller
from earlysign.framework.projector import ProtocolProjector
from earlysign.framework.session import Session
from earlysign.parts.trackers.binomial import Scoreboard as BinomialScoreboard
from earlysign.parts.trackers.continuous import Scoreboard as ContinuousScoreboard
from earlysign.schema.ES3.Binomial import BinomialArmData
from earlysign.schema.ES3.Continuous import ContinuousArmData


class BinomialConfidenceSequenceWaudbySmith2021Controller(Controller[Protocol]):
    """Controller for Binomial Confidence Sequence (Waudby-Smith 2021).

    Uses GAVI framework with Known/Bounded Variance.
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        arms: ES3_BASE.ArmStructure,
        alpha: float,
        variance: float,  # Must be provided (e.g. 0.25)
        max_n: int,
        sides: Literal["one", "two"] = "two",
        burn_in: int = 100,
    ) -> Protocol:
        """
        Design a Confidence Sequence experiment.
        """
        method = GAVIMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=Sides(sides),
            max_n=max_n,
            burn_in=burn_in,
        )
        task = TaskSpec(arms=arms, response_type=ResponseType.BINARY)
        return Protocol(name="CS (Waudby-Smith 2021)", task=task, method=method)

    def update(self, batch: List[BinomialArmData]) -> None:
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(Protocol)).data
            metrics = sess.read(BinomialScoreboard(identity="metrics"))

            if not isinstance(protocol.task.arms, ES3_BASE.TwoArmComparison):
                raise NotImplementedError(
                    f"ConfidenceSequence (Binomial) on {type(protocol.task.arms).__name__} is not yet supported in this controller. "
                    "Currently, only TwoArmComparison is supported."
                )

            # Run GAVI Engine (calculates CS boundary)
            engine = GAVIEngine(protocol)
            sess.call_and_commit(LookResult, engine.run, metrics=metrics)

    def report_progress(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")


class ContinuousConfidenceSequenceWaudbySmith2021Controller(Controller[Protocol]):
    """Controller for Continuous Confidence Sequence (Waudby-Smith 2021).

    Uses GAVI framework with Known Variance.
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        arms: ES3_BASE.ArmStructure,
        alpha: float,
        variance: float,
        max_n: int,
        sides: Literal["one", "two"] = "two",
    ) -> Protocol:
        method = GAVIMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=Sides(sides),
            max_n=max_n,
        )
        task = TaskSpec(arms=arms, response_type=ResponseType.CONTINUOUS)
        return Protocol(
            name="Continuous CS (Waudby-Smith 2021)", task=task, method=method
        )

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
                    f"ConfidenceSequence (Continuous) on {type(protocol.task.arms).__name__} is not yet supported in this controller. "
                    "Currently, only TwoArmComparison is supported."
                )

            engine = GAVIEngine(protocol)
            sess.call_and_commit(LookResult, engine.run, metrics=metrics)

    def report_progress(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")
