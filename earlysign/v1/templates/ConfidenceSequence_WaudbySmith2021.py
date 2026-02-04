"""Confidence Sequence (Waudby-Smith 2021).

This template implements Confidence Sequences (CS) derived using the
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
    >>> from earlysign.v1.templates.ConfidenceSequence_WaudbySmith2021 import BinomialConfidenceSequenceWaudbySmith2021Template
    >>> from earlysign.schema.ES3.Binomial import ArmData

    >>> conn = ibis.connect("duckdb://:memory:")
    >>> ledger = Ledger(conn, "events")
    >>> ledger.ensure()
    >>> ledger = ledger.bind(experiment_id="doctest_cs_ws2021")
    >>> template = BinomialConfidenceSequenceWaudbySmith2021Template(ledger)
    >>>
    >>> # Design CS
    >>> protocol = template.design(
    ...     arms=["C", "T"],
    ...     alpha=0.05,
    ...     variance=0.25, # Max variance for Bernoulli
    ...     sides="two",
    ...     max_n=1000
    ... )
    >>> template.set_protocol(protocol)
    >>>
    >>> # Update
    >>> batch = [ArmData(n=50, success=10, arm="C"), ArmData(n=50, success=15, arm="T")]
    >>> template.update(batch)
"""

from typing import Any, Dict, List, Literal

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.AVI import (
    GAVIMethodSpec,
    Protocol,
    TaskSpec,
)
from earlysign.schema.ES3.AVI.Log import LookResult
from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
from earlysign.schema.ES3.Continuous import ArmData as ContinuousArmData
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.AVI import GAVIEngine
from earlysign.v1.methods.AVI.reporting import FinalProjector, ProgressProjector
from earlysign.v1.methods.binomial import Scoreboard as BinomialScoreboard
from earlysign.v1.methods.continuous import Scoreboard as ContinuousScoreboard
from earlysign.v1.templates.base import TemplateBase


class BinomialConfidenceSequenceWaudbySmith2021Template(TemplateBase[Protocol]):
    """Template for Binomial Confidence Sequence (Waudby-Smith 2021).

    Uses GAVI framework with Known/Bounded Variance.
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        arms: List[str],
        alpha: float,
        variance: float,  # Must be provided (e.g. 0.25)
        max_n: int,
        sides: Literal["one", "two"] = "two",
    ) -> Protocol:
        """
        Design a Confidence Sequence experiment.
        """
        method = GAVIMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            max_n=max_n,
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="binary")
        return Protocol(name="CS (Waudby-Smith 2021)", task=task, method=method)

    def update(self, batch: List[BinomialArmData]) -> None:
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(Protocol)).data
            metrics = sess.read(BinomialScoreboard(identity="metrics"))

            # Run GAVI Engine (calculates CS boundary)
            engine = GAVIEngine(protocol)
            sess.call_and_commit(LookResult, engine.run, metrics=metrics)

    def report_progress(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")


class ContinuousConfidenceSequenceWaudbySmith2021Template(TemplateBase[Protocol]):
    """Template for Continuous Confidence Sequence (Waudby-Smith 2021).

    Uses GAVI framework with Known Variance.
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design(
        cls,
        arms: List[str],
        alpha: float,
        variance: float,
        max_n: int,
        sides: Literal["one", "two"] = "two",
    ) -> Protocol:
        method = GAVIMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            max_n=max_n,
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="continuous")
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

            engine = GAVIEngine(protocol)
            sess.call_and_commit(LookResult, engine.run, metrics=metrics)

    def report_progress(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")
