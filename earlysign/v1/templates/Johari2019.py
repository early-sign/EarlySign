from typing import Any, Dict, List, Literal

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.AVI import (
    MSPRTMethodSpec,
    Protocol,
    TaskSpec,
)
from earlysign.schema.ES3.AVI.Log import LookResult
from earlysign.v1.framework.projector import ProtocolProjector
from earlysign.v1.framework.session import Session
from earlysign.v1.methods.AVI import mSPRTEngine
from earlysign.v1.methods.AVI.reporting import FinalProjector, ProgressProjector
from earlysign.v1.methods.binomial import Scoreboard as BinomialScoreboard
from earlysign.v1.templates.base import TemplateBase


class Johari2019Template(TemplateBase[Protocol]):
    """
    Template for Anytime-Valid Inference (AVI) using mSPRT as described in Johari et al. (2019).

    This template implements the Mixture Sequential Probability Ratio Test (mSPRT) which
    allows for continuous monitoring of A/B tests. Unlike traditional fixed-horizon tests,
    mSPRT produces "anytime-valid" p-values that allow you to stop the test as soon as
    significance is observed, without inflating the false positive rate.

    The method uses a Normal mixing distribution N(0, tau^2) over the effect size.
    The parameter `tau` determines the "characteristic scale" of the effect we are
    looking for. It is mathematically equivalent to the Minimum Detectable Effect (MDE)
    parameter in other AVI systems.

    Reference:
        Johari, R., Pekelis, L., & Walsh, J. (2019).
        Always Valid Inference: Continuous Monitoring of A/B Tests.
        https://arxiv.org/abs/1512.04922

    Doctests:
        This example replicates a scenario similar to the Johari et al. blog post.
        We simulate an A/B test with a control conversion rate of 0.25 and a
        treatment rate of 0.35 (a 10 percentage point lift).

        >>> import ibis
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData

        # 1. Setup Ledger
        # We use an in-memory DuckDB for this demonstration.
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="johari_2019_blog_scenario")

        # 2. Design Experiment
        # We set alpha=0.05. The 'tau' parameter is set to 0.1, reflecting
        # our expected lift magnitude.
        >>> template = Johari2019Template(ledger)
        >>> protocol = Johari2019Template.design_binomial(
        ...     arms=["control", "treatment"],
        ...     alpha=0.05,
        ...     tau=0.1,
        ...     sides="two"
        ... )
        >>> template.set_protocol(protocol)

        # 3. Initial Monitoring (Small Sample Size)
        # Even with unequal allocation (Control: 200, Treatment: 100),
        # the mSPRT correctly adjusts the confidence boundary.
        # Control: 50/200 (25%), Treatment: 35/100 (35%)
        >>> batch1 = [
        ...     BinomialArmData(n=200, success=50, arm="control"),
        ...     BinomialArmData(n=100, success=35, arm="treatment"),
        ... ]
        >>> template.update(batch1)
        >>> report1 = template.report_progress()
        >>> report1["status"]
        'continue'
        >>> abs(report1["trajectory"] - 0.1) < 1e-9
        True
        >>> report1["sample_n"]  # Total samples across arms
        300

        # 4. Final Detection (Accumulated Evidence)
        # As more data arrives, the trajectory persists at +10% lift.
        # Total Control: 250/1000 (25%), Total Treatment: 350/1000 (35%)
        # The anytime-valid boundary will eventually be crossed.
        >>> batch2 = [
        ...     BinomialArmData(n=800, success=200, arm="control"),
        ...     BinomialArmData(n=900, success=315, arm="treatment"),
        ... ]
        >>> template.update(batch2)
        >>> report2 = template.report_progress()
        >>> report2["status"]
        'stop_efficacy'
        >>> report2["sample_n"]
        2000
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design_binomial(
        cls,
        arms: List[str],
        alpha: float,
        tau: float = 0.1,
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
            variance=None,  # Estimated adaptiveley
            sides=sides,
            mde=tau,  # tau maps to mde in our mSPRTEngine
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="binary")
        return Protocol(name="Johari2019 mSPRT", task=task, method=method)

    def update(self, batch: List[Any]) -> None:
        """
        Update the ledger with new data and run the AVI engine.
        Supports unequal allocation naturally.
        """
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    sess.Commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.Read(ProtocolProjector(Protocol)).data
            metrics = sess.Read(BinomialScoreboard(identity="metrics"))

            # Johari 2019 is primarily mSPRT
            engine = mSPRTEngine(protocol)
            sess.CallAndCommit(LookResult, engine.run, metrics=metrics)

    def report_progress(self) -> Dict[str, Any]:
        """Returns the current interim report."""
        with Session(self.ledger) as sess:
            return sess.Read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        """Returns the final study result."""
        with Session(self.ledger) as sess:
            return sess.Read(FinalProjector()).data.model_dump(mode="json")
