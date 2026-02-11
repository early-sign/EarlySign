from typing import Any, Dict, List, Literal, Optional

import earlysign.schema.ES3.Base as ES3_BASE
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


class AsymptoticConfidenceSequenceMaharaj2023Template(TemplateBase[Protocol]):
    r"""Template for Asymptotic Confidence Sequences (Maharaj et al. 2023).

    This template implements the practical, variance-adaptive approaches described
    in Maharaj et al. (2023) for constructing Asymptotic Confidence Sequences (CS).
    It is specifically designed for **Unknown Variance** scenarios, estimating
    effective variance dynamically from the data stream.

    It supports two distinct design philosophies:

    1. **Budget-based (`design_from_budget`)**: Optimizes the CS boundary for a
       fixed target sample size ($max\_n$). Best when resources/time are constrained.
    2. **Effect-based (`design_from_effect`)**: Optimizes the CS boundary for a
       specific Minimum Detectable Effect ($mde$). Best when business sensitivity
       drives the experiment.

    Reference:
        Maharaj, A., et al. (2023). Anytime-Valid Confidence Sequences in an
        Enterprise A/B Testing Platform. WWW '23 Companion.
        https://doi.org/10.1145/3543873.3584635

    Examples:
        >>> import ibis, duckdb  # noqa: F401
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.schema.ES3.AVI.Log import DecisionStatus
        >>> from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
        >>> from earlysign.v1.templates.AsymptoticConfidenceSequence_Maharaj2023 import AsymptoticConfidenceSequenceMaharaj2023Template
        >>>
        >>> # Setup ledger
        >>> conn = ibis.connect("duckdb://:memory:")
        >>> ledger = Ledger(conn, "events")
        >>> ledger.ensure()
        >>> ledger = ledger.bind(experiment_id="maharaj_test_001")
        >>>
        >>> # Initialize template
        >>> template = AsymptoticConfidenceSequenceMaharaj2023Template(ledger)
        >>>
        >>> # 1. Design from BUDGET (optimizing for max_n=1000)
        >>> protocol = template.design_from_budget(
        ...     arms=ES3_BASE.TwoArmComparison(control_arm_name="control", treatment_arm_name="treatment"),
        ...     alpha=0.05,
        ...     max_n=1000,
        ...     variance=None, # Estimated from data
        ...     sides="two"
        ... )
        >>> template.set_protocol(protocol)
        >>>
        >>> # 2. Simulate Data Update
        >>> batch = [BinomialArmData(n=800, success=400, arm="control"),
        ...          BinomialArmData(n=800, success=480, arm="treatment")]
        >>> template.update(batch)
        >>> report = template.report_progress()
        >>> report["status"]
        'stop_efficacy'
    """

    _protocol_class = Protocol

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @classmethod
    def design_from_budget(
        cls,
        arms: ES3_BASE.ArmStructure,
        alpha: float,
        max_n: int,
        variance: Optional[float] = None,
        sides: Literal["one", "two"] = "two",
    ) -> Protocol:
        """
        Design an Asymptotic CS optimized for a fixed budget.

        **Design Principle:**
        Leverages the **GAVI (Generalized Anytime-Valid Inference)** framework.
        While GAVI defines the statistical construction of the boundary,
        this design path optimizes GAVI's tuning parameters to achieve
        the maximum possible sensitivity (narrowest boundary) at a
        pre-defined target sample size ($max\\_n$).

        **Philosophy:**
        "I have a budget of $N$ samples. Use the GAVI framework to give me
        the tightest possible boundary at that specific point."

        **Parameter Role:**

        - `max_n`: Acts as the optimization target for the GAVI boundary.
          The boundary minimization (using Lambert W approximation) is
          centered around this value to ensure maximal sensitivity when
          the budget is reached.

        **Mental Model:**
        Use this when you have a fixed experimental window or resource constraint.
        GAVI's flexibility allows us to "aim" the statistical power at the
        expected end-of-experiment, while maintaining anytime-validity.

        Args:
            arms: List of arm names.
            alpha: Type-1 error rate.
            max_n: The target/maximum sample size to optimize for.
            variance: If provided, uses this fixed variance. If None, estimates from data.
            sides: "one" or "two" sided test.
        """
        method = GAVIMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            max_n=max_n,
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="binary")
        return Protocol(
            name="Asymptotic CS (Budget-based GAVI)", task=task, method=method
        )

    @classmethod
    def design_from_effect(
        cls,
        arms: ES3_BASE.ArmStructure,
        alpha: float,
        mde: float,
        variance: Optional[float] = None,
        sides: Literal["one", "two"] = "two",
    ) -> Protocol:
        """
        Design an Asymptotic CS optimized to detect a target effect.

        **Design Principle:**
        Leverages the **mSPRT (Mixture Sequential Probability Ratio Test)**
        framework. This design path focuses on detecting a specific effect
        magnitude as efficiently as possible by tuning the prior mixing
        distribution of the likelihood ratio.

        **Philosophy:**
        "I need to detect a change of at least $X$. Use the mSPRT framework
        to give me the most efficient boundary for that magnitude of effect."

        **Parameter Role:**

        - `mde`: The Minimum Detectable Effect. It determines the prior mixing
          variance (tau^2) in the mSPRT mixture likelihood. A smaller MDE
          leads to a boundary that is more sensitive to small effects but
          potentially slower to cross for larger ones.

        **Mental Model:**
        Use this when the experiment is driven by business sensitivity to
        a specific lift. mSPRT is the canonical choice when you want to
        minimize the expected time to detect a target effect size.

        Args:
            arms: List of arm names.
            alpha: Type-1 error rate.
            mde: The minimum detectable effect size to optimize for.
            variance: If provided, uses this fixed variance. If None, estimates from data.
            sides: "one" or "two" sided test.
        """
        method = MSPRTMethodSpec(
            alpha=alpha,
            variance=variance,
            sides=sides,
            mde=mde,
        )
        task = TaskSpec(kind="AVI", arms=arms, response_type="binary")
        return Protocol(
            name="Asymptotic CS (Effect-based mSPRT)", task=task, method=method
        )

    def update(self, batch: List[Any]) -> None:
        """
        Update the ledger with new data and run the AVI engine.
        Accepts list of ArmData (Binomial or Continuous).
        """
        if batch:
            with Session(self.ledger) as sess:
                for item in batch:
                    # Determine trace if needed, empty for now
                    sess.commit(item, trace=[])

        with Session(self.ledger) as sess:
            protocol = sess.read(ProtocolProjector(Protocol)).data

            # Determine response type to pick correct scoreboard
            # This logic mimics the standard AVI template but is simplified here
            response_type = getattr(protocol.task, "response_type", "binary")

            if response_type == "binary":
                metrics = sess.read(BinomialScoreboard(identity="metrics"))
            else:
                metrics = sess.read(ContinuousScoreboard(identity="metrics"))

            if not isinstance(protocol.task.arms, ES3_BASE.TwoArmComparison):
                raise NotImplementedError(
                    f"Asymptotic CS on {type(protocol.task.arms).__name__} is not yet supported in this template. "
                    "Currently, only TwoArmComparison is supported."
                )

            # Select Engine
            if protocol.method.kind == "GAVI":
                engine = GAVIEngine(protocol)
            elif protocol.method.kind == "mSPRT":
                engine = mSPRTEngine(protocol)
            else:
                raise ValueError(f"Unknown AVI method kind: {protocol.method.kind}")

            sess.call_and_commit(LookResult, engine.run, metrics=metrics)

    def report_progress(self) -> Dict[str, Any]:
        """Returns the current interim report."""
        with Session(self.ledger) as sess:
            return sess.read(ProgressProjector()).data.model_dump(mode="json")

    def report_result(self) -> Dict[str, Any]:
        with Session(self.ledger) as sess:
            return sess.read(FinalProjector()).data.model_dump(mode="json")
