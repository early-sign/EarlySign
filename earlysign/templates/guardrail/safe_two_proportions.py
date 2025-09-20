"""
Guardrail: Safe Testing (Anytime-Valid, Ville) for two-proportions.

This is a convenience wrapper over the ABTesting safe pipeline, but emits
"alert"/"ok" style signals for monitoring use-cases.
"""

from typing import Optional, Tuple

from earlysign.core.ledger import Ledger
from earlysign.framework.templates import TemplateBase, AnalysisResult
from earlysign.templates.ab_testing.safe_two_proportions import (
    SafeTestingTwoProportions,
)


class SafeTwoProportions(TemplateBase):
    """
    Guardrail facade that reuses SafeTestingTwoProportions under the hood.

    Signals
    -------
    - "alert"   : when AB safe test returns "reject"
    - "ok"      : when AB safe test returns "continue"
    - "hold"    : when AB safe test returns "stop_futility" (optional policy)
    """

    def build_components(self) -> None:
        # Delegate: keep an internal AB-safe template with its own ids/params
        self._ab = SafeTestingTwoProportions(
            self.experiment_id, namespace=self.namespace + ":ab"
        )
        self._ab.setup(self.scoped)

    # Design / configuration are thin passthroughs
    def design_safe(self, *, alpha: float) -> None:
        self._ab.design_safe(alpha=alpha)

    def set_priors(
        self, *, null: Tuple[float, float], alt: Tuple[float, float]
    ) -> None:
        self._ab.set_priors(null=null, alt=alt)

    def set_gate(self, *, enabled: bool, min_total: int) -> None:
        self._ab.set_gate(enabled=enabled, min_total=min_total)

    # Data IO
    def add_observations(self, *, nA: int, mA: int, nB: int, mB: int) -> None:
        self._ab.add_observations(nA=nA, mA=mA, nB=nB, mB=mB)

    # Analyze → map signals to guardrail terms
    def run_analysis(self) -> AnalysisResult:
        res = self._ab.analyze()
        if res.signal == "reject":
            return AnalysisResult(
                signal="alert",
                reason="ville",
                stopped=True,
                summary=res.summary,
                raw=res.raw,
            )
        if res.signal == "stop_futility":
            return AnalysisResult(
                signal="hold",
                reason="futility",
                stopped=True,
                summary=res.summary,
                raw=res.raw,
            )
        return AnalysisResult(
            signal="ok", reason="none", stopped=False, summary=res.summary, raw=res.raw
        )
