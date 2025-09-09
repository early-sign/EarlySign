"""
earlysign.runtime.runners
=========================

Generic runners that execute experiment templates with different strategies.

Runners provide the execution environment while templates define the experiment logic.
This separation allows for different execution patterns (sequential, parallel, batch, etc.)
while keeping experiment definitions portable.

Examples
--------
>>> from earlysign.runtime.experiment_template import ExperimentTemplate
>>> from earlysign.runtime.runners import SequentialRunner
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>>
>>> # runner = SequentialRunner(custom_template, PolarsLedger())
>>> # runner.add_observations(**data)
>>> # result = runner.analyze()
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable

from earlysign.core.ledger import Ledger
from earlysign.runtime.experiment_template import ExperimentTemplate, AnalysisResult


class SequentialRunner:
    """
    Basic sequential experiment runner.

    Provides a simple execution environment for experiment templates with:
    - Setup and teardown
    - Logging and monitoring
    - Error handling
    - Result aggregation
    """

    def __init__(self, template: ExperimentTemplate, ledger: Optional[Ledger] = None):
        self.template = template
        self._ledger = ledger
        self._results_history: List[AnalysisResult] = []

        if ledger is not None:
            self.setup(ledger)

    def setup(self, ledger: Ledger) -> None:
        """Setup the runner with a specific ledger backend."""
        self._ledger = ledger
        self.template.setup(ledger)

    def add_observations(self, **kwargs: Any) -> None:
        """Add observations through the template."""
        if self._ledger is None:
            raise RuntimeError(
                "Runner not setup. Call setup(ledger) first or provide ledger in constructor."
            )

        self.template.add_observations(**kwargs)

    def analyze(self) -> AnalysisResult:
        """Run analysis and store results."""
        if self._ledger is None:
            raise RuntimeError(
                "Runner not setup. Call setup(ledger) first or provide ledger in constructor."
            )

        result = self.template.analyze()
        self._results_history.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary including template and runner state."""
        summary = self.template.get_summary()
        summary.update(
            {
                "runner_type": "sequential",
                "total_looks": len(self._results_history),
                "is_stopped": (
                    self._results_history[-1].should_stop
                    if self._results_history
                    else False
                ),
            }
        )
        return summary

    def get_results_history(self) -> List[AnalysisResult]:
        """Get history of all analysis results."""
        return self._results_history.copy()

    def reset(self) -> None:
        """Reset both template and runner state."""
        self.template.reset()
        self._results_history.clear()


class BatchRunner:
    """
    Runner for batch processing multiple experiment templates.

    Useful for simulation studies or comparing different experiment designs.
    """

    def __init__(
        self,
        templates: List[ExperimentTemplate],
        ledger_factory: Optional[Callable[[], Optional[Ledger]]] = None,
    ):
        self.templates = templates
        self.ledger_factory = ledger_factory or (lambda: None)
        self.runners: List[SequentialRunner] = []

    def setup(self) -> None:
        """Setup all templates with separate ledger instances."""
        for template in self.templates:
            ledger = self.ledger_factory()
            runner = SequentialRunner(template, ledger)
            self.runners.append(runner)

    def add_observations_all(self, **kwargs: Any) -> None:
        """Add the same observations to all templates."""
        for runner in self.runners:
            runner.add_observations(**kwargs)

    def analyze_all(self) -> List[AnalysisResult]:
        """Analyze all templates and return results."""
        return [runner.analyze() for runner in self.runners]

    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get comparative summary across all templates."""
        summaries = [runner.get_summary() for runner in self.runners]

        return {
            "total_templates": len(self.templates),
            "templates": summaries,
            "stopped_count": sum(1 for s in summaries if s.get("is_stopped", False)),
        }
