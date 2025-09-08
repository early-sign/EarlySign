"""
earlysign.runtime.runners
=========================

Generic runners that execute experiment modules with different strategies.

Runners provide the execution environment while modules define the experiment logic.
This separation allows for different execution patterns (sequential, parallel, batch, etc.)
while keeping experiment definitions portable.

Examples
--------
>>> from earlysign.runtime.experiment_module import ExperimentModule
>>> from earlysign.runtime.runners import SequentialRunner
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>>
>>> # runner = SequentialRunner(custom_module, PolarsLedger())
>>> # runner.add_observations(**data)
>>> # result = runner.analyze()
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable

from earlysign.core.ledger import Ledger
from earlysign.runtime.experiment_module import ExperimentModule, AnalysisResult


class SequentialRunner:
    """
    Basic sequential experiment runner.

    Provides a simple execution environment for experiment modules with:
    - Setup and teardown
    - Logging and monitoring
    - Error handling
    - Result aggregation
    """

    def __init__(self, module: ExperimentModule, ledger: Optional[Ledger] = None):
        self.module = module
        self._ledger = ledger
        self._results_history: List[AnalysisResult] = []

        if ledger is not None:
            self.setup(ledger)

    def setup(self, ledger: Ledger) -> None:
        """Setup the runner with a specific ledger backend."""
        self._ledger = ledger
        self.module.setup(ledger)

    def add_observations(self, **kwargs: Any) -> None:
        """Add observations through the module."""
        if self._ledger is None:
            raise RuntimeError(
                "Runner not setup. Call setup(ledger) first or provide ledger in constructor."
            )

        self.module.add_observations(**kwargs)

    def analyze(self) -> AnalysisResult:
        """Run analysis and store results."""
        if self._ledger is None:
            raise RuntimeError(
                "Runner not setup. Call setup(ledger) first or provide ledger in constructor."
            )

        result = self.module.analyze()
        self._results_history.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary including module and runner state."""
        summary = self.module.get_summary()
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
        """Reset both module and runner state."""
        self.module.reset()
        self._results_history.clear()


class BatchRunner:
    """
    Runner for batch processing multiple experiment modules.

    Useful for simulation studies or comparing different experiment designs.
    """

    def __init__(
        self,
        modules: List[ExperimentModule],
        ledger_factory: Optional[Callable[[], Optional[Ledger]]] = None,
    ):
        self.modules = modules
        self.ledger_factory = ledger_factory or (lambda: None)
        self.runners: List[SequentialRunner] = []

    def setup(self) -> None:
        """Setup all modules with separate ledger instances."""
        for module in self.modules:
            ledger = self.ledger_factory()
            runner = SequentialRunner(module, ledger)
            self.runners.append(runner)

    def add_observations_all(self, **kwargs: Any) -> None:
        """Add the same observations to all modules."""
        for runner in self.runners:
            runner.add_observations(**kwargs)

    def analyze_all(self) -> List[AnalysisResult]:
        """Analyze all modules and return results."""
        return [runner.analyze() for runner in self.runners]

    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get comparative summary across all modules."""
        summaries = [runner.get_summary() for runner in self.runners]

        return {
            "total_modules": len(self.modules),
            "modules": summaries,
            "stopped_count": sum(1 for s in summaries if s.get("is_stopped", False)),
        }
