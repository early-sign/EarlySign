"""
earlysign.runtime.sequential_runner
===================================

Sequential experiment runner implementation.

This module provides the main SequentialRunner class for executing experiments
in a sequential manner, processing observations one batch at a time.

Examples
--------
>>> from earlysign.runtime.experiment_module import ExperimentModule
>>> from earlysign.runtime.sequential_runner import SequentialRunner
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>>
>>> # runner = SequentialRunner(custom_module, PolarsLedger())
>>> # runner.add_observations(**data)
>>> # result = runner.analyze()
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

from earlysign.core.ledger import Ledger
from earlysign.runtime.experiment_module import ExperimentModule, AnalysisResult


class SequentialRunner:
    """
    Runner for sequential experiments with real-time analysis.

    This runner is designed for experiments where:
    - Observations arrive over time
    - Analysis happens after each batch of observations
    - Decisions are made based on sequential looks

    This provides the execution environment while the ExperimentModule
    provides the experiment logic.
    """

    def __init__(self, module: ExperimentModule, ledger: Optional[Ledger] = None):
        self.module = module
        self.ledger = ledger

        if ledger is not None:
            self.setup(ledger)

    def setup(self, ledger: Ledger) -> None:
        """Setup the runner with a ledger backend."""
        self.ledger = ledger
        self.module.setup(ledger)

    def add_observations(self, **kwargs: Any) -> None:
        """Add a batch of observations to the experiment."""
        if self.ledger is None:
            raise RuntimeError("Runner not setup. Call setup(ledger) first.")

        self.module.add_observations(**kwargs)

    def analyze(self) -> AnalysisResult:
        """Run analysis and return results."""
        if self.ledger is None:
            raise RuntimeError("Runner not setup. Call setup(ledger) first.")

        return self.module.analyze()

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return self.module.get_summary()

    def reset(self) -> None:
        """Reset the experiment state."""
        self.module.reset()
        if self.ledger:
            # Clear ledger data for this experiment
            # This is backend-specific - for now, just reset module state
            pass


class BatchRunner:
    """
    Runner for batch experiments with predefined observation schedules.

    This runner is designed for experiments where:
    - All observations are available upfront or in large batches
    - Analysis happens at predetermined intervals
    - Focus is on efficient batch processing

    Useful for simulation studies, historical data analysis, or
    experiments with batch data collection.
    """

    def __init__(self, module: ExperimentModule, ledger: Optional[Ledger] = None):
        self.module = module
        self.ledger = ledger
        self._batch_queue: List[Dict[str, Any]] = []

        if ledger is not None:
            self.setup(ledger)

    def setup(self, ledger: Ledger) -> None:
        """Setup the runner with a ledger backend."""
        self.ledger = ledger
        self.module.setup(ledger)

    def add_batch(self, **kwargs: Any) -> None:
        """Queue a batch of observations for later processing."""
        self._batch_queue.append(kwargs)

    def process_batches(self) -> Dict[int, AnalysisResult]:
        """Process all queued batches and return analysis results."""
        if self.ledger is None:
            raise RuntimeError("Runner not setup. Call setup(ledger) first.")

        results = {}

        for i, batch_kwargs in enumerate(self._batch_queue, 1):
            self.module.add_observations(**batch_kwargs)
            results[i] = self.module.analyze()

        # Clear processed batches
        self._batch_queue = []

        return results

    def run_simulation(
        self,
        observation_batches: List[Dict[str, Any]],
        analyze_at: Optional[List[int]] = None,
    ) -> Dict[int, AnalysisResult]:
        """
        Run a complete simulation with predefined observation batches.

        Parameters
        ----------
        observation_batches : List[Dict[str, Any]]
            List of dictionaries, each containing observations for one batch.
        analyze_at : List[int], optional
            List of batch numbers to analyze at. If None, analyzes after each batch.

        Returns
        -------
        Dict[int, AnalysisResult]
            Analysis results keyed by batch number.
        """
        if self.ledger is None:
            raise RuntimeError("Runner not setup. Call setup(ledger) first.")

        if analyze_at is None:
            analyze_at = list(range(1, len(observation_batches) + 1))

        results = {}

        for i, batch_kwargs in enumerate(observation_batches, 1):
            self.module.add_observations(**batch_kwargs)

            if i in analyze_at:
                results[i] = self.module.analyze()

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        summary = self.module.get_summary()
        summary.update({"queued_batches": len(self._batch_queue)})
        return summary

    def reset(self) -> None:
        """Reset the experiment state."""
        self.module.reset()
        self._batch_queue = []
        if self.ledger:
            # Clear ledger data for this experiment
            # This is backend-specific - for now, just reset module state
            pass
