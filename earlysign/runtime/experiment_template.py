"""
earlysign.runtime.experiment_template
====================================

Base classes and infrastructure for experiment templates.

This module provides the foundational building blocks for creating portable,
reusable experiment templates.

Examples
--------
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.runtime.experiment_template import ExperimentTemplate
>>>
>>> # Example template implementation (inherit from ExperimentTemplate)
>>> class MyTemplate(ExperimentTemplate):
...     def analyze(self, ledger):
...         # Your analysis logic here
...         return {"result": "example"}
...     def _populate_batch(self, ledger, batch_data):
...         # Required abstract method implementation
...         pass
...     def configure_components(self):
...         # Required abstract method implementation
...         return {}
...     def extract_results(self, results):
...         # Required abstract method implementation
...         return results
...     def register_design(self, ledger):
...         # Required abstract method implementation
...         pass
>>>
>>> template = MyTemplate("my_experiment")
>>> conn = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(conn, "test")
>>> template.setup(ledger)
>>> template._is_setup
True

Examples
--------
>>> from earlysign.runtime.experiment_template import ExperimentTemplate, AnalysisResult
>>> from earlysign.core.components import Observation
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>>
>>> class MyTemplate(ExperimentTemplate):
...     def configure_components(self): return {"ingestor": Observation()}
...     def register_design(self, ledger): pass
...     def extract_results(self, ledger): return AnalysisResult(should_stop=False, statistic_value=0.0, threshold_value=None, look_number=1)
...     def _populate_batch(self, batch, **kwargs): pass
>>>
>>> template = MyTemplate("test")
>>> conn = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(conn, "test")
>>> template.setup(ledger)
>>> template._is_setup
True
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any

from earlysign.core.ledger import Ledger
from earlysign.core.ledger import Namespace


@dataclass
class AnalysisResult:
    """Results from a single analysis look."""

    # Core results
    should_stop: bool
    statistic_value: float
    threshold_value: Optional[float]
    look_number: int

    # Additional context
    group_a_rate: Optional[float] = None
    group_b_rate: Optional[float] = None
    total_sample_size: Optional[int] = None

    # Method-specific results
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    # Raw event data for advanced use
    statistic_event: Optional[Any] = None
    criteria_event: Optional[Any] = None
    signal_event: Optional[Any] = None


class ExperimentTemplate(ABC):
    """
    Base class for portable experiment templates.

    Encapsulates all the logic for a specific type of experiment including:
    - Design parameters and validation
    - Data ingestion and validation
    - Component configuration (statistics, criteria, signalers)
    - Analysis pipeline coordination
    - Results interpretation

    Users can subclass this to create portable, shareable experiment definitions.
    """

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.ledger: Optional[Ledger] = None
        self._is_setup = False
        self._current_look = 0

    @abstractmethod
    def configure_components(self) -> Dict[str, Any]:
        """
        Configure the components (statistics, criteria, signalers, ingestors).

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys: 'statistic', 'criteria', 'signaler', 'ingestor'
            and optionally 'recommender' or other custom components.
        """
        pass

    @abstractmethod
    def register_design(self, ledger: Ledger) -> None:
        """Register the experimental design to the ledger."""
        pass

    @abstractmethod
    def extract_results(self, ledger: Ledger) -> AnalysisResult:
        """Extract analysis results from ledger events."""
        pass

    def setup(self, ledger: Ledger) -> None:
        """Setup the template with a specific ledger backend."""
        self.ledger = ledger
        self.components = self.configure_components()
        self.register_design(ledger)
        self._is_setup = True

    def add_observations(self, **kwargs: Any) -> None:
        """Add observations using the configured ingestor."""
        if not self._is_setup or self.ledger is None:
            raise RuntimeError("Template not setup. Call setup(ledger) first.")

        self._current_look += 1
        time_index = f"t{self._current_look}"
        step_key = f"look-{self._current_look}"

        # Use the configured ingestor
        ingestor = self.components["ingestor"]
        batch = ingestor.create_batch()

        # Let subclasses handle the specific observation format
        self._populate_batch(batch, **kwargs)

        # Register the batch
        success = ingestor.register_batch(
            self.ledger, self.experiment_id, step_key, time_index, batch
        )

        if not success:
            raise ValueError(
                f"Failed to register observations: {batch.validation_errors}"
            )

    @abstractmethod
    def _populate_batch(self, batch: Any, **kwargs: Any) -> None:
        """Populate the observation batch with data. Subclasses implement specific logic."""
        pass

    def analyze(self) -> AnalysisResult:
        """Run the analysis pipeline and return results."""
        if not self._is_setup or self.ledger is None:
            raise RuntimeError("Template not setup. Call setup(ledger) first.")

        if self._current_look == 0:
            raise ValueError(
                "No observations registered yet. Call add_observations() first."
            )

        time_index = f"t{self._current_look}"
        step_key = f"look-{self._current_look}"

        # Run component pipeline
        for component_name, component in self.components.items():
            if hasattr(component, "step"):
                component.step(
                    self.ledger, str(self.experiment_id), str(step_key), str(time_index)
                )

        # Extract and return results
        return self.extract_results(self.ledger)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment state."""
        if not self._is_setup:
            return {
                "experiment_id": str(self.experiment_id),
                "status": "not_setup",
                "current_look": self._current_look,
            }

        # Base summary - subclasses can extend
        return {
            "experiment_id": str(self.experiment_id),
            "status": "ready" if self._is_setup else "not_setup",
            "current_look": self._current_look,
            "components": list(self.components.keys()),
        }

    def reset(self) -> None:
        """Reset the template to initial state (but keep configuration)."""
        self._current_look = 0
        # Note: This doesn't clear the ledger - that's the runner's responsibility
