"""
Base workflow template for EarlySign.

This module defines an abstract base class for *templates* that orchestrate
scheme-specific records and operators into a ready-to-use analysis pipeline.
Templates are intended to be portable facades over lower-level framework modules.
"""

from typing import Any, Dict

from ibis import BaseBackend


class TemplateBase:
    def __init__(
        self,
        connector: BaseBackend | str,
        experiment_id: str,
        table_name: str | None = None,
    ) -> None:
        raise NotImplementedError()

    def update(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def set_design(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError()

    def status(self) -> Any:
        raise NotImplementedError()

    def report_results(self) -> str:
        """Check final result"""
        raise NotImplementedError()

    def report_history(self) -> str:
        """Visualize test history"""
        raise NotImplementedError()
