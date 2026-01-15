from typing import Any, Dict

import pandas as pd
from ibis import BaseBackend


class TemplateBase:
    """
    Base workflow template for EarlySign.

    This class defines an abstract base class for *templates* that orchestrate
    scheme-specific records and operators into a ready-to-use analysis pipeline.
    """

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

    def get_history(self) -> pd.DataFrame:
        """Get complete history of the trial with all key metrics."""
        raise NotImplementedError()
