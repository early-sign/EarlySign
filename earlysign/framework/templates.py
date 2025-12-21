from typing import Any, Dict, List, Mapping, Optional

import ibis
import pandas as pd
from ibis import BaseBackend
from ibis.expr.types import Table as TableExpr


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

    @staticmethod
    def _format_history_table(
        records: Mapping[str, Any],
        *,
        index_col: str = "look",
    ) -> TableExpr:
        """
        Join multiple records by their sequence (look) number.

        Parameters
        ----------
        records : Mapping[str, Any]
            Dict of alias -> LedgerRecord
        index_col : str, optional
            Name of the look index column, by default "look"

        Returns
        -------
        TableExpr
            Joined table expression.
        """
        views: List[TableExpr] = []
        for alias, rec in records.items():
            t = rec.order_by_ts(ascending=True, explode=True)
            t = t.mutate(**{index_col: ibis.row_number().over(order_by="ts") + 1})
            # Select look, ts, and everything in payload
            cols = {index_col: t[index_col], "ts": t.ts}
            for field in rec.schema.keys():
                cols[f"{alias}_{field}" if len(records) > 1 else field] = t[field]
            views.append(t.select(**cols))

        if not views:
            raise ValueError("No records provided for joining.")

        result = views[0]
        for v in views[1:]:
            result = result.inner_join(v, [result[index_col] == v[index_col]])

        # Clean up: keep only one version of look and ts (greatest)
        look_col = result[index_col]
        ts_expr = result.ts
        for v in views[1:]:
            ts_expr = ibis.greatest(ts_expr, v.ts)

        # Drop redundant ts columns from joins if they exist
        to_drop = [c for c in result.columns if c.startswith("ts_") or c == "ts"]
        result = result.drop(*to_drop).mutate(**{index_col: look_col, "ts": ts_expr})

        return result


def join_sequential_history(
    records: Mapping[str, Any],
    *,
    index_col: str = "look",
) -> TableExpr:
    """Helper for joining multiple records by their sequence number."""
    return TemplateBase._format_history_table(records, index_col=index_col)

