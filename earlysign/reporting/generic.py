"""
earlysign.reporting.generic
===========================

A scheme-agnostic reporter that shows raw ledger events and
namespace x kind counts. Works with ibis-based ledgers.

Examples
--------
>>> import ibis
>>> from earlysign.core.ledger import Ledger, Namespace
>>> from earlysign.reporting.generic import LedgerReporter
>>> conn = ibis.duckdb.connect(":memory:")
>>> L = Ledger(conn, "test")
>>> rep = LedgerReporter(L)  # Direct ledger usage
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

import ibis

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger


@dataclass
class LedgerReporter:
    """
    A generic, scheme-agnostic reporter for any experiment ledger.
    Works with ibis-based ledgers for backend-agnostic operations.
    """

    ledger: "Ledger"

    def ledger_table(self) -> Any:
        """Return the underlying ledger table as ibis expression."""
        return self.ledger.table

    def unique_entities(self) -> list[str]:
        """List all unique experiment entities."""
        table = self.ledger.table
        try:
            # Check if entity column exists and get unique values
            unique_values = table.select(table.entity).distinct().execute()
            entities = [row.entity for row in unique_values if row.entity is not None]
            return sorted(entities)
        except Exception:
            # Column doesn't exist or other error
            return []

    def unique_namespaces(self) -> list[str]:
        """List all unique event namespaces."""
        table = self.ledger.table
        try:
            # Get unique namespace values
            unique_values = table.select(table.namespace).distinct().execute()
            namespaces = [
                row.namespace for row in unique_values if row.namespace is not None
            ]
            return sorted(namespaces)
        except Exception:
            return []

    def unique_kinds(self) -> list[str]:
        """List all unique event kinds."""
        table = self.ledger.table
        try:
            # Get unique kind values
            unique_values = table.select(table.kind).distinct().execute()
            kinds = [row.kind for row in unique_values if row.kind is not None]
            return sorted(kinds)
        except Exception:
            return []

    def namespace_kind_counts(self) -> Any:
        """
        Return counts of events grouped by namespace and kind.

        Returns
        -------
        ibis.Table
            Table with namespace, kind, and count columns
        """
        table = self.ledger.table
        try:
            # Group by namespace and kind, count events
            counts = (
                table.group_by([table.namespace, table.kind])
                .aggregate(count=ibis._.count())
                .order_by([table.namespace, table.kind])
            )
            return counts
        except Exception:
            # Return empty table if error occurs
            return table.limit(0).select(
                namespace=ibis.literal("").cast("string"),
                kind=ibis.literal("").cast("string"),
                count=ibis.literal(0).cast("int64"),
            )
