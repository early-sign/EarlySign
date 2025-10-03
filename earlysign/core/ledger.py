"""
High level Ledger facade over a backend table (Ibis).

Design
------
- Append-only event ledger with JSON payload and labels.
- Labels can be "bound" (like a scope) so that queries and inserts
  automatically apply those filters / label merges.
- `bind(**labels)` returns a new Ledger with additional labels bound.
- `unbind(*selectors)` removes bound labels:

Table contract
--------------
Base table has columns:
  - uuid: string (auto-generated)
  - ts: timestamp (UTC) (auto-generated)
  - pkg_version: string (auto-generated)
  - payload_type: string
  - payload: json
  - labels: json

Doctests
--------
>>> import ibis, duckdb  # noqa: F401
>>> from earlysign.core.ledger import Ledger
>>> import re

# Setup
>>> con = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(con, "events"); ledger.ensure()

# Bind two labels, then drop one (exact key)
>>> scoped = ledger.bind(experiment_id="exp1", env="prod")
>>> scoped.labels == {"experiment_id": "exp1", "env": "prod"}
True
>>> scoped2 = scoped.unbind("env")
>>> scoped2.labels == {"experiment_id": "exp1"}
True

# Insert with only remaining bound label applied
>>> _ = scoped2.insert(payload_type="X", payload={"a": 1})

# Some backends differ in JSON key equality semantics; materialize and check in Python.
>>> df = ledger.t.execute()
>>> any(rec["labels"].get("experiment_id") == "exp1" for rec in df.to_dict("records"))
True

# Regex unbind: drop all keys starting with 'site_'
>>> scoped3 = scoped.bind(site_eu=True, site_us=True)
>>> scoped3.labels == {"experiment_id": "exp1", "env": "prod", "site_eu": True, "site_us": True}
True
>>> scoped4 = scoped3.unbind(r"^site_.*")
>>> "site_eu" in scoped4.labels or "site_us" in scoped4.labels
False
>>> set(scoped4.labels.keys()) == {"experiment_id", "env"}
True

# Regex unbind with compiled pattern
>>> p = re.compile(r"^exp.*")
>>> scoped5 = scoped4.unbind(p)
>>> "experiment_id" in scoped5.labels
False

# Drop all bound labels
>>> unscoped = scoped.unbind()
>>> unscoped.labels
{'experiment_id': 'exp1', 'env': 'prod'}
"""

import re
import uuid as uuidlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Pattern, Union

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.expr.types import Table as TableExpr

from earlysign import __version__


@dataclass(frozen=True)
class Ledger:
    """
    A thin, JSON-backed append-only ledger.

    - `labels`: bound labels (scope). They are merged into every insert and
      also applied as filters when reading via `t`.
    - All rows have: uuid (str), ts (ISO8601 str UTC), payload_type (str),
      payload (json), labels (json).
    """

    connector: ibis.BaseBackend | None = None
    table_name: str = "events"
    labels: Dict[str, Any] = field(default_factory=dict)

    # --------- lifecycle ----------
    def set_connector(self, connector: ibis.BaseBackend) -> "Ledger":
        return Ledger(connector, self.table_name, dict(self.labels))

    def use_default_table(self, name: str = "events") -> "Ledger":
        return Ledger(self.connector, name, dict(self.labels))

    def ensure(self) -> None:
        """Ensure the ledger table (with standard schema) exists."""
        if self.connector is None:
            raise RuntimeError("Ledger connector not set")
        if self.table_name in self.connector.list_tables():
            return
        schema = sch.schema(
            dict(
                uuid=dt.string,
                ts=dt.timestamp(
                    timezone="UTC"
                ),  # ISO8601 string to avoid tz/precision drift across backends
                pkg_version=dt.string,
                payload_type=dt.string,
                payload=dt.json,
                labels=dt.json,
            )
        )
        self.connector.create_table(self.table_name, schema=schema)

    # --------- binding / scoping ----------
    def bind(self, **labels: Any) -> "Ledger":
        """Return a new Ledger whose scope includes the given labels."""
        merged = dict(self.labels)
        merged.update(labels)
        return Ledger(self.connector, self.table_name, merged)

    def unbind(self, *patterns: Union[str, Pattern[str]]) -> "Ledger":
        """
        Return a new Ledger with bound labels removed if their keys match ANY pattern.
        Keys for which ANY compiled pattern .search(key) succeeds will be removed.

        Each argument in `patterns` may be:
          - str: compiled via re.compile(...)
          - re.Pattern[str]: used as-is

        Examples:
        >>> from earlysign.core.ledger import Ledger
        >>> import ibis, duckdb, re
        >>> con = ibis.duckdb.connect(":memory:")
        >>> base = Ledger(con, "events"); base.ensure()
        >>> scoped = base.bind(experiment_id="exp1", env="prod")
        >>> _ = scoped.insert(payload_type="X", payload={"a": 1})
        >>> df = base.t.execute()
        >>> any(rec["labels"].get("experiment_id") == "exp1" for rec in df.to_dict("records"))
        True

        # Regex unbind: drop all keys starting with 'site_'
        >>> scoped3 = scoped.bind(site_eu=True, site_us=True)
        >>> scoped3.labels == {"experiment_id": "exp1", "env": "prod", "site_eu": True, "site_us": True}
        True
        >>> scoped4 = scoped3.unbind(r"^site_.*")
        >>> "site_eu" in scoped4.labels or "site_us" in scoped4.labels
        False
        """
        if not patterns:
            return self  # nothing to drop

        compiled: list[Pattern[str]] = []
        for p in patterns:
            if isinstance(p, str):
                compiled.append(re.compile(p))
            else:
                # already a Pattern[str]
                compiled.append(p)

        def _keep_key(k: str) -> bool:
            return not any(rx.match(k) for rx in compiled)

        remaining = {k: v for k, v in self.labels.items() if _keep_key(k)}
        return Ledger(self.connector, self.table_name, remaining)

    # --------- table view ----------
    @property
    def t(self) -> TableExpr:
        """Return a scoped TableExpr filtered by bound labels."""
        if self.connector is None:
            raise RuntimeError("Ledger connector not set")
        t = self.connector.table(self.table_name)
        for k, v in self.labels.items():
            # compare as strings for backend portability
            t = t.filter(t.labels[k].str == str(v))
        return t

    # --------- write ----------
    def insert(
        self,
        payload_type: str,
        payload: Mapping[str, Any],
        labels: Mapping[str, Any] = {},
    ) -> None:
        """
        Insert one row (append-only). Auto-fills uuid and ts.
        The current scope labels (self.labels) are ALWAYS merged into `labels`.
        """
        if self.connector is None:
            raise RuntimeError("Ledger connector not set")

        labels = {**self.labels, **labels}
        row = {
            "uuid": uuidlib.uuid4().hex,
            "ts": datetime.now(timezone.utc),
            "pkg_version": f"earlysign=={__version__}",
            "payload_type": payload_type,
            "payload": dict(payload),
            "labels": labels if labels else None,
        }
        self.connector.insert(self.table_name, [row])
