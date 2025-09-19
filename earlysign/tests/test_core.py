"""
Doctests for LedgerDF

>>> import ibis, duckdb  # noqa: F401
>>> from earlysign.core.ledger_df import LEDGER_SCHEMA, LedgerDF

# -- JSON strategy: payload/labels access behaves like Ibis JSON column --
>>> con = ibis.duckdb.connect(":memory:")
>>> _ = con.create_table("events", schema=LEDGER_SCHEMA)
>>> df = LedgerDF(connector=con, table_name="events")
>>> df.ensure()  # should be a no-op since table exists

# Empty select should still compile and execute
>>> q = df.select(
...     df.t.uuid,
...     x = df.payload["x"],          # JSON scalar (string) extraction
...     kind = df.labels["kind"],     # JSON scalar from labels
... )
>>> out = q.execute()
>>> list(out.columns)
['uuid', 'x', 'kind']

# Insert a couple of JSON events and read back via JSON accessors
>>> rows = [
...     dict(payload={"x":"A"}, labels={"kind":"k1"}),
...     dict(payload={"x":"B"}, labels={"kind":"k2"}),
... ]
>>> df.append_rows(rows)  # strategy handles persist details
>>> got = df.select(df.payload["x"].name("x")).execute().to_dict("records")
>>> sorted(v["x"] for v in got)
['A', 'B']

# -- Delegation check: these methods must return Ibis TableExpr unmodified --
>>> from ibis.expr.types import Table as TableExpr
>>> isinstance(df.select("uuid"), TableExpr)
True
>>> isinstance(df.filter(df.t.payload_type == "json"), TableExpr)
True
>>> isinstance(df.limit(1), TableExpr)
True
>>> df.t  # the underlying ibis table
DatabaseTable: events
  uuid         string
  ts           string
  payload_type string
  labels       json
  payload      json

# Doctests for Ledger

>>> import ibis, duckdb  # noqa: F401
>>> from earlysign.core.ledger_df import LEDGER_SCHEMA, LedgerDF
>>> from earlysign.core.ledger import Ledger

# -- Setup with JSON-based ledger --
>>> con = ibis.duckdb.connect(":memory:")
>>> _ = con.create_table("events", schema=LEDGER_SCHEMA)
>>> L = Ledger().set_connector(con).use_default_table("events")
>>> # ensure must succeed when base table already exists
>>> L.ensure()

# -- Insert multiple json events --
>>> _ = L.insert_events([
...   dict(payload={"nA": 100, "mA": 38, "nB": 120, "mB": 51}, labels={"kind":"observation","batch":1}),
...   dict(payload={"nA":  80, "mA": 22, "nB":  90, "mB": 30}, labels={"kind":"observation","batch":2}),
... ])

# Read using JSON accessors
>>> obs = (
...   L.t
...     .select(
...       "uuid",
...       nA=L.t.payload["nA"].cast("int64"),
...       mA=L.t.payload["mA"].cast("int64"),
...       nB=L.t.payload["nB"].cast("int64"),
...       mB=L.t.payload["mB"].cast("int64"),
...     )
...     .execute()
... )
>>> set({"uuid","nA","mA","nB","mB"}) <= set(obs.columns)
True

# Insert more data and verify JSON access works
>>> _ = L.insert_event(
...   payload_type="TwoPropObsBatch",
...   payload={"nA":150,"mA":60,"nB":140,"mB":48},
...   labels={"kind":"observation","batch":3}
... )
>>> q2 = (
...   L.t
...     .filter(L.t.payload_type == "TwoPropObsBatch")
...     .select(n_treat=L.t.payload["nA"].cast("int64"))
... )
>>> rows = q2.execute().to_dict("records")
>>> rows[0]["n_treat"]
150

# Simple verification that basic operations work
>>> len_before = len(L.t.execute())
>>> _ = L.insert_event(payload={"test": True}, labels={"kind": "test"})
>>> len_after = len(L.t.execute())
>>> len_after > len_before
True
"""
