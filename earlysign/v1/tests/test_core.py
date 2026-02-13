"""
Doctests

>>> import ibis, duckdb  # noqa: F401

# -- JSON strategy: payload/labels access behaves like Ibis JSON column --
>>> from earlysign.core.ledger import Ledger
>>> con = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(con, "events")
>>> ledger.ensure()  # should be a no-op since table exists

# Empty select should still compile and execute
>>> df = ledger.t
>>> q = df.select(
...     df.uuid,
...     x = df.payload["x"],          # JSON scalar (string) extraction
...     kind = df.attributes["kind"],     # JSON scalar from attributes
... )
>>> out = q.execute()
>>> list(out.columns)
['uuid', 'x', 'kind']

# Insert a couple of JSON events and read back via JSON accessors
>>> rows = [
...     dict(type="", payload={"x":"A"}, attributes={"kind":"k1"}),
...     dict(type="", payload={"x":"B"}, attributes={"kind":"k2"}),
... ]
>>> ledger.insert(data=rows[0]["payload"], attributes=rows[0]["attributes"])
UUID(...)
>>> ledger.insert(data=rows[1]["payload"], attributes=rows[1]["attributes"])
UUID(...)
>>> got = df.select(df.payload["x"].name("x")).execute().to_dict("records")
>>> sorted(v["x"] for v in got)
['A', 'B']

# -- Delegation check: these methods must return Ibis TableExpr unmodified --
>>> from ibis.expr.types import Table as TableExpr
>>> isinstance(df.select("uuid"), TableExpr)
True
>>> isinstance(df.filter(df.type == "json"), TableExpr)
True
>>> isinstance(df.limit(1), TableExpr)
True
>>> df  # the underlying ibis table
DatabaseTable: events
  uuid      string
  type      string
  payload   json
  attributes    json
  timestamp timestamp('UTC', 6)
  metadata  json

# Doctests for Ledger

>>> import ibis, duckdb  # noqa: F401
>>> from earlysign.core.ledger import Ledger

# -- Setup with JSON-based ledger --
>>> con = ibis.duckdb.connect(":memory:")
>>> L = Ledger(con, "events")
>>> # ensure must succeed when base table already exists
>>> L.ensure()

# -- Insert multiple json events --
>>> _ = [L.insert(data=row["payload"], attributes=row["attributes"]) for row in [
...   dict(payload={"nA": 100, "mA": 38, "nB": 120, "mB": 51}, attributes={"kind":"observation","batch":1}),
...   dict(payload={"nA":  80, "mA": 22, "nB":  90, "mB": 30}, attributes={"kind":"observation","batch":2}),
... ]]

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
>>> L.insert(
...   data={"nA":150,"mA":60,"nB":140,"mB":48},
...   attributes={"kind":"observation","batch":3}
... )
UUID(...)
>>> q2 = (
...     L.t.filter(L.t.type == "dict")
...     .order_by(L.t.timestamp.desc())
...     .limit(1)
...     .select(n_treat=L.t.payload["nA"].cast("int64"))
... )
>>> rows = q2.execute().to_dict("records")
>>> rows[0]["n_treat"]
150

# Simple verification that basic operations work
>>> len_before = len(L.t.execute())
>>> L.insert(data={"test": True}, attributes={"kind": "test"})
UUID(...)
>>> len_after = len(L.t.execute())
>>> len_after > len_before
True
"""
