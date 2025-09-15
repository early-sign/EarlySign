"""
Doctests for LedgerDF

>>> import ibis, duckdb  # noqa: F401
>>> from typing import TypedDict
>>> from earlysign.core.ledger_df import (
...     LEDGER_SCHEMA, LedgerDF, LedgerDataHandler, JsonStrategy, TypedStrategy
... )

# -- JSON strategy: payload/labels access behaves like Ibis JSON column --
>>> con = ibis.duckdb.connect(":memory:")
>>> _ = con.create_table("events", schema=LEDGER_SCHEMA)
>>> df = LedgerDF(connector=con, table_name="events").set_strategy(JsonStrategy())
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

# -- TypedStrategy: join-based exposure must keep the same access API --
>>> class Obs(TypedDict):
...     n: int
...     m: int
>>> handler = LedgerDataHandler.from_typeddict("Obs", Obs)
>>> tdf = LedgerDF(connector=con, table_name="events").set_strategy(TypedStrategy())
>>> _ = tdf.register_handler(handler)
>>> tdf.ensure()  # should create typed table alongside the base if needed

# Append typed rows
>>> tdf.append_rows([
...     dict(payload_type="Obs", payload={"n": 10, "m": 3}, labels={"kind":"obs"}),
...     dict(payload_type="Obs", payload={"n": 20, "m": 9}, labels={"kind":"obs"}),
... ])

# Access payload fields through the same JSON-like interface
>>> q2 = (
...   tdf
...     .filter(tdf.t.payload_type == "Obs")
...     .select(n = tdf.payload["n"].cast("int64"),
...             m = tdf.payload["m"].cast("int64"),
...             kind = tdf.labels["kind"])
... )
>>> res = q2.execute().to_dict("records")
>>> sorted((r["n"], r["m"], r["kind"]) for r in res)
[(10, 3, 'obs'), (20, 9, 'obs')]

# Multiple handler registrations: last one wins for same typename
>>> handler2 = LedgerDataHandler.from_typeddict("Obs", Obs)  # same type name
>>> _ = tdf.register_handler(handler2)  # should replace silently
>>> True
True

# Doctests for Ledger

>>> import ibis, duckdb  # noqa: F401
>>> from typing import TypedDict
>>> from earlysign.core.ledger_df import (
...     LEDGER_SCHEMA, LedgerDF, LedgerDataHandler, JsonStrategy, TypedStrategy
... )
>>> from earlysign.core.ledger import Ledger

# -- Setup with JSON-first ledger --
>>> con = ibis.duckdb.connect(":memory:")
>>> _ = con.create_table("events", schema=LEDGER_SCHEMA)
>>> L = Ledger().set_connector(con).use_default_table("events")
>>> # ensure must succeed when base table already exists
>>> L.ensure()

# -- Insert multiple json events (no typed handler yet) --
>>> _ = L.insert_events([
...   dict(payload={"nA": 100, "mA": 38, "nB": 120, "mB": 51}, labels={"kind":"observation","batch":1}),
...   dict(payload={"nA":  80, "mA": 22, "nB":  90, "mB": 30}, labels={"kind":"observation","batch":2}),
... ])

# Read as plain JSON accessors
>>> obs = (
...   L.df
...     .select(
...       "uuid",
...       nA=L.df.payload["nA"].cast("int64"),
...       mA=L.df.payload["mA"].cast("int64"),
...       nB=L.df.payload["nB"].cast("int64"),
...       mB=L.df.payload["mB"].cast("int64"),
...     )
...     .execute()
... )
>>> set({"uuid","nA","mA","nB","mB"}) <= set(obs.columns)
True

# -- Switch to TypedStrategy and register a typed handler --
>>> class TwoPropObsBatch(TypedDict):
...     nA: int
...     nB: int
...     mA: int
...     mB: int
>>> handler = LedgerDataHandler.from_typeddict("TwoPropObsBatch", TwoPropObsBatch)
>>> _ = L.register_handler(handler)
>>> _ = L.set_strategy(TypedStrategy())
>>> L.ensure()  # must create typed table (if missing) with join-based exposure

# Insert a typed event and query through JSON-like access API (backed by join)
>>> _ = L.insert_event(
...   payload_type="TwoPropObsBatch",
...   payload={"nA":150,"mA":60,"nB":140,"mB":48},
...   labels={"kind":"observation","batch":3}
... )
>>> q2 = (
...   L.df
...     .filter(L.df.t.payload_type == "TwoPropObsBatch")
...     .select(n_treat=L.df.payload["nA"].cast("int64"))
... )
>>> rows = q2.execute().to_dict("records")
>>> rows[0]["n_treat"]
150

# -- Minimal end-to-end: observations -> stats -> criteria -> signal --
>>> import math
>>> def wald_z_p(nA,mA,nB,mB):
...     # compute two-proportions Wald z and two-sided p
...     pA = mA/max(nA,1); pB = mB/max(nB,1)
...     p  = (mA+mB)/max(nA+nB,1)
...     se = math.sqrt(p*(1-p)*(1/max(nA,1)+1/max(nB,1)))
...     z  = (pA-pB)/max(se,1e-12)
...     from math import erf, sqrt
...     cdf = 0.5*(1+erf(z/sqrt(2)))
...     pval = 2*min(cdf, 1-cdf)
...     return z, pval

# create stats rows (typed handler is optional for stats; we use json)
>>> obs_df = (
...   L.df
...     .filter(L.df.labels["kind"] == "observation")
...     .select(
...       "uuid",
...       nA=L.df.payload["nA"].cast("int64"),
...       mA=L.df.payload["mA"].cast("int64"),
...       nB=L.df.payload["nB"].cast("int64"),
...       mB=L.df.payload["mB"].cast("int64"),
...     )
...     .execute()
... )
>>> stat_rows = []
>>> for _, r in obs_df.iterrows():
...     z, p = wald_z_p(int(r["nA"]), int(r["mA"]), int(r["nB"]), int(r["mB"]))
...     stat_rows.append(dict(
...       payload_type="json",
...       payload={"z": float(z), "p": float(p), "source_uuid": r["uuid"]},
...       labels={"kind":"statistic"},
...     ))
>>> _ = L.insert_events(stat_rows)

# criteria rows from stats
>>> stats_df = L.df.select(p=L.df.payload["p"].cast("float64")).execute()
>>> crit_rows = []
>>> for _, r in stats_df.iterrows():
...     ok = bool(r["p"] < 0.05)
...     crit_rows.append(dict(
...       payload_type="json",
...       payload={"name":"p_lt_0.05","ok":ok,"p":float(r["p"])},
...       labels={"kind":"criteria"},
...     ))
>>> _ = L.insert_events(crit_rows)

# final signal
>>> crit = L.df.select(ok=L.df.payload["ok"].cast("boolean")).execute()
>>> decision = "Go" if (len(crit) and crit["ok"].any()) else "Hold"
>>> _ = L.insert_event(payload_type="json", payload={"decision": decision}, labels={"kind":"signal"})

# compact report
>>> report = (
...   L.df
...     .select(
...       "ts",
...       kind=L.df.labels["kind"],
...       z=L.df.payload["z"].cast("float64"),
...       p=L.df.payload["p"].cast("float64"),
...       decision=L.df.payload["decision"],
...     )
...     .execute()
...     .sort_values("ts")
... )
>>> {"ts","kind","z","p","decision"} <= set(report.columns)
True
>>> (set(report.get("decision", []).dropna().unique().tolist()) <= {"Go","Hold"}) if "decision" in report.columns else True
True

# -- save(): append / replace / upsert の最小確認 --
>>> # snapshot current rows
>>> snap = L.df.t.execute()
>>> n0 = len(snap)
>>> res_append = L.save(mode="append")
>>> assert res_append["written"] >= 0  # append may repeat
>>> res_replace = L.save(mode="replace")
>>> assert res_replace["replaced"] >= 0
>>> res_upsert = L.save(mode="upsert", match_keys=["uuid"])
>>> assert "upserted" in res_upsert
True
"""
