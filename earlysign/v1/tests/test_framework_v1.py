"""
Doctests for earlysign.v1.framework integration.

Covers the flow:
  Session -> Ingest -> Read (Projector) -> WriteModel (derived fact)

>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.v1.framework.session import Session
>>> from earlysign.v1.framework.write_models import WriteModel
>>> from earlysign.v1.methods.actions import Ingest, Decision
>>> from earlysign.v1.methods.binomial import BatchObservation, BinomialSummaryFact
>>> from earlysign.v1.methods.group_sequential.binomial import BinomialZProjector

# --- Setup in-memory ledger ---
>>> con = ibis.duckdb.connect(":memory:")
>>> base_ledger = Ledger(con, "events")
>>> base_ledger.ensure()
>>> ledger = base_ledger.bind(experiment_id="v1_demo")

# --- Step 1. Ingest Data ---
>>> with Session(ledger) as sess:
...     _ = Ingest(sess, BatchObservation(n=100, success=38, arm="C"))
...     _ = Ingest(sess, BatchObservation(n=120, success=51, arm="T"))

# --- Step 2. Read and Analyze (Tier 1 Projection) ---
>>> with Session(ledger) as sess:
...     summary_c = sess.Read(BinomialSummaryFact(identity="s_c", filter_arm="C")).data
...     summary_t = sess.Read(BinomialSummaryFact(identity="s_t", filter_arm="T")).data
...     print(f"C: {summary_c.n}, {summary_c.successes}")
...     print(f"T: {summary_t.n}, {summary_t.successes}")
C: 100, 38
T: 120, 51

# --- Step 3. Tier 2 Projection (Statistical Inference) ---
>>> with Session(ledger) as sess:
...     analysis = sess.Read(BinomialZProjector(boundary=1.96)).data
...     print(round(analysis.z_stat, 3))
...     print(analysis.is_rejected)
0.677
False

# --- Step 4. Record Decision ---
>>> from pydantic import BaseModel
>>> class MyDecision(BaseModel):
...     status: str
>>> with Session(ledger) as sess:
...     analysis_traced = sess.Read(BinomialZProjector(boundary=1.96))
...     if not analysis_traced.data.is_rejected:
...         _ = Decision(sess, MyDecision(status="CONTINUE"), trace=analysis_traced.trace)

# --- Inspect Ledger ---
>>> df = ledger.t.execute()
>>> # 2 BatchObservations + 1 Decision
>>> len(df)
3
"""
