"""
Doctests for earlysign.v1.framework integration.

Covers the flow:
  Session -> Ingest -> Read (Projector) -> Writer (derived fact)

>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.v1.framework.session import Session
>>> from earlysign.v1.framework.writer import Writer
>>> from earlysign.v1.methods.actions import Ingest, Decision, UpdateProtocol
>>> from earlysign.v1.methods.binomial import BatchObservation, BinomialSummaryFact
>>> from earlysign.v1.templates.binomial_ab import BinomialABTemplate, BinomialABTaskSpec, BinomialABProtocol
>>> import earlysign.schema.ES3.GST as GST
>>> from earlysign.schema.ES3.GST.Log import DecisionStatus

# --- Setup in-memory ledger ---
>>> con = ibis.duckdb.connect(":memory:")
>>> base_ledger = Ledger(con, "events")
>>> base_ledger.ensure()
>>> ledger = base_ledger.bind(experiment_id="v1_demo")

# --- Step 0. Define and Ingest Protocol ---
>>> # Use Template factory to ensure valid schema
>>> task = BinomialABTaskSpec(
...     arms=["C", "T"],
...     response_type=GST.ResponseType.BINARY,
...     efficacy=GST.EfficacyRequirement(alpha=0.05),
...     futility=GST.FutilityRequirement(power=0.8),
...     hypotheses=GST.HypothesisSpec(
...         h_null="Diff <= 0",
...         h_alt="Diff > 0.01",
...         test_logic=GST.SuperiorityHypothesis(superiority_margin=0.01),
...         target_effect=GST.BinaryEffectSize(
...             proportions={"C": 0.1, "T": 0.11}
...         )
...     )
... )
>>> # Design a simple 1-look protocol (effectively fixed sample / simple Z-test)
>>> protocol = BinomialABTemplate.design(
...     task, looks=1, spending_function="pocock",
...     designer_params={"model": "canonical_joint"}
... )
>>> with Session(ledger) as sess:
...     _ = UpdateProtocol(sess, protocol)

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
# Create a template instance to access reporting logic
>>> trial = BinomialABTemplate(ledger)
>>> prog = trial.report_progress()
>>> print(round(prog['z_stat'], 3))
0.677
>>> print(prog['status'])
continue

# --- Step 4. Record Decision ---
# The template handles decision logic inside update() or we can check status
>>> # BinomialABTemplate.update() handles decisions when batches are processed.
>>> # Here we manually just check that the status is CONTINUE.
>>> if prog['status'] == DecisionStatus.CONTINUE_:
...     print("Decision: CONTINUE")
Decision: CONTINUE

# --- Inspect Ledger ---
>>> df = ledger.t.execute()
>>> # 1 Protocol + 2 BatchObservations
>>> len(df)
3
"""
