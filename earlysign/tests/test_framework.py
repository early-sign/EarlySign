"""
Doctests for earlysign.framework integration (current API).

Covers the flow:
  Ledger.ensure() → Ledger.bind() → Records → Operators → Derived Records.

Steps:
  1. Insert binomial observation snapshots (two-proportions scheme).
  2. Compute Wald Z statistic via operator.
  3. Compute information time (counts-based: n_total & planned_max_n).
  4. Compute GS boundary from design and make a decision.

>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.integration.execution.schemes.two_proportions.records import BinomialCountsRecord
>>> from earlysign.integration.execution.methods.group_sequential.records.statistics import WaldZStatisticRecord
>>> from earlysign.integration.execution.schemes.two_proportions.operators import WaldZStatistic
>>> from earlysign.integration.execution.methods.group_sequential.operators.decision import GSDecisionFromWaldZ
>>> from earlysign.integration.execution.methods.group_sequential.records.boundary import GroupSequentialBoundaryRecord
>>> from earlysign.integration.execution.methods.group_sequential.records.decision import GroupSequentialDecisionSignalRecord
>>> from earlysign.integration.execution.methods.group_sequential.records.design import GroupSequentialDesignRecord
>>> from earlysign.integration.execution.methods.group_sequential.records.info import InformationTimeRecord
>>> from earlysign.integration.execution.schemes.two_proportions.operators import InformationTime
>>> from earlysign.integration.execution.methods.group_sequential.operators.boundary import BoundaryFromDesign


# --- Setup in-memory ledger ---------------------------------------------------
>>> con = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(con, "events")
>>> ledger.ensure()
>>> scoped = ledger.bind(experiment_id="exp_demo")

# --- Step 1. Insert binomial observation snapshots ----------------------------
>>> counts = BinomialCountsRecord(name="counts1").attach(scoped)
>>> _ = counts.insert({"nA": 100, "mA": 38, "nB": 120, "mB": 51})
>>> _ = counts.insert({"nA": 150, "mA": 60, "nB": 140, "mB": 48})

# --- Step 2. Compute Wald Z statistic -----------------------------------------
# Operator signature in the new framework takes the scoped ledger and out_id.
>>> _ = WaldZStatistic(scoped, cum_counts=counts, out_id="wald1", pooled=True).run()

# --- Step 3. Compute information time (counts-based) --------------------------
# InformationTime now takes cum_counts record and planned_max_n directly.
>>> _ = InformationTime(scoped, out_id="info1", cum_counts=counts, planned_max_n=300).run()

# --- Step 4. GS design → boundary → decision ---------------------------------
>>> design_payload = {
...     "alpha": 0.05,
...     "hypothesis": {"structure": "two_sided_symmetric"},
...     "statistic": {"kind": "wald_z", "scale": "z"},
...     "efficacy": {"style": "alpha_spending", "family": "obf"},
...     "futility": {"mode": "symmetric", "binding_mode": "non_binding"},
...     "planned_max_n": 600,
...     "planned_info_times": [0.5, 1.0],
... }
>>> design_rec = GroupSequentialDesignRecord(name="design1").attach(scoped)
>>> _ = design_rec.insert(design_payload)
>>> _ = BoundaryFromDesign(
...         scoped,
...         design=GroupSequentialDesignRecord(name="design1").attach(scoped),
...         info=InformationTimeRecord(name="info1").attach(scoped),
...         out_id="bound1",
...     ).run()
>>> _ = GSDecisionFromWaldZ(
...         scoped,
...         wald=WaldZStatisticRecord(name="wald1").attach(scoped),
...         boundary=GroupSequentialBoundaryRecord(name="bound1").attach(scoped),
...         info=InformationTimeRecord(name="info1").attach(scoped),
...         out_id="dec1",
...         value_scale="z",
...     ).run()

# --- Inspect final decision ---------------------------------------------------
>>> dec = GroupSequentialDecisionSignalRecord(name="dec1").attach(scoped)
>>> df = dec.latest().select(signal=dec.t.payload["signal"]).execute()
>>> df["signal"].iloc[0] in ("continue", "stop_efficacy", "stop_futility")
True
"""
