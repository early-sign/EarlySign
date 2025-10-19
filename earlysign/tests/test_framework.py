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
>>> from earlysign.stats.schemes.two_proportions.records import (
...     BinomialCountsRecord, WaldZStatisticRecord
... )
>>> from earlysign.stats.schemes.two_proportions.operators import WaldZStatistic
>>> from earlysign.stats.schemes.two_proportions.group_sequential import GSDecisionFromWaldZ
>>> from earlysign.stats.common.group_sequential.records import (
...     InformationTimeRecord, GroupSequentialDesignRecord,
...     GroupSequentialBoundaryRecord, GroupSequentialDecisionSignalRecord
... )
>>> from earlysign.stats.common.group_sequential.operators.info_op import InformationTime
>>> from earlysign.stats.common.group_sequential.operators.design_op import GroupSequentialDesign
>>> from earlysign.stats.common.group_sequential.operators.boundary_op import BoundaryFromDesign


# --- Setup in-memory ledger ---------------------------------------------------
>>> con = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(con, "events")
>>> ledger.ensure()
>>> scoped = ledger.bind(experiment_id="exp_demo")

# --- Step 1. Insert binomial observation snapshots ----------------------------
>>> counts = BinomialCountsRecord(id="counts1").attach(scoped)
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
...     "alpha": 0.05, "tails": 2, "scale": "z",
...     "efficacy": {"style": "alpha_spending", "family": "obf"},
...     "futility": {"mode": "symmetric"},
... }
>>> _ = GroupSequentialDesign(scoped, out_id="design1", design=design_payload).run()
>>> _ = BoundaryFromDesign(
...         scoped,
...         design=GroupSequentialDesignRecord(id="design1").attach(scoped),
...         info=InformationTimeRecord(id="info1").attach(scoped),
...         out_id="bound1",
...     ).run()
>>> _ = GSDecisionFromWaldZ(
...         scoped,
...         wald=WaldZStatisticRecord(id="wald1").attach(scoped),
...         boundary=GroupSequentialBoundaryRecord(id="bound1").attach(scoped),
...         info=InformationTimeRecord(id="info1").attach(scoped),
...         out_id="dec1",
...         value_scale="z",
...     ).run()

# --- Inspect final decision ---------------------------------------------------
>>> dec = GroupSequentialDecisionSignalRecord(id="dec1").attach(scoped)
>>> df = dec.latest().select(signal=dec.t.payload["signal"]).execute()
>>> df["signal"].iloc[0] in ("continue", "stop_efficacy", "stop_futility")
True
"""
