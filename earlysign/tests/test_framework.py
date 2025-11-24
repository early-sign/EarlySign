"""
Doctests for earlysign.framework integration (current API).

Covers the flow:
  Ledger.ensure() → Ledger.bind() → Records → Operators → Derived Records.

Steps:
  1. Insert binomial observation snapshots (two-proportions scheme).
  2. Compute Wald Z statistic via operator.
  3. Compute information time (counts-based: total_trials & planned_max_n).
  4. Compute GS boundary from design and make a decision.

>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.integration.execution.schemes.two_proportions.binomial_arms import BinomialArmSnapshot
>>> from earlysign.integration.execution.schemes.two_proportions.wald_z import (
...     BinomialWaldZ,
...     WaldZStatisticRecord,
... )
>>> from earlysign.integration.execution.methods.group_sequential.decision import GSDecisionFromWaldZ
>>> from earlysign.integration.execution.methods.group_sequential.boundary import GroupSequentialBoundaryRecord
>>> from earlysign.integration.execution.methods.group_sequential.decision import GroupSequentialDecisionSignalRecord
>>> from earlysign.integration.execution.methods.group_sequential.records.design import GroupSequentialDesignRecord
>>> from earlysign.integration.execution.methods.group_sequential.information_time import (
...     InformationTime,
...     InformationTimeRecord,
... )
>>> from earlysign.integration.execution.methods.group_sequential.boundary import BoundaryFromDesign


# --- Setup in-memory ledger ---------------------------------------------------
>>> con = ibis.duckdb.connect(":memory:")
>>> base_ledger = Ledger(con, "events")
>>> base_ledger.ensure()
>>> ledger = base_ledger.bind(experiment_id="exp_demo")

# --- Step 1. Insert binomial observation snapshots ----------------------------
>>> control_snapshot = BinomialArmSnapshot(name="control_counts").attach(ledger)
>>> variant_snapshot = BinomialArmSnapshot(name="variant_counts").attach(ledger)
>>> _ = control_snapshot.insert({"trial": 100, "success": 38})
>>> _ = variant_snapshot.insert({
...     "trial": 120,
...     "success": 51,
... })
>>> _ = control_snapshot.insert({
...     "trial": 250,
...     "success": 98,
... })
>>> _ = variant_snapshot.insert({
...     "trial": 260,
...     "success": 99,
... })

# --- Step 2. Compute Wald Z statistic -----------------------------------------
# Operator signature in the new framework takes the ledger and out_id.
>>> _ = BinomialWaldZ(
...         ledger,
...         control=control_snapshot,
...         variant=variant_snapshot,
...         out_id="wald1",
...         pooled=True,
...     ).run()

# --- Step 3. Compute information time (counts-based) --------------------------
# InformationTime now takes the control/variants records and planned_max_n directly.
>>> _ = InformationTime(
...         ledger,
...         out_id="info1",
...         control=control_snapshot,
...         variants=[variant_snapshot],
...         planned_max_n=300,
...     ).run()

# --- Step 4. GS design → boundary → decision ---------------------------------
>>> design_payload = {
...     "alpha": 0.05,
...     "hypothesis": {"structure": "two_sided_symmetric"},
...     "statistic": {"kind": "wald_z", "scale": "z"},
...     "efficacy": {"style": "alpha_spending", "family": "obrien_fleming"},
...     "futility": {"mode": "symmetric", "binding_mode": "non_binding"},
...     "planned_max_n": 600,
...     "planned_info_times": [0.5, 1.0],
... }
>>> design_rec = GroupSequentialDesignRecord(name="design1").attach(ledger)
>>> _ = design_rec.insert(design_payload)
>>> _ = BoundaryFromDesign(
...         ledger,
...         design=GroupSequentialDesignRecord(name="design1").attach(ledger),
...         info=InformationTimeRecord(name="info1").attach(ledger),
...         out_id="bound1",
...     ).run()
>>> _ = GSDecisionFromWaldZ(
...         ledger,
...         wald=WaldZStatisticRecord(name="wald1").attach(ledger),
...         boundary=GroupSequentialBoundaryRecord(name="bound1").attach(ledger),
...         info=InformationTimeRecord(name="info1").attach(ledger),
...         out_id="dec1",
...         value_scale="z",
...     ).run()

# --- Inspect final decision ---------------------------------------------------
>>> dec = GroupSequentialDecisionSignalRecord(name="dec1").attach(ledger)
>>> df = dec.latest().select(signal=dec.t.payload["signal"]).execute()
>>> df["signal"].iloc[0] in ("continue", "stop_efficacy", "stop_futility")
True
"""
