# """
# Doctests for earlysign.framework integration.

# Covers the flow:
#   Ledger.ensure() → Ledger.bind() → Records → Operators → Derived Records.

# Steps:
#   1. Insert binomial observation snapshots.
#   2. Compute Wald Z statistic via operator.
#   3. Compute information time.
#   4. Gate decision using GST-like boundary.

# >>> import ibis, duckdb
# >>> from earlysign.core.ledger import Ledger
# >>> from earlysign.stats.schemes.two_binomials.records import BinomialCountsRecord, WaldZStatisticRecord
# >>> from earlysign.stats.schemes.two_binomials.operators import WaldZStatistic
# >>> from earlysign.stats.common.group_sequential.records import InformationTimeRecord, GroupSequentialBoundaryRecord, GroupSequentialDecisionSignalRecord
# >>> from earlysign.stats.common.group_sequential.info_time import InformationTime
# >>> from earlysign.stats.common.group_sequential.decision import Decision

# # --- Setup in-memory ledger ---------------------------------------------------
# >>> con = ibis.duckdb.connect(":memory:")
# >>> ledger = Ledger(con, "events")
# >>> ledger.ensure()
# >>> scoped = ledger.bind(experiment_id="exp_demo")

# # --- Step 1. Insert binomial observation snapshots ----------------------------
# >>> counts = BinomialCountsRecord(id="counts1").attach(scoped)
# >>> _ = counts.insert(nA=100, mA=38, nB=120, mB=51)
# >>> _ = counts.insert(nA=150, mA=60, nB=140, mB=48)

# # --- Step 2. Compute Wald Z statistic -----------------------------------------
# >>> wald_rec = WaldZStatisticRecord(id="wald1").attach(scoped)
# >>> wald = WaldZStatistic(counts=counts, out=wald_rec)
# >>> _ = wald.run()

# # --- Step 3. Compute information time -----------------------------------------
# >>> info_rec = InformationTimeRecord(id="info1").attach(scoped)
# >>> info = InformationTime(counts=counts, out=info_rec)
# >>> _ = info.run()

# # --- Step 4. Gate decision using GST-like boundary ----------------------------
# >>> bound_rec = GSTBoundaryRecord(id="bound1").attach(scoped)
# >>> bound = GSTBoundary(info=info_rec, out=bound_rec)
# >>> _ = bound.run()

# >>> dec_rec  = DecisionSignalRecord(id="dec1").attach(scoped)
# >>> decision = Decision(wald=wald_rec, info=info_rec, bound=bound_rec, out=dec_rec)
# >>> _ = decision.run()

# # --- Inspect ledger contents --------------------------------------------------
# >>> scoped.t.order_by(scoped.t.ts)["payload_type"].execute().to_list()
# ['BinomCounts', 'BinomCounts', 'WaldZ', 'InfoTime', 'GSTBoundary', 'DecisionSignal']

# # --- Inspect final decision ---------------------------------------------------
# >>> t = dec_rec.latest()
# >>> t.select(
# ...     signal=t.payload["signal"].cast("string"),
# ...     wald_z=t.payload["wald_z"].cast("float64"),
# ...     info_time=t.payload["info_time"].cast("float64"),
# ... ).execute()
#   signal   wald_z  info_time
# 0  ...      ...       ...
# """
