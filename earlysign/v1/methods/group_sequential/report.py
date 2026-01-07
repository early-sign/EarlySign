from typing import Any, List, Optional, Tuple, Union

import ibis
import numpy as np
import pandas as pd
from pydantic import BaseModel

from earlysign.schema.ES3.GST import DecisionStatus
from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.methods.binomial import BinomialSummaryFact


class ABDecisionRecord(BaseModel):
    """
    Structured record of a decision event (e.g. stopping for efficacy/futility).
    """

    status: Union[DecisionStatus, str]
    message: str


class BinomialProgressReport(BaseModel):
    """Interim progress report for Binomial A/B tests."""

    look: Optional[int]
    n_c: int
    n_t: int
    z_stat: Optional[float]
    boundary: Optional[float]
    info_frac: float
    status: Union[DecisionStatus, str]


class BinomialFinalReport(BaseModel):
    """Comprehensive final summary report for Binomial A/B tests."""

    n_c: int
    n_t: int
    successes_c: int
    successes_t: int
    p_hat_c: float
    p_hat_t: float
    delta_hat: float
    z_stat: float
    is_rejected: bool
    final_status: Union[DecisionStatus, str]


class BinomialProgressProjector(Projector[BinomialProgressReport]):
    """
    Tier 2 Projector for interim monitoring.
    Stateless: Reconstructs the report from Protocol and Summary facts.
    """

    def __init__(self, protocol_type: Any = None):
        import earlysign.schema.ES3.GST as GST

        self.protocol_type = protocol_type or GST.Protocol

    def project(self, table: ibis.Expr) -> ProjectionResult[BinomialProgressReport]:
        from earlysign.v1.framework.projector import ProtocolProjector

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(self.protocol_type).project(table)
        p = protocol_traced.data

        # 2. Read Summary
        sc_traced = BinomialSummaryFact(identity="summary_c", filter_arm="C").project(
            table
        )
        st_traced = BinomialSummaryFact(identity="summary_t", filter_arm="T").project(
            table
        )
        sc, st = sc_traced.data, st_traced.data

        # 3. Calculate Operating Stats
        # Extract n_max and milestones from ES3 Protocol
        schedule = p.method.efficacy.schedule
        if schedule.unit == "sample_size" and schedule.interim_points:
            look_ns = [int(n) for n in schedule.interim_points]
            n_max = max(look_ns)
            milestones = [n / n_max for n in look_ns]
            boundaries_vals = []
            # Reconstruct boundaries using GSTStoppingRuleEngine for accuracy.
            from earlysign.v1.methods.group_sequential.engine import (
                GSTStoppingRuleEngine,
            )

            alpha = p.task.efficacy.alpha
            engine = GSTStoppingRuleEngine(
                p.method.efficacy, "efficacy", total_budget=alpha
            )
            boundaries_vals = [
                engine.get_boundary_at_look(i, m) for i, m in enumerate(milestones)
            ]
        else:
            n_max = 0
            milestones = []
            boundaries_vals = []

        n_c, n_t = sc.n, st.n
        n_total = n_c + n_t
        info_frac = n_total / n_max if n_max > 0 else 0.0

        # Determine current look and boundary
        look_num = None
        boundary = None

        # Determine look: latest milestone passed by current sample size.
        for i, m_frac in enumerate(milestones):
            m_n = look_ns[i]
            if n_total >= m_n:
                look_num = i + 1
                boundary = boundaries_vals[i]
            else:
                break

        z_stat = None
        if n_c >= 2 and n_t >= 2:
            p_pool = (sc.successes + st.successes) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            z_stat = float((st.p_hat - sc.p_hat) / se) if se > 0 else 0.0

        status = DecisionStatus.CONTINUE
        if look_num:
            if boundary and z_stat is not None and abs(z_stat) > boundary:
                status = DecisionStatus.STOP_EFFICACY
            elif look_num == len(milestones):
                status = DecisionStatus.STOP_PLAN_END_REACHED

        report = BinomialProgressReport(
            look=look_num,
            n_c=n_c,
            n_t=n_t,
            z_stat=z_stat,
            boundary=boundary,
            info_frac=info_frac,
            status=status,
        )
        return ProjectionResult(
            data=report, trace=sc_traced.trace + st_traced.trace + protocol_traced.trace
        )


class BinomialFinalProjector(Projector[BinomialFinalReport]):
    """
    Tier 2 Projector for final study summary.
    """

    def __init__(self, protocol_type: Any = None):
        import earlysign.schema.ES3.GST as GST

        self.protocol_type = protocol_type or GST.Protocol

    def project(self, table: ibis.Expr) -> ProjectionResult[BinomialFinalReport]:
        from earlysign.v1.framework.projector import ProtocolProjector

        protocol_res = ProtocolProjector(self.protocol_type).project(table)
        p = protocol_res.data

        sc = BinomialSummaryFact(identity="summary_c", filter_arm="C").project(table)
        st = BinomialSummaryFact(identity="summary_t", filter_arm="T").project(table)

        n_c, n_t = sc.data.n, st.data.n
        z_stat = 0.0
        if n_c >= 2 and n_t >= 2:
            p_pool = (sc.data.successes + st.data.successes) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            z_stat = float((st.data.p_hat - sc.data.p_hat) / se) if se > 0 else 0.0

        # Determinie Final Status from Ledger (Decision Event or Max Sample)
        is_rejected = False
        final_status = "COMPLETED"  # Default if finished without decision

        # 1. Check for explicit Decision
        try:
            # Use generic ProtocolProjector to find last ABDecisionRecord
            decision_res = ProtocolProjector(ABDecisionRecord).project(table)
            decision = decision_res.data
            if decision.status == DecisionStatus.STOP_EFFICACY:
                is_rejected = True
                final_status = DecisionStatus.STOP_EFFICACY
            elif decision.status == DecisionStatus.STOP_FUTILITY:
                final_status = DecisionStatus.STOP_FUTILITY
        except RuntimeError:
            # No decision record found.
            # Check if we reached max samples (Implicit Futility / Completion)
            n_total = n_c + n_t

            # Extract n_max
            schedule = p.method.efficacy.schedule
            n_max = schedule.interim_points[-1] if schedule.interim_points else 0

            if n_max > 0 and n_total >= n_max:
                final_status = DecisionStatus.STOP_PLAN_END_REACHED
            else:
                # Still running or just arbitrarily requested final report
                final_status = DecisionStatus.CONTINUE

        report = BinomialFinalReport(
            n_c=n_c,
            n_t=n_t,
            successes_c=sc.data.successes,
            successes_t=st.data.successes,
            p_hat_c=sc.data.p_hat,
            p_hat_t=st.data.p_hat,
            delta_hat=st.data.p_hat - sc.data.p_hat,
            z_stat=z_stat,
            is_rejected=is_rejected,
            final_status=final_status,
        )

        trace = sc.trace + st.trace + protocol_res.trace
        if "decision_res" in locals():
            trace += decision_res.trace

        return ProjectionResult(data=report, trace=trace)


def reconstruct_binomial_z_history(
    table: ibis.Expr, protocol: Any
) -> Tuple[List[int], List[float]]:
    """
    Helper to reconstruct the Z-statistic history at look milestones from the ledger.
    Supports both generic GSTProtocol and ES3.GST.Protocol.
    """
    df = table.execute()

    # Check table columns. Assuming generic ledger schema or implicit conversion from Ingest
    if "arm" not in df.columns or "n" not in df.columns:
        # Fallback empty if schema mismatch
        return [], []

    df_c = df[df["arm"] == "C"].copy()
    df_t = df[df["arm"] == "T"].copy()

    # Build timeline
    df_all = pd.concat([df_c.assign(grp="C"), df_t.assign(grp="T")]).sort_index()
    df_all["n_total"] = df_all["n"].cumsum()

    # Adapt to Schema Version
    # ES3
    if hasattr(protocol, "method") and hasattr(protocol.method, "efficacy"):
        schedule = protocol.method.efficacy.schedule
        if schedule.unit == "sample_size" and schedule.interim_points:
            # ES3 stores N directly in interim_points
            look_ns = [int(n) for n in schedule.interim_points]
            n_max = max(look_ns)
        else:
            return [], []
    else:
        # Legacy GSTProtocol
        milestones = protocol.milestones
        n_max = protocol.n_max
        look_ns = [int(m * n_max) for m in milestones]

    history_z = []
    history_n = []

    # Calculate Z at each look milestone that we have passed
    current_look_idx = 0

    for idx, row in df_all.iterrows():
        if current_look_idx >= len(look_ns):
            break

        n_target = look_ns[current_look_idx]
        if row["n_total"] >= n_target:
            # Calculate stats
            sub = df_all.loc[:idx]
            sum_c = sub[sub["grp"] == "C"][["n", "successes"]].sum()
            sum_t = sub[sub["grp"] == "T"][["n", "successes"]].sum()

            nc, sc = sum_c["n"], sum_c["successes"]
            nt, st = sum_t["n"], sum_t["successes"]

            z = 0.0
            if nc > 0 and nt > 0:
                p_pool = (sc + st) / (nc + nt)
                se = np.sqrt(p_pool * (1 - p_pool) * (1 / nc + 1 / nt))
                if se > 0:
                    z = (st / nt - sc / nc) / se

            history_z.append(z)
            history_n.append(row["n_total"])
            current_look_idx += 1

    return history_n, history_z


def plot_gst_summary(
    protocol: Any,
    history_n: List[int],
    history_z: List[float],
    title: str = "GST Monitoring",
    **kwargs: Any,
) -> Any:
    """
    Generates a standard summary plot for Group Sequential Test results.

    Args:
        protocol: The GSTProtocol object containing design parameters.
        history_n: List of sample sizes where Z-stats were computed.
        history_z: List of Z-statistics corresponding to history_n.
        title: Plot title.
        **kwargs: Additional keyword arguments passed to matplotlib (not fully used yet, but for future extensibility).

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract Protocol Params (Adapter)
    if hasattr(protocol, "method") and hasattr(protocol.method, "efficacy"):
        # ES3
        # Using GSTStoppingRuleEngine to compute theoretical boundaries at look points.
        from earlysign.v1.methods.group_sequential.engine import GSTStoppingRuleEngine

        # Assuming Efficacy Engine for plot
        alpha = protocol.task.efficacy.alpha if protocol.task.efficacy else 0.05
        engine = GSTStoppingRuleEngine(
            protocol.method.efficacy, "efficacy", total_budget=alpha
        )

        schedule = protocol.method.efficacy.schedule
        if schedule.unit == "sample_size" and schedule.interim_points:
            look_ns = [int(n) for n in schedule.interim_points]
            n_max = max(look_ns)
            milestones = [n / n_max for n in look_ns]

        # Calculate Efficacy Boundaries
        boundaries = []
        for i, m in enumerate(milestones):
            b = engine.get_boundary_at_look(i, m)
            boundaries.append(b if b is not None else 0.0)

        # Calculate Futility Boundaries if applicable
        futility_boundaries = []
        if protocol.method.futility:
            beta_budget = 1.0 - (
                protocol.task.futility.power if protocol.task.futility else 0.8
            )
            fut_engine = GSTStoppingRuleEngine(
                protocol.method.futility, "futility", total_budget=beta_budget
            )
            for i, m in enumerate(milestones):
                b = fut_engine.get_boundary_at_look(i, m)
                futility_boundaries.append(b if b is not None else -np.inf)

    else:
        # Legacy
        milestones = protocol.milestones
        n_max = int(protocol.n_max)
        boundaries = protocol.boundaries
        look_ns = [int(m * n_max) for m in milestones]

    # 1. Boundaries
    if look_ns:
        if boundaries:
            ax.plot(look_ns, boundaries, "r--", label="Efficacy Boundary")

        # Plot Futility if exists (ES3)
        if "futility_boundaries" in locals() and futility_boundaries:
            # Check if we have valid boundaries (not -inf)
            valid_fut = [
                b for b in futility_boundaries if b > -100
            ]  # Simple filter for plotting
            if valid_fut:
                ax.plot(look_ns, futility_boundaries, "k--", label="Futility Boundary")

        elif hasattr(protocol, "boundaries") and protocol.boundaries:
            pass

    # 2. Trajectory using realized history
    # Add origin
    plot_ns = [0] + history_n
    plot_zs = [0.0] + history_z

    ax.plot(plot_ns, plot_zs, "b.-", label="Z-Statistic")

    # 3. Points at looks (highlighted)
    if history_n:
        ax.scatter(history_n, history_z, color="blue", zorder=5)

    # Axes
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Z-Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linestyle=":", alpha=0.3)

    # Info Time Axis
    def n_to_info(x: Any) -> Any:
        return x / n_max if n_max > 0 else 0

    def info_to_n(x: Any) -> Any:
        return x * n_max

    secax = ax.secondary_xaxis("top", functions=(n_to_info, info_to_n))
    secax.set_xlabel("Information Time")

    return fig
