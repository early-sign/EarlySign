from typing import Optional, List, Any, Dict, Tuple

import ibis
import numpy as np
import pandas as pd
from pydantic import BaseModel

from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.methods.binomial import BinomialSummaryFact


class ABDecisionRecord(BaseModel):
    """
    Structured record of a decision event (e.g. stopping for efficacy/futility).
    """
    status: str
    message: str


class BinomialProgressReport(BaseModel):
    """Interim progress report for Binomial A/B tests."""

    look: Optional[int]
    n_c: int
    n_t: int
    z_stat: Optional[float]
    boundary: Optional[float]
    info_frac: float
    status: str


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
    final_status: str


class BinomialProgressProjector(Projector[BinomialProgressReport]):
    """
    Tier 2 Projector for interim monitoring.
    Stateless: Reconstructs the report from Protocol and Summary facts.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[BinomialProgressReport]:
        from earlysign.v1.framework.projector import ProtocolProjector
        from earlysign.v1.methods.group_sequential.protocol import GSTProtocol

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(GSTProtocol).project(table)
        p = protocol_traced.data

        # 2. Read Summary
        sc_traced = BinomialSummaryFact(
            identity="summary_c", filter_variant="C"
        ).project(table)
        st_traced = BinomialSummaryFact(
            identity="summary_t", filter_variant="T"
        ).project(table)
        sc, st = sc_traced.data, st_traced.data

        # 3. Calculate Operating Stats
        n_c, n_t = sc.n, st.n
        n_total = n_c + n_t
        info_frac = n_total / p.n_max if p.n_max > 0 else 0.0

        # Determine current look and boundary
        look_num = None
        boundary = None
        for i, m in enumerate(p.milestones):
            if info_frac >= m:
                look_num = i + 1
                boundary = p.boundaries[i]

        z_stat = None
        if n_c >= 2 and n_t >= 2:
            p_pool = (sc.successes + st.successes) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            z_stat = float((st.p_hat - sc.p_hat) / se) if se > 0 else 0.0

        status = "MONITORING"
        if look_num:
            if boundary and z_stat is not None and abs(z_stat) > boundary:
                status = "STOP_EFFICACY"
            elif look_num == len(p.milestones):
                status = "STOP_FINAL"

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

    def project(self, table: ibis.Expr) -> ProjectionResult[BinomialFinalReport]:
        from earlysign.v1.framework.projector import ProtocolProjector
        from earlysign.v1.methods.group_sequential.protocol import GSTProtocol

        protocol_res = ProtocolProjector(GSTProtocol).project(table)
        p = protocol_res.data
        
        sc = BinomialSummaryFact(identity="summary_c", filter_variant="C").project(
            table
        )
        st = BinomialSummaryFact(identity="summary_t", filter_variant="T").project(
            table
        )

        n_c, n_t = sc.data.n, st.data.n
        z_stat = 0.0
        if n_c >= 2 and n_t >= 2:
            p_pool = (sc.data.successes + st.data.successes) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            z_stat = float((st.data.p_hat - sc.data.p_hat) / se) if se > 0 else 0.0

        # Determinie Final Status from Ledger (Decision Event or Max Sample)
        is_rejected = False
        final_status = "COMPLETED" # Default if finished without decision

        # 1. Check for explicit Decision
        try:
             # Use generic ProtocolProjector to find last ABDecisionRecord
            decision_res = ProtocolProjector(ABDecisionRecord).project(table)
            decision = decision_res.data
            if decision.status == "STOP":
                 # Assuming message or context implies rejection if STOP_EFFICACY
                 if "Rejected" in decision.message:
                     is_rejected = True
                     final_status = "STOP_EFFICACY"
                 else:
                     final_status = "STOP_FUTILITY" # or similar
        except RuntimeError:
            # No decision record found.
            # Check if we reached max samples (Implicit Futility / Completion)
            n_total = n_c + n_t
            if p.n_max > 0 and n_total >= p.n_max:
                final_status = "STOP_FINAL"
            else:
                 # Still running or just arbitrarily requested final report
                 final_status = "MONITORING"

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
        if 'decision_res' in locals():
            trace += decision_res.trace

        return ProjectionResult(data=report, trace=trace)


def reconstruct_binomial_z_history(table: ibis.Expr, protocol: Any) -> Tuple[List[int], List[float]]:
    """
    Helper to reconstruct the Z-statistic history at look milestones from the ledger.
    """
    df = table.execute()
    
    # Check table columns. Assuming generic ledger schema or implicit conversion from Ingest
    if 'variant' not in df.columns or 'n' not in df.columns:
        # Fallback empty if schema mismatch
        return [], []
            
    df_c = df[df['variant'] == 'C'].copy()
    df_t = df[df['variant'] == 'T'].copy()
    
    # Build timeline
    df_all = pd.concat([df_c.assign(grp='C'), df_t.assign(grp='T')]).sort_index()
    df_all['n_total'] = df_all['n'].cumsum()
    
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
        # Check if we just crossed the target or are past it (and haven't recorded yet)
        if row['n_total'] >= n_target:
            # Calculate stats
            sub = df_all.loc[:idx]
            sum_c = sub[sub['grp']=='C'][['n', 'successes']].sum()
            sum_t = sub[sub['grp']=='T'][['n', 'successes']].sum()
            
            nc, sc = sum_c['n'], sum_c['successes']
            nt, st = sum_t['n'], sum_t['successes']
            
            z = 0.0
            if nc > 0 and nt > 0:
                 p_pool = (sc+st)/(nc+nt)
                 se = np.sqrt(p_pool*(1-p_pool)*(1/nc + 1/nt))
                 if se > 0:
                    z = (st/nt - sc/nc) / se
            
            history_z.append(z)
            history_n.append(row['n_total'])
            current_look_idx += 1
            
    return history_n, history_z


def plot_gst_summary(
    protocol: Any,
    history_n: List[int],
    history_z: List[float],
    title: str = "GST Monitoring",
    **kwargs
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
    
    # Extract Protocol Params
    milestones = protocol.milestones
    n_max = protocol.n_max
    boundaries = protocol.boundaries
    look_ns = [int(m * n_max) for m in milestones]
    
    # 1. Boundaries
    ax.plot(look_ns, boundaries, 'r--', label='Upper Boundary')
    ax.plot(look_ns, [-b for b in boundaries], 'r--', label='Lower Boundary')
    
    # 2. Trajectory using realized history
    # Add origin
    plot_ns = [0] + history_n
    plot_zs = [0.0] + history_z
    
    ax.plot(plot_ns, plot_zs, 'b.-', label='Z-Statistic')
    
    # 3. Points at looks (highlighted)
    if history_n:
        ax.scatter(history_n, history_z, color='blue', zorder=5)
    
    # Axes
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Z-Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    
    # Info Time Axis
    def n_to_info(x): return x / n_max if n_max > 0 else 0
    def info_to_n(x): return x * n_max
    
    secax = ax.secondary_xaxis('top', functions=(n_to_info, info_to_n))
    secax.set_xlabel('Information Time')
    
    return fig
