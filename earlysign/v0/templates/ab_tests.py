from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
)

import ibis
import pandas as pd
from ibis import BaseBackend
from matplotlib.figure import Figure

from earlysign.core.ledger import Ledger
from earlysign.core.util.ibis_cache import IbisCache
from earlysign.v0.framework import templates as tpl
from earlysign.v0.methods.group_sequential.boundary import BoundaryFromDesign
from earlysign.v0.methods.group_sequential.decision import (
    GroupSequentialDecisionSignalRecord,
    GSDecisionFromWaldZ,
)
from earlysign.v0.methods.group_sequential.design.initial_design.scenarios.binomial import (
    create_binomial_design,
)
from earlysign.v0.methods.group_sequential.design.initial_design.scenarios.fst_to_gst import (
    AddInterimToFixedSampleTest,
)
from earlysign.v0.methods.group_sequential.design.records.design import (
    DesignPayloadModel,
    GroupSequentialDesignRecord,
)
from earlysign.v0.methods.group_sequential.info_time import (
    InformationTime,
    InformationTimeRecord,
)
from earlysign.v0.methods.group_sequential.report.plot_design_boundaries import (
    plot_design_boundaries,
)
from earlysign.v0.methods.group_sequential.schemes.two_proportions.binomial_arms import (
    BinomialArmResultRecord,
    BinomialArmSnapshot,
)
from earlysign.v0.methods.group_sequential.schemes.two_proportions.wald_z import (
    BinomialWaldZ,
    WaldZStatisticRecord,
)


@dataclass
class State:
    stop_recommended: bool = False


@dataclass
class BinomialGSTDesignInterface:
    """Wrapper for binomial GST design helper to maintain API compatibility."""

    design: AddInterimToFixedSampleTest

    def build_design_payload(
        self,
        info_times: Sequence[float],
        planned_max_n: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Proxy to the underlying design helper."""
        return self.design.build_design_payload(
            info_times=info_times,
            planned_max_n=planned_max_n,
            metadata=metadata,
        )


class BinomialABTest(tpl.TemplateBase):
    """
    Example:
    >>> import ibis
    >>> BinomialABTest(ibis.connect("duckdb://:memory:"), "my_exp")
    <earlysign.v0.templates.ab_tests.BinomialABTest object at 0x...>
    >>> BinomialABTest("duckdb://:memory:", "my_exp")
    <earlysign.v0.templates.ab_tests.BinomialABTest object at 0x...>
    """

    def __init__(
        self,
        connector: BaseBackend | str,
        experiment_id: str,
        table_name: str | None = None,
        *,
        ibis_cache: Optional[IbisCache] = None,
    ) -> None:
        """Initialize BinomialABTest.

        Parameters
        ----------
        connector : BaseBackend | str
            Database connector or connection string
        experiment_id : str
            Unique identifier for this experiment
        table_name : str | None
            Custom table name for ledger (defaults to experiment_id)

        Examples
        --------
        >>> import ibis
        >>> test = BinomialABTest(ibis.connect("duckdb://:memory:"), "exp1")
        >>> test.experiment_id
        'exp1'
        """
        if isinstance(connector, str):
            self.connector = ibis.connect(connector)
        elif isinstance(connector, BaseBackend):
            self.connector = connector
        self.experiment_id = experiment_id

        ## Use experiment_id if table_name is not specified
        self.ledger = Ledger(
            self.connector, table_name if table_name is not None else experiment_id
        ).bind(experiment_id=self.experiment_id)
        self.ledger.ensure()
        self._ibis_cache = (
            ibis_cache
            if ibis_cache is not None
            else IbisCache(self.connector, mode="execute")
        )

    @staticmethod
    def design_interface(**kwargs: Any) -> BinomialGSTDesignInterface:
        """
        Build a group sequential design helper for binomial metrics.
        (Kept for backward compatibility and tutorial usage)
        """
        helper = create_binomial_design(**kwargs)
        return BinomialGSTDesignInterface(design=helper)

    def set_design(self, payload: Dict[str, Any]) -> None:
        """Set the group sequential design.

        Parameters
        ----------
        payload : Dict[str, Any]
            Design configuration matching ``DesignPayloadModel``::

                {
                    "alpha": 0.05,
                    "hypothesis": {"structure": "two_sided_symmetric"},
                    "statistic": {"kind": "wald_z", "scale": "z"},
                    "efficacy": {"style": "alpha_spending", "family": "obrien_fleming"},
                    "futility": {"mode": "none", "binding_mode": "non_binding"},
                    "planned_max_n": 1000,
                    "planned_info_times": [0.33, 0.67, 1.0]
                }

        Examples
        --------
        >>> import ibis
        >>> test = BinomialABTest(ibis.connect("duckdb://:memory:"), "exp1")
        >>> design = {
        ...     "alpha": 0.05,
        ...     "hypothesis": {"structure": "two_sided_symmetric"},
        ...     "statistic": {"kind": "wald_z", "scale": "z"},
        ...     "efficacy": {"style": "alpha_spending", "family": "obrien_fleming"},
        ...     "futility": {"mode": "symmetric", "binding_mode": "non_binding"},
        ...     "planned_max_n": 1000,
        ...     "planned_info_times": [0.33, 0.67, 1.0]
        ... }
        >>> test.set_design(design)
        """
        design = GroupSequentialDesignRecord("design").attach(self.ledger)
        design_model = DesignPayloadModel.model_validate(payload)
        design.insert(design_model.to_payload())

    def _execute_expr(self, expr: Any) -> Any:
        with self._ibis_cache as cached_execute:
            return cached_execute(expr)

    def update(self, payload: Dict[str, Any]) -> None:
        """Update experiment with new observations.

        Records incremental observations and automatically triggers analysis
        (boundary computation and decision) when current information time
        reaches or exceeds a planned look that has not yet been executed.

        Parameters
        ----------
        payload : Dict[str, Any]
            Incremental counts: {"nA": int, "mA": int, "nB": int, "mB": int}

        Returns
        -------
        None
            This method performs ledger writes (observations, statistics,
            information time, and decisions when triggered) and does not
            return a value. Query the ledger for status after calling.

        Notes
        -----
        Analysis is triggered when:
        1. current_info_time >= planned_info_time[i]
        2. Look i has not been executed yet
        3. i is the smallest index satisfying conditions 1 and 2

        This means you can update with arbitrary batch sizes, and the system
        will automatically execute the appropriate analysis when each planned
        information time is reached.

        Examples
        --------
        >>> import ibis
        >>> test = BinomialABTest(ibis.connect("duckdb://:memory:"), "exp1")
        >>> design = {
        ...     "alpha": 0.05,
        ...     "hypothesis": {"structure": "two_sided_symmetric"},
        ...     "statistic": {"kind": "wald_z", "scale": "z"},
        ...     "efficacy": {"style": "alpha_spending", "family": "obrien_fleming"},
        ...     "futility": {"mode": "symmetric", "binding_mode": "non_binding"},
        ...     "planned_max_n": 1000,
        ...     "planned_info_times": [0.5, 1.0]
        ... }
        >>> test.set_design(design)
        >>> # First update doesn't trigger (info_time < 0.5)
        >>> test.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
        >>> # Second update may trigger first look (info_time >= 0.5)
        >>> test.update({"nA": 150, "mA": 15, "nB": 150, "mB": 18})
        """
        if not {"nA", "mA", "nB", "mB"}.issubset(payload.keys()):
            raise ValueError(
                "Payload must include counts for control (nA, mA) and variant (nB, mB)."
            )

        control_trials = int(payload["nA"])
        control_successes = int(payload["mA"])
        variant_trials = int(payload["nB"])
        variant_successes = int(payload["mB"])

        ## Record incremental observation (delta) per arm
        control_obs = BinomialArmResultRecord(name="observation_A").attach(self.ledger)
        variant_obs = BinomialArmResultRecord(name="observation_B").attach(self.ledger)
        control_obs.insert(
            trial=control_trials,
            success=control_successes,
            labels={"arm_name": "A"},
        )
        variant_obs.insert(
            trial=variant_trials,
            success=variant_successes,
            labels={"arm_name": "B"},
        )

        ## Compute cumulative snapshots for each arm
        control_snapshot_record = BinomialArmSnapshot(
            name="snapshot_A",
            ledger=self.ledger,
        )
        control_snapshot_record.update_from_obs(control_obs, arm_name="A")

        variant_snapshot_record = BinomialArmSnapshot(
            name="snapshot_B",
            ledger=self.ledger,
        )
        variant_snapshot_record.update_from_obs(variant_obs, arm_name="B")

        ## Compute statistic (using snapshot)
        stat_op = BinomialWaldZ(
            self.ledger,
            control=control_snapshot_record,
            variant=variant_snapshot_record,
            pooled=True,
            out_id="statistic",
        )
        stat_op.run()
        stat_record = stat_op.outputs.wald

        ## Read design info
        design_record = GroupSequentialDesignRecord("design").attach(self.ledger)
        try:
            design_payload = design_record.latest_payload()
        except LookupError as exc:
            raise ValueError("Design must be set before calling update().") from exc
        design_model = DesignPayloadModel.model_validate(design_payload)
        planned_max_n = int(design_model.planned_max_n)
        planned_info_times: list[float] = list(design_model.planned_info_times)

        ## Compute information time
        info_op = InformationTime(
            self.ledger,
            out_id="info_time",
            control=control_snapshot_record,
            variants=[variant_snapshot_record],
            planned_max_n=planned_max_n,
        )
        info_op.run()
        info_record = info_op.outputs.info

        # Get current info time
        info_payload = info_record.latest_payload()
        current_info_time = float(info_payload["info_time"])

        # Use latest decision's timestamp to determine if a new look is due
        decision_record = GroupSequentialDecisionSignalRecord("decision").attach(
            self.ledger
        )

        try:
            decision_payload, _last_decision_ts = decision_record.latest_payload(
                include_ts=True
            )
            last_info_time_before_decision = float(
                decision_payload.get("info_time", 0.0)
            )
        except LookupError:
            last_info_time_before_decision = 0.0

        # Trigger only if any planned look is newly due since last decision
        if not any(
            last_info_time_before_decision < planned_time <= current_info_time
            for planned_time in planned_info_times
        ):
            return

        ## Compute boundary
        boundary_op = BoundaryFromDesign(
            self.ledger,
            design=design_record,
            info=info_record,
            out_id="boundary",
        )
        boundary_op.run()
        boundary_record = boundary_op.outputs.boundary

        ## Record the decision
        decision_op = GSDecisionFromWaldZ(
            self.ledger,
            wald=stat_record,
            boundary=boundary_record,
            out_id="decision",
        )
        decision_op.run()

    def status(self) -> State:
        decision_record = GroupSequentialDecisionSignalRecord("decision").attach(
            self.ledger
        )
        latest_decision = self._execute_expr(decision_record.latest())
        if len(latest_decision) == 0:
            return State(stop_recommended=False)
        if "stop" in latest_decision["signal"].iloc[0]:
            return State(stop_recommended=True)
        else:
            return State(stop_recommended=False)

    def plot_design(self, n_points: int = 50) -> Figure:
        """Plot the group sequential design boundaries.

        Parameters
        ----------
        n_points : int, optional
            Number of information time points to evaluate boundaries at, by default 50

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object
        """
        # Read design from ledger
        design_record = GroupSequentialDesignRecord("design").attach(self.ledger)
        design_df = self._execute_expr(
            design_record.latest(explode=False).select(payload=design_record.t.payload)
        )
        if len(design_df) == 0:
            raise ValueError("No design found in ledger. Call set_design() first.")
        design_model = DesignPayloadModel.model_validate(design_df.iloc[0]["payload"])

        # Delegate to reporting module
        return plot_design_boundaries(
            design=design_model.boundary_spec(),
            n_points=n_points,
        )

    def get_history(self) -> pd.DataFrame:
        """Get complete history of the trial with all key metrics."""
        records = {
            "control": BinomialArmSnapshot(name="snapshot_A").attach(self.ledger),
            "treatment": BinomialArmSnapshot(name="snapshot_B").attach(self.ledger),
            "stat": WaldZStatisticRecord("statistic").attach(self.ledger),
            "info": InformationTimeRecord("info_time").attach(self.ledger),
            "decision": GroupSequentialDecisionSignalRecord("decision").attach(
                self.ledger
            ),
        }

        result = tpl.join_sequential_history(records)
        result = result.mutate(
            nA_cum=result.control_trial,
            mA_cum=result.control_success,
            nB_cum=result.treatment_trial,
            mB_cum=result.treatment_success,
            wald_z=result.stat_wald_z,
            info_time=result.info_info_time,
            signal=result.decision_signal,
        )
        result = result.mutate(
            pA=(lambda t: (t.mA_cum / t.nA_cum).fill_null(0.0)),  # type: ignore[operator]
            pB=(lambda t: (t.mB_cum / t.nB_cum).fill_null(0.0)),  # type: ignore[operator]
        )
        result = result.mutate(diff=lambda t: t.pB - t.pA)  # type: ignore[operator]

        cols = [
            "look",
            "nA_cum",
            "mA_cum",
            "nB_cum",
            "mB_cum",
            "pA",
            "pB",
            "diff",
            "wald_z",
            "info_time",
            "signal",
        ]
        df_result = self._execute_expr(result.select(*cols).order_by("look"))
        return df_result if isinstance(df_result, pd.DataFrame) else pd.DataFrame()

    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive summary of trial results."""
        design_rec = GroupSequentialDesignRecord("design").attach(self.ledger)
        design_df = self._execute_expr(
            design_rec.latest().select(payload=design_rec.t.payload)
        )
        design_payload = (
            DesignPayloadModel.model_validate(design_df.iloc[0]["payload"]).to_payload()
            if len(design_df) > 0
            else {}
        )

        history_df = self.get_history()
        if len(history_df) == 0:
            return {
                "design": design_payload,
                "n_looks": 0,
                "stopped": False,
                "final_signal": "unknown",
                "final_stats": {},
                "history": history_df,
            }

        final = history_df.iloc[-1]
        return {
            "design": design_payload,
            "n_looks": len(history_df),
            "stopped": "stop" in final["signal"],
            "final_signal": final["signal"],
            "final_stats": {
                "nA": int(final["nA_cum"]),
                "mA": int(final["mA_cum"]),
                "nB": int(final["nB_cum"]),
                "mB": int(final["mB_cum"]),
                "pA": float(final["pA"]),
                "pB": float(final["pB"]),
                "diff": float(final["diff"]),
                "wald_z": float(final["wald_z"]) if final["wald_z"] else None,
            },
            "history": history_df,
        }

    def print_results(self) -> None:
        """Print formatted summary of trial results."""
        res = self.get_results()
        stats = res.get("final_stats", {})
        if not stats:
            print("No results found.")
            return

        history_lines = [
            f"\nLook {int(r['look'])} (Info time: {r['info_time']:.3f}):\n"
            f"  Control:   {r['mA_cum']}/{r['nA_cum']} = {r['pA']:.3%}\n"
            f"  Treatment: {r['mB_cum']}/{r['nB_cum']} = {r['pB']:.3%}\n"
            f"  Z-stat:    {r['wald_z']:.4f}\n"
            f"  Signal:    {r['signal']}"
            for _, r in res["history"].iterrows()
        ]

        wald_z_text = (
            f"  Wald Z:        {stats['wald_z']:.4f}\n"
            if stats.get("wald_z") is not None
            else ""
        )

        summary = f"""
{'=' * 70}
TRIAL RESULTS: {self.experiment_id}
{'=' * 70}
Looks: {res['n_looks']} | Stopped: {res['stopped']} | Signal: {res['final_signal']}

FINAL STATISTICS:
  Control (A):   {stats['mA']}/{stats['nA']} = {stats['pA']:.3%}
  Treatment (B): {stats['mB']}/{stats['nB']} = {stats['pB']:.3%}
  Difference:    {stats['diff']:.3%}
{wald_z_text}
ANALYSIS HISTORY:
{"".join(history_lines)}
{'=' * 70}
"""
        print(summary)
