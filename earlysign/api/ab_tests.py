from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

import ibis
import pandas as pd
from ibis import BaseBackend
from matplotlib.figure import Figure

from earlysign.core.ledger import Ledger
from earlysign.core.util.ibis_cache import IbisCache
from earlysign.framework import templates as tpl
from earlysign.reporting.group_sequential import plot_design_boundaries
from earlysign.stats.applications.design.group_sequential.initial_design.scenarios.fst_to_gst import (
    AddInterimToFixedSampleTest,
)
from earlysign.stats.applications.design.group_sequential.initial_design.schema import (
    DesignPayloadModel,
)
from earlysign.stats.applications.execution.methods.group_sequential.operators.boundary import (
    BoundaryFromDesign,
)
from earlysign.stats.applications.execution.methods.group_sequential.operators.decision import (
    GSDecisionFromWaldZ,
)
from earlysign.stats.applications.execution.methods.group_sequential.records.decision import (
    GroupSequentialDecisionSignalRecord,
)
from earlysign.stats.applications.execution.methods.group_sequential.records.design import (
    GroupSequentialDesignRecord,
)
from earlysign.stats.applications.execution.methods.group_sequential.records.statistics import (
    WaldZStatisticRecord,
)
from earlysign.stats.applications.execution.schemes.two_proportions.operators import (
    BinomialCountsSnapshot,
    InformationTime,
    WaldZStatistic,
)
from earlysign.stats.applications.execution.schemes.two_proportions.records import (
    BinomialCountsRecord,
    BinomialCountsSnapshotRecord,
)
from earlysign.stats.essentials.methods.group_sequential.asn import ASNCalculator
from earlysign.stats.essentials.methods.group_sequential.spending import (
    SpendingFunction,
    get_spending_class,
)
from earlysign.stats.essentials.schemes.two_proportions.asn import (
    build_asn_calculator,
)
from earlysign.stats.essentials.schemes.two_proportions.design import (
    build_two_proportions_scheme,
)


@dataclass
class BinomialGSTDesignInterface:
    """Structured accessor for the canonical binomial design helper."""

    design: AddInterimToFixedSampleTest
    spending: SpendingFunction
    asn_calculator_factory: Callable[[], ASNCalculator] = field(repr=False)

    def new_asn_calculator(self) -> ASNCalculator:
        """Return a fresh ASN calculator using the stored factory."""

        return self.asn_calculator_factory()

    def spending_family(self) -> str:
        """Return canonical spending-family key understood by payload builders."""

        return self.spending.name

    def build_design_payload(
        self,
        info_times: Sequence[float],
        planned_max_n: int,
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> MutableMapping[str, object]:
        """Construct a minimal JSON-serialisable design payload."""

        payload: MutableMapping[str, object] = {
            "alpha": float(self.design.alpha),
            "hypothesis": {"structure": "two_sided_symmetric"},
            "statistic": {"kind": "wald_z", "scale": "z"},
            "efficacy": {
                "style": "alpha_spending",
                "family": self.spending_family(),
            },
            "futility": {"mode": "none", "binding_mode": "non_binding"},
            "planned_max_n": int(planned_max_n),
            "planned_info_times": [float(x) for x in info_times],
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return payload


@dataclass
class State:
    stop_recommended: bool = False


class BinomialABTest(tpl.TemplateBase):
    """
    Example:
    >>> import ibis
    >>> BinomialABTest(ibis.connect("duckdb://:memory:"), "my_exp")
    <earlysign.api.ab_tests.BinomialABTest object at 0x...>
    >>> BinomialABTest("duckdb://:memory:", "my_exp")
    <earlysign.api.ab_tests.BinomialABTest object at 0x...>
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
                    "efficacy": {"style": "alpha_spending", "family": "obf"},
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
        ## Record incremental observation (delta)
        obs = BinomialCountsRecord("observation").attach(self.ledger)
        obs.insert(**payload)

        ## Compute cumulative snapshot
        snapshot_op = BinomialCountsSnapshot(self.ledger, obs=obs, out_id="snapshot")
        snapshot_op.run()
        snapshot_record = snapshot_op.outputs.snapshot

        ## Compute statistic (using snapshot)
        stat_op = WaldZStatistic(
            self.ledger, cum_counts=snapshot_record, pooled=True, out_id="statistic"
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
            cum_counts=snapshot_record,
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

    @classmethod
    def design_interface(
        cls,
        *,
        alpha: float,
        delta: float,
        power: float,
        p_control: float,
        allocation_ratio: float = 1.0,
        spending: Union[SpendingFunction, Type[SpendingFunction], str] = "pocock",
        effect_sizes: Optional[Sequence[float]] = None,
        n_sim: int = 200,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        design_payload_builder: Optional[
            Callable[[Sequence[float], int], Mapping[str, Any]]
        ] = None,
    ) -> BinomialGSTDesignInterface:
        """Return a configured binomial GST design helper.

        This surfaces the class-based ``AddInterimToFixedSampleTest`` flow
        through a stable API so callers (including notebooks) no longer need
        to replicate the spending/procedure wiring.
        """

        if hasattr(spending, "cumulative") and hasattr(
            spending, "boundaries_from_stage_alpha"
        ):
            spending_obj = cast(SpendingFunction, spending)
        else:
            spending_cls = (
                cast(Type[SpendingFunction], spending)
                if isinstance(spending, type)
                else get_spending_class(str(spending))
            )
            spending_obj = spending_cls(alpha=alpha)

        base_scheme = build_two_proportions_scheme(
            p_control=p_control,
            target_effect=delta,
            effect_sizes=effect_sizes or [delta],
            alpha=alpha,
            power=power,
            allocation_ratio=allocation_ratio,
        )
        resolved_scheme = base_scheme.with_effect_sizes(effect_sizes)

        def _asn_factory() -> ASNCalculator:
            return build_asn_calculator(
                alpha=alpha,
                beta=1.0 - power,
                sided=2,
                p_control=p_control,
                effect_size=delta,
                allocation_ratio=allocation_ratio,
                spending=spending_obj,
            )

        procedure_factory = resolved_scheme.procedure_factory_builder(
            spending_obj, allocation_ratio
        )

        design = AddInterimToFixedSampleTest(
            alpha=alpha,
            power=power,
            allocation_ratio=allocation_ratio,
            scheme=resolved_scheme,
            procedure_factory=procedure_factory,
            asn_calculator_factory=_asn_factory,
            n_sim=n_sim,
            batch_size=batch_size,
            seed=seed,
            design_payload_builder=design_payload_builder,
        )

        return BinomialGSTDesignInterface(
            design=design,
            spending=spending_obj,
            asn_calculator_factory=_asn_factory,
        )

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
        """Get complete history of the trial with all key metrics.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: look, nA_cum, mA_cum, nB_cum, mB_cum,
            pA, pB, diff, wald_z, info_time, signal

        Examples
        --------
        >>> conn = ibis.connect("duckdb://:memory:")  # doctest: +SKIP
        >>> test = BinomialABTest(conn, "exp1")  # doctest: +SKIP
        ... # ... run experiment ...
        >>> history = test.get_history()  # doctest: +SKIP
        >>> print(history)  # doctest: +SKIP
        """
        # Get all records with exploded payloads as ibis expressions
        snapshot_rec = BinomialCountsSnapshotRecord("snapshot").attach(self.ledger)
        stat_rec = WaldZStatisticRecord("statistic").attach(self.ledger)
        decision_rec = GroupSequentialDecisionSignalRecord("decision").attach(
            self.ledger
        )

        # Use ibis to join the tables by timestamp and uuid
        snapshots = snapshot_rec.order_by_ts(ascending=True, explode=True)
        stats = stat_rec.order_by_ts(ascending=True, explode=True)
        decisions = decision_rec.order_by_ts(ascending=True, explode=True)

        # Join, compute, and select - all in one chain
        # Note: Type ignores needed due to mypy not fully understanding ibis column operations
        result = (
            snapshots.inner_join(
                stats, [snapshots.ts == stats.ts, snapshots.uuid == stats.uuid]
            )
            .inner_join(
                decisions,
                [snapshots.ts == decisions.ts, snapshots.uuid == decisions.uuid],
            )
            .select(
                nA_cum=snapshots.nA,
                mA_cum=snapshots.mA,
                nB_cum=snapshots.nB,
                mB_cum=snapshots.mB,
                wald_z=stats.wald_z,
                info_time=decisions.info_time,
                signal=decisions.signal,
                ts=snapshots.ts,
            )
            .order_by("ts")
            .mutate(
                pA=(lambda t: (t.mA_cum / t.nA_cum).fill_null(0.0)),  # type: ignore[operator]
                pB=(lambda t: (t.mB_cum / t.nB_cum).fill_null(0.0)),  # type: ignore[operator]
            )
            .mutate(diff=lambda t: t.pB - t.pA)  # type: ignore[operator]
            .mutate(look=ibis.row_number().over(order_by="ts"))
            .select(
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
            )
        )

        df_result = self._execute_expr(result)
        return df_result if isinstance(df_result, pd.DataFrame) else pd.DataFrame()

    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive summary of trial results.

        Returns
        -------
        dict
            Dictionary containing:
            - design: Design parameters
            - n_looks: Number of analyses conducted
            - stopped: Whether trial stopped early
            - final_signal: Final decision signal
            - final_stats: Final cumulative statistics
            - history: Full history DataFrame

        Examples
        --------
        >>> conn = ibis.connect("duckdb://:memory:")  # doctest: +SKIP
        >>> test = BinomialABTest(conn, "exp1")  # doctest: +SKIP
        ... # ... run experiment ...
        >>> results = test.get_results()  # doctest: +SKIP
        >>> print(results['n_looks'])  # doctest: +SKIP
        >>> print(results['final_signal'])  # doctest: +SKIP
        """
        # Get design
        design_rec = GroupSequentialDesignRecord("design").attach(self.ledger)
        design_df = self._execute_expr(
            design_rec.latest().select(payload=design_rec.t.payload)
        )
        if len(design_df) > 0:
            design_payload = DesignPayloadModel.model_validate(
                design_df.iloc[0]["payload"]
            ).to_payload()
        else:
            design_payload = {}

        # Get history (this already does all the joining and computation)
        history_df = self.get_history()

        if len(history_df) == 0:
            # No data yet
            return {
                "design": design_payload,
                "n_looks": 0,
                "stopped": False,
                "final_signal": "unknown",
                "final_stats": {
                    "nA": 0,
                    "mA": 0,
                    "nB": 0,
                    "mB": 0,
                    "pA": 0.0,
                    "pB": 0.0,
                    "diff": 0.0,
                    "wald_z": None,
                },
                "history": history_df,
            }

        # Extract final row (already computed in get_history)
        final_row = history_df.iloc[-1]
        stopped = "stop" in final_row["signal"]

        return {
            "design": design_payload,
            "n_looks": len(history_df),
            "stopped": stopped,
            "final_signal": final_row["signal"],
            "final_stats": {
                "nA": int(final_row["nA_cum"]),
                "mA": int(final_row["mA_cum"]),
                "nB": int(final_row["nB_cum"]),
                "mB": int(final_row["mB_cum"]),
                "pA": float(final_row["pA"]),
                "pB": float(final_row["pB"]),
                "diff": float(final_row["diff"]),
                "wald_z": (
                    float(final_row["wald_z"])
                    if final_row["wald_z"] is not None
                    else None
                ),
            },
            "history": history_df,
        }

    def print_results(self) -> None:
        """Print formatted summary of trial results.

        Examples
        --------
        >>> conn = ibis.connect("duckdb://:memory:")  # doctest: +SKIP
        >>> test = BinomialABTest(conn, "exp1")  # doctest: +SKIP
        ... # ... run experiment ...
        >>> test.print_results()  # doctest: +SKIP
        """
        results = self.get_results()
        stats = results["final_stats"]

        # Build compact history text
        history = results["history"]
        history_lines = []
        for _, row in history.iterrows():
            history_lines.append(
                f"\nLook {int(row['look'])} (Info time: {row['info_time']:.3f}):\n"
                f"  Control:   {row['mA_cum']}/{row['nA_cum']} = {row['pA']:.3%}\n"
                f"  Treatment: {row['mB_cum']}/{row['nB_cum']} = {row['pB']:.3%}\n"
                f"  Z-stat:    {row['wald_z']:.4f}\n"
                f"  Signal:    {row['signal']}"
            )
        history_text = "".join(history_lines)

        wald_z_text = f"Wald Z:        {stats['wald_z']:.4f}" if stats["wald_z"] else ""

        summary = f"""
{'=' * 70}
TRIAL RESULTS SUMMARY
{'=' * 70}

Experiment ID: {self.experiment_id}
Number of analyses: {results['n_looks']}
Stopped early: {'Yes' if results['stopped'] else 'No'}
Final decision: {results['final_signal']}

{'-' * 70}
FINAL STATISTICS
{'-' * 70}
Control (A):   {stats['mA']}/{stats['nA']} = {stats['pA']:.3%}
Treatment (B): {stats['mB']}/{stats['nB']} = {stats['pB']:.3%}
Difference:    {stats['diff']:.3%}
{wald_z_text}

{'-' * 70}
ANALYSIS HISTORY
{'-' * 70}
{history_text}

{'=' * 70}
"""
        print(summary)
