"""Convert a fixed-sample two-proportion design into group-sequential procedures.

This module focuses on :class:`AddInterimToFixedSampleTest`, a helper class used
throughout the tutorials and APIs. The class is intentionally generic but the
concrete implementation below targets the two-proportions scheme using the
essentials/applications building blocks in the repository.

Depending on the ``keep_power`` flag supplied to the class constructor,
each instance operates in one of two modes:

- ``keep_power=False``: build k-look GSTs that adopt the fixed-sample
  design budget as the maximum sample size. The helper optimizes information
  timing (via ``MinimizeASNOptimizer``), constructs the procedure/boundaries,
  and runs Monte-Carlo OC simulations at the requested effect sizes.
- ``keep_power=True``: after optimizing timing the helper performs a
  binary search over candidate maximum sample sizes to locate the minimal budget
  that still attains the target power at the design effect. Each candidate is
  evaluated through the simulator and the search stops once the tolerance band
  is satisfied.

Notes
-----
All imports are restricted to modules under
``earlysign.stats.essentials`` and ``earlysign.integration`` as
requested. The implementations prefer existing building blocks:

- timing optimizer: ``MinimizeASNOptimizer`` (applications/.../minimize_asn.py)
- two-proportions ASN factory: ``build_asn_calculator`` (essentials/.../asn.py)
- Wald Z computation: ``compute_wald_z`` (essentials schemes two_proportions)
- simulator: ``TwoProportionsSimulator`` (essentials/schemes/two_proportions)
- plotting: ``OCCurvePlotter`` (applications/report/group_sequential)

The module purposely keeps a small, local ``Procedure`` implementation that
implements the `Procedure` protocol expected by the simulator (``ingest``,
``should_stop``, ``reset``). The logic is described in the docstrings of the
exported functions below.
"""

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import ibis
from tqdm.auto import tqdm

from earlysign.integration.design.group_sequential.initial_design.helpers.protocols import (
    ProcedureFactory,
    ProcedureLike,
)
from earlysign.integration.design.group_sequential.initial_design.helpers.scheme import (
    GSTSchemeHooks,
)
from earlysign.integration.design.group_sequential.initial_design.workflows.optimize_timing.minimize_asn import (
    MinimizeASNOptimizer,
)
from earlysign.integration.design.group_sequential.initial_design.workflows.plan_max_sample_size import (
    MonteCarloPowerEstimator,
    PlanMaxSampleSizeWorkflow,
)
from earlysign.integration.report.group_sequential.plot_oc_curve import (
    OCCurvePlotter,
)
from earlysign.framework.templates import TemplateBase
from earlysign.stats.essentials.methods.group_sequential import simulation
from earlysign.stats.essentials.methods.group_sequential.asn import ASNCalculator
from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    OCPointResult,
    Simulator,
)

# Module logger - consumers should configure logging for the project (handlers/formatters)
logger = logging.getLogger(__name__)


@dataclass
class SamplingPlan:
    """Container for sampling strategy metadata used during simulations."""

    strategy: simulation.SamplingStrategy
    sample_sizes: List[int]
    metadata: Dict[str, Any]

    def allocation_ratio(self) -> Optional[float]:
        value = self.metadata.get("allocation_ratio")
        return float(value) if value is not None else None


@dataclass
class DesignSearchContext:
    """Bundle design-time inputs shared across estimation routines."""

    info_times: List[float]
    planned_max_n: int
    sampling_plan: SamplingPlan


class TemplateProcedureAdapter:
    """Adapter that exposes a TemplateBase as a Procedure for the simulator.

    Parameters
    - template_factory: Callable[[connector, experiment_id, table_name], TemplateBase]
        Factory that returns a TemplateBase instance when given an ibis connector
        (e.g. ``ibis.duckdb.connect(':memory:')``), an experiment id and an
        optional table name.
    - experiment_id, table_name: passed to the factory when creating the
        template instance.
    - stop_decision_fn: Callable[[TemplateBase, int], Optional[Dict[str, Any]]]
        Function that inspects the template instance and the current look
        index and returns None to continue or a dict describing the stop
        decision (for example {'reject': True}). This predicate must be
        provided by the caller because the terminal state depends on the
        concrete template implementation.

    Behavior
    - reset(): create a new in-memory DuckDB ibis backend and instantiate a
      fresh TemplateBase through the provided factory.
    - ingest(cumulative): delegate to template.update(payload)
    - should_stop(look): call stop_decision_fn(template, look) and return its
      result (None or a dict). The adapter does not attempt to interpret
      template internals.
    """

    def __init__(
        self,
        template_factory: Callable[[Any, str, Optional[str]], TemplateBase],
        experiment_id: str,
        table_name: Optional[str],
        stop_decision_fn: Callable[[TemplateBase, int], Optional[Dict[str, Any]]],
        design_payload: Mapping[str, Any],
        rng_seed: Optional[int] = None,
    ) -> None:
        self.template_factory = template_factory
        self.experiment_id = experiment_id
        self.table_name = table_name
        self.stop_decision_fn = stop_decision_fn
        self.design_payload: Dict[str, Any] = dict(design_payload or {})
        self.rng_seed = rng_seed

        self.backend: Optional[Any] = None
        self.template: Optional[TemplateBase] = None

        self.reset()

    def _initial_design_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self.design_payload)
        if self.rng_seed is not None and "rng_seed" not in payload:
            payload["rng_seed"] = int(self.rng_seed)
        return payload

    def reset(self) -> None:
        try:
            backend = ibis.duckdb.connect(":memory:")
        except Exception:
            backend = ":memory:"

        self.backend = backend
        self.template = self.template_factory(
            backend, self.experiment_id, self.table_name
        )
        if self.template is not None:
            try:
                self.template.set_design(self._initial_design_payload())
            except Exception:
                pass

    def ingest(self, cumulative: Mapping[str, Any]) -> None:
        if self.template is None:
            raise RuntimeError("Template not initialized; call reset() first")
        self.template.update(dict(cumulative))

    def should_stop(self, look: int) -> Optional[Dict[str, Any]]:
        if self.template is None:
            raise RuntimeError("Template not initialized; call reset() first")
        return self.stop_decision_fn(self.template, look)

    def snapshot_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "design_payload": dict(self.design_payload),
        }
        if self.template is not None:
            try:
                status = self.template.status()
                metadata["template_status"] = status
            except Exception:
                metadata["template_status"] = None
        return metadata


@dataclass
class AddInterimToFixedSampleTest:
    """Class-based API to convert a generic fixed design into GST designs."""

    alpha: float
    power: float
    scheme: GSTSchemeHooks
    procedure_factory: ProcedureFactory
    asn_calculator_factory: Callable[[], ASNCalculator]
    simulator: Simulator
    keep_power: bool = False
    max_multiplier: int = 4
    tol: float = 0.01
    batch_size: Optional[int] = None
    seed: Optional[int] = None
    design_payload_builder: Optional[
        Callable[[Sequence[float], int], Mapping[str, Any]]
    ] = None
    power_estimator: Optional[Callable[[Sequence[float], int], float]] = None

    def __post_init__(self) -> None:
        self.batch_size = None if self.batch_size is None else int(self.batch_size)
        self.max_multiplier = max(2, int(self.max_multiplier))
        self.power_tolerance = float(self.tol)
        if self.power_estimator is None:
            self.power_estimator = MonteCarloPowerEstimator(
                build_design_context=self._build_design_context,
                build_design_payload=self._build_design_payload,
                make_procedure=self._make_procedure,
                build_simulation_request=lambda effect, n, strategy: self._build_simulation_request(
                    effect_size=effect,
                    planned_max_n=n,
                    sampling_strategy=strategy,
                ),
                simulator=self.simulator,
                target_effect=float(self.scheme.target_effect),
                seed=self.seed,
                logger=logger,
            )
        self.power_searcher = PlanMaxSampleSizeWorkflow(
            estimate_power=self.power_estimator,
            target_power=float(self.power),
            tolerance=self.power_tolerance,
            max_multiplier=self.max_multiplier,
        )

    @cached_property
    def fsd_design(self) -> Dict[str, Any]:
        return dict(self.scheme.fsd_planner())

    def _build_design_payload(
        self, info_times: Sequence[float], planned_max_n: int
    ) -> Mapping[str, Any]:
        if self.design_payload_builder is None:
            raw_payload: Mapping[str, Any] = {
                "info_times": list(map(float, info_times)),
                "planned_max_n": int(planned_max_n),
            }
        else:
            raw_payload = self.design_payload_builder(info_times, planned_max_n)
        return {str(key): value for key, value in dict(raw_payload).items()}

    def _make_procedure(
        self,
        info_times: Sequence[float],
        planned_max_n: int,
        payload: Mapping[str, Any],
    ) -> ProcedureLike:
        return self.procedure_factory(
            info_times, int(planned_max_n), payload, self.seed
        )

    def _build_sampling_plan(
        self,
        info_times: Sequence[float],
        planned_max_n: int,
    ) -> SamplingPlan:
        builder = self.scheme.sampling_strategy_builder
        rates = [float(x) for x in info_times]
        strategy = builder(
            rates,
            int(planned_max_n),
            self.batch_size,
        )
        metadata_raw = strategy.metadata()
        metadata: Dict[str, Any] = (
            {str(key): value for key, value in metadata_raw.items()}
            if isinstance(metadata_raw, Mapping)
            else {}
        )
        sample_sizes = metadata.get("cumulative_sizes")
        if sample_sizes is None:
            sample_sizes = simulation.compute_cumulative_sample_sizes(
                rates, int(planned_max_n)
            )
        sample_size_list = [int(x) for x in sample_sizes]
        if "cumulative_sizes" not in metadata:
            metadata["cumulative_sizes"] = sample_size_list
        return SamplingPlan(
            strategy=strategy, sample_sizes=sample_size_list, metadata=metadata
        )

    def _build_design_context(
        self, info_times: Sequence[float], planned_max_n: int
    ) -> DesignSearchContext:
        info_times_list = [float(x) for x in info_times]
        sampling_plan = self._build_sampling_plan(info_times_list, planned_max_n)
        return DesignSearchContext(
            info_times=info_times_list,
            planned_max_n=int(planned_max_n),
            sampling_plan=sampling_plan,
        )

    def _build_simulation_request(
        self,
        *,
        effect_size: float,
        planned_max_n: int,
        sampling_strategy: simulation.SamplingStrategy,
    ) -> Any:
        return self.scheme.simulation_request_builder(
            float(effect_size),
            int(planned_max_n),
            sampling_strategy,
            int(
                getattr(
                    self.simulator,
                    "n_simulations",
                    getattr(self.scheme, "n_simulations", 1),
                )
            ),
        )

    def compare_interim(
        self,
        k: int,
        plot_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute OC curve and related metadata for a given k.

        If the instance is configured with ``keep_power=False`` the GST uses the FSD total as the
        planned maximum (power may change). If True, the method searches for
        the minimal `planned_max_n` that preserves power at the design H1
        (`scheme.target_effect`) approximately.

        Note on the search behaviour and reproducibility
        -----------------------------------------------
        When ``keep_power=True`` the method performs a small binary
        search over candidate budgets. It uses an early-stop strategy: the
        first candidate `planned_max_n` whose estimated power falls in the
        acceptance interval [target, target + tol] will be accepted and the
        search will terminate. This makes the routine faster and aligns with
        the common "first-match" usage pattern, but it also means the
        returned budget depends on the order of mid-point evaluations and the
        Monte-Carlo RNG. For reproducible results callers should provide and
        manage `seed` consistently across runs.
        """
        logger.info(
            "Starting compare_interim(k=%s, keep_power=%s)",
            k,
            self.keep_power,
        )
        # When searching for a budget that preserves the target power, log the
        # requested target and tolerance so the Monte-Carlo estimates can be
        # interpreted relative to the goal.
        if self.keep_power:
            logger.info(
                "Target power=%s, tolerance=%s (accept range: [target, target+tol])",
                float(self.power),
                float(self.power_tolerance),
            )

        optimized_info_times = MinimizeASNOptimizer(
            calculator=self.asn_calculator_factory(),
            k_max=int(k),
            seed=self.seed,
        ).minimize()

        if self.keep_power is False:
            planned_max_n = int(self.fsd_design["sample_size"])
        else:
            planned_max_n = self.power_searcher.search(
                info_times=optimized_info_times,
                k=int(k),
                fsd_total=int(self.fsd_design["sample_size"]),
            )

        logger.info("Using planned_max_n=%s for k=%s", planned_max_n, k)

        context = self._build_design_context(optimized_info_times, planned_max_n)
        design_payload = self._build_design_payload(
            context.info_times, context.planned_max_n
        )

        sampling_plan = context.sampling_plan
        base_proc = self._make_procedure(
            context.info_times, context.planned_max_n, design_payload
        )
        raw_metadata = {
            str(key): value for key, value in base_proc.snapshot_metadata().items()
        }
        default_metadata: Dict[str, Any] = {
            "info_times": list(context.info_times),
            "planned_max_n": int(context.planned_max_n),
            "sample_size": int(self.fsd_design["sample_size"]),
            "sample_sizes": list(sampling_plan.sample_sizes),
            "target_power": float(self.power),
            "target_effect": float(self.scheme.target_effect),
            "alpha": float(self.alpha),
        }
        if "n_fsd_per_group" in self.fsd_design:
            default_metadata["n_fsd_per_group"] = int(
                self.fsd_design["n_fsd_per_group"]
            )
        alloc_meta = sampling_plan.allocation_ratio()
        if alloc_meta is not None:
            default_metadata["allocation_ratio"] = alloc_meta
        base_metadata = dict(default_metadata)
        base_metadata.update(raw_metadata)

        effect_grid = [float(es) for es in self.scheme.effect_sizes]
        oc_results: List[OCPointResult] = []
        if effect_grid:
            simulator = self.simulator
            procedure = base_proc
            procedure.reset()
            requests = [
                self._build_simulation_request(
                    effect_size=float(es),
                    planned_max_n=context.planned_max_n,
                    sampling_strategy=sampling_plan.strategy,
                )
                for es in effect_grid
            ]
            progress_active = tqdm is not None and logger.isEnabledFor(logging.INFO)
            points = simulator.simulate(
                procedure,
                requests=requests,
                rng_seed=self.seed,
            )
            if len(points) != len(effect_grid):
                raise RuntimeError("Simulator returned an unexpected number of results")
            effect_iterable = list(zip(effect_grid, points))
            iterable = (
                tqdm(
                    effect_iterable,
                    desc=f"Simulating effect sizes (k={k})",
                    unit="effect",
                    leave=False,
                )
                if progress_active
                else effect_iterable
            )
            for idx, (es, point) in enumerate(iterable):
                logger.debug(
                    "Simulated effect_size=%s (idx=%s) with planned_max_n=%s -> power=%s",
                    es,
                    idx,
                    context.planned_max_n,
                    float(point.power),
                )
                point.metadata = {
                    **base_metadata,
                    **(
                        {str(key): value for key, value in point.metadata.items()}
                        if point.metadata
                        else {}
                    ),
                    "sample_sizes": base_metadata.get("sample_sizes"),
                    "effect_size": float(es),
                    "n_looks": len(context.info_times),
                    "planned_max_n": int(context.planned_max_n),
                }
                oc_results.append(point)

        closest = min(
            oc_results,
            key=lambda r: abs(r.effect_size - float(self.scheme.target_effect)),
        )

        plotter = OCCurvePlotter()
        plot_err: Optional[str] = None
        try:
            if self.scheme.null_reference is None:
                raise ValueError("scheme.null_reference must not be None")
            plot_kwargs: Dict[str, Any] = {
                "null_value": float(self.scheme.null_reference),
            }
            ax = plotter.plot_oc_curve(
                oc_results,
                target_effect=float(self.scheme.target_effect),
                plot_options=plot_options,
                **plot_kwargs,
            )
        except Exception as e:
            # Do not silently swallow plotting errors; return them so callers
            # (and notebooks) can surface the root cause and fix simulator
            # metadata or data formatting issues.
            ax = None
            plot_err = repr(e)
            logger.exception("Failed to plot OC curve: %s", e)

        per_analysis_total = float(context.planned_max_n) / max(1, k)
        n_per_analysis = max(1, int(round(per_analysis_total)))

        return {
            "info_times": list(map(float, context.info_times)),
            "boundaries": base_metadata.get("boundaries"),
            "design_payload": design_payload,
            "procedure_metadata": base_metadata,
            "oc_results": oc_results,
            "power_at_delta": float(closest.power),
            "plot_axes": ax,
            "n_per_analysis": n_per_analysis,
            "planned_max_n": int(context.planned_max_n),
            "plot_error": plot_err,
        }
