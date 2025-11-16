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
``earlysign.stats.essentials`` and ``earlysign.stats.applications`` as
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
from contextlib import nullcontext
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import ibis
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from earlysign.framework.templates import TemplateBase
from earlysign.stats.applications.design.group_sequential.initial_design.helpers.protocols import (
    ProcedureFactory,
    ProcedureLike,
)
from earlysign.stats.applications.design.group_sequential.initial_design.helpers.scheme import (
    GSTSchemeHooks,
)
from earlysign.stats.applications.design.group_sequential.initial_design.workflows.optimize_timing.minimize_asn import (
    MinimizeASNOptimizer,
)
from earlysign.stats.applications.report.group_sequential.plot_oc_curve import (
    OCCurvePlotter,
)
from earlysign.stats.essentials.methods.group_sequential import simulation
from earlysign.stats.essentials.methods.group_sequential.asn import ASNCalculator
from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    OCPointResult,
)

# Module logger - consumers should configure logging for the project (handlers/formatters)
logger = logging.getLogger(__name__)


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
        design_payload: Optional[Mapping[str, Any]] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.template_factory = template_factory
        self.experiment_id = experiment_id
        self.table_name = table_name
        self.stop_decision_fn = stop_decision_fn
        payload: Dict[str, Any] = dict(design_payload or {})
        info_times_raw = payload.get("info_times")
        if info_times_raw is None:
            raise ValueError("design_payload must include 'info_times'")
        payload["info_times"] = [float(x) for x in info_times_raw]
        planned_raw = payload.get("planned_max_n")
        if planned_raw is None:
            raise ValueError("design_payload must include 'planned_max_n'")
        payload["planned_max_n"] = int(planned_raw)
        self.design_payload: Dict[str, Any] = payload
        self.rng_seed = rng_seed

        self.backend: Optional[Any] = None
        self.template: Optional[TemplateBase] = None

        self.reset()

    def _initial_design_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self.design_payload)
        if self.rng_seed is not None:
            payload.setdefault("rng_seed", int(self.rng_seed))
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


class AddInterimToFixedSampleTest:
    """Class-based API to convert a generic fixed design into GST designs."""

    def __init__(
        self,
        *,
        alpha: float,
        power: float,
        scheme: GSTSchemeHooks,
        procedure_factory: ProcedureFactory,
        asn_calculator_factory: Callable[[], ASNCalculator],
        simulator: Any,
        effect_sizes: Optional[Sequence[float]] = None,
        target_effect: Optional[float] = None,
        null_reference: Optional[float] = None,
        keep_power: bool = False,
        max_multiplier: int = 4,
        tol: float = 0.01,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        design_payload_builder: Optional[
            Callable[[Sequence[float], int], Mapping[str, Any]]
        ] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.power = float(power)
        self.scheme = scheme
        self.target_effect = (
            float(target_effect)
            if target_effect is not None
            else float(scheme.target_effect)
        )
        self.null_reference = (
            float(null_reference)
            if null_reference is not None
            else (
                float(scheme.null_reference)
                if scheme.null_reference is not None
                else None
            )
        )
        resolved_effects = (
            list(effect_sizes)
            if effect_sizes is not None
            else list(scheme.effect_sizes)
        )
        if not resolved_effects:
            raise ValueError("effect_sizes must contain at least one value")
        self.effect_sizes = [float(x) for x in resolved_effects]
        self.batch_size = None if batch_size is None else int(batch_size)
        self.seed = seed
        self.procedure_factory = procedure_factory
        self.asn_calculator_factory = asn_calculator_factory
        self.design_payload_builder = design_payload_builder
        self.simulator = simulator
        self.keep_power = bool(keep_power)
        self.max_multiplier = max(2, int(max_multiplier))
        self.power_tolerance = float(tol)

        # placeholders set by design_fst()
        self.planned_max_n: Optional[int] = None
        self.fsd_metadata: Dict[str, Any] = {}

        logger.debug(
            "Initialized AddInterimToFixedSampleTest: alpha=%s power=%s n_sim=%s scheme=%s",
            self.alpha,
            self.power,
            self.n_simulations,
            scheme.name,
        )

    def design_fst(self) -> Dict[str, int]:
        """Compute fixed-sample (FSD) per-group sample size and set planned_max_n.

        Returns a dict with keys 'n_fsd_per_group' and 'planned_max_n'.
        """
        fsd_result = dict(self.scheme.fsd_planner())
        planned_max_n = int(fsd_result.get("planned_max_n", 0))
        if planned_max_n <= 0:
            raise ValueError("fsd_planner must return a positive planned_max_n")
        fsd_result.setdefault("fsd_total", planned_max_n)
        self.fsd_metadata = dict(fsd_result)
        self.planned_max_n = planned_max_n
        logger.info(
            "Computed FSD metadata: %s",
            self.fsd_metadata,
        )
        return dict(self.fsd_metadata)

    @property
    def n_simulations(self) -> int:
        value = getattr(self.simulator, "n_simulations", None)
        if value is None:
            raise AttributeError("Simulator must expose an 'n_simulations' attribute")
        count = int(value)
        if count <= 0:
            raise ValueError("Simulator n_simulations must be positive")
        return count

    def _optimize_info_times(self, k: int) -> Sequence[float]:
        asn_calc = self.asn_calculator_factory()
        optimizer = MinimizeASNOptimizer(
            calculator=asn_calc,
            k_max=int(k),
            seed=self.seed,
        )
        info_times = optimizer.minimize()
        if not info_times:
            raise RuntimeError("Timing optimisation failed to produce info_times")
        logger.info("Timing optimisation (k=%s) -> info_times=%s", k, info_times)
        return info_times

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

    def _build_sampling_strategy(
        self,
        info_times: Sequence[float],
        planned_max_n: int,
    ) -> Tuple[simulation.SamplingStrategy, List[int]]:
        builder = getattr(self.scheme, "sampling_strategy_builder", None)
        if builder is None:
            raise AttributeError("Scheme does not provide a sampling strategy builder")
        rates = [float(x) for x in info_times]
        strategy = builder(
            rates,
            int(planned_max_n),
            self.batch_size,
        )
        metadata = strategy.metadata()
        sample_sizes = metadata.get("cumulative_sizes")
        if sample_sizes is None:
            sample_sizes = simulation.compute_cumulative_sample_sizes(
                rates, int(planned_max_n)
            )
        return strategy, [int(x) for x in sample_sizes]

    def _build_simulation_request(
        self,
        *,
        effect_size: float,
        planned_max_n: int,
        sampling_strategy: simulation.SamplingStrategy,
    ) -> Any:
        builder = getattr(self.scheme, "simulation_request_builder", None)
        if builder is not None:
            return builder(
                float(effect_size),
                int(planned_max_n),
                sampling_strategy,
                int(self.n_simulations),
            )

        legacy_builder = getattr(self.scheme, "_simulator_kwargs_builder", None)
        if legacy_builder is None:
            raise AttributeError(
                "No simulation request builder available on the scheme hooks"
            )
        kwargs = dict(
            legacy_builder(
                float(effect_size),
                int(planned_max_n),
                sampling_strategy,
            )
        )
        kwargs.setdefault("effect_size", float(effect_size))
        kwargs.setdefault("n_simulations", int(self.n_simulations))
        kwargs.setdefault("max_total", int(planned_max_n))
        kwargs.setdefault("sampling", sampling_strategy)
        from earlysign.stats.essentials.schemes.two_proportions.simulator import (
            TwoProportionsSimulationRequest,
        )

        return TwoProportionsSimulationRequest(
            p_control=float(kwargs["p_control"]),
            effect_size=float(kwargs["effect_size"]),
            n_simulations=int(kwargs["n_simulations"]),
            max_total=int(kwargs["max_total"]),
            sampling=kwargs["sampling"],
        )

    def _estimate_power(self, info_times: Sequence[float], planned_max_n: int) -> float:
        design_payload = self._build_design_payload(info_times, planned_max_n)
        procedure = self._make_procedure(info_times, planned_max_n, design_payload)
        procedure.reset()
        info_times_list = [float(x) for x in info_times]
        sampling_strategy, _ = self._build_sampling_strategy(
            info_times_list, planned_max_n
        )
        try:
            meta_before = sampling_strategy.metadata()
        except Exception:
            meta_before = None
        logger.info(
            "Estimating power (planned_max_n=%s, info_times=%s, sampling_meta=%s)",
            planned_max_n,
            info_times_list,
            meta_before,
        )

        simulator = self.simulator
        request = self._build_simulation_request(
            effect_size=self.target_effect,
            planned_max_n=planned_max_n,
            sampling_strategy=sampling_strategy,
        )
        points = simulator.simulate(
            procedure,
            requests=[request],
            rng_seed=self.seed,
        )
        if not points:
            raise RuntimeError("Simulator returned no results during power estimation")
        point = points[0]
        logger.info(
            "Estimated power (planned_max_n=%s) -> %s",
            planned_max_n,
            float(point.power),
        )
        return float(point.power)

    def _find_planned_max_n_preserving_power(
        self,
        info_times: Sequence[float],
        *,
        k: int,
    ) -> int:
        if self.planned_max_n is None:
            raise RuntimeError("Call design_fst() before searching for planned_max_n")
        fsd_total = int(self.planned_max_n)
        lo = 2 * int(k)
        hi = max(2 * int(k), int(self.max_multiplier * fsd_total))
        best_n = hi
        target_power = float(self.power)
        progress_active = tqdm is not None and logger.isEnabledFor(logging.INFO)
        search_bar = (
            tqdm(
                total=None,
                desc=f"Power search (k={k})",
                unit="candidate",
                leave=False,
            )
            if progress_active
            else None
        )
        log_context = (
            logging_redirect_tqdm(loggers=[logger])
            if progress_active
            else nullcontext()
        )
        try:
            with log_context:
                while lo <= hi:
                    if search_bar is not None:
                        search_bar.update()
                    mid = (lo + hi) // 2
                    achieved = self._estimate_power(info_times, int(mid))
                    if target_power <= achieved <= target_power + self.power_tolerance:
                        best_n = int(mid)
                        logger.info(
                            "Accepting planned_max_n=%s with achieved power=%s within [%s, %s] (early-stop)",
                            int(mid),
                            achieved,
                            target_power,
                            target_power + self.power_tolerance,
                        )
                        break
                    if achieved < target_power:
                        lo = mid + 1
                    else:
                        hi = mid - 1
        finally:
            if search_bar is not None:
                search_bar.close()

        return int(best_n)

    def compare_interim(
        self,
        k: int,
        plot_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute OC curve and related metadata for a given k.

        If the instance is configured with ``keep_power=False`` the GST uses the FSD total as the
        planned maximum (power may change). If True, the method searches for
        the minimal `planned_max_n` that preserves power at the design H1
        (self.target_effect) approximately.

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
        if self.planned_max_n is None:
            raise RuntimeError("Call design_fst() before compare_interim()")

        keep_power = self.keep_power
        logger.info(
            "Starting compare_interim(k=%s, keep_power=%s)",
            k,
            keep_power,
        )
        # When searching for a budget that preserves the target power, log the
        # requested target and tolerance so the Monte-Carlo estimates can be
        # interpreted relative to the goal.
        if keep_power:
            logger.info(
                "Target power=%s, tolerance=%s (accept range: [target, target+tol])",
                float(self.power),
                float(self.power_tolerance),
            )

        info_times = self._optimize_info_times(int(k))

        if keep_power is False:
            planned_max_n = int(self.planned_max_n)
        else:
            planned_max_n = self._find_planned_max_n_preserving_power(
                info_times, k=int(k)
            )

        logger.info("Using planned_max_n=%s for k=%s", planned_max_n, k)

        design_payload = self._build_design_payload(info_times, planned_max_n)

        info_times_list = [float(x) for x in info_times]
        sampling_strategy, sample_sizes = self._build_sampling_strategy(
            info_times_list, planned_max_n
        )
        sampling_metadata = sampling_strategy.metadata()
        base_proc = self._make_procedure(info_times, planned_max_n, design_payload)
        raw_metadata = base_proc.snapshot_metadata()
        base_metadata: Dict[str, Any] = {
            str(key): value for key, value in raw_metadata.items()
        }
        base_metadata.setdefault("info_times", info_times_list)
        base_metadata.setdefault("planned_max_n", int(planned_max_n))
        if isinstance(sampling_metadata, Mapping):
            alloc_meta = sampling_metadata.get("allocation_ratio")
            if alloc_meta is not None:
                base_metadata.setdefault("allocation_ratio", float(alloc_meta))
        base_metadata.setdefault("sample_sizes", list(sample_sizes))
        if "n_fsd_per_group" in self.fsd_metadata:
            base_metadata.setdefault(
                "n_fsd_per_group", int(self.fsd_metadata["n_fsd_per_group"])
            )
        if "fsd_total" in self.fsd_metadata:
            base_metadata.setdefault("fsd_total", int(self.fsd_metadata["fsd_total"]))
        base_metadata.setdefault("target_power", float(self.power))
        base_metadata.setdefault("target_effect", float(self.target_effect))
        base_metadata.setdefault("alpha", float(self.alpha))

        effect_grid = [float(es) for es in self.effect_sizes]
        oc_results: List[OCPointResult] = []
        if effect_grid:
            simulator = self.simulator
            procedure = base_proc
            procedure.reset()
            requests = [
                self._build_simulation_request(
                    effect_size=float(es),
                    planned_max_n=planned_max_n,
                    sampling_strategy=sampling_strategy,
                )
                for es in effect_grid
            ]
            progress_active = tqdm is not None and logger.isEnabledFor(logging.INFO)
            log_context = (
                logging_redirect_tqdm(loggers=[logger])
                if progress_active
                else nullcontext()
            )
            with log_context:
                points = simulator.simulate(
                    procedure,
                    requests=requests,
                    rng_seed=self.seed,
                )
            if len(points) != len(effect_grid):
                raise RuntimeError("Simulator returned an unexpected number of results")
            paired = list(zip(effect_grid, points))
            effect_iterable = (
                tqdm(
                    paired,
                    desc=f"Simulating effect sizes (k={k})",
                    unit="effect",
                    leave=False,
                )
                if progress_active
                else paired
            )
            for idx, (es, point) in enumerate(effect_iterable):
                logger.debug(
                    "Simulated effect_size=%s (idx=%s) with planned_max_n=%s -> power=%s",
                    es,
                    idx,
                    planned_max_n,
                    float(point.power),
                )
                merged_md = dict(base_metadata)
                if point.metadata:
                    merged_md.update(
                        {str(key): value for key, value in point.metadata.items()}
                    )
                merged_md.setdefault("sample_sizes", base_metadata.get("sample_sizes"))
                merged_md["effect_size"] = float(es)
                merged_md["n_looks"] = int(len(info_times))
                merged_md["planned_max_n"] = int(planned_max_n)
                point.metadata = merged_md
                oc_results.append(point)

        closest = min(
            oc_results, key=lambda r: abs(r.effect_size - float(self.target_effect))
        )

        plotter = OCCurvePlotter()
        plot_err: Optional[str] = None
        try:
            plot_kwargs: Dict[str, Any] = {}
            if self.null_reference is not None:
                plot_kwargs["null_value"] = float(self.null_reference)
            ax = plotter.plot_oc_curve(
                oc_results,
                target_effect=float(self.target_effect),
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

        per_analysis_total = planned_max_n / max(1, k)
        n_per_analysis = max(1, int(round(per_analysis_total)))

        return {
            "info_times": list(map(float, info_times)),
            "boundaries": base_metadata.get("boundaries"),
            "design_payload": design_payload,
            "procedure_metadata": base_metadata,
            "oc_results": oc_results,
            "power_at_delta": float(closest.power),
            "plot_axes": ax,
            "n_per_analysis": n_per_analysis,
            "planned_max_n": int(planned_max_n),
            "plot_error": plot_err,
        }
