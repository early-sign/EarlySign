"""Convert a fixed-sample two-proportion design into group-sequential procedures.

This module provides two high-level scenario functions used by the tutorial
and UI flows. Both functions are intentionally generic but the concrete
implementation below targets the two-proportions scheme using the
essentials/applications building blocks in the repository.

Implemented functions
---------------------
- add_interim(...)
    Build k-look group-sequential procedures taking the fixed-sample-design
    (FSD) sample size as the maximum sample size. For each k the function
    optimizes information timing (using the MinimizeASN workflow via the
    two-proportions normal approximation), forms a boundary + Procedure, runs
    Monte-Carlo operating-characteristic simulations and returns the OC curve
    results together with the realized power at the target effect (δ).

- add_interim_keep_power(...)
    Instead of fixing the GST maximum sample size to the FSD total, this
    scenario finds (for each k) the minimal maximum sample size that still
    attains the target power at the specified effect δ. Timing is optimized
    first (same optimizer) and then a small binary search over maximum sample
    size is used to locate the required budget.

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
import os
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

import ibis

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
from earlysign.stats.essentials.methods.group_sequential.spending import (
    SpendingFunction,
    get_spending_class,
)

# Module logger - consumers should configure logging for the project (handlers/formatters)
logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm as _tqdm_impl
    from tqdm.contrib.logging import (
        logging_redirect_tqdm as _logging_redirect_tqdm_impl,
    )
except Exception:  # pragma: no cover
    _tqdm_impl = None
    _logging_redirect_tqdm_impl = None

_FORCE_ENABLE_TQDM = os.environ.get("EARLYSIGN_ENABLE_TQDM") == "1"
_FORCE_DISABLE_TQDM = os.environ.get("EARLYSIGN_DISABLE_TQDM") == "1"
_TQDM_AVAILABLE = _tqdm_impl is not None


def _progress_enabled() -> bool:
    if _FORCE_DISABLE_TQDM:
        return False
    if not _TQDM_AVAILABLE:
        return False
    if _FORCE_ENABLE_TQDM:
        return True
    return logger.isEnabledFor(logging.INFO)


def _iter_with_progress(
    iterable: Iterable[Any], *, desc: str, unit: str, leave: bool = False
) -> Iterable[Any]:
    if not _progress_enabled():
        return iterable
    if _tqdm_impl is None:
        return iterable
    return cast(Iterable[Any], _tqdm_impl(iterable, desc=desc, unit=unit, leave=leave))


@contextmanager
def _tqdm_logging(loggers: Sequence[logging.Logger]) -> Iterator[None]:
    if not _progress_enabled():
        yield
        return
    if _logging_redirect_tqdm_impl is None:
        yield
        return
    with _logging_redirect_tqdm_impl(loggers=loggers):
        yield


def _create_progress_bar(*args: Any, **kwargs: Any) -> Any:
    if not _progress_enabled():
        return None
    if _tqdm_impl is None:
        return None
    return _tqdm_impl(*args, **kwargs)


def _update_progress(bar: Any, *, n: int = 1) -> None:
    if bar is not None:
        bar.update(n)


def _close_progress(bar: Any) -> None:
    if bar is not None:
        bar.close()


def _materialize_spending(
    spending: Union[str, SpendingFunction, Type[SpendingFunction]], *, alpha: float
) -> SpendingFunction:
    if hasattr(spending, "cumulative") and hasattr(
        spending, "boundaries_from_stage_alpha"
    ):
        return cast(SpendingFunction, spending)
    if isinstance(spending, type):
        spending_cls = cast(Type[SpendingFunction], spending)
    else:
        spending_cls = get_spending_class(str(spending))
    ctor = cast(Any, spending_cls)
    return cast(SpendingFunction, ctor(alpha=alpha))


class TemplateProcedureAdapter:
    """Adapter that exposes a TemplateBase as a Procedure for the simulator.

    Parameters
    - template_factory: Callable[[connector, experiment_id, table_name], TemplateBase]
        Factory that returns a TemplateBase instance when given an ibis connector
        (e.g. ``ibis.duckdb.connect(':memory:')``), an experiment id and an
        optional table name.
    - experiment_id, table_name: passed to the factory when creating the
        template instance.
    - terminate_fn: Callable[[TemplateBase, int], Optional[Dict[str, Any]]]
        Function that inspects the template instance and the current look
        index and returns None to continue or a dict describing the stop
        decision (for example {'reject': True}). This predicate must be
        provided by the caller because the terminal state depends on the
        concrete template implementation.

    Behavior
    - reset(): create a new in-memory DuckDB ibis backend and instantiate a
      fresh TemplateBase through the provided factory.
    - ingest(cumulative): delegate to template.update(payload)
    - should_stop(look): call terminate_fn(template, look) and return its
      result (None or a dict). The adapter does not attempt to interpret
      template internals.
    """

    def __init__(
        self,
        template_factory: Callable[[Any, str, Optional[str]], TemplateBase],
        experiment_id: str,
        table_name: Optional[str],
        terminate_fn: Callable[[TemplateBase, int], Optional[Dict[str, Any]]],
        info_times: Sequence[float],
        planned_max_n: int,
        design_payload: Optional[Mapping[str, Any]],
        rng_seed: Optional[int],
    ) -> None:
        if terminate_fn is None:
            raise ValueError("terminate_fn must be provided")
        self._factory = template_factory
        self._experiment_id = experiment_id
        self._table_name = table_name
        self._terminate_fn = terminate_fn
        self._info_times = [float(x) for x in info_times]
        self._planned_max_n = int(planned_max_n)
        self._design_payload = dict(design_payload or {})
        self._rng_seed = rng_seed

        self._backend: Optional[Any] = None
        self._template: Optional[TemplateBase] = None

        self.reset()

    def _initial_design_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self._design_payload)
        payload.setdefault("info_times", list(self._info_times))
        payload.setdefault("planned_max_n", int(self._planned_max_n))
        if self._rng_seed is not None:
            payload.setdefault("rng_seed", int(self._rng_seed))
        return payload

    def reset(self) -> None:
        try:
            backend = ibis.duckdb.connect(":memory:")
        except Exception:
            backend = ":memory:"

        self._backend = backend
        self._template = self._factory(backend, self._experiment_id, self._table_name)
        if self._template is not None:
            try:
                self._template.set_design(self._initial_design_payload())
            except Exception:
                pass

    def ingest(self, cumulative: Mapping[str, Any]) -> None:
        if self._template is None:
            raise RuntimeError("Template not initialized; call reset() first")
        self._template.update(dict(cumulative))

    def should_stop(self, look: int) -> Optional[Dict[str, Any]]:
        if self._template is None:
            raise RuntimeError("Template not initialized; call reset() first")
        return self._terminate_fn(self._template, look)

    def snapshot_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "info_times": list(self._info_times),
            "planned_max_n": int(self._planned_max_n),
            "design_payload": dict(self._design_payload),
        }
        if self._template is not None:
            try:
                status = self._template.status()
                metadata["template_status"] = status
            except Exception:
                metadata["template_status"] = None
        return metadata

    @property
    def template(self) -> TemplateBase:
        if self._template is None:
            raise RuntimeError("Template not initialized; call reset() first")
        return self._template


class AddInterimToFixedSampleTest:
    """Class-based API to convert a generic fixed design into GST designs."""

    def __init__(
        self,
        *,
        alpha: float,
        power: float,
        allocation_ratio: float,
        scheme: GSTSchemeHooks,
        procedure_factory: ProcedureFactory,
        asn_calculator_factory: Callable[[], ASNCalculator],
        effect_sizes: Optional[Sequence[float]] = None,
        target_effect: Optional[float] = None,
        null_reference: Optional[float] = None,
        n_sim: int = 200,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        design_payload_builder: Optional[
            Callable[[Sequence[float], int], Mapping[str, Any]]
        ] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.power = float(power)
        self.allocation_ratio = float(allocation_ratio)
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
        self.n_sim = int(n_sim)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.seed = seed
        self._procedure_factory = procedure_factory
        self._asn_calculator_factory = asn_calculator_factory
        self._design_payload_builder = design_payload_builder
        self._fsd_planner = scheme.fsd_planner
        self._simulator_factory = scheme.simulator_factory
        self._simulator_kwargs_builder = scheme.simulator_kwargs_builder

        # placeholders set by design_fst()
        self.planned_max_n: Optional[int] = None
        self._fsd_metadata: Dict[str, Any] = {}

        logger.debug(
            "Initialized AddInterimToFixedSampleTest: alpha=%s power=%s n_sim=%s scheme=%s",
            self.alpha,
            self.power,
            self.n_sim,
            scheme.name,
        )

    def design_fst(self) -> Dict[str, int]:
        """Compute fixed-sample (FSD) per-group sample size and set planned_max_n.

        Returns a dict with keys 'n_fsd_per_group' and 'planned_max_n'.
        """
        fsd_result = dict(self._fsd_planner())
        planned_max_n = int(fsd_result.get("planned_max_n", 0))
        if planned_max_n <= 0:
            raise ValueError("fsd_planner must return a positive planned_max_n")
        fsd_result.setdefault("fsd_total", planned_max_n)
        self._fsd_metadata = dict(fsd_result)
        self.planned_max_n = planned_max_n
        logger.info(
            "Computed FSD metadata: %s",
            self._fsd_metadata,
        )
        return dict(self._fsd_metadata)

    def _optimize_info_times(self, k: int) -> Sequence[float]:
        asn_calc = self._asn_calculator_factory()
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
        if self._design_payload_builder is None:
            raw_payload: Mapping[str, Any] = {
                "info_times": list(map(float, info_times)),
                "planned_max_n": int(planned_max_n),
            }
        else:
            raw_payload = self._design_payload_builder(info_times, planned_max_n)
        return {str(key): value for key, value in dict(raw_payload).items()}

    def _make_procedure(
        self,
        info_times: Sequence[float],
        planned_max_n: int,
        payload: Mapping[str, Any],
    ) -> ProcedureLike:
        return self._procedure_factory(
            info_times, int(planned_max_n), payload, self.seed
        )

    def _build_simulator(self) -> Any:
        return self._simulator_factory(self.n_sim, self.allocation_ratio)

    def _build_simulator_kwargs(
        self,
        *,
        effect_size: float,
        planned_max_n: int,
        sampling_strategy: simulation.SamplingStrategy,
    ) -> Dict[str, Any]:
        kwargs = dict(
            self._simulator_kwargs_builder(
                float(effect_size), int(planned_max_n), sampling_strategy
            )
        )
        kwargs.setdefault("effect_size", float(effect_size))
        kwargs.setdefault("n_simulations", int(self.n_sim))
        if self.seed is not None:
            kwargs.setdefault("rng_seed", self.seed)
        kwargs.setdefault("max_total", int(planned_max_n))
        kwargs.setdefault("sampling", sampling_strategy)
        return kwargs

    def _estimate_power(self, info_times: Sequence[float], planned_max_n: int) -> float:
        design_payload = self._build_design_payload(info_times, planned_max_n)
        procedure = self._make_procedure(info_times, planned_max_n, design_payload)
        procedure.reset()
        info_times_list = [float(x) for x in info_times]
        sampling_strategy: simulation.SamplingStrategy
        if self.batch_size is None:
            sampling_strategy = simulation.InfoTimeSampling(
                info_times=info_times_list,
                planned_max_n=int(planned_max_n),
                allocation_ratio=float(self.allocation_ratio),
            )
        else:
            sampling_strategy = simulation.FixedBatchSampling(
                size=int(self.batch_size),
                allocation_ratio=float(self.allocation_ratio),
                total=int(planned_max_n),
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

        simulator = self._build_simulator()
        kwargs = self._build_simulator_kwargs(
            effect_size=self.target_effect,
            planned_max_n=planned_max_n,
            sampling_strategy=sampling_strategy,
        )
        point = simulator.simulate(
            procedure,
            **kwargs,
        )
        logger.info(
            "Estimated power (planned_max_n=%s) -> %s",
            planned_max_n,
            float(point.power),
        )
        return float(point.power)

    def compare_interim(
        self,
        k: int,
        keep_power_at_H1: bool = False,
        plot_options: Optional[Dict[str, Any]] = None,
        *,
        max_multiplier: int = 4,
        tol: float = 0.01,
    ) -> Dict[str, Any]:
        """Compute OC curve and related metadata for a given k.

        If `keep_power_at_H1` is False, the GST uses the FSD total as the
        planned maximum (power may change). If True, the method searches for
        the minimal `planned_max_n` that preserves power at the design H1
        (self.target_effect) approximately.

        Note on the search behaviour and reproducibility
        -----------------------------------------------
        When `keep_power_at_H1=True` the method performs a small binary
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

        logger.info(
            "Starting compare_interim(k=%s, keep_power_at_H1=%s)", k, keep_power_at_H1
        )
        # When searching for a budget that preserves the target power, log the
        # requested target and tolerance so the Monte-Carlo estimates can be
        # interpreted relative to the goal.
        if keep_power_at_H1:
            logger.info(
                "Target power=%s, tolerance=%s (accept range: [target, target+tol])",
                float(self.power),
                float(tol),
            )

        info_times = self._optimize_info_times(int(k))

        if keep_power_at_H1 is False:
            planned_max_n = int(self.planned_max_n)
        else:
            fsd_total = int(self.planned_max_n)
            lo = 2 * int(k)
            hi = max(2 * int(k), int(max_multiplier * fsd_total))
            best_n = hi
            # Binary search for the minimal planned_max_n that attains the
            # target power. Accept a configuration when the estimated power
            # lies in [target_power, target_power + tol]. If the estimate is
            # below the target, increase lower bound; if it's greater than
            # target + tol, reduce the upper bound to try a smaller budget.
            target_power = float(self.power)
            search_bar = _create_progress_bar(
                total=None,
                desc=f"Power search (k={k})",
                unit="candidate",
                leave=False,
            )
            try:
                with _tqdm_logging([logger]):
                    while lo <= hi:
                        _update_progress(search_bar)
                        mid = (lo + hi) // 2
                        achieved = self._estimate_power(info_times, int(mid))
                        # If achieved is within [target, target + tol], accept this mid
                        if (
                            achieved >= target_power
                            and achieved <= target_power + float(tol)
                        ):
                            best_n = int(mid)
                            logger.info(
                                "Accepting planned_max_n=%s with achieved power=%s within [%s, %s] (early-stop)",
                                int(mid),
                                achieved,
                                target_power,
                                target_power + float(tol),
                            )
                            # Early-stop: accept the first candidate that meets the
                            # acceptance interval and terminate the search. This makes
                            # the behaviour deterministic w.r.t. the mid evaluation
                            # order and relies on caller-managed RNG seed for
                            # reproducibility.
                            break
                        elif achieved < target_power:
                            # Not enough power, increase budget
                            lo = mid + 1
                        else:
                            # Achieved > target + tol: we might be able to reduce budget
                            hi = mid - 1
            finally:
                _close_progress(search_bar)

            planned_max_n = int(best_n)

        logger.info("Using planned_max_n=%s for k=%s", planned_max_n, k)

        design_payload = self._build_design_payload(info_times, planned_max_n)

        info_times_list = [float(x) for x in info_times]
        sampling_strategy: simulation.SamplingStrategy
        if self.batch_size is None:
            sample_sizes = simulation.compute_cumulative_sample_sizes(
                info_times_list, planned_max_n
            )
            sampling_strategy = simulation.InfoTimeSampling(
                info_times=info_times_list,
                planned_max_n=int(planned_max_n),
                allocation_ratio=float(self.allocation_ratio),
            )
        else:
            sampling_strategy = simulation.FixedBatchSampling(
                size=int(self.batch_size),
                allocation_ratio=float(self.allocation_ratio),
                total=int(planned_max_n),
            )
            sample_sizes = sampling_strategy.metadata().get("cumulative_sizes", [])
        base_proc = self._make_procedure(info_times, planned_max_n, design_payload)
        base_proc_for_loop: Optional[ProcedureLike] = base_proc
        raw_metadata = base_proc.snapshot_metadata()
        base_metadata: Dict[str, Any] = {
            str(key): value for key, value in raw_metadata.items()
        }
        base_metadata.setdefault("info_times", info_times_list)
        base_metadata.setdefault("planned_max_n", int(planned_max_n))
        base_metadata.setdefault("allocation_ratio", float(self.allocation_ratio))
        if "n_fsd_per_group" in self._fsd_metadata:
            base_metadata.setdefault(
                "n_fsd_per_group", int(self._fsd_metadata["n_fsd_per_group"])
            )
        if "fsd_total" in self._fsd_metadata:
            base_metadata.setdefault("fsd_total", int(self._fsd_metadata["fsd_total"]))
        base_metadata.setdefault("target_power", float(self.power))
        base_metadata.setdefault("target_effect", float(self.target_effect))
        base_metadata.setdefault("alpha", float(self.alpha))
        base_metadata.setdefault("sample_sizes", list(sample_sizes))

        oc_results = []
        effect_grid = list(self.effect_sizes)
        effect_iterable = _iter_with_progress(
            effect_grid,
            desc=f"Simulating effect sizes (k={k})",
            unit="effect",
            leave=False,
        )
        with _tqdm_logging([logger]):
            for idx, es in enumerate(effect_iterable):
                procedure = (
                    base_proc_for_loop
                    if base_proc_for_loop is not None
                    else self._make_procedure(info_times, planned_max_n, design_payload)
                )
                procedure.reset()
                logger.debug(
                    "Simulating effect_size=%s (idx=%s) with planned_max_n=%s",
                    es,
                    idx,
                    planned_max_n,
                )
                simulator = self._build_simulator()
                kwargs = self._build_simulator_kwargs(
                    effect_size=float(es),
                    planned_max_n=planned_max_n,
                    sampling_strategy=sampling_strategy,
                )
                point = simulator.simulate(
                    procedure,
                    **kwargs,
                )
                logger.debug(
                    "Simulated effect_size=%s -> power=%s", es, float(point.power)
                )
                base_proc_for_loop = None
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

        per_group_total = planned_max_n / (1.0 + float(self.allocation_ratio))
        n_per_analysis = max(1, int(round(per_group_total / max(1, k))))

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


def add_interim(
    *,
    alpha: float,
    power: float,
    ks: Sequence[int],
    spending: Any,
    scheme: GSTSchemeHooks,
    allocation_ratio: float,
    effect_sizes: Optional[Sequence[float]] = None,
    n_sim: int = 200,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    plot_options: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Scenario A: build GSTs using the fixed-sample budget as the maximum."""
    spending_obj = _materialize_spending(spending, alpha=alpha)
    resolved_scheme = scheme.with_effect_sizes(effect_sizes)
    procedure_factory = resolved_scheme.procedure_factory_builder(
        spending_obj, allocation_ratio
    )
    asn_calculator_factory = resolved_scheme.asn_factory_builder(spending_obj)

    inst = AddInterimToFixedSampleTest(
        alpha=alpha,
        power=power,
        allocation_ratio=allocation_ratio,
        scheme=resolved_scheme,
        procedure_factory=procedure_factory,
        asn_calculator_factory=asn_calculator_factory,
        n_sim=n_sim,
        batch_size=batch_size,
        seed=seed,
    )
    inst.design_fst()

    results: Dict[int, Dict[str, Any]] = {}
    ks_iterable = [int(value) for value in ks]
    with _tqdm_logging([logger]):
        for k_value in _iter_with_progress(
            ks_iterable,
            desc="compare_interim (fixed budget)",
            unit="design",
            leave=False,
        ):
            logger.debug("Running compare_interim for k=%s (fixed budget)", k_value)
            results[int(k_value)] = inst.compare_interim(
                k=int(k_value), keep_power_at_H1=False, plot_options=plot_options
            )

    return results


def add_interim_keep_power(
    *,
    alpha: float,
    power: float,
    ks: Sequence[int],
    spending: Any,
    scheme: GSTSchemeHooks,
    allocation_ratio: float,
    effect_sizes: Optional[Sequence[float]] = None,
    n_sim: int = 200,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    max_multiplier: int = 4,
    tol: float = 0.01,
    plot_options: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Scenario B: search for the minimal budget that preserves target power."""
    spending_obj = _materialize_spending(spending, alpha=alpha)
    resolved_scheme = scheme.with_effect_sizes(effect_sizes)
    procedure_factory = resolved_scheme.procedure_factory_builder(
        spending_obj, allocation_ratio
    )
    asn_calculator_factory = resolved_scheme.asn_factory_builder(spending_obj)

    inst = AddInterimToFixedSampleTest(
        alpha=alpha,
        power=power,
        allocation_ratio=allocation_ratio,
        scheme=resolved_scheme,
        procedure_factory=procedure_factory,
        asn_calculator_factory=asn_calculator_factory,
        n_sim=n_sim,
        batch_size=batch_size,
        seed=seed,
    )
    inst.design_fst()

    results: Dict[int, Dict[str, Any]] = {}
    ks_iterable = [int(value) for value in ks]
    with _tqdm_logging([logger]):
        for k_value in _iter_with_progress(
            ks_iterable,
            desc="compare_interim (keep power)",
            unit="design",
            leave=False,
        ):
            logger.debug("Running compare_interim for k=%s (keep power)", k_value)
            results[int(k_value)] = inst.compare_interim(
                k=int(k_value),
                keep_power_at_H1=True,
                plot_options=plot_options,
                max_multiplier=max_multiplier,
                tol=tol,
            )

    return results
