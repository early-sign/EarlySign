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
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

import ibis
import numpy as np

from earlysign.framework.templates import TemplateBase
from earlysign.stats.applications.design.group_sequential.initial_design.helpers.protocols import (
    ProcedureFactory,
    ProcedureLike,
)
from earlysign.stats.applications.design.group_sequential.initial_design.workflows.optimize_timing.minimize_asn import (
    MinimizeASNOptimizer,
)
from earlysign.stats.applications.report.group_sequential.plot_oc_curve import (
    OCCurvePlotter,
)
from earlysign.stats.essentials.methods.group_sequential import simulation
from earlysign.stats.essentials.methods.group_sequential.asn import ASNCalculator
from earlysign.stats.essentials.methods.group_sequential.boundary import (
    BoundaryCalculator,
    BoundaryCalculatorSpec,
    EfficacySpec,
    FutilitySpec,
)
from earlysign.stats.essentials.methods.group_sequential.spending import (
    HSDSpending,
    OBFSpending,
    PocockSpending,
    SpendingFunction,
)
from earlysign.stats.essentials.schemes.two_proportions.asn import (
    build_asn_calculator,
)
from earlysign.stats.essentials.schemes.two_proportions.effect_size import (
    TwoProportionsEffectSizeCalculator,
)
from earlysign.stats.essentials.schemes.two_proportions.simulator import (
    TwoProportionsSimulator,
)
from earlysign.stats.essentials.schemes.two_proportions.wald_z import (
    compute_wald_z,
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


def _spending_factory(
    spending: Union[SpendingFunction, str, Any], *, alpha: float
) -> SpendingFunction:
    """Return a SpendingFunction instance from a simple spec.

    Accepts either an already-instantiated SpendingFunction or a string key
    ('obrien_fleming' / 'obf', 'pocock', 'hsd'). When 'hsd' is chosen we use
    the default gamma provided by HSDSpending constructor.
    """
    # Prefer duck-typing over isinstance checks against Protocols which are
    # not runtime-checkable in some type-checking configurations.
    if hasattr(spending, "cumulative") and hasattr(
        spending, "boundaries_from_stage_alpha"
    ):
        # assume it's a SpendingFunction-like object; cast for the type checker
        return cast(SpendingFunction, spending)
    key = str(spending).strip().lower()
    if key in ("obf", "obrien_fleming", "o'brien-fleming", "obrien-fleming"):
        return OBFSpending(alpha=alpha, sided=2)
    if key == "pocock":
        return PocockSpending(alpha=alpha)
    if key == "hsd":
        return HSDSpending(alpha=alpha)
    # Fallback: default to OBF
    return OBFSpending(alpha=alpha, sided=2)


def _spending_family(spending_obj: SpendingFunction) -> str:
    """Return canonical family key for a SpendingFunction instance.

    This maps concrete spending classes to the string keys expected by
    BoundaryCalculator (e.g. 'obf', 'pocock', 'hsd').
    """
    if isinstance(spending_obj, OBFSpending):
        return "obf"
    if isinstance(spending_obj, PocockSpending):
        return "pocock"
    if isinstance(spending_obj, HSDSpending):
        return "hsd"
    # conservative default
    return "obf"


def _build_two_prop_procedure_factory(
    *,
    spending_obj: SpendingFunction,
    alpha: float,
    allocation_ratio: float,
) -> ProcedureFactory:
    """Return a :class:`ProcedureFactory` tailored to two-proportion tests.

    Examples
    --------
    >>> from earlysign.stats.essentials.methods.group_sequential.spending import OBFSpending
    >>> factory = _build_two_prop_procedure_factory(
    ...     spending_obj=OBFSpending(alpha=0.05, sided=2),
    ...     alpha=0.05,
    ...     allocation_ratio=1.0,
    ... )
    >>> procedure = factory([0.5, 1.0], 200, None, None)
    >>> metadata = procedure.snapshot_metadata()
    >>> (metadata["planned_max_n"], round(metadata["boundaries"]["upper"][0], 10))
    (200, 2.7718076487)
    >>> metadata["boundaries"]["lower"][0] == float("-inf")
    True
    """
    family = _spending_family(spending_obj)

    def _factory(
        info_times: Sequence[float],
        planned_max_n: int,
        design_payload: Optional[Mapping[str, Any]],
        rng_seed: Optional[int],
    ) -> ProcedureLike:
        rates = np.asarray(info_times, dtype=float)
        eff = EfficacySpec(style="alpha_spending", family=family)
        fut = FutilitySpec(mode="none")
        spec = BoundaryCalculatorSpec(alpha=alpha, tails=2, efficacy=eff, futility=fut)
        calculator = BoundaryCalculator(spec)
        boundaries = calculator.compute_boundaries(rates)

        payload = {str(key): value for key, value in dict(design_payload or {}).items()}
        payload.setdefault("info_times", [float(x) for x in rates.tolist()])
        payload.setdefault("planned_max_n", int(planned_max_n))
        payload.setdefault("spending_family", family)
        upper = [float(x) for x in boundaries["upper"]]
        lower_raw = boundaries.get("lower")
        payload["boundaries"] = {
            "upper": upper,
            "lower": [float(x) for x in lower_raw] if lower_raw is not None else None,
        }

        return _TwoPropProcedure(
            info_times=rates.tolist(),
            z_upper=list(boundaries["upper"]),
            z_lower=(
                list(boundaries["lower"])
                if boundaries.get("lower") is not None
                else None
            ),
            planned_max_n=int(planned_max_n),
            allocation_ratio=float(allocation_ratio),
            design_payload=payload,
        )

    return _factory


@dataclass
class _TwoPropProcedure:
    """Procedure implementation for the two-proportions simulator."""

    info_times: Sequence[float]
    z_upper: Sequence[float]
    z_lower: Optional[Sequence[float]]
    planned_max_n: int
    allocation_ratio: float = 1.0
    pooled: bool = True
    design_payload: Optional[Mapping[str, Any]] = None

    _cum_nA: int = field(default=0, init=False)
    _cum_mA: int = field(default=0, init=False)
    _cum_nB: int = field(default=0, init=False)
    _cum_mB: int = field(default=0, init=False)
    _last_checked_idx: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._info_times = np.asarray(list(map(float, self.info_times)), dtype=float)
        if self._info_times.size == 0 or not np.isclose(self._info_times[-1], 1.0):
            raise ValueError("info_times must be non-empty and end at 1.0")
        self._z_upper = [float(x) for x in self.z_upper]
        self._z_lower = (
            [float(x) for x in self.z_lower] if self.z_lower is not None else None
        )
        self._planned_max_n = int(self.planned_max_n)
        self._sample_n_total = self._compute_sample_sizes()
        self._metadata_snapshot = self._build_metadata_snapshot()

    def _compute_sample_sizes(self) -> np.ndarray:
        schedule = simulation.compute_cumulative_sample_sizes(
            [float(x) for x in self.info_times], self._planned_max_n
        )
        return np.asarray(schedule, dtype=int)

    def _build_metadata_snapshot(self) -> Dict[str, Any]:
        raw_payload = dict(self.design_payload or {})
        payload: Dict[str, Any] = {
            str(key): value for key, value in raw_payload.items()
        }
        sample_sizes = [int(x) for x in self._sample_n_total.tolist()]
        payload.setdefault("info_times", [float(x) for x in self._info_times])
        payload.setdefault("planned_max_n", int(self._planned_max_n))
        payload.setdefault("sample_sizes", sample_sizes)
        payload.setdefault(
            "boundaries",
            {
                "upper": [float(x) for x in self._z_upper],
                "lower": (
                    [float(x) for x in self._z_lower]
                    if self._z_lower is not None
                    else None
                ),
            },
        )
        payload.setdefault("allocation_ratio", float(self.allocation_ratio))
        payload.setdefault("n_looks", len(sample_sizes))
        return payload

    def ingest(self, cumulative: Mapping[str, Any]) -> None:
        self._cum_nA += int(cumulative.get("nA", 0))
        self._cum_mA += int(cumulative.get("mA", 0))
        self._cum_nB += int(cumulative.get("nB", 0))
        self._cum_mB += int(cumulative.get("mB", 0))

    def should_stop(self, look: int) -> Optional[Dict[str, Any]]:
        total = int(self._cum_nA + self._cum_nB)
        idx = 0
        for analysis_idx, required in enumerate(self._sample_n_total, start=1):
            if total >= int(required):
                idx = analysis_idx
        if idx <= self._last_checked_idx:
            return None

        for analysis in range(self._last_checked_idx + 1, idx + 1):
            z = compute_wald_z(
                nA=self._cum_nA,
                mA=self._cum_mA,
                nB=self._cum_nB,
                mB=self._cum_mB,
                pooled=self.pooled,
            )
            upper = float(self._z_upper[analysis - 1])
            lower = (
                float(self._z_lower[analysis - 1])
                if self._z_lower is not None and analysis - 1 < len(self._z_lower)
                else None
            )
            self._last_checked_idx = analysis
            if z >= upper:
                return {
                    "reject": True,
                    "reason": "efficacy",
                    "analysis": analysis,
                    "z": float(z),
                }
            if lower is not None and z <= lower:
                return {
                    "reject": False,
                    "reason": "futility",
                    "analysis": analysis,
                    "z": float(z),
                }
        return None

    def reset(self) -> None:
        self._cum_nA = 0
        self._cum_mA = 0
        self._cum_nB = 0
        self._cum_mB = 0
        self._last_checked_idx = 0

    def snapshot_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata_snapshot)


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
    """Class-based API to convert an FSD (two proportions) into GST designs.

    Usage:
        inst = AddInterimToFixedSampleTest(
            alpha=..., delta=..., power=..., p_control=..., allocation_ratio=...,
            procedure_factory=..., asn_calculator_factory=...
        )
        inst.design_fst()
        res = inst.compare_interim(k=3, keep_power_at_H1=False)

    The class stores the FSD baseline (per-group sample size) after
    `design_fst()` and exposes `compare_interim()` to compute operating
    characteristics for a given number of looks `k` either keeping the
    original power at H1 or fixing the maximum sample size to the FSD total.
    """

    def __init__(
        self,
        *,
        alpha: float,
        delta: float,
        power: float,
        p_control: float,
        allocation_ratio: float,
        procedure_factory: ProcedureFactory,
        asn_calculator_factory: Callable[[], ASNCalculator],
        effect_sizes: Optional[Sequence[float]] = None,
        n_sim: int = 200,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        design_payload_builder: Optional[
            Callable[[Sequence[float], int], Mapping[str, Any]]
        ] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.power = float(power)
        self.p_control = float(p_control)
        self.allocation_ratio = float(allocation_ratio)
        self.effect_sizes = (
            list(effect_sizes)
            if effect_sizes is not None
            else list(np.linspace(0.0, max(self.delta * 2.0, 0.02), num=11))
        )
        self.n_sim = int(n_sim)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.seed = seed
        self._procedure_factory = procedure_factory
        self._asn_calculator_factory = asn_calculator_factory
        self._design_payload_builder = design_payload_builder

        # placeholders set by design_fst()
        self.n_fsd_per_group: Optional[int] = None
        self.planned_max_n: Optional[int] = None

        self._simulator = TwoProportionsSimulator(
            effect_size=float(self.delta),
            n_simulations=int(self.n_sim),
            allocation_ratio=float(self.allocation_ratio),
            strategy=None,
        )
        logger.debug(
            "Initialized AddInterimToFixedSampleTest: alpha=%s delta=%s power=%s n_sim=%s",
            self.alpha,
            self.delta,
            self.power,
            self.n_sim,
        )

    def design_fst(self) -> Dict[str, int]:
        """Compute fixed-sample (FSD) per-group sample size and set planned_max_n.

        Returns a dict with keys 'n_fsd_per_group' and 'planned_max_n'.
        """
        effect_calc = TwoProportionsEffectSizeCalculator(p_control=self.p_control)
        n_fsd_per_group = effect_calc.calculate_sample_size(
            effect_size=self.delta, alpha=self.alpha, power=self.power
        )
        planned_max_n = int(2 * int(n_fsd_per_group))

        self.n_fsd_per_group = int(n_fsd_per_group)
        self.planned_max_n = int(planned_max_n)
        logger.info(
            "Computed FSD: n_fsd_per_group=%s planned_max_n=%s",
            self.n_fsd_per_group,
            self.planned_max_n,
        )
        return {
            "n_fsd_per_group": int(self.n_fsd_per_group),
            "planned_max_n": int(self.planned_max_n),
        }

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

        point = self._simulator.simulate(
            procedure,
            p_control=float(self.p_control),
            effect_size=float(self.delta),
            n_simulations=int(self.n_sim),
            rng_seed=self.seed,
            max_total=int(planned_max_n),
            sampling=sampling_strategy,
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
        (self.delta) approximately.

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
        if self.n_fsd_per_group is not None:
            base_metadata.setdefault("n_fsd_per_group", int(self.n_fsd_per_group))
            base_metadata.setdefault("fsd_total", int(2 * self.n_fsd_per_group))
        base_metadata.setdefault("target_power", float(self.power))
        base_metadata.setdefault("target_effect", float(self.delta))
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
                point = self._simulator.simulate(
                    procedure,
                    p_control=float(self.p_control),
                    effect_size=float(es),
                    n_simulations=int(self.n_sim),
                    rng_seed=self.seed,
                    max_total=int(planned_max_n),
                    sampling=sampling_strategy,
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

        closest = min(oc_results, key=lambda r: abs(r.effect_size - float(self.delta)))

        plotter = OCCurvePlotter()
        plot_err: Optional[str] = None
        try:
            ax = plotter.plot_oc_curve(
                oc_results,
                target_effect=float(self.delta),
                null_value=float(self.p_control),
                plot_options=plot_options,
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
    delta: float,
    power: float,
    ks: Sequence[int],
    spending: Any,
    p_control: float,
    effect_sizes: Optional[Sequence[float]] = None,
    n_sim: int = 200,
    batch_size: Optional[int] = None,
    allocation_ratio: float,
    seed: Optional[int] = None,
    plot_options: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Scenario A: build GSTs using FSD sample size as maximum and plot OC curves.

    Algorithm (high-level / pseudo):
    1. Compute FSD per-group sample size `n_fsd` for two-proportions using the
       normal-approximation calculator with inputs (p_control, delta, alpha,
       power). The FSD total maximum sample size used for GST is
       `planned_max_n = 2 * n_fsd`.
    2. For each desired number of looks k in `ks`:
       a. Build an ASN calculator adapted for two proportions
          (``build_asn_calculator``) and instantiate the
          ``MinimizeASNOptimizer`` with `k`.
       b. Run timing optimization to get information fractions (rates).
       c. Convert the timing + spending policy to Z-boundaries using
          ``BoundaryCalculator``.
       d. Form a lightweight ``Procedure`` (see `_TwoPropProcedure`) using
          the boundaries and a per-analysis per-group sample size computed as
          floor((planned_max_n/2) / k).
       e. Evaluate the procedure using ``TwoProportionsSimulator`` across
          a set of effect sizes (defaults chosen relative to `delta`) and
          collect the OC results.
    3. Plot the OC curves for each k using ``OCCurvePlotter`` and return a
       dictionary keyed by k containing info_times, boundaries, the
       OC results and estimated power at `delta`.

    Returns
    -------
    Mapping: k -> {
        'info_times': list[float],
        'boundaries': dict,  # boundary dict returned by BoundaryCalculator.compute_boundaries
        'oc_results': List[OCPointResult],
        'power_at_delta': float,
    }
    """
    # Refactor note: use class-based API for clearer stages. Keep thin wrapper
    # for backward compatibility by delegating to AddInterimToFixedSampleTest.
    # Create concrete spending instance and factories to inject into the
    # class so the class itself remains agnostic to spending/futility policy.
    spending_obj = _spending_factory(spending, alpha=alpha)

    def asn_calculator_factory() -> Any:
        return build_asn_calculator(
            alpha=alpha,
            beta=1.0 - power,
            sided=2,
            p_control=p_control,
            effect_size=delta,
            allocation_ratio=allocation_ratio,
            spending=spending_obj,
        )

    procedure_factory = _build_two_prop_procedure_factory(
        spending_obj=spending_obj,
        alpha=alpha,
        allocation_ratio=allocation_ratio,
    )

    inst = AddInterimToFixedSampleTest(
        alpha=alpha,
        delta=delta,
        power=power,
        p_control=p_control,
        allocation_ratio=allocation_ratio,
        procedure_factory=procedure_factory,
        asn_calculator_factory=asn_calculator_factory,
        effect_sizes=effect_sizes,
        n_sim=n_sim,
        batch_size=batch_size,
        seed=seed,
    )

    # ensure FSD design computed
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
    delta: float,
    power: float,
    ks: Sequence[int],
    spending: Any,
    p_control: float,
    effect_sizes: Optional[Sequence[float]] = None,
    n_sim: int = 200,
    batch_size: Optional[int] = None,
    allocation_ratio: float,
    seed: Optional[int] = None,
    max_multiplier: int = 4,
    tol: float = 0.01,
    plot_options: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Scenario B: design GST maximum sample size to keep the same power at δ.

    Algorithm (high-level):
    1. For each k in ks, run the same timing optimization as in `add_interim` to
       obtain information fractions.
    2. Given the timings, perform a binary search over the maximum total sample
       size `planned_max_n` (search range: [2, max_multiplier * fsd_total]) to
       find the minimal budget where the simulated power at effect δ is at
       least the requested `power` (within tolerance `tol`). For each trial in
       the binary search we form a `_TwoPropProcedure` using per-analysis
       per-group sample sizes = floor((planned_max_n/2) / k) and evaluate it
       with the two-proportions simulator.
    3. Return the found `planned_max_n`, the boundaries, OC curve at the final
       budget and the estimated power at δ.

    The search uses Monte-Carlo estimates and therefore may be noisy; for
    production use increase `n_sim` or tighten `tol`.
    """
    spending_obj = _spending_factory(spending, alpha=alpha)

    if effect_sizes is None:
        effect_sizes = [delta]

    def asn_calculator_factory() -> Any:
        return build_asn_calculator(
            alpha=alpha,
            beta=1.0 - power,
            sided=2,
            p_control=p_control,
            effect_size=delta,
            allocation_ratio=allocation_ratio,
            spending=spending_obj,
        )

    procedure_factory = _build_two_prop_procedure_factory(
        spending_obj=spending_obj,
        alpha=alpha,
        allocation_ratio=allocation_ratio,
    )

    inst = AddInterimToFixedSampleTest(
        alpha=alpha,
        delta=delta,
        power=power,
        p_control=p_control,
        allocation_ratio=allocation_ratio,
        procedure_factory=procedure_factory,
        asn_calculator_factory=asn_calculator_factory,
        effect_sizes=effect_sizes,
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
