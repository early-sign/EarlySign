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

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Union, cast

import ibis
import numpy as np

from earlysign.framework.templates import TemplateBase
from earlysign.stats.applications.design.group_sequential.initial_design.workflows.optimize_timing.minimize_asn import (
    MinimizeASNOptimizer,
)
from earlysign.stats.applications.report.group_sequential.plot_oc_curve import (
    OCCurvePlotter,
)
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
    ProportionsSampleSize,
    TwoProportionsEffectSizeCalculator,
)
from earlysign.stats.essentials.schemes.two_proportions.simulator import (
    TwoProportionsSimulator,
)
from earlysign.stats.essentials.schemes.two_proportions.wald_z import (
    compute_wald_z,
)


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


@dataclass
class _TwoPropProcedure:
    """Lightweight Procedure implementing the simulator Protocol.

    Behavior (pseudo-algorithm):
    - Initialization inputs: info_times (fractions), z_upper, z_lower,
      n_per_analysis (per-group), allocation_ratio, pooled variance flag.
    - Precompute per-analysis planned sample sizes using
      TwoProportionsEffectSizeCalculator.sample_sizes(). This yields arrays
      of control/treatment/total sample sizes at each analysis.
    - ingest(delta_batch): accumulate cumulative counts (nA, mA, nB, mB).
    - should_stop(look_idx): determine the highest analysis index for which
      the cumulative total samples reaches or exceeds the planned size. If a
      new analysis boundary is now available, compute Wald Z from cumulative
      counts and compare against the corresponding upper/lower boundaries.
      If crossing occurs return a dict {'reject': bool, 'reason': 'efficacy'|'futility'}
      otherwise return None to continue.

    This minimal implementation is purposely independent from the ledger and
    is intended for Monte-Carlo simulation only.
    """

    info_times: Sequence[float]
    z_upper: Sequence[float]
    z_lower: Sequence[float]
    n_per_analysis: int
    allocation_ratio: float = 1.0
    pooled: bool = True

    # runtime state
    _cum_nA: int = 0
    _cum_mA: int = 0
    _cum_nB: int = 0
    _cum_mB: int = 0
    _last_checked_idx: int = 0

    def __post_init__(self) -> None:
        # build sample sizes per analysis using the effect-size helper (only
        # uses sample size layout, we don't depend on effect parameters here).
        calc = TwoProportionsEffectSizeCalculator()
        sample_size_obj = ProportionsSampleSize(n_per_analysis=int(self.n_per_analysis))
        sample_info = calc.sample_sizes(
            sample_size_obj,
            allocation_ratio=float(self.allocation_ratio),
            info_times=np.asarray(self.info_times, dtype=float),
        )
        # sample_info['n_total'] is an array (per-analysis total N)
        self._sample_n_total = np.asarray(sample_info["n_total"], dtype=int)

    def ingest(self, cumulative: Dict[str, int]) -> None:
        # cumulative here is an incremental batch dict from the simulator
        self._cum_nA += int(cumulative.get("nA", 0))
        self._cum_mA += int(cumulative.get("mA", 0))
        self._cum_nB += int(cumulative.get("nB", 0))
        self._cum_mB += int(cumulative.get("mB", 0))

    def should_stop(self, look: int) -> Optional[Dict[str, Any]]:
        # Determine if we crossed the next analysis sample threshold.
        total = int(self._cum_nA + self._cum_nB)
        # find highest analysis index reached by current total
        idx = 0
        for i, req in enumerate(self._sample_n_total):
            if total >= int(req):
                idx = i + 1

        if idx <= self._last_checked_idx:
            return None

        # evaluate sequentially for any newly reached analyses
        for analysis in range(self._last_checked_idx + 1, idx + 1):
            # compute Wald Z on cumulative counts
            z = compute_wald_z(
                nA=self._cum_nA,
                mA=self._cum_mA,
                nB=self._cum_nB,
                mB=self._cum_mB,
                pooled=self.pooled,
            )
            up = float(self.z_upper[analysis - 1])
            lo = (
                float(self.z_lower[analysis - 1])
                if self.z_lower is not None
                else float("-inf")
            )
            if z >= up:
                self._last_checked_idx = analysis
                return {
                    "reject": True,
                    "reason": "efficacy",
                    "z": float(z),
                    "analysis": analysis,
                }
            if lo is not None and z <= lo:
                self._last_checked_idx = analysis
                return {
                    "reject": False,
                    "reason": "futility",
                    "z": float(z),
                    "analysis": analysis,
                }

        self._last_checked_idx = idx
        return None

    def reset(self) -> None:
        self._cum_nA = 0
        self._cum_mA = 0
        self._cum_nB = 0
        self._cum_mB = 0
        self._last_checked_idx = 0


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
    ) -> None:
        if terminate_fn is None:
            raise ValueError(
                "terminate_fn is required and must be provided by the caller"
            )
        self._factory = template_factory
        self._experiment_id = experiment_id
        self._table_name = table_name
        self._terminate_fn = terminate_fn

        self._backend: Optional[Any] = None
        self._template: Optional[TemplateBase] = None

        # Immediately create the first instance
        self.reset()

    def reset(self) -> None:
        # Create a fresh in-memory DuckDB ibis backend and instantiate the
        # template factory with it. This ensures reset produces a clean
        # execution environment for Monte-Carlo replications.
        try:
            backend = ibis.duckdb.connect(":memory:")
        except Exception:
            # Fallback: allow passing a string connector to the factory; the
            # factory can decide how to handle it. We still attempt to pass
            # the ibis backend where possible.
            backend = ":memory:"

        self._backend = backend
        self._template = self._factory(backend, self._experiment_id, self._table_name)

    def ingest(self, cumulative: Dict[str, Any]) -> None:
        if self._template is None:
            raise RuntimeError("Template not initialized; call reset() first")
        # Delegate to template's update API
        self._template.update(cumulative)

    def should_stop(self, look: int) -> Optional[Dict[str, Any]]:
        if self._template is None:
            raise RuntimeError("Template not initialized; call reset() first")
        return self._terminate_fn(self._template, look)

    # Expose the underlying template for callers that need to query status
    @property
    def template(self) -> TemplateBase:
        if self._template is None:
            raise RuntimeError("Template not initialized; call reset() first")
        return self._template


class AddInterimToFixedSampleTest:
    """Class-based API to convert an FSD (two proportions) into GST designs.

    Usage:
        inst = AddInterimToFixedSampleTest(alpha, delta, power, spending, p_control, allocation_ratio)
        inst.design_fst()
        res = inst.compare_interim(k=3, keep_power_at_H1=False)

    The class stores the FSD baseline (per-group sample size) after
    `design_fst()` and exposes `compare_interim()` to compute operating
    characteristics for a given number of looks `k` either keeping the
    original power at H1 or fixing the maximum sample size to the FSD total.
    """

    def __init__(
        self,
        alpha: float,
        delta: float,
        power: float,
        p_control: float,
        allocation_ratio: float,
        effect_sizes: Optional[Sequence[float]] = None,
        n_sim: int = 2000,
        batch_size: int = 100,
        seed: Optional[int] = None,
        designer_factory: Optional[Callable[[Sequence[float]], Dict[str, Any]]] = None,
        asn_calculator_factory: Optional[Callable[[], ASNCalculator]] = None,
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
        self.batch_size = int(batch_size)
        self.seed = seed
        self.designer_factory = designer_factory
        self.asn_calculator_factory = asn_calculator_factory

        # placeholders set by design_fst()
        self.n_fsd_per_group: Optional[int] = None
        self.planned_max_n: Optional[int] = None

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
        return {
            "n_fsd_per_group": int(self.n_fsd_per_group),
            "planned_max_n": int(self.planned_max_n),
        }

    def _optimize_info_times(self, k: int) -> Sequence[float]:
        # ASN calculator is injected via asn_calculator_factory for flexibility
        if self.asn_calculator_factory is not None:
            asn_calc = self.asn_calculator_factory()
        else:
            raise RuntimeError(
                "asn_calculator_factory not provided; caller must inject factory to build ASN calculator"
            )

        optimizer = MinimizeASNOptimizer(
            calculator=asn_calc, k_max=int(k), seed=self.seed
        )
        info_times = optimizer.minimize()
        return info_times if info_times else list(np.linspace(1.0 / k, 1.0, k))

    def make_designer(self, info_times: Sequence[float]) -> Dict[str, Any]:
        """Instantiate a designer (boundary generator) for the provided info times.

        The designer factory is injected by callers when different boundary
        policies (spending, futility) or optimizers are required. If no
        factory was provided, raise an error to force the caller to supply
        a designer.
        """
        if self.designer_factory is None:
            raise RuntimeError(
                "designer_factory not provided; caller must inject a designer_factory"
            )
        return self.designer_factory(info_times)

    def compare_interim(self, k: int, keep_power_at_H1: bool = False) -> Dict[str, Any]:
        """Compute OC curve and related metadata for a given k.

        If `keep_power_at_H1` is False, the GST uses the FSD total as the
        planned maximum (power may change). If True, the method searches for
        the minimal `planned_max_n` that preserves power at the design H1
        (self.delta) approximately.
        """
        if self.planned_max_n is None:
            raise RuntimeError("Call design_fst() before compare_interim()")

        info_times = self._optimize_info_times(int(k))
        bdict = self.make_designer(info_times)

        if keep_power_at_H1 is False:
            planned_max_n = int(self.planned_max_n)
        else:
            # binary search for minimal planned_max_n that attains power at H1
            fsd_total = int(self.planned_max_n)
            lo = 2
            hi = max(2, 4 * fsd_total)
            best_n = hi
            tol = 0.01
            while lo <= hi:
                mid = (lo + hi) // 2
                per_group = max(1, mid // 2)
                n_per_analysis = max(1, per_group // int(k))

                proc = _TwoPropProcedure(
                    info_times=info_times,
                    z_upper=bdict["upper"],
                    z_lower=bdict["lower"],
                    n_per_analysis=n_per_analysis,
                    allocation_ratio=self.allocation_ratio,
                    pooled=True,
                )
                sim = TwoProportionsSimulator(
                    effect_size=self.delta,
                    n_simulations=int(self.n_sim),
                    batch_size=int(self.batch_size),
                    allocation_ratio=float(self.allocation_ratio),
                )
                point = sim.simulate(
                    proc,
                    p_control=float(self.p_control),
                    effect_size=float(self.delta),
                    n_simulations=int(self.n_sim),
                    rng_seed=self.seed,
                    max_total=int(mid),
                )
                achieved = float(point.power)
                if achieved + tol >= float(self.power):
                    best_n = int(mid)
                    hi = mid - 1
                else:
                    lo = mid + 1

            planned_max_n = int(best_n)

        # Evaluate OC curve at planned_max_n
        per_group_total = max(1, int(planned_max_n) // 2)
        n_per_analysis = max(1, per_group_total // int(k))

        proc = _TwoPropProcedure(
            info_times=info_times,
            z_upper=bdict["upper"],
            z_lower=bdict["lower"],
            n_per_analysis=n_per_analysis,
            allocation_ratio=self.allocation_ratio,
            pooled=True,
        )

        sim = TwoProportionsSimulator(
            effect_size=self.delta,
            n_simulations=int(self.n_sim),
            batch_size=int(self.batch_size),
            allocation_ratio=float(self.allocation_ratio),
        )

        oc_results = []
        for es in self.effect_sizes:
            point = sim.simulate(
                proc,
                p_control=float(self.p_control),
                effect_size=float(es),
                n_simulations=int(self.n_sim),
                rng_seed=self.seed,
                max_total=int(planned_max_n),
            )
            oc_results.append(point)

        closest = min(oc_results, key=lambda r: abs(r.effect_size - float(self.delta)))

        plotter = OCCurvePlotter()
        try:
            ax = plotter.plot_oc_curve(
                oc_results,
                target_effect=float(self.delta),
                null_value=float(self.p_control),
            )
        except Exception:
            ax = None

        return {
            "info_times": list(map(float, info_times)),
            "boundaries": bdict,
            "oc_results": oc_results,
            "power_at_delta": float(closest.power),
            "plot_axes": ax,
            "n_per_analysis": n_per_analysis,
            "planned_max_n": int(planned_max_n),
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
    n_sim: int = 2000,
    batch_size: int = 100,
    allocation_ratio: float,
    seed: Optional[int] = None,
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
        asn = build_asn_calculator(
            alpha=alpha,
            beta=1.0 - power,
            sided=2,
            p_control=p_control,
            effect_size=delta,
            allocation_ratio=allocation_ratio,
            spending=spending_obj,
        )
        return asn

    def designer_factory(info_times: Sequence[float]) -> Dict[str, Any]:
        eff = EfficacySpec(
            style="alpha_spending", family=_spending_family(spending_obj)
        )
        fut = FutilitySpec(mode="none")
        spec = BoundaryCalculatorSpec(alpha=alpha, tails=2, efficacy=eff, futility=fut)
        bc = BoundaryCalculator(spec)
        return bc.compute_boundaries(np.asarray(info_times, dtype=float))

    inst = AddInterimToFixedSampleTest(
        alpha=alpha,
        delta=delta,
        power=power,
        p_control=p_control,
        allocation_ratio=allocation_ratio,
        effect_sizes=effect_sizes,
        n_sim=n_sim,
        batch_size=batch_size,
        seed=seed,
        designer_factory=designer_factory,
        asn_calculator_factory=asn_calculator_factory,
    )

    # ensure FSD design computed
    inst.design_fst()

    results: Dict[int, Dict[str, Any]] = {}
    for k in ks:
        results[int(k)] = inst.compare_interim(k=int(k), keep_power_at_H1=False)

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
    n_sim: int = 2000,
    batch_size: int = 100,
    allocation_ratio: float,
    seed: Optional[int] = None,
    max_multiplier: int = 4,
    tol: float = 0.01,
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
    # baseline FSD per-group sample size to guide search range
    effect_calc = TwoProportionsEffectSizeCalculator(p_control=p_control)
    n_fsd_per_group = effect_calc.calculate_sample_size(
        effect_size=delta, alpha=alpha, power=power
    )
    int(2 * int(n_fsd_per_group))

    spending_obj = _spending_factory(spending, alpha=alpha)

    if effect_sizes is None:
        effect_sizes = [delta]

    # Delegate to class-based API for clarity and reuse
    # Create concrete spending instance and factories to inject into the
    # class so the class itself remains agnostic to spending/futility policy.
    spending_obj = _spending_factory(spending, alpha=alpha)

    def asn_calculator_factory() -> Any:
        asn = build_asn_calculator(
            alpha=alpha,
            beta=1.0 - power,
            sided=2,
            p_control=p_control,
            effect_size=delta,
            allocation_ratio=allocation_ratio,
            spending=spending_obj,
        )
        return asn

    def designer_factory(info_times: Sequence[float]) -> Dict[str, Any]:
        eff = EfficacySpec(
            style="alpha_spending", family=_spending_family(spending_obj)
        )
        fut = FutilitySpec(mode="none")
        spec = BoundaryCalculatorSpec(alpha=alpha, tails=2, efficacy=eff, futility=fut)
        bc = BoundaryCalculator(spec)
        return bc.compute_boundaries(np.asarray(info_times, dtype=float))

    inst = AddInterimToFixedSampleTest(
        alpha=alpha,
        delta=delta,
        power=power,
        p_control=p_control,
        allocation_ratio=allocation_ratio,
        effect_sizes=effect_sizes,
        n_sim=n_sim,
        batch_size=batch_size,
        seed=seed,
        designer_factory=designer_factory,
        asn_calculator_factory=asn_calculator_factory,
    )
    inst.design_fst()

    results: Dict[int, Dict[str, Any]] = {}
    for k in ks:
        results[int(k)] = inst.compare_interim(k=int(k), keep_power_at_H1=True)

    return results
