#!/usr/bin/env python
"""Benchmark direct GST design helpers against the ledger-backed template adapter.

This script compares the runtime of the classic `_TwoPropProcedure` workflow
(driven directly via `AddInterimToFixedSampleTest`) and the template-backed
`TemplateProcedureAdapter`. It keeps the Monte-Carlo simulation counts small so
it finishes quickly while still showcasing the relative slowdown seen in the
ledger-backed path. A lightweight cProfile report for the template path is
written next to this script to help identify bottlenecks.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Sequence

import ibis
import numpy as np
import pstats

from earlysign.api.ab_tests import BinomialABTest
from earlysign.core.util.ibis_cache import CacheEntry, IbisCache
from earlysign.integration.design.group_sequential.initial_design.helpers.template_helpers import (
    build_template_procedure_factory,
)
from earlysign.integration.design.group_sequential.initial_design.helpers.scheme import (
    GSTSchemeHooks,
)
from earlysign.integration.design.group_sequential.initial_design.scenarios.fst_to_gst import (
    AddInterimToFixedSampleTest,
)
from earlysign.stats.methods.group_sequential.spending import (
    SpendingFunction,
    get_spending_class,
)
from earlysign.stats.schemes.two_proportions.design import (
    build_two_proportions_scheme,
)

LOGGER = logging.getLogger(__name__)


def _configure_logging(*, level: str, enable_tqdm: bool) -> None:
    """Configure root logging and tqdm behaviour for the benchmark."""

    normalized = level.upper()
    numeric_level = getattr(logging, normalized, logging.INFO)

    if not enable_tqdm:
        os.environ.setdefault("EARLYSIGN_DISABLE_TQDM", "1")

    logging.basicConfig(level=numeric_level)
    LOGGER.setLevel(numeric_level)


def _direct_compare_runtime(
    *,
    alpha: float,
    delta: float,
    power: float,
    p_control: float,
    allocation_ratio: float,
    effect_sizes: Optional[Sequence[float]],
    n_sim: int,
    batch_size: Optional[int],
    seed: int,
    k: int,
) -> float:
    LOGGER.debug(
        "Direct flow: preparing AddInterimToFixedSampleTest (n_sim=%s, k=%s)",
        n_sim,
        k,
    )
    spending_cls = get_spending_class("pocock")
    spending_obj = spending_cls(alpha=alpha)
    scheme = build_two_proportions_scheme(
        p_control=p_control,
        target_effect=delta,
        effect_sizes=effect_sizes or [delta],
        alpha=alpha,
        power=power,
        allocation_ratio=allocation_ratio,
    )
    resolved_scheme = scheme.with_effect_sizes(effect_sizes)
    procedure_factory = resolved_scheme.procedure_factory_builder(
        spending_obj, allocation_ratio
    )
    asn_factory = resolved_scheme.asn_factory_builder(spending_obj)

    inst = AddInterimToFixedSampleTest(
        alpha=alpha,
        power=power,
        scheme=resolved_scheme,
        procedure_factory=procedure_factory,
        asn_calculator_factory=asn_factory,
        simulator=resolved_scheme.simulator_factory(n_sim, allocation_ratio),
        batch_size=batch_size,
        seed=seed,
    )
    start = time.perf_counter()
    LOGGER.debug("Direct flow: calling compare_interim")
    inst.compare_interim(k=k, plot_options=None)
    return time.perf_counter() - start


def _template_payload(
    info_times: Sequence[float],
    planned_max_n: int,
    *,
    alpha: float,
    spending_family: str,
) -> Dict[str, Any]:
    return {
        "alpha": float(alpha),
        "hypothesis": {"structure": "two_sided_symmetric"},
        "statistic": {"kind": "wald_z", "scale": "z"},
        "efficacy": {"style": "alpha_spending", "family": spending_family},
        "futility": {"mode": "none", "binding_mode": "non_binding"},
        "planned_max_n": int(planned_max_n),
        "planned_info_times": [float(x) for x in info_times],
        "metadata": {"generated_from": "benchmark_template_adapter"},
    }


def _template_compare_runtime_with_profile(
    *,
    alpha: float,
    delta: float,
    power: float,
    p_control: float,
    allocation_ratio: float,
    effect_sizes: Optional[Sequence[float]],
    n_sim: int,
    batch_size: Optional[int],
    seed: int,
    k: int,
    profile_limit: int,
    enable_profile: bool,
) -> tuple[float, Optional[str]]:
    LOGGER.debug(
        "Template flow: preparing AddInterimToFixedSampleTest (n_sim=%s, k=%s)",
        n_sim,
        k,
    )
    spending_cls = get_spending_class("pocock")
    spending_obj = spending_cls(alpha=alpha)
    spending_family = spending_obj.name
    scheme = build_two_proportions_scheme(
        p_control=p_control,
        target_effect=delta,
        effect_sizes=effect_sizes or [delta],
        alpha=alpha,
        power=power,
        allocation_ratio=allocation_ratio,
    )
    resolved_scheme = scheme.with_effect_sizes(effect_sizes)
    asn_factory = resolved_scheme.asn_factory_builder(spending_obj)

    template_cache_store: MutableMapping[str, CacheEntry] = {}

    def _template_factory(backend: Any, experiment_id: str, table_name: Optional[str]):
        ibis_cache = IbisCache(backend, mode="execute", cache=template_cache_store)
        return BinomialABTest(
            backend,
            experiment_id,
            table_name,
            ibis_cache=ibis_cache,
        )

    def _stop_decision_fn(template: Any, look: int) -> Optional[Dict[str, Any]]:
        status = template.status()
        if getattr(status, "stop_recommended", False):
            return {"reject": True, "reason": "template_stop", "look": int(look)}
        return None

    template_proc_factory = build_template_procedure_factory(
        template_factory=_template_factory,
        stop_decision_fn=_stop_decision_fn,
        experiment_id="benchmark_exp",
        table_name=None,
    )

    inst = AddInterimToFixedSampleTest(
        alpha=alpha,
        power=power,
        scheme=resolved_scheme,
        procedure_factory=template_proc_factory,
        asn_calculator_factory=asn_factory,
        simulator=resolved_scheme.simulator_factory(n_sim, allocation_ratio),
        batch_size=batch_size,
        seed=seed,
        design_payload_builder=lambda info_times, planned_max_n: _template_payload(
            info_times,
            planned_max_n,
            alpha=alpha,
            spending_family=spending_family,
        ),
    )
    profiler: Optional[cProfile.Profile]
    if enable_profile:
        profiler = cProfile.Profile()
        profiler.enable()
    else:
        profiler = None
    start = time.perf_counter()
    inst.compare_interim(k=k, plot_options=None)
    elapsed = time.perf_counter() - start
    if profiler is None:
        return elapsed, None

    profiler.disable()

    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).strip_dirs().sort_stats(
        "cumtime"
    ).print_stats(profile_limit)
    return elapsed, stats_stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark direct GST helpers vs ledger-backed template adapter."
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--p-control", type=float, default=0.10, dest="p_control")
    parser.add_argument("--allocation-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument(
        "--effect-sizes",
        type=float,
        nargs="*",
        default=None,
        help="Optional effect size grid. Defaults to a single point at delta.",
    )
    parser.add_argument("--n-sim-direct", type=int, default=1)
    parser.add_argument("--n-sim-template", type=int, default=1)
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=30,
        help="Number of functions to display in the profile summary.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile for the template path (disabled by default).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Root log level for both implementations (default: INFO).",
    )
    parser.add_argument(
        "--enable-tqdm",
        action="store_true",
        help="Allow tqdm progress bars (disabled by default).",
    )
    args = parser.parse_args()

    _configure_logging(level=args.log_level, enable_tqdm=args.enable_tqdm)

    effect_sizes = args.effect_sizes if args.effect_sizes else [float(args.delta)]

    LOGGER.info(
        "Running direct procedure benchmark (n_sim=%s, k=%s)...",
        args.n_sim_direct,
        args.k,
    )
    direct_seconds = _direct_compare_runtime(
        alpha=args.alpha,
        delta=args.delta,
        power=args.power,
        p_control=args.p_control,
        allocation_ratio=args.allocation_ratio,
        effect_sizes=effect_sizes,
        n_sim=args.n_sim_direct,
        batch_size=args.batch_size,
        seed=args.seed,
        k=args.k,
    )

    LOGGER.info(
        "Running template adapter benchmark (n_sim=%s, k=%s, profile=%s)...",
        args.n_sim_template,
        args.k,
        args.profile,
    )
    template_seconds, profile_text = _template_compare_runtime_with_profile(
        alpha=args.alpha,
        delta=args.delta,
        power=args.power,
        p_control=args.p_control,
        allocation_ratio=args.allocation_ratio,
        effect_sizes=effect_sizes,
        n_sim=args.n_sim_template,
        batch_size=args.batch_size,
        seed=args.seed,
        k=args.k,
        profile_limit=args.profile_limit,
        enable_profile=args.profile,
    )

    ratio = template_seconds / direct_seconds if direct_seconds > 0 else float("inf")

    print("Direct compare_interim runtime (s):", round(direct_seconds, 3))
    print("Template adapter runtime (s):", round(template_seconds, 3))
    print("Relative slowdown (template/direct):", round(ratio, 1), "x")
    print("Direct config -> n_sim:", args.n_sim_direct, "effect_sizes:", effect_sizes)
    print(
        "Template config -> n_sim:", args.n_sim_template, "effect_sizes:", effect_sizes
    )

    if profile_text is not None:
        profile_path = Path(__file__).with_name("template_adapter_profile.txt")
        profile_path.write_text(profile_text)
        print(
            f"Wrote template profile summary to {profile_path.relative_to(Path.cwd())}"
        )


if __name__ == "__main__":
    main()
