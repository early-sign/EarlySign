"""
Integration-style checks for the binomial GST design helper.

These tests exercise :meth:`BinomialGSTDesignInterface.design.compare_interim`
to ensure it returns sensible OC curves and that the lightweight plotting
utilities in ``earlysign.methods.group_sequential.report`` consume the results.
"""

import matplotlib

# Force a non-interactive backend so pyplot calls work in CI.
matplotlib.use("Agg")

import numpy as np
import pytest

from earlysign.methods.group_sequential.design.initial_design.scenarios.binomial import (
    create_binomial_design,
)
from earlysign.methods.group_sequential.report.plot_oc_curve import OCCurvePlotter


def _run_compare_interim(
    *,
    alpha: float,
    delta: float,
    power: float,
    p_control: float,
    effect_sizes: list[float],
    k: int,
    n_sim: int = 40,
    seed: int = 123,
):
    interface = create_binomial_design(
        alpha=alpha,
        delta=delta,
        power=power,
        p_control=p_control,
        effect_sizes=list(effect_sizes),
        n_sim=n_sim,
        seed=seed,
    )
    return interface.compare_interim(k=k)



@pytest.mark.basic
def test_gst_operating_characteristics() -> None:
    """Compare-interim returns monotonic power and plots without error."""

    alpha = 0.05
    power = 0.8
    p0 = 0.2
    delta = 0.1
    effect_sizes = np.linspace(0.02, 0.12, 6).tolist()

    comparison = _run_compare_interim(
        alpha=alpha,
        delta=delta,
        power=power,
        p_control=p0,
        effect_sizes=effect_sizes,
        k=2,
        n_sim=50,
        seed=321,
    )

    oc_results = comparison["oc_results"]
    assert len(oc_results) == len(effect_sizes)

    powers = [point.power for point in oc_results]
    # Allow a small tolerance because the simulation is stochastic.
    assert all(
        powers[i] <= powers[i + 1] + 0.05 for i in range(len(powers) - 1)
    ), powers

    ess = [point.expected_sample_size for point in oc_results]
    assert ess[-1] <= comparison["planned_max_n"]
    assert (
        comparison["planned_max_n"] >= comparison["procedure_metadata"]["sample_size"]
    )
    assert comparison["plot_error"] is None

    plotter = OCCurvePlotter(figsize=(6, 4))
    ax = plotter.plot_oc_curve(
        oc_results,
        target_effect=delta,
        null_value=p0,
    )
    # With a non-zero null reference the plot should relabel the axis.
    assert ax.get_xlabel() == "Treated proportion"


@pytest.mark.basic
def test_gst_operating_characteristics_realistic_ctr() -> None:
    """Low-CTR scenarios still emit OC curves and usable metadata."""

    p0 = 0.005  # 0.5% baseline CTR
    delta = 0.002  # aim for +0.2pp
    alpha = 0.05
    power = 0.8
    effect_sizes = [0.0002, 0.0005, 0.001, delta]

    comparison = _run_compare_interim(
        alpha=alpha,
        delta=delta,
        power=power,
        p_control=p0,
        effect_sizes=effect_sizes,
        k=3,
        n_sim=40,
        seed=99,
    )

    oc_results = comparison["oc_results"]
    assert len(oc_results) == len(effect_sizes)
    assert comparison["planned_max_n"] > 0

    # At least one effect size should contain non-empty stop-distribution info.
    assert any(point.stop_distribution for point in oc_results)

    plotter = OCCurvePlotter(figsize=(5, 4))
    ax = plotter.plot_oc_curve(
        oc_results,
        target_effect=delta,
        null_value=p0,
        plot_options={"y_bottom": comparison["planned_max_n"] * 0.5},
    )
    bottom, top = ax.get_ylim()
    assert bottom <= comparison["planned_max_n"] <= top
