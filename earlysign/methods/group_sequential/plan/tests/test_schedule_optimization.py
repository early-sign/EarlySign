"""Tests for GSD Schedule Optimization."""

import numpy as np
import pytest

from earlysign.methods.group_sequential.plan.schedule_optimization import (
    optimize_schedule,
)
from earlysign.methods.group_sequential.shared.spending import (
    OBrienFlemingSpending,
    PocockSpending,
)


def test_optimization_beats_uniform() -> None:
    """Verify that optimization finds a schedule better than uniform."""
    k = 3
    alpha = 0.025
    drift = 3.0  # Strong signal
    eff_spend = OBrienFlemingSpending(alpha)
    tails = 1

    # Run optimization
    res = optimize_schedule(
        k_looks=k,
        efficacy_spending=eff_spend,
        alpha=alpha,
        drift=drift,
        tails=tails,
        method="simulation",
        n_sims=5000,
        seed=42,
    )
    opt_t = res.schedule
    opt_asn = res.asn

    # Compare with Uniform
    from earlysign.methods.group_sequential.shared.canonical_joint_model import (
        CanonicalJointModel,
        Config,
    )

    t_uniform = np.linspace(1 / k, 1.0, k)
    model_config = Config(
        info_times=t_uniform,
        alpha=alpha,
        efficacy_spending=eff_spend,
        tails=tails,
        n_sims=5000,
        rng_seed=42,
    )
    model = CanonicalJointModel(model_config)
    a, _ = model.solve_boundaries(drift=0.0, method="simulation")

    # Calculate Uniform ASN
    # Re-use logic from estimator
    # Let's just use the optimizer's estimator class for fair comparison
    from earlysign.methods.group_sequential.plan.schedule_optimization import (
        OptimizationConfig,
        SequentialASNEstimator,
    )

    estimator = SequentialASNEstimator(
        k_looks=k,
        alpha=alpha,
        efficacy_spending=eff_spend,
        futility_spending=None,
        drift=drift,
        tails=tails,
        config=OptimizationConfig(method="simulation", n_sims=5000, rng_seed=42),
    )

    uniform_asn = estimator.calculate(np.full(k, 1 / k))

    print(f"\nUniform ASN: {uniform_asn:.4f}")
    print(f"Optimal ASN: {opt_asn:.4f}")
    print(f"Optimal Power: {res.power:.4f}")
    print(f"Optimal Schedule: {opt_t}")

    assert (
        opt_asn <= uniform_asn + 1e-3
    )  # Allow small noise, but generally should be better
    assert opt_t[-1] == pytest.approx(1.0)
    assert np.all(np.diff(opt_t) > 0)


def test_integration_method_smoke() -> None:
    """Smoke test for numerical integration method."""
    k = 2
    alpha = 0.025
    eff_spend = PocockSpending(alpha)

    res = optimize_schedule(
        k_looks=k,
        efficacy_spending=eff_spend,
        alpha=alpha,
        drift=2.0,
        method="numerical_integration",
        seed=123,
    )

    assert len(res.schedule) == k
    assert res.schedule[-1] == pytest.approx(1.0)
    assert 0.0 <= res.power <= 1.0
