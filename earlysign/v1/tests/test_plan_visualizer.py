import numpy as np

from earlysign.schema.ES3.GST import (
    AlphaSpendingStrategy,
    Protocol,
    SpendingFunctionType,
)
from earlysign.v1.methods.group_sequential.plan.gs_design_converter import (
    convert_gs_design_to_protocol,
)
from earlysign.v1.methods.group_sequential.plan.operating_characteristics.engines import (
    AsymptoticSimulator,
)
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
    Config,
)


# Test Cases
def test_convert_gs_design_simple():
    """Test converting a simple gsDesign-like dictionary."""
    gs_data = {
        "k": 3,
        "test.type": 1,  # One-sided
        "alpha": 0.025,
        "beta": 0.2,  # Power 0.8
        "n.I": [100, 200, 300],
        "timing": [0.33, 0.67, 1.0],
        "delta": 0.1,
        # Mock upper structure
        "upper": {"sfFunction": "sfLDOF"},
    }

    protocol = convert_gs_design_to_protocol(gs_data, "Test Protocol")

    assert isinstance(protocol, Protocol)
    assert protocol.task.efficacy.alpha == 0.025
    assert protocol.method.stopping_policy.schedule.analyses == [0.33, 0.67, 1.0]

    strategy = protocol.method.stopping_policy.strategy
    assert isinstance(strategy, AlphaSpendingStrategy)
    assert strategy.spending_fn.family == SpendingFunctionType.OBRIEN_FLEMING

    # Check Max Sample Size
    timer = protocol.method.stopping_policy.timer
    assert timer.max_sample_size == 300


def test_convert_gs_design_two_sided_asymmetric():
    """Test converting a test.type=4 (Asymmetric non-binding) design."""
    gs_data = {
        "k": 2,
        "test.type": 4,  # Non-binding futility
        "alpha": 0.05,
        "beta": 0.1,
        "n.I": [50, 100],
        # Actually converter defaults if missing, but let's provide array
        "timing": np.array([0.5, 1.0]),
        "delta": 0.2,
    }

    protocol = convert_gs_design_to_protocol(gs_data, "Test Type 4")

    # Needs to be AlphaBeta
    strategy = protocol.method.stopping_policy.strategy
    assert strategy.kind == "alpha_beta_spending"
    assert strategy.beta_binding is False  # Non-binding


def test_calculate_stopping_probabilities_sum_to_one():
    """Verify that stopping probabilities sum to 1 (accounting for forced stop)."""
    k = 5
    info_times = np.linspace(0.2, 1.0, k)
    # Symmetric Pocock bounds (approx)
    # Just use arbitrary bounds
    upper = np.array([3.0, 3.0, 3.0, 2.5, 2.0])
    lower = np.array([-3.0, -3.0, -3.0, -2.5, -2.0])

    # Under H0 (drift=0)
    sim = AsymptoticSimulator(
        model=CanonicalJointModel(
            Config(info_times=info_times, alpha=0.05, tails=2, rng_seed=42, n_sims=5000)
        ),
        upper_boundaries=upper,
        lower_boundaries=lower,
        seed=42,
        n_sims=5000,
    )
    res_h0 = sim.evaluate_point(drift=0.0)
    probs_h0 = res_h0.prob_stop_total

    assert len(probs_h0) == k
    assert np.isclose(np.sum(probs_h0), 1.0)

    # Under H1 (drift=3.0)
    res_h1 = sim.evaluate_point(drift=3.0)
    probs_h1 = res_h1.prob_stop_total
    assert np.isclose(np.sum(probs_h1), 1.0)


def test_calculate_stopping_prob_values():
    """Check values for a known restrictive case."""
    # If bounds are very tight, we stop early.
    info_times = np.array([0.5, 1.0])
    upper = np.array([0.1, 0.1])  # Very tight, should stop almost immediately
    lower = np.array([-0.1, -0.1])

    sim = AsymptoticSimulator(
        model=CanonicalJointModel(
            Config(info_times=info_times, alpha=0.05, tails=2, rng_seed=42, n_sims=2000)
        ),
        upper_boundaries=upper,
        lower_boundaries=lower,
        seed=42,
        n_sims=2000,
    )
    res = sim.evaluate_point(drift=0.0)
    probs = res.prob_stop_total

    # Most should stop at look 1
    assert probs[0] > 0.8
    assert np.isclose(np.sum(probs), 1.0)
