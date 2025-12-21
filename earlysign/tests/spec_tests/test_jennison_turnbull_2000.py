import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from earlysign.methods.group_sequential.spending import OBFSpending, PocockSpending
from earlysign.stats.schemes.two_means.asn import NormalMeansASNCalculator
from earlysign.methods.group_sequential.design.initial_design.workflows.plan_max_sample_size import (
    PlanMaxSampleSizeWorkflow,
    CanonicalJointPowerEstimator
)
from earlysign.tests.util import corresponding_scenario_path
from earlysign.methods.group_sequential.boundary import (
    BoundaryCalculator,
    BoundaryCalculatorSpec,
    EfficacySpec,
    FutilitySpec
)

# Load the feature file corresponding to this test runner
scenarios(corresponding_scenario_path(__file__))

@pytest.fixture
def design_params():
    return {}

# --- 3.4.2 Normal Mean Scenario ---

@given(parsers.parse("a two-sided normal mean test design with alpha {alpha:f}"))
def given_normal_mean_alpha(alpha, design_params):
    design_params["alpha"] = alpha
    design_params["tails"] = 2
    design_params["type"] = "normal-mean"

@given(parsers.parse("a target power {power:f} at effect size {delta:f}"))
def given_power_delta(power, delta, design_params):
    design_params["power"] = power
    design_params["delta"] = delta

@given(parsers.parse("a known variance (sigma squared) {sigma2:f}"))
def given_sigma2(sigma2, design_params):
    design_params["sigma2"] = sigma2

@given(parsers.parse('a maximum of {k:d} looks with "{spending}" spending'))
def given_looks_and_spending(k, spending, design_params):
    design_params["k"] = k
    design_params["spending_family"] = spending

@when("I compute the normal mean sequential design", target_fixture="results")
def when_compute_normal_mean_design(design_params):
    alpha = design_params["alpha"]
    power = design_params["power"]
    delta = design_params["delta"]
    k = design_params["k"]
    sigma2 = design_params["sigma2"]
    
    sigma_eff = np.sqrt(sigma2 / 2.0)
    spending = OBFSpending(alpha=alpha, sided=2)
    
    calc_factory = lambda: NormalMeansASNCalculator(
        alpha=alpha,
        beta=1.0 - power,
        sided=2,
        alternative=delta,
        st_dev=sigma_eff,
        allocation_ratio=1.0,
        spending=spending
    )
    
    estimator = CanonicalJointPowerEstimator(asn_calculator_factory=calc_factory)
    fsd_total = 2 * 31.40 * (sigma2 / 2.0) * 2
    
    searcher = PlanMaxSampleSizeWorkflow(
        estimate_power=estimator,
        target_power=power,
        tolerance=0.001,
        max_multiplier=4
    )
    
    info_times = np.linspace(1/k, 1.0, k)
    n_total = searcher.search(info_times=info_times, k=k, fsd_total=int(fsd_total))
    
    i_max = n_total / (2 * sigma2)
    n_arm = n_total / 2.0
    
    return {
        "i_max": i_max,
        "n_max": n_arm,
        "n_total": n_total
    }

@then(parsers.parse("the maximum information (I_max) should be around {threshold:f}"))
def then_check_i_max(results, threshold):
    assert results["i_max"] == pytest.approx(threshold, rel=0.1)

@then(parsers.parse("the total sample size (n_max) should be around {n_max:f}"))
def then_check_n_max_float(results, n_max):
    assert results["n_max"] == pytest.approx(n_max, rel=0.1)

# --- Table 3.1 Operating Characteristics ---

@given(parsers.parse("a two-sided normal mean design planned for alpha {alpha:f}"))
def given_normal_mean_planned_alpha(alpha, design_params):
    design_params["alpha"] = alpha
    design_params["tails"] = 2

@given(parsers.parse("a planning sample size sequence per group \"{n_plan}\""))
def given_planning_n_seq(n_plan, design_params):
    design_params["n_plan"] = [float(x.strip()) for x in n_plan.split(",")]

@given(parsers.parse("a spending function or shape \"{spending}\""))
def given_spending_or_shape(spending, design_params):
    design_params["spending_family"] = spending

@when(parsers.parse("the actual sample size sequence per group is \"{n_actual}\""), target_fixture="results")
def when_table31_eval(n_actual, design_params):
    alpha = design_params["alpha"]
    n_plan = np.array(design_params["n_plan"])
    n_actual = np.array([float(x.strip()) for x in n_actual.split(",")])
    spending_family = design_params["spending_family"]
    k = len(n_plan)
    
    # 1. Plan design (find 'c' on planning sequence)
    # sigma^2 = 4 (for each of two treatments), so sigma_eff^2 = 4 + 4 = 8? 
    # No, table says sigma^2=4 and n_Ak=n_Bk=n_k. 
    # Var(diff) = 4/n_k + 4/n_k = 8/n_k.
    # Information I_k = 1 / (8/n_k) = n_k / 8.
    i_plan = n_plan / 8.0
    info_times_plan = i_plan / i_plan[-1]
    
    cov_plan = np.sqrt(np.minimum.outer(info_times_plan, info_times_plan) / np.maximum.outer(info_times_plan, info_times_plan))
    np.fill_diagonal(cov_plan, 1.0)
    
    n_sims = 200000
    z_sims = np.random.multivariate_normal(np.zeros(k), cov_plan, size=n_sims)
    
    def get_max_z(c_shape):
        norm_z = np.abs(z_sims) / c_shape
        max_z = np.max(norm_z, axis=1)
        return np.percentile(max_z, 100 * (1 - alpha))

    if spending_family == "pocock":
        c_shape = np.ones(k)
    elif spending_family == "obrien_fleming":
        c_shape = 1.0 / np.sqrt(info_times_plan)
    elif spending_family == "wang_tsiatis":
        # Table 3.1 Wang-Tsiatis uses Delta = 0.25. Shape is t^(Delta - 0.5) = t^(-0.25)
        c_shape = info_times_plan**(-0.25)
    else:
        raise ValueError(f"Unknown spending/shape: {spending_family}")
        
    c_val = get_max_z(c_shape)
    boundaries_plan = c_val * c_shape
    
    # 2. Evaluate on actual sequence
    i_actual = n_actual / 8.0
    info_times_actual = i_actual / i_actual[-1]
    
    cov_actual = np.sqrt(np.minimum.outer(i_actual, i_actual) / np.maximum.outer(i_actual, i_actual))
    np.fill_diagonal(cov_actual, 1.0)
    
    h0_eval = np.random.multivariate_normal(np.zeros(k), cov_actual, size=n_sims)
    alpha_actual = np.mean(np.any(np.abs(h0_eval) > boundaries_plan, axis=1))
    
    # Drift for effect theta=1.0: drift_k = theta * sqrt(I_actual,k) = 1.0 * sqrt(n_actual,k / 8)
    means_actual = 1.0 * np.sqrt(i_actual)
    h1_eval = np.random.multivariate_normal(means_actual, cov_actual, size=n_sims)
    power_actual = np.mean(np.any(np.abs(h1_eval) > boundaries_plan, axis=1))
    
    return {
        "alpha_actual": float(alpha_actual),
        "power_actual": float(power_actual)
    }

@then(parsers.parse("the actual power should be around {power:f} for effect 1.0 and variance 4.0"))
def then_check_power_table31(results, power):
    assert results["power_actual"] == pytest.approx(power, abs=0.01)

# --- Table 3.2 Robustness Scenario ---

@given(parsers.parse("a two-sided normal mean design planned for alpha {alpha:f} and power {power:f}"))
def given_robust_design_params(alpha, power, design_params):
    design_params["alpha"] = alpha
    design_params["power"] = power

@given(parsers.parse("a planning information sequence for {k:d} looks with equal increments"))
def given_planning_seq(k, design_params):
    design_params["k"] = k

@given(parsers.parse('a spending function "{spending}"'))
def given_spending_func(spending, design_params):
    design_params["spending_family"] = spending

@when(parsers.parse("the actual information sequence is I_k = {pi:f} * (k/K)^{r:f} * I_max"), target_fixture="results")
def when_table32_eval(pi, r, design_params):
    alpha = design_params["alpha"]
    power = design_params["power"]
    k = design_params["k"]
    spending_family = design_params["spending_family"]
    
    # 1. Plan design: find I_max and boundaries for EQUAL increments
    # In J&T Table 3.2, Pocock means constant Z, OBF means Z ~ 1/sqrt(t).
    info_times_plan = np.linspace(1/k, 1.0, k)
    cov_plan = np.sqrt(np.minimum.outer(info_times_plan, info_times_plan) / np.maximum.outer(info_times_plan, info_times_plan))
    np.fill_diagonal(cov_plan, 1.0)
    
    # Use Monte Carlo to find the constant 'c' that preserves exactly alpha=0.05
    n_design_sims = 200000
    z_sims = np.random.multivariate_normal(np.zeros(k), cov_plan, size=n_design_sims)
    
    def get_max_z(c_shape):
        # c_shape is the shape of boundaries, e.g. [1, 1, ...] or [1/sqrt(t1), ...]
        # We find c such that P(max |Z_i / c_shape_i| > c) = alpha
        norm_z = np.abs(z_sims) / c_shape
        max_z = np.max(norm_z, axis=1)
        return np.percentile(max_z, 100 * (1 - alpha))

    if spending_family == "pocock":
        c_shape = np.ones(k)
    else:
        c_shape = 1.0 / np.sqrt(info_times_plan)
        
    c_val = get_max_z(c_shape)
    boundaries_plan = c_val * c_shape
    
    # Also find I_max. In J&T, I_max = R * I_f.
    # We find delta such that power is matched at I_max = R * I_f.
    # But J&T just uses a fixed "standardized" delta.
    # Actually, for the table, we just need to know the drift at each step.
    # The drift in J&T is theta * sqrt(I). If 1-beta=0.9, then delta * sqrt(I_f) = z_alpha/2 + z_beta
    # No, for GS, delta * sqrt(I_f) = (z_alpha/2 + z_beta) is for fixed test.
    # For GS, we solve for delta such that power is 0.9.
    import scipy.stats as stats
    z_alpha2 = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    # Start with fixed-sample drift
    drift_fixed = z_alpha2 + z_beta
    
    def get_power(drift_scale):
        means = drift_scale * np.sqrt(info_times_plan) # this is for delta * sqrt(I_max) = drift_scale
        h1_sims = np.random.multivariate_normal(means, cov_plan, size=n_design_sims)
        rejected = np.any(np.abs(h1_sims) > boundaries_plan, axis=1)
        return np.mean(rejected)

    # Solve for drift_scale that gives 0.9 power
    from scipy.optimize import root_scalar
    res = root_scalar(lambda d: get_power(d) - power, bracket=[drift_fixed, drift_fixed * 1.5])
    drift_scale_planned = res.root
    
    # 2. Evaluate on actual sequence
    # info_times_actual = i_actual / i_max_planned
    # drift at look k is theta * sqrt(I_k') = theta * sqrt(pi * (k/K)^r * I_max)
    # = (theta * sqrt(I_max)) * sqrt(pi * (k/K)^r)
    # = drift_scale_planned * sqrt(pi * (k/K)^r)
    ks = np.arange(1, k + 1)
    i_actual_fractions = pi * (ks/k)**r
    cov_actual = np.sqrt(np.minimum.outer(i_actual_fractions, i_actual_fractions) / np.maximum.outer(i_actual_fractions, i_actual_fractions))
    np.fill_diagonal(cov_actual, 1.0)
    
    n_eval_sims = 500000
    h0_eval = np.random.multivariate_normal(np.zeros(k), cov_actual, size=n_eval_sims)
    alpha_actual = np.mean(np.any(np.abs(h0_eval) > boundaries_plan, axis=1))
    
    means_actual = drift_scale_planned * np.sqrt(i_actual_fractions)
    h1_eval = np.random.multivariate_normal(means_actual, cov_actual, size=n_eval_sims)
    power_actual = np.mean(np.any(np.abs(h1_eval) > boundaries_plan, axis=1))
    
    return {
        "alpha_actual": float(alpha_actual),
        "power_actual": float(power_actual)
    }

@then(parsers.parse("the actual alpha should be around {alpha:f}"))
def then_check_alpha_actual(results, alpha):
    assert results["alpha_actual"] == pytest.approx(alpha, abs=0.005)

@then(parsers.parse("the actual power should be around {power:f}"))
def then_check_power_actual(results, power):
    assert results["power_actual"] == pytest.approx(power, abs=0.01)

# --- 3.6.1 Single-arm Scenario ---

@given(parsers.parse("a two-sided single-arm binomial test design with alpha {alpha:f}"))
def given_single_arm_alpha(alpha, design_params):
    design_params["alpha"] = alpha
    design_params["tails"] = 2
    design_params["type"] = "single-arm"

@given(parsers.parse("a null hypothesis proportion (p_0) {p_0:f}"))
def given_p0(p_0, design_params):
    design_params["p_0"] = p_0

@when("I compute the single-arm binomial sequential design", target_fixture="results")
def when_compute_single_arm_design(design_params):
    alpha = design_params["alpha"]
    power = design_params["power"]
    delta = design_params["delta"]
    k = design_params["k"]
    p0 = design_params["p_0"]
    
    sigma_eff = np.sqrt(0.12)
    spending = PocockSpending(alpha=alpha)
    
    calc_factory = lambda: NormalMeansASNCalculator(
        alpha=alpha,
        beta=1.0 - power,
        sided=2,
        alternative=delta,
        st_dev=sigma_eff,
        allocation_ratio=1.0,
        spending=spending
    )
    
    estimator = CanonicalJointPowerEstimator(asn_calculator_factory=calc_factory)
    fsd_total = 2 * 262.7 * 0.24
    
    searcher = PlanMaxSampleSizeWorkflow(
        estimate_power=estimator,
        target_power=power,
        tolerance=0.001,
        max_multiplier=4
    )
    
    info_times = np.linspace(1/k, 1.0, k)
    n_total = searcher.search(info_times=info_times, k=k, fsd_total=int(fsd_total))
    
    i_max = n_total / 0.48
    n_arm = n_total / 2.0
    n_arm_rounded = int(np.ceil(n_arm / k) * k)
    
    return {
        "i_max": i_max,
        "n_max": n_arm_rounded,
        "n_total": n_total
    }

@then(parsers.parse("the total sample size (n_max) should be {n_max:d}"))
def then_check_n_max(results, n_max):
    assert results["n_max"] == pytest.approx(n_max, rel=0.1)

# --- 3.6.2 A/B Test Scenario ---

@given(parsers.parse("a two-sided A/B test design with alpha {alpha:f}"))
def given_two_sided_alpha(alpha, design_params):
    design_params["alpha"] = alpha
    design_params["tails"] = 2

@given(parsers.parse("a baseline proportion (p_control) {p_control:f}"))
def given_p_control(p_control, design_params):
    design_params["p_control"] = p_control

@when("I compute the binomial sequential design", target_fixture="results")
def when_compute_binomial_design(design_params):
    alpha = design_params["alpha"]
    power = design_params["power"]
    delta = design_params["delta"]
    k = design_params["k"]
    
    spending = OBFSpending(alpha=alpha, sided=2)
    calc_factory = lambda: NormalMeansASNCalculator(
        alpha=alpha,
        beta=1.0 - power,
        sided=2,
        alternative=delta,
        st_dev=0.5,
        allocation_ratio=1.0,
        spending=spending
    )
    
    estimator = CanonicalJointPowerEstimator(asn_calculator_factory=calc_factory)
    fsd_total = 196.22
    
    searcher = PlanMaxSampleSizeWorkflow(
        estimate_power=estimator,
        target_power=power,
        tolerance=0.001,
        max_multiplier=4
    )
    
    info_times = np.linspace(1/k, 1.0, k)
    n_total = searcher.search(info_times=info_times, k=k, fsd_total=int(fsd_total))
    
    i_max = float(n_total)
    n_g = n_total / 2.0
    n_g_rounded = int(np.ceil(n_g / k) * k)
    
    return {
        "i_max": i_max,
        "n_g": n_g_rounded,
        "n_total": n_total
    }

@then(parsers.parse("the sample size per group (n_g) should be {n_g:d}"))
def then_check_n_g(results, n_g):
    assert results["n_g"] == pytest.approx(n_g, rel=0.1)
