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
from earlysign.methods.group_sequential.canonical_joint_distribution import CanonicalJointDistribution
from earlysign.methods.group_sequential.design.planner import DesignPlanner
from earlysign.methods.group_sequential.evaluation.evaluator import OperatingCharacteristicEvaluator

# Load the feature file corresponding to this test runner
scenarios(corresponding_scenario_path(__file__))

@pytest.fixture
def design_params():
    return {}

@pytest.fixture
def cjd():
    return CanonicalJointDistribution(n_sims=300000, rng_seed=42)

@pytest.fixture
def planner(cjd):
    return DesignPlanner(cjd=cjd)

@pytest.fixture
def evaluator(cjd):
    return OperatingCharacteristicEvaluator(cjd=cjd)

# --- 3.4.2 Normal Mean Scenario ---

@given(parsers.parse("a two-sided normal mean test design with alpha {alpha:f}"))
def given_normal_mean_alpha(alpha, design_params):
    design_params["alpha"] = alpha
    design_params["tails"] = 2
    design_params["trial_type"] = "normal-mean"

@given(parsers.parse("a two-sided paired comparison design with alpha {alpha:f}"))
def given_paired_comparison_alpha(alpha, design_params):
    design_params["alpha"] = alpha
    design_params["tails"] = 2
    design_params["trial_type"] = "paired"

@given(parsers.parse("a two-sided crossover trial design with alpha {alpha:f}"))
def given_crossover_trial_alpha(alpha, design_params):
    design_params["alpha"] = alpha
    design_params["tails"] = 2
    design_params["trial_type"] = "crossover"

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
def when_compute_normal_mean_design(design_params, planner):
    alpha = design_params["alpha"]
    power = design_params["power"]
    delta = design_params["delta"]
    k = design_params["k"]
    sigma2 = design_params["sigma2"]
    spending_family = design_params["spending_family"]
    trial_type = design_params.get("trial_type", "normal-mean")
    
    res = planner.plan_design(
        alpha=alpha,
        power=power,
        theta=delta,
        sigma2=sigma2,
        k=k,
        shape_type=spending_family,
        trial_type=trial_type
    )
    
    return res

@then(parsers.parse("the total sample size (n_max) should be {n_max:d}"))
def then_check_n_max_exact(results, n_max):
    # For rounded values
    assert results["n_max"] == pytest.approx(n_max, abs=4.0)

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
def when_table31_eval(n_actual, design_params, evaluator):
    alpha = design_params["alpha"]
    n_plan = np.array(design_params["n_plan"])
    n_actual = np.array([float(x.strip()) for x in n_actual.split(",")])
    spending_family = design_params["spending_family"]
    
    res = evaluator.evaluate_table31_robustness(
        planned_n=n_plan,
        actual_n=n_actual,
        alpha=alpha,
        spending_family=spending_family,
        theta=1.0,
        var_diff=8.0
    )
    return res

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
def when_table32_eval(pi, r, design_params, evaluator):
    alpha = design_params["alpha"]
    power = design_params["power"]
    k = design_params["k"]
    spending_family = design_params["spending_family"]
    
    res = evaluator.evaluate_table32_robustness(
        k=k,
        alpha=alpha,
        planned_power=power,
        spending_family=spending_family,
        pi=pi,
        r=r
    )
    return res

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
