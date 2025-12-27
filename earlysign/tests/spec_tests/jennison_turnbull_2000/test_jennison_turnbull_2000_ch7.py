import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from scipy.stats import norm

from earlysign.methods.group_sequential.canonical_joint_distribution import (
    CanonicalJointDistribution,
)
from earlysign.methods.group_sequential.evaluation.evaluator import (
    OperatingCharacteristicEvaluator,
)
from earlysign.tests.util import corresponding_scenario_path

# Load the feature file
scenarios(corresponding_scenario_path(__file__))


@pytest.fixture
def design_params():
    return {}


@pytest.fixture
def cjd(design_params):
    # Use values from design_params if available, else default
    n_sims = design_params.get("n_sims", 10000)
    seed = design_params.get("rng_seed", 42)
    return CanonicalJointDistribution(n_sims=n_sims, rng_seed=seed)


@pytest.fixture
def evaluator(cjd):
    return OperatingCharacteristicEvaluator(cjd=cjd)


@given(parsers.parse("simulation precision with {n} samples"))
def given_precision(n, design_params):
    design_params["n_sims"] = int(n)


@given(parsers.parse("a target power {power} at effect size {delta}"))
def given_power_delta(power, delta, design_params):
    design_params["power"] = float(power)
    design_params["delta"] = float(delta)


@given(parsers.parse("a two-sided maximum information test with alpha {alpha:f}"))
def given_alpha(alpha, design_params):
    design_params["alpha"] = alpha
    design_params["tails"] = 2


@given(parsers.parse("a target power {power} at some effect size"))
def given_power(power, design_params):
    design_params["power"] = float(power)


@given(
    parsers.re(
        r"a maximum of (?P<k>\d+) looks with rho-family spending (?P<rho>[\d.]+)$"
    )
)
def given_looks_and_rho(k, rho, design_params):
    design_params["k"] = int(k)
    design_params["rho"] = float(rho)


@when("I compute the inflation factor R_LD", target_fixture="results")
def when_compute_r_ld(design_params, evaluator):
    alpha = design_params["alpha"]
    power = design_params["power"]
    k = design_params["k"]
    rho = design_params["rho"]

    from earlysign.methods.group_sequential.spending import RhoFamilySpending

    spending = RhoFamilySpending(alpha=alpha, rho=rho)
    info_times = np.linspace(1 / k, 1.0, k)

    metrics = evaluator.evaluate_design_characteristics(
        info_times=info_times,
        spending=spending,
        target_power=power,
        alpha=alpha,
        tails=2,
    )

    return {"R_LD": metrics["r_ld"]}


@when(
    "I evaluate the expected sample size relative to fixed design",
    target_fixture="results",
)
def when_evaluate_asn(design_params, evaluator):
    alpha = design_params["alpha"]
    power = design_params["power"]
    k = design_params["k"]
    rho = design_params["rho"]

    from earlysign.methods.group_sequential.spending import RhoFamilySpending

    spending = RhoFamilySpending(alpha=alpha, rho=rho)
    info_times = np.linspace(1 / k, 1.0, k)

    metrics = evaluator.evaluate_design_characteristics(
        info_times=info_times,
        spending=spending,
        target_power=power,
        alpha=alpha,
        tails=2,
    )

    r_ld = metrics["r_ld"]
    asn_looks = metrics["asn_looks"]

    # ASN% = 100 * R_LD * (E[look] / K)
    def get_asn_percent(look_val):
        return 100.0 * r_ld * (look_val / k)

    return {
        "R_LD_pct": 100.0 * r_ld,
        "ASN_0": get_asn_percent(asn_looks["0"]),
        "ASN_05delta": get_asn_percent(asn_looks["0.5delta"]),
        "ASN_delta": get_asn_percent(asn_looks["delta"]),
        "ASN_15delta": get_asn_percent(asn_looks["1.5delta"]),
    }


@then(parsers.parse("the result should be {r_ld:f} with {atol:f} precision"))
def then_check_r_ld(results, r_ld, atol):
    # Calibrated tolerance from Gherkin
    assert results["R_LD"] == pytest.approx(float(r_ld), abs=atol)


@then(
    parsers.parse(
        "the maximum information (R_LD) should be {r_ld:f} percent with {atol:f} precision"
    )
)
@then(
    parsers.parse(
        "the maximum information (R_OS) should be {r_ld:f} percent with {atol:f} precision"
    )
)
def then_check_r_ld_pct(results, r_ld, atol):
    # Try R_LD_pct first, then R_OS_pct
    val = results.get("R_LD_pct", results.get("R_OS_pct"))
    assert val == pytest.approx(float(r_ld), abs=atol)


@then(
    parsers.parse(
        "the expected sample size at theta={condition} should be {asn:f} percent with {atol:f} precision"
    )
)
def then_check_asn(results, condition, asn, atol):
    # The condition in feature file is "0", "0.5δ", "δ", "1.5δ"
    # Mapping to results keys
    mapping = {
        "0": "ASN_0",
        "0.5δ": "ASN_05delta",
        "δ": "ASN_delta",
        "1.5δ": "ASN_15delta",
    }
    actual_key = mapping.get(condition)
    # Calibrated tolerance from Gherkin
    assert results[actual_key] == pytest.approx(asn, abs=atol)


# --- Subsection 7.2.2 Steps ---


@given(parsers.parse("a two-sided A/B test with alpha {alpha}"))
def given_ab_alpha(alpha, design_params):
    design_params["alpha"] = float(alpha)
    design_params["tails"] = 2
    design_params["trial_type"] = "binomial-ab"  # Using 2-arm logic


@given(parsers.parse("a target power {power} at effect size {delta}"))
def given_power_delta_ch7(power, delta, design_params):
    design_params["power"] = float(power)
    design_params["delta"] = float(delta)


@given(parsers.parse("a known variance (sigma squared) {sigma2}"))
def given_sigma2_ch7(sigma2, design_params):
    design_params["sigma2"] = float(sigma2)


@given(parsers.parse("a maximum information (I_max) constrained to {i_max}"))
def given_i_max_constrained(i_max, design_params):
    design_params["i_max_constrained"] = float(i_max)


@when(
    parsers.parse(
        "I analyze the Lan-DeMets design from section 7.2.2 with K={k} and rho={rho}"
    ),
    target_fixture="results",
)
def when_analyze_722(k, rho, design_params, evaluator):
    alpha = design_params["alpha"]
    power = design_params["power"]
    delta = design_params["delta"]
    sigma2 = design_params["sigma2"]

    k_val = int(k)
    rho_val = float(rho)

    # Fixed sample information
    eta_fixed = norm.ppf(1 - alpha / 2.0) + norm.ppf(power)
    i_fixed = (eta_fixed / delta) ** 2

    from earlysign.methods.group_sequential.spending import RhoFamilySpending

    spending = RhoFamilySpending(alpha=alpha, rho=rho_val)
    info_times = np.linspace(1 / k_val, 1.0, k_val)

    metrics = evaluator.evaluate_design_characteristics(
        info_times=info_times,
        spending=spending,
        target_power=power,
        alpha=alpha,
        tails=2,
    )

    r_ld = metrics["r_ld"]
    i_max = r_ld * i_fixed

    # n = 2 * sigma2 * I
    n_fixed = 2 * sigma2 * i_fixed
    n_max = 2 * sigma2 * i_max

    return {
        "i_fixed": i_fixed,
        "n_fixed": n_fixed,
        "i_max": i_max,
        "n_max": n_max,
        "r_ld": r_ld,
    }


@when(
    parsers.parse(
        "I evaluate the required effect size for power {power} using K={k} and rho={rho}"
    ),
    target_fixture="results",
)
def when_solve_delta(power, k, rho, design_params, evaluator):
    alpha = design_params["alpha"]
    i_max_const = design_params["i_max_constrained"]

    p_val = float(power)
    k_val = int(k)
    rho_val = float(rho)

    from earlysign.methods.group_sequential.spending import RhoFamilySpending

    spending = RhoFamilySpending(alpha=alpha, rho=rho_val)
    info_times = np.linspace(1 / k_val, 1.0, k_val)

    metrics = evaluator.evaluate_design_characteristics(
        info_times=info_times,
        spending=spending,
        target_power=p_val,
        alpha=alpha,
        tails=2,
    )

    r_ld = metrics["r_ld"]
    i_fixed_required = i_max_const / r_ld

    # delta = eta_fixed / sqrt(I_fixed)
    eta_fixed = norm.ppf(1 - alpha / 2.0) + norm.ppf(p_val)
    delta_required = eta_fixed / np.sqrt(i_fixed_required)

    return {"delta": delta_required, "r_ld": r_ld}


@then(
    parsers.parse(
        "the fixed sample information (I_f) should be {i_f} with {atol:f} precision"
    )
)
def then_check_i_f_ch7(results, i_f, atol):
    assert results["i_fixed"] == pytest.approx(float(i_f), abs=atol)


@then(
    parsers.parse(
        "the fixed sample size per group should be {n} with {atol:f} precision"
    )
)
def then_check_n_fixed_ch7(results, n, atol):
    assert results["n_fixed"] == pytest.approx(float(n), abs=atol)


@then(
    parsers.parse(
        "the maximum information (I_max) should be {i_max} with {atol:f} precision"
    )
)
def then_check_i_max_722(results, i_max, atol):
    assert results["i_max"] == pytest.approx(float(i_max), abs=atol)


@then(
    parsers.parse(
        "the maximum sample size per group should be {n} with {atol:f} precision"
    )
)
def then_check_n_max(results, n, atol):
    # Calibrated tolerance from Gherkin
    assert results["n_max"] == pytest.approx(float(n), abs=atol)


@then(
    parsers.parse(
        "the required effect size (delta) should be {delta} with {atol:f} precision"
    )
)
def then_check_delta_required(results, delta, atol):
    assert results["delta"] == pytest.approx(float(delta), abs=atol)


@then(
    parsers.parse("the inflation factor R_LD should be {r_ld} with {atol:f} precision")
)
def then_check_r_ld_722(results, r_ld, atol):
    assert results["r_ld"] == pytest.approx(float(r_ld), abs=atol)


# --- Under-running and Over-running Steps ---


@given(parsers.parse("a planned maximum information {i_max}"))
def given_planned_i_max(i_max, design_params):
    design_params["planned_i_max"] = float(i_max)


@given(parsers.parse("a rho-family spending function with rho {rho}"))
def given_rho_spending(rho, design_params):
    design_params["rho"] = float(rho)


@when(
    parsers.parse("I perform a trial with actual information sequence {sequence}"),
    target_fixture="results",
)
def when_perform_mismatched_trial(sequence, design_params, evaluator):
    # Parse sequence "1.125, 2.25, ..."
    actual_info = np.array([float(s.strip()) for s in sequence.split(",")])
    planned_i_max = design_params["planned_i_max"]
    alpha = design_params["alpha"]
    rho = design_params["rho"]

    from earlysign.methods.group_sequential.spending import RhoFamilySpending

    spending = RhoFamilySpending(alpha=alpha, rho=rho)

    # Example 7.2.2 evaluates power at delta=1.0
    metrics = evaluator.evaluate_mismatched_design(
        actual_info=actual_info,
        planned_i_max=planned_i_max,
        spending=spending,
        theta=1.0,
        tails=2,
    )
    return metrics


@then(
    parsers.parse(
        "the power at delta {theta} should be {power:f} with {atol:f} precision"
    )
)
def then_check_power_722_robust(results, theta, power, atol):
    assert results["rejection_probability"] == pytest.approx(float(power), abs=atol)


@when(
    parsers.parse(
        "I perform a trial with group size {n} per stage until I_max {i_max_target} is reached"
    ),
    target_fixture="results",
)
def when_perform_overrunning_trial(n, i_max_target, design_params, evaluator):
    n_per_stage = float(n)
    i_max_planned = float(i_max_target)
    sigma2 = design_params["sigma2"]
    alpha = design_params["alpha"]
    rho = design_params["rho"]

    # Generate information until I >= I_max_planned
    info_step = n_per_stage / (2.0 * sigma2)
    actual_info = []
    i_curr = 0.0
    while (
        i_curr < i_max_planned - 1e-9
    ):  # tiny epsilon to avoid floating point issues at terminal
        i_curr += info_step
        actual_info.append(i_curr)
        if len(actual_info) > 100:
            break

    actual_info = np.array(actual_info)

    from earlysign.methods.group_sequential.spending import RhoFamilySpending

    spending = RhoFamilySpending(alpha=alpha, rho=rho)

    metrics = evaluator.evaluate_mismatched_design(
        actual_info=actual_info,
        planned_i_max=i_max_planned,
        spending=spending,
        theta=1.0,
        tails=2,
    )
    metrics["num_looks"] = len(actual_info)
    return metrics


@then(parsers.parse("the trial should stop at look {look}"))
def then_check_num_looks(results, look):
    assert results["num_looks"] == int(look)


@then(
    parsers.parse(
        "the final maximum information should be {i_max:f} with {atol:f} precision"
    )
)
def then_check_final_info(results, i_max, atol):
    assert results["actual_final_info"] == pytest.approx(float(i_max), abs=atol)


# --- Table 7.4 Robustness Steps ---


@when(
    parsers.parse(
        "I evaluate Table 7.4 robustness with K_tilde={k}, rho={rho}, r={r}, and pi={pi}"
    ),
    target_fixture="results",
)
def when_evaluate_table_7_4(k, rho, r, pi, design_params, evaluator):
    alpha = design_params["alpha"]
    target_p = design_params["power"]
    delta_val = design_params["delta"]  # matches textbook theta=1.0 case

    k_tilde = int(k)
    rho_val = float(rho)
    r_val = float(r)
    pi_val = float(pi)

    # 1. Planned design: equidistant info times
    t_planned = np.linspace(1 / k_tilde, 1.0, k_tilde)
    from earlysign.methods.group_sequential.spending import RhoFamilySpending

    spending = RhoFamilySpending(alpha=alpha, rho=rho_val)

    # Use the library to get the inflationary factor for the planned design
    planned_metrics = evaluator.evaluate_design_characteristics(
        info_times=t_planned,
        spending=spending,
        target_power=target_p,
        alpha=alpha,
        tails=2,
    )

    # Planned I_max (assuming I_fixed is for target_p at delta_val)
    eta_fixed = norm.ppf(1 - alpha / 2.0) + norm.ppf(target_p)
    i_fixed = (eta_fixed / delta_val) ** 2
    planned_i_max = planned_metrics["r_ld"] * i_fixed

    # 2. Actual information sequence
    steps = np.arange(1, k_tilde + 1)
    # I_k = pi * (k / k_tilde)^r * planned_I_max
    actual_info = pi_val * ((steps / k_tilde) ** r_val) * planned_i_max

    # 3. Evaluate mismatched performance
    metrics = evaluator.evaluate_mismatched_design(
        actual_info=actual_info,
        planned_i_max=planned_i_max,
        spending=spending,
        theta=delta_val,
        tails=2,
    )
    return metrics


@then(parsers.parse("the resulting power should be {power} with {atol:f} precision"))
@then(parsers.parse("the power at delta 1.0 should be {power} with {atol:f} precision"))
def then_check_resulting_power(results, power, atol):
    # Calibrated tolerance from Gherkin
    assert results["rejection_probability"] == pytest.approx(float(power), abs=atol)


# --- Table 7.5 Robustness Steps ---


@when(
    parsers.parse(
        "I evaluate Table 7.5 robustness with K_tilde={k_tilde}, K={k}, and rho={rho}"
    ),
    target_fixture="results",
)
def when_evaluate_table_7_5(k_tilde, k, rho, design_params, evaluator):
    alpha = design_params["alpha"]
    target_p = design_params["power"]
    delta_val = design_params["delta"]

    k_tilde_val = int(k_tilde)
    k_val = int(k)
    rho_val = float(rho)

    # 1. Planned design: equidistant info times for K_tilde looks
    t_planned = np.linspace(1 / k_tilde_val, 1.0, k_tilde_val)
    from earlysign.methods.group_sequential.spending import RhoFamilySpending

    spending = RhoFamilySpending(alpha=alpha, rho=rho_val)

    planned_metrics = evaluator.evaluate_design_characteristics(
        info_times=t_planned,
        spending=spending,
        target_power=target_p,
        alpha=alpha,
        tails=2,
    )

    eta_fixed = norm.ppf(1 - alpha / 2.0) + norm.ppf(target_p)
    i_fixed = (eta_fixed / delta_val) ** 2
    planned_i_max = planned_metrics["r_ld"] * i_fixed

    # 2. Actual design: equidistant info times for K looks, ending at planned_I_max
    actual_info = np.linspace(planned_i_max / k_val, planned_i_max, k_val)

    # 3. Evaluate mismatched performance
    metrics = evaluator.evaluate_mismatched_design(
        actual_info=actual_info,
        planned_i_max=planned_i_max,
        spending=spending,
        theta=delta_val,
        tails=2,
    )
    return metrics


# --- BHAT Trial Steps (Subsection 7.2.3) ---


@given(parsers.parse("the BHAT trial setup with total duration {months} months"))
def given_bhat_setup(months, design_params):
    design_params["bhat_t_max"] = float(months)


@given(parsers.parse("rho-family spending rho {rho} based on calendar time"))
def given_bhat_rho(rho, design_params):
    design_params["bhat_rho"] = float(rho)


@given(parsers.parse("information estimated as deaths divided by {divisor}"))
def given_bhat_divisor(divisor, design_params):
    design_params["bhat_divisor"] = float(divisor)


@when(
    parsers.parse("I analyze the BHAT trial with calendar months {months}"),
    target_fixture="bhat_calendar",
)
def when_analyze_bhat_months(months, design_params):
    t_list = np.array([float(m.strip()) for m in months.split(",")])
    design_params["bhat_t_fractions"] = t_list / design_params["bhat_t_max"]
    return t_list


@when(parsers.parse("the observed death counts are {deaths}"), target_fixture="results")
def when_bhat_deaths(deaths, design_params, evaluator):
    d_list = np.array([float(d.strip()) for d in deaths.split(",")])
    # Use explicit divisor
    divisor = design_params.get("bhat_divisor", 4.0)
    info_sequence = d_list / divisor

    from earlysign.methods.group_sequential.spending import RhoFamilySpending

    # Use explicit rho
    rho = design_params.get("bhat_rho", 1.0)
    spending = RhoFamilySpending(alpha=design_params["alpha"], rho=rho)

    metrics = evaluator.evaluate_dual_scale_design(
        info_sequence=info_sequence,
        spending_fractions=design_params["bhat_t_fractions"],
        spending=spending,
        tails=2,
    )
    return metrics


@then(
    parsers.parse(
        'the the BHAT boundaries should be "{b_list}" with {atol:f} precision'
    )
)
@then(parsers.parse('the boundaries should be "{b_list}" with {atol:f} precision'))
def then_check_bhat_boundaries(results, b_list, atol):
    expected = np.array([float(b.strip()) for b in b_list.split(",")])
    actual = results["boundaries"]
    # Calibrated tolerance from Gherkin
    assert actual == pytest.approx(expected, abs=atol)


@then(
    parsers.parse("the observed Z-statistics {z_list} should reject H0 at look {look}")
)
def then_check_bhat_rejection(results, z_list, look):
    z_obs = np.array([float(z.strip()) for z in z_list.split(",")])
    boundaries = results["boundaries"]
    look_idx = int(look) - 1

    # Check if any prior look rejected
    rejected_early = np.any(np.abs(z_obs[:look_idx]) > boundaries[:look_idx])
    assert not rejected_early, "Should not have rejected before the specified look"

    # Check rejection at specified look
    assert (
        np.abs(z_obs[look_idx]) > boundaries[look_idx]
    ), f"Should have rejected at look {look}"


@given(
    parsers.parse(
        "a one-sided maximum information test with alpha {alpha:f} and beta {beta:f}"
    ),
    target_fixture="design_params",
)
def given_one_sided_max_info_beta(alpha, beta):
    return {"alpha": alpha, "beta": beta, "tails": 1}


@given(
    parsers.re(
        r"a maximum of (?P<k>\d+) looks with rho-family spending (?P<rho>[\d.]+) for both errors"
    )
)
def given_max_looks_rho_both(k, rho, design_params):
    design_params["k"] = int(k)
    design_params["rho"] = float(rho)


@when("I compute the inflation factor R_OS", target_fixture="results")
def when_compute_ros(design_params, cjd):
    k = design_params["k"]
    alpha = design_params["alpha"]
    beta = design_params["beta"]
    rho = design_params["rho"]

    ros = cjd.solve_ros_inflation_factor(k, alpha, beta, rho)
    return {"R_LD": ros}


@when(
    "I evaluate the one-sided expected sample size relative to fixed design",
    target_fixture="results",
)
def when_evaluate_os_asn(design_params, cjd):
    k = design_params["k"]
    alpha = design_params["alpha"]
    beta = design_params["beta"]
    rho = design_params["rho"]

    metrics = cjd.evaluate_ros_design_characteristics(k, alpha, beta, rho)

    r_os = metrics["r_os"]

    # ASN% = 100 * R_OS * (E[look] / K)
    def get_asn_percent(look_val):
        return 100.0 * r_os * (look_val / k)

    return {
        "R_OS_pct": 100.0 * r_os,
        "ASN_0": get_asn_percent(metrics["asn_0"]),
        "ASN_05delta": get_asn_percent(metrics["asn_05delta"]),
        "ASN_delta": get_asn_percent(metrics["asn_delta"]),
    }
