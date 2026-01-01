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


@given(parsers.re(r"a target power (?P<power>[\d.]+) at effect size (?P<delta>[\d.]+)$"))
def given_power_delta(power, delta, design_params):
    design_params["power"] = float(power)
    design_params["delta"] = float(delta)


@given(parsers.parse("a two-sided maximum information test with alpha {alpha:f}"))
def given_alpha(alpha, design_params):
    design_params["alpha"] = alpha
    design_params["tails"] = 2


@given(
    parsers.re(
        r"(?i)(?:a target power |the design is planned to attain power )(?P<power>[\d.]+) at (?:theta\s*=\s*(?:±)?δ|some effect size|effect size delta|mu_A - mu_B = ±(?P<delta>[\d.]+))"
    )
)
def given_power_ch7(power, delta, design_params):
    design_params["power"] = float(power)
    if delta:
        design_params["delta"] = float(delta)
    elif "delta" not in design_params:
        design_params["delta"] = 1.0


@given(parsers.re(r"(?i)delta is (?P<delta>[\d.]+)"))
def given_delta_value_ch7(delta, design_params):
    design_params["delta"] = float(delta)


@given(parsers.re(r"(?i)the sample size is designed to attain this power at theta = ±δ"))
def given_sample_size_at_delta_ch7(design_params):
    # This is primarily informational in the Gherkin to match textbook phrasing
    pass


@given(parsers.re(r"(?i)a planned maximum information (?P<i_max>[\d.]+)"))
def given_planned_i_max_ch7(i_max, design_params):
    design_params["planned_i_max"] = float(i_max)


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
        "the maximum sample size relative to fixed design (R_LD) should be {r_ld:f} percent with {atol:f} precision"
    )
)
@then(
    parsers.parse(
        "the maximum sample size relative to fixed design (R_OS) should be {r_ld:f} percent with {atol:f} precision"
    )
)
def then_check_r_ld_pct(results, r_ld, atol):
    # Try R_LD_pct first, then R_OS_pct
    val = results.get("R_LD_pct", results.get("R_OS_pct"))
    assert val == pytest.approx(float(r_ld), abs=float(atol))


@then(
    parsers.re(
        r"the expected sample size at theta\s*=\s*(?P<condition>[^ ]+) should be (?P<asn>[\d.]+) percent with (?P<atol>[\d.]+) precision"
    )
)
def then_check_asn(results, condition, asn, atol):
    # Standardize condition string: remove spaces and normalize characters
    cleaned_cond = condition.replace(" ", "")

    # Mapping to results keys
    mapping = {
        "0": "ASN_0",
        "0.5δ": "ASN_05delta",
        "±0.5δ": "ASN_05delta",
        "δ": "ASN_delta",
        "±δ": "ASN_delta",
        "1.5δ": "ASN_15delta",
        "±1.5δ": "ASN_15delta",
    }
    actual_key = mapping.get(cleaned_cond)

    if actual_key is None:
        # Fallback for Table 7.9 or others if they use different notation
        # e.g. "0.5δ" might be passed if Gherkin has "theta = 0.5δ"
        # Let's also support the raw condition if it matches a key in results
        actual_key = cleaned_cond

    # Calibrated tolerance from Gherkin
    assert results[actual_key] == pytest.approx(float(asn), abs=float(atol))


# --- Subsection 7.2.2 Steps ---


@given(
    parsers.re(
        r"(?i)a two-sided (?:maximum information test|normal mean comparison|A/B test) with alpha (?P<alpha>[\d.]+)"
    ),
    target_fixture="design_params",
)
def given_two_sided_test_ch7(alpha, design_params):
    design_params["alpha"] = float(alpha)
    design_params["tails"] = 2
    return design_params


@given(
    parsers.re(
        r"(?i)a target power (?P<power>[\d.]+) at effect size mu_A - mu_B = ±(?P<delta>[\d.]+)"
    )
)
def given_power_delta_ch7(power, delta, design_params):
    design_params["power"] = float(power)
    design_params["delta"] = float(delta)


@given(
    parsers.re(
        r"(?i)(?:the responses|a) (?:have )?known variance \(sigma squared\) (?P<sigma2>[\d.]+)"
    )
)
def given_sigma2_ch7(sigma2, design_params):
    design_params["sigma2"] = float(sigma2)


@given(parsers.re(r"(?i)(?:we use a|a) Lan-DeMets rho-family spending function with rho (?P<rho>[\d.]+)"))
@given(parsers.re(r"(?i)(?:we use a|a) rho-family spending function with rho (?P<rho>[\d.]+)"))
def given_rho_spending_ch7(rho, design_params):
    design_params["rho"] = float(rho)


@given(parsers.re(r"(?i)(?:we plan for a|a) maximum of (?:K = )?(?P<k>\d+) analyses"))
def given_max_analyses_ch7(k, design_params):
    design_params["k"] = int(k)


@given(parsers.re(r"(?i)a total sample size budget of (?P<total>\d+) observations \((?P<per_arm>\d+) per arm\)"))
def given_sample_budget_ch7(total, per_arm, design_params):
    design_params["i_max_constrained"] = float(per_arm) / (2 * design_params["sigma2"])


@given(parsers.parse("a maximum information (I_max) constrained to {i_max}"))
def given_i_max_constrained(i_max, design_params):
    design_params["i_max_constrained"] = float(i_max)


@when("I compute the group sequential design parameters", target_fixture="results")
@when(
    parsers.parse(
        "I analyze the Lan-DeMets design from section 7.2.2 with K={k} and rho={rho}"
    ),
)
def when_analyze_722(design_params, evaluator, k=None, rho=None):
    alpha = design_params["alpha"]
    power = design_params["power"]
    delta = design_params["delta"]
    sigma2 = design_params["sigma2"]

    k_val = int(k) if k is not None else design_params["k"]
    rho_val = float(rho) if rho is not None else design_params["rho"]

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
    parsers.re(r"I evaluate the required effect size for power (?P<power>[\d.]+)"),
    target_fixture="results",
)
@when(
    parsers.parse(
        "I evaluate the required effect size for power {power} using K={k} and rho={rho}"
    ),
)
def when_solve_delta(power, design_params, evaluator, k=None, rho=None):
    alpha = design_params["alpha"]
    i_max_const = design_params["i_max_constrained"]

    p_val = float(power)
    k_val = int(k) if k is not None else design_params["k"]
    rho_val = float(rho) if rho is not None else design_params["rho"]

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

    return {
        "delta": delta_required,
        "r_ld": r_ld,
        "i_max": i_max_const,
        "i_fixed": i_fixed_required,
    }


@then(
    parsers.re(
        r"the fixed sample information \(I_f\) should be (?P<i_f>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_i_f_ch7(results, i_f, atol):
    assert results["i_fixed"] == pytest.approx(float(i_f), abs=float(atol))


@then(
    parsers.re(
        r"the fixed sample size per group (\(n_f\) )?should be (?P<n>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_n_fixed_ch7(results, n, atol):
    assert results["n_fixed"] == pytest.approx(float(n), abs=float(atol))


@then(
    parsers.re(
        r"the (?:resulting |final )?maximum information \(I_max\) should be (?P<i_max>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_i_max_ch7(results, i_max, atol):
    assert results["i_max"] == pytest.approx(float(i_max), abs=float(atol))


@then(
    parsers.re(
        r"the maximum sample size per group (\(n_max\) )?should be (?P<n>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_n_max(results, n, atol):
    # Calibrated tolerance from Gherkin
    assert results["n_max"] == pytest.approx(float(n), abs=float(atol))


@then(
    parsers.re(
        r"the required effect size \(delta\) should be (?P<delta>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_delta_required(results, delta, atol):
    assert results["delta"] == pytest.approx(float(delta), abs=float(atol))


@then(
    parsers.re(
        r"the inflation factor R_LD should be (?P<r_ld>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_r_ld_722(results, r_ld, atol):
    assert results["r_ld"] == pytest.approx(float(r_ld), abs=float(atol))


# --- Under-running and Over-running Steps ---


@given(parsers.parse("a planned maximum information {i_max}"))
def given_planned_i_max(i_max, design_params):
    design_params["planned_i_max"] = float(i_max)


@given(parsers.parse("a rho-family spending function with rho {rho}"))
def given_rho_spending(rho, design_params):
    design_params["rho"] = float(rho)


@when(
    parsers.re(
        r"the trial ends at cumulative information (?P<cum_info>[\d.]+) after (?P<looks>\d+) looks"
    ),
    target_fixture="results",
)
@when(
    parsers.parse("I perform a trial with actual information sequence {sequence}"),
)
def when_perform_mismatched_trial(design_params, evaluator, sequence=None, cum_info=None, looks=None):
    if sequence:
        # Parse sequence "1.125, 2.25, ..."
        actual_info = np.array([float(s.strip()) for s in sequence.split(",")])
    else:
        # Generate sequence ending at cum_info with 'looks' looks
        # Textbook context for 7.2.2 under-running: first 9 looks are 1.125 each (total 10.125),
        # look 10 ends at 10.6.
        k_val = int(looks)
        info_per_look = 1.125 # assumed for 7.2.2
        actual_info = []
        for i in range(k_val - 1):
            actual_info.append((i+1) * info_per_look)
        actual_info.append(float(cum_info))
        actual_info = np.array(actual_info)

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
    parsers.re(
        r"the power at (?:theta = ±1|mu_A - mu_B = ±1|theta = δ|delta [\d.]+) should be (?P<power>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
@then(
    parsers.re(
        r"the (?:resulting |attained )?power at (?:the same |active )?effect size delta should be (?P<power>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_power_722_robust(results, power, atol):
    assert results["rejection_probability"] == pytest.approx(float(power), abs=float(atol))


@when(
    parsers.re(
        r"I perform a trial with group size (?P<n>[\d.]+) per stage until I_max (?P<i_max_target>[\d.]+) is reached"
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


@given(
    parsers.re(
        r"(?i)(?:the design is )?planned for K_tilde (?P<k_tilde>\d+) equidistant analyses?(?: reaching I_max)?"
    )
)
def when_planned_k_tilde(k_tilde, design_params):
    design_params["k"] = int(k_tilde)


@given(
    parsers.re(
        r"(?i)(?:the design |the test |we )?uses? a (?:Lan-DeMets )?rho-family spending function with rho (?P<rho>[\d.]+)"
    )
)
def given_rho_ch7(rho, design_params):
    design_params["rho"] = float(rho)


@when(
    parsers.re(
        r"(?i)the actual information accrual follows schedule r (?P<r>[\d.]+) and pi (?P<pi>[\d.]+)"
    ),
    target_fixture="results",
)
def when_actual_schedule_re(r, pi, design_params, evaluator):
    return when_evaluate_table_7_4(
        design_params["k"], design_params["rho"], float(r), float(pi), 
        design_params, evaluator
    )


@when(
    parsers.re(
        r"(?i)actually K (?P<k>\d+) equidistant analyses occur reaching I_max"
    ),
    target_fixture="results",
)
def when_actual_k_re(k, design_params, evaluator):
    return when_evaluate_table_7_5(
        design_params["k"], int(k), design_params["rho"],
        design_params, evaluator
    )


@when(
    parsers.re(
        r"(?i)the information levels follow schedule r (?P<r>[\d.]+) and pi (?P<pi>[\d.]+) with (?P<k>\d+) analyses and rho (?P<rho>[\d.]+)"
    ),
    target_fixture="results",
)
def when_evaluate_schedule_robustness(k, rho, r, pi, design_params, evaluator):
    # Backward compatibility for existing feature file runs
    return when_evaluate_table_7_4(k, rho, r, pi, design_params, evaluator)


@when(
    parsers.re(
        r"I evaluate Table 7.4 robustness with K_tilde=(?P<k>\d+), rho=(?P<rho>[\d.]+), r=(?P<r>[\d.]+), and pi=(?P<pi>[\d.]+)"
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


@then(
    parsers.re(
        r"the power at delta 1.0 should be (?P<power>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_resulting_power_alt(results, power, atol):
    assert results["rejection_probability"] == pytest.approx(float(power), abs=float(atol))


# --- Table 7.5 Robustness Steps ---


@when(
    parsers.re(
        r"(?i)the design is planned for K_tilde (?P<k_tilde>\d+) but actually has (?P<k>\d+) looks with rho (?P<rho>[\d.]+)"
    ),
    target_fixture="results",
)
def when_evaluate_k_robustness(k_tilde, k, rho, design_params, evaluator):
    # Backward compatibility
    return when_evaluate_table_7_5(k_tilde, k, rho, design_params, evaluator)


@when(
    parsers.re(
        r"I evaluate Table 7.5 robustness with K_tilde=(?P<k_tilde>\d+), K=(?P<k>\d+), and rho=(?P<rho>[\d.]+)"
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


@given(
    parsers.re(
        r"(?i)(?:use the error spending function f\(t\) = alpha min\(t, 1\)|rho-family spending rho (?P<rho>[\d.]+)) based on calendar time"
    )
)
def given_bhat_rho(rho, design_params):
    # If no rho provided (alpha min(t, 1) case), it is effectively rho=1
    design_params["bhat_rho"] = float(rho) if rho else 1.0


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
    parsers.re(
        r"the (?:observed Z-statistics|sequence of standardized log-rank statistics at the interim analyses) (?P<z_list>[\d.,\s-]+) should reject H0 at look (?P<look>\d+)"
    )
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
    parsers.re(
        r"a one-sided maximum information test with alpha (?P<alpha>[\d.]+) and power (?P<power>[\d.]+)(?: at theta = δ)?"
    ),
    target_fixture="design_params",
)
def given_one_sided_max_info_power(alpha, power):
    alpha_val = float(alpha)
    power_val = float(power)
    return {"alpha": alpha_val, "power": power_val, "beta": 1.0 - power_val, "tails": 1}


@given(
    parsers.re(
        r"a maximum of (?P<k>\d+) (?:equally-spaced )?looks with rho-family spending (?P<rho>[\d.]+) for both (?:type-I and type-II )?errors"
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
