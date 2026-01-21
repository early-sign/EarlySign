from typing import Any, Dict

import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from scipy import stats
from scipy.stats import norm

from earlysign.v1.methods.group_sequential.plan.simulator import (
    OperatingCharacteristicSimulator,
)
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
    Config,
)
from earlysign.v1.methods.group_sequential.shared.spending import PowerFamilySpending
from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess
from earlysign.v1.tests.util import corresponding_scenario_path

# Load the normalized feature file
scenarios(str(corresponding_scenario_path(__file__)))


@pytest.fixture
def ch7_params() -> Dict[str, Any]:
    # Ultra-high precision for benchmark matching
    return {"n_sims": 20000, "rng_seed": 42, "tails": 2, "alpha": 0.05, "power": 0.9}


# --- GIVEN: Setup ---


@given(parsers.re(r"(?i)simulation precision with (?P<n>\d+) samples"))
def given_precision(n: str, ch7_params: Dict[str, Any]) -> None:
    ch7_params["n_sims"] = int(n)


@given(parsers.re(r"(?i)a (?P<design_type>.*) design with alpha (?P<alpha>[\d.]+)"))
def given_design_alpha(
    design_type: str, alpha: str, ch7_params: Dict[str, Any]
) -> None:
    ch7_params.update({"alpha": float(alpha)})
    if "one-sided" in design_type.lower():
        ch7_params["tails"] = 1
    elif "two-sided" in design_type.lower():
        ch7_params["tails"] = 2


@given(
    parsers.re(
        r"(?i)(?:a )?target power (?P<power>[\d.]+) at (?:effect|theta =) (?P<desc>.*)"
    )
)
def given_power_effect(power: str, ch7_params: Dict[str, Any], desc: str = "") -> None:
    ch7_params["power"] = float(power)


@given(
    parsers.re(
        r"(?i)(?:a maximum of )?(?P<K>\d+) (?:equally[- ]spaced )?looks with rho-family spending (?P<rho>[\d.]+)\s*$"
    )
)
def given_looks_rho(K: str, rho: str, ch7_params: Dict[str, Any]) -> None:
    ch7_params.update({"k": int(K), "rho": float(rho)})


@given(
    parsers.re(
        r"(?i)we use a Lan-DeMets rho-family spending function with rho (?P<rho>[\d.]+)"
    )
)
@given(
    parsers.re(
        r"(?i)the design uses a rho-family spending function with rho (?P<rho>[\d.]+)"
    )
)
@given(
    parsers.re(
        r"(?i)rho-family spending (?:rho )?(?P<rho>[\d.]+) based on calendar time"
    )
)
@given(
    parsers.re(
        r"(?i)(?:a maximum of )?(?P<K>\d+) (?:equally[- ]spaced )?looks with rho-family spending (?P<rho>[\d.]+) for both type-I and type-II errors"
    )
)
def given_rho_config(rho: str, ch7_params: Dict[str, Any], K: str = None) -> None:
    ch7_params.update({"rho": float(rho)})
    if K is not None:
        ch7_params["k"] = int(K)
        ch7_params["symmetric_futility"] = True


@given(parsers.re(r"(?i)we plan for a maximum of K = (?P<K>\d+) analyses"))
def given_plan_k(ch7_params: Dict[str, Any], K: str) -> None:
    ch7_params["k"] = int(K)


@given(
    parsers.re(
        r"(?i)the design is planned for K_tilde (?P<K_tilde>\d+) (?:equidistant )?analyses"
    )
)
def given_plan_k_tilde(ch7_params: Dict[str, Any], K_tilde: str) -> None:
    ch7_params["k_planned"] = int(K_tilde)


@given(
    parsers.re(
        r"(?i)a two-sided (?:normal mean comparison|maximum information (?:design|test)) with alpha (?P<alpha>[\d.]+)"
    )
)
def given_two_sided_normal(alpha: str, ch7_params: Dict[str, Any]) -> None:
    ch7_params.update({"alpha": float(alpha), "tails": 2})


@given(
    parsers.re(
        r"(?i)a one-sided (?:maximum information (?:design|test)) with alpha (?P<alpha>[\d.]+)"
    )
)
def given_one_sided_normal(alpha: str, ch7_params: Dict[str, Any]) -> None:
    ch7_params.update({"alpha": float(alpha), "tails": 1})


@given(
    parsers.re(
        r"(?i)the responses have known variance \(sigma squared\) (?P<sigma2>[\d.]+)"
    )
)
def given_variance(sigma2: str, ch7_params: Dict[str, Any]) -> None:
    ch7_params["sigma2"] = float(sigma2)


@given(
    parsers.re(
        r"(?i)a total sample size budget of (?P<budget>\d+) observations \((?P<per_arm>\d+) per arm\)"
    )
)
def given_budget(budget: str, per_arm: str, ch7_params: Dict[str, Any]) -> None:
    ch7_params["n_per_arm_budget"] = int(per_arm)


@given(parsers.re(r"(?i)a planned maximum information (?P<i_max>[\d.]+)"))
def given_planned_imax(i_max: str, ch7_params: Dict[str, Any]) -> None:
    ch7_params["planned_imax"] = float(i_max)


@given(parsers.re(r"(?i)the BHAT trial setup with total duration 48 months"))
def given_bhat(ch7_params: Dict[str, Any]) -> None:
    ch7_params.update({"bhat": True, "tails": 2, "alpha": 0.05, "rho": 1.0, "k": 7})


@given(parsers.re(r"(?i)information estimated as deaths divided by (?P<denom>[\d.]+)"))
def given_bhat_info_denom(denom: str, ch7_params: Dict[str, Any]) -> None:
    ch7_params["bhat_info_denom"] = float(denom)


# --- WHEN: Compute ---


@when(parsers.re(r"(?i)I compute R_LD"), target_fixture="results")
@when(parsers.re(r"(?i)I compute the inflation factor R_LD"), target_fixture="results")
@when(
    parsers.re(r"(?i)I compute the group sequential design parameters"),
    target_fixture="results",
)
def when_compute_rld(ch7_params: Dict[str, Any]) -> Dict[str, Any]:
    alpha, power, k, rho, tails = (
        ch7_params["alpha"],
        ch7_params.get("power", 0.9),
        ch7_params.get("k", 5),
        ch7_params.get("rho", 2.0),
        ch7_params.get("tails", 2),
    )
    t = np.linspace(1 / k, 1.0, k)
    eff_sf = PowerFamilySpending(alpha, rho)
    model = CanonicalJointModel(
        Config(
            t,
            alpha=alpha,
            efficacy_spending=eff_sf,
            tails=tails,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    i_fixed = (norm.ppf(1 - alpha / tails) + norm.ppf(power)) ** 2
    a, _ = model.solve_boundaries(drift=0.0, method="simulation")
    drift = model.solve_drift(
        t.tolist(),
        a.tolist() if a is not None else [],
        target_power=power,
        tails=tails,
        method="simulation",
    )
    i_max = drift**2
    res = {
        "R_LD": i_max / i_fixed,
        "i_max": i_max,
        "i_fixed": i_fixed,
        "alpha": alpha,
        "tails": tails,
        "rho": rho,
        "k": k,
        "planned_drift": drift,
    }
    if "sigma2" in ch7_params:
        sigma2 = ch7_params["sigma2"]
        res["n_f"] = i_fixed * 2 * sigma2
        res["n_max"] = i_max * 2 * sigma2
    return res


@when(
    parsers.re(r"(?i)I evaluate ASN at theta = (?P<thetas>.*) \* delta"),
    target_fixture="results",
)
def when_eval_asn(thetas: str, ch7_params: Dict[str, Any]) -> Dict[str, Any]:
    res = when_compute_rld(ch7_params)
    parsed_thetas = [float(x.strip().strip('"')) for x in thetas.strip("[]").split(",")]
    asns = []
    t = np.linspace(1 / res["k"], 1.0, res["k"])
    model = CanonicalJointModel(
        Config(
            t,
            alpha=res["alpha"],
            efficacy_spending=PowerFamilySpending(res["alpha"], res["rho"]),
            tails=res["tails"],
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a, _ = model.solve_boundaries(drift=0.0, method="simulation")
    for m in parsed_thetas:
        drift = m * res["planned_drift"]
        asn_look = model.evaluate_asn(
            t.tolist(),
            a.tolist() if a is not None else [],
            drift=drift,
            tails=res["tails"],
        )
        expected_info = asn_look * (res["i_max"] / res["k"])
        asns.append(expected_info / res["i_fixed"] * 100)
    return {"relative_asn": asns}


@when(
    parsers.re(r"(?i)I evaluate the required effect size for power (?P<p>[\d.]+)"),
    target_fixture="results",
)
def when_eval_delta(ch7_params: Dict[str, Any], p: str) -> Dict[str, Any]:
    ch7_params["power"] = float(p)
    res = when_compute_rld(ch7_params)
    i_max = ch7_params["n_per_arm_budget"] / (2 * ch7_params["sigma2"])
    i_fixed = i_max / res["R_LD"]
    delta = (norm.ppf(1 - res["alpha"] / 2) + norm.ppf(float(p))) / np.sqrt(i_fixed)
    return {
        "delta": delta,
        "i_fixed": i_fixed,
        "R_LD": res["R_LD"],
        "power": float(p),
        "i_max": i_max,
    }


@when(
    parsers.re(
        r"(?i)the trial ends at cumulative information (?P<i_curr>[\d.]+) after (?P<k_looks>\d+) looks"
    ),
    target_fixture="results",
)
def when_under_run(
    i_curr: str, k_looks: str, ch7_params: Dict[str, Any]
) -> Dict[str, Any]:
    alpha, power, rho, k_plan, tails = (
        ch7_params["alpha"],
        ch7_params["power"],
        ch7_params["rho"],
        ch7_params.get("k", 5),
        ch7_params.get("tails", 2),
    )
    t_plan = np.linspace(1 / k_plan, 1.0, k_plan)
    model_plan = CanonicalJointModel(
        Config(
            t_plan,
            alpha=alpha,
            efficacy_spending=PowerFamilySpending(alpha, rho),
            tails=tails,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a_plan, _ = model_plan.solve_boundaries(drift=0.0, method="simulation")
    planned_drift = model_plan.solve_drift(
        t_plan.tolist(),
        a_plan.tolist() if a_plan is not None else [],
        target_power=power,
        tails=tails,
    )

    rel_total_info = float(i_curr) / ch7_params["planned_imax"]
    t_actual = np.linspace(rel_total_info / int(k_looks), rel_total_info, int(k_looks))
    model_act = CanonicalJointModel(
        Config(
            t_actual,
            alpha=alpha,
            efficacy_spending=PowerFamilySpending(alpha, rho),
            tails=tails,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a, _ = model_act.solve_boundaries(drift=0.0, method="simulation")
    prob = model_act.compute_rejection_probability(
        t_actual.tolist(),
        a.tolist() if a is not None else [],
        drift=planned_drift,
        tails=tails,
    )
    return {"power": prob}


@when(
    parsers.re(
        r"(?i)I perform a trial with group size (?P<m>\d+) per stage until I_max (?P<planned_imax>[\d.]+) is reached"
    ),
    target_fixture="results",
)
def when_over_run(
    m: str, planned_imax: str, ch7_params: Dict[str, Any]
) -> Dict[str, Any]:
    alpha, power, rho, k_plan, tails = (
        ch7_params["alpha"],
        ch7_params["power"],
        ch7_params["rho"],
        ch7_params.get("k", 5),
        ch7_params.get("tails", 2),
    )
    t_plan = np.linspace(1 / k_plan, 1.0, k_plan)
    model_plan = CanonicalJointModel(
        Config(
            t_plan,
            alpha=alpha,
            efficacy_spending=PowerFamilySpending(alpha, rho),
            tails=tails,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a_plan, _ = model_plan.solve_boundaries(drift=0.0, method="simulation")
    planned_drift = model_plan.solve_drift(
        t_plan.tolist(),
        a_plan.tolist() if a_plan is not None else [],
        target_power=power,
        tails=tails,
    )

    info_step = int(m) / (2 * ch7_params["sigma2"])
    looks = int(np.ceil(float(planned_imax) / info_step))
    i_final = looks * info_step
    rel_final_info = i_final / float(planned_imax)
    t_actual = np.linspace(rel_final_info / looks, rel_final_info, looks)
    model_act = CanonicalJointModel(
        Config(
            t_actual,
            alpha=alpha,
            efficacy_spending=PowerFamilySpending(alpha, rho),
            tails=tails,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a, _ = model_act.solve_boundaries(drift=0.0, method="simulation")
    prob = model_act.compute_rejection_probability(
        t_actual.tolist(),
        a.tolist() if a is not None else [],
        drift=planned_drift,
        tails=tails,
    )
    return {"looks": looks, "i_max": i_final, "power": prob}


@when(
    parsers.re(
        r"(?i)the actual information accrual follows schedule r (?P<r>[\d.]+) and pi (?P<pi>[\d.]+)"
    ),
    target_fixture="results",
)
def when_mismatched_schedule(
    r: str, pi: str, ch7_params: Dict[str, Any]
) -> Dict[str, Any]:
    alpha, power, rho, k_plan, tails = (
        ch7_params["alpha"],
        ch7_params["power"],
        ch7_params["rho"],
        ch7_params.get("k_planned", 5),
        ch7_params.get("tails", 1),
    )
    t_plan = np.linspace(1 / k_plan, 1.0, k_plan)
    model_plan = CanonicalJointModel(
        Config(
            t_plan,
            alpha=alpha,
            efficacy_spending=PowerFamilySpending(alpha, rho),
            tails=tails,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a_plan, _ = model_plan.solve_boundaries(drift=0.0, method="simulation")
    planned_drift = model_plan.solve_drift(
        t_plan.tolist(),
        a_plan.tolist() if a_plan is not None else [],
        target_power=power,
        tails=tails,
    )

    # pi is info factor relative to I_max,planned
    pi_val = float(pi)
    actual_drift = planned_drift * np.sqrt(pi_val)
    t_spend = (np.arange(1, k_plan + 1) / k_plan) ** float(r)

    model_act = CanonicalJointModel(
        Config(
            t_spend,
            alpha=alpha,
            efficacy_spending=PowerFamilySpending(alpha, rho),
            tails=tails,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a, _ = model_act.solve_boundaries(drift=0.0, method="simulation")
    prob = model_act.compute_rejection_probability(
        t_spend.tolist(),
        a.tolist() if a is not None else [],
        drift=actual_drift,
        tails=tails,
    )
    return {"power": prob}


@when(
    parsers.re(r"(?i)actually K (?P<K_val>\d+) .* analyses occur.*"),
    target_fixture="results",
)
def when_different_k(K_val: str, ch7_params: Dict[str, Any]) -> Dict[str, Any]:
    alpha, power, rho, k_plan = (
        ch7_params["alpha"],
        ch7_params["power"],
        ch7_params["rho"],
        ch7_params.get("k_planned", 5),
    )
    t_plan = np.linspace(1 / k_plan, 1.0, k_plan)
    model_plan = CanonicalJointModel(
        Config(
            t_plan,
            alpha=alpha,
            efficacy_spending=PowerFamilySpending(alpha, rho),
            tails=2,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a_plan, _ = model_plan.solve_boundaries(drift=0.0, method="simulation")
    planned_drift = model_plan.solve_drift(
        t_plan.tolist(),
        a_plan.tolist() if a_plan is not None else [],
        target_power=power,
        tails=2,
        method="simulation",
    )

    K_actual = int(K_val)
    t_act = np.linspace(1 / K_actual, 1.0, K_actual)
    model_act = CanonicalJointModel(
        Config(
            t_act,
            alpha=alpha,
            efficacy_spending=PowerFamilySpending(alpha, rho),
            tails=2,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a, _ = model_act.solve_boundaries(drift=0.0, method="simulation")
    prob = model_act.compute_rejection_probability(
        t_act.tolist(),
        a.tolist() if a is not None else [],
        drift=planned_drift,
        tails=2,
    )
    return {"power": prob}


@when(
    parsers.re(r"(?i)I analyze the BHAT trial with calendar months (?P<months>.*)"),
    target_fixture="results",
)
def when_analyze_bhat(months: str, ch7_params: Dict[str, Any]) -> Dict[str, Any]:
    ch7_params["bhat_months"] = [float(x.strip().strip('"')) for x in months.split(",")]
    return {}


@when(
    parsers.re(r"(?i)the observed death counts are (?P<deaths>.*)"),
    target_fixture="results",
)
def when_bhat_deaths(deaths: str, ch7_params: Dict[str, Any]) -> Dict[str, Any]:
    parsed_deaths = [float(x.strip().strip('"')) for x in deaths.split(",")]
    # Information estimated as deaths / 4.0?
    # If 400 is max, then deaths / 4.0 is percentage (0-100).
    denom = ch7_params.get("bhat_info_denom", 400.0)
    t_info = np.array(parsed_deaths) / denom
    # calendar spending: months / 48
    t_spend = np.array(ch7_params["bhat_months"]) / 48.0
    rho = ch7_params.get("rho", 1.0)
    model = CanonicalJointModel(
        Config(
            t_info,
            spending_times=t_spend,
            alpha=0.05,
            efficacy_spending=PowerFamilySpending(0.05, rho),
            tails=2,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a, _ = model.solve_boundaries(drift=0.0, method="simulation")
    return {"boundaries": a}


@when(parsers.re(r"(?i)I compute the inflation factor R_OS"), target_fixture="results")
def when_compute_ros(ch7_params: Dict[str, Any]) -> Dict[str, Any]:
    alpha, power, rho, k = (
        ch7_params.get("alpha", 0.05),
        ch7_params.get("power", 0.9),
        ch7_params.get("rho", 2.0),
        ch7_params.get("k", 5),
    )
    t = np.linspace(1 / k, 1.0, k)
    i_fixed = (norm.ppf(1 - alpha) + norm.ppf(power)) ** 2
    from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess

    gpt0 = CanonicalGaussianProcess(
        drift=0.0, rng=np.random.default_rng(ch7_params["rng_seed"])
    )
    z_h0 = gpt0.sample(t, ch7_params["n_sims"])

    def f(r_os: float) -> float:
        drift = np.sqrt(r_os * i_fixed)
        z_h1 = z_h0 + drift * np.sqrt(t)
        a_tmp, b_tmp = np.zeros(k), np.zeros(k)
        stop0, stop1 = np.zeros(len(z_h0), dtype=bool), np.zeros(len(z_h0), dtype=bool)
        rej0, fut1 = np.zeros(len(z_h0), dtype=bool), np.zeros(len(z_h1), dtype=bool)
        a_cum, b_cum = alpha * (t**rho), (1 - power) * (t**rho)
        for i in range(k):
            frac0 = (a_cum[i] * len(z_h0) - np.sum(rej0)) / max(1, np.sum(~stop0))
            if frac0 >= 1.0 or np.sum(~stop0) < 10:
                a_tmp[i] = -10.0 if frac0 > 0 else 10.0
            else:
                a_tmp[i] = np.percentile(
                    z_h0[~stop0, i], 100 * max(0, min(1, 1 - frac0))
                )
            frac1 = (b_cum[i] * len(z_h1) - np.sum(fut1)) / max(1, np.sum(~stop1))
            if frac1 >= 1.0 or np.sum(~stop1) < 10:
                b_tmp[i] = 10.0 if frac1 > 0 else -10.0
            else:
                b_tmp[i] = np.percentile(z_h1[~stop1, i], 100 * max(0, min(1, frac1)))

            j0 = (~stop0) & (z_h0[:, i] > a_tmp[i])
            rej0 |= j0
            stop0 |= j0
            jf0 = (~stop0) & (z_h0[:, i] < b_tmp[i])
            stop0 |= jf0

            j1 = (~stop1) & (z_h1[:, i] > a_tmp[i])
            stop1 |= j1
            jf1 = (~stop1) & (z_h1[:, i] < b_tmp[i])
            fut1 |= jf1
            stop1 |= jf1

        return float(a_tmp[-1] - b_tmp[-1])

    from scipy.optimize import brentq

    try:
        r_os = brentq(f, 0.4, 4.0, xtol=1e-2)
    except Exception:
        v04, v40 = f(0.4), f(4.0)
        r_os = 0.4 if abs(v04) < abs(v40) else 4.0
    return {
        "R_OS": r_os,
        "i_fixed": i_fixed,
        "alpha": alpha,
        "power": power,
        "rho": rho,
        "k": k,
    }


@when(
    parsers.re(
        r"(?i)I evaluate the one-sided expected sample size relative to fixed design"
    ),
    target_fixture="results",
)
def when_eval_ros_asn(ch7_params: Dict[str, Any]) -> Dict[str, Any]:
    res = when_compute_ros(ch7_params)
    r_os, alpha, power, rho, k = (
        res["R_OS"],
        res["alpha"],
        res["power"],
        res["rho"],
        res["k"],
    )
    t = np.linspace(1 / k, 1.0, k)
    drift_h1 = np.sqrt(r_os * res["i_fixed"])
    model = CanonicalJointModel(
        Config(
            t,
            alpha=alpha,
            power=power,
            efficacy_spending=PowerFamilySpending(alpha, rho),
            futility_spending=PowerFamilySpending(1 - power, rho),
            tails=1,
            n_sims=ch7_params["n_sims"],
            rng_seed=42,
        )
    )
    a, b = model.solve_boundaries(drift=drift_h1, method="simulation")
    res_asn = {"R_OS": r_os * 100}
    for m in [0.0, 0.5, 1.0]:
        from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess

        gpt = CanonicalGaussianProcess(
            drift=m * drift_h1, rng=np.random.default_rng(42)
        )
        z = gpt.sample(t, ch7_params["n_sims"])
        stop_looks = np.full(len(z), k, dtype=int)
        stopped = np.zeros(len(z), dtype=bool)
        for i in range(k):
            if a is None or b is None:
                continue
            cr = (z[:, i] > a[i]) | (z[:, i] < b[i])
            if i == k - 1:
                cr = np.ones(len(z), dtype=bool)
            js = cr & ~stopped
            stop_looks[js] = i + 1
            stopped |= cr
        res_asn[f"ASN_{m}"] = np.mean(stop_looks) * (r_os / k) * 100
    return res_asn


# --- THEN: Verify ---


@then(
    parsers.re(
        r"(?i)^(?:the )?.*?\b(?P<key>R_LD|R_OS|I_f|I_max|n_f|n_max|power|attained power at effect size delta|required effect size \(delta\))\)?\s+should be (?P<val>[-+]?\d*\.\d+|\d+) with (?P<atol>[-+]?\d*\.\d+|\d+) precision"
    )
)
@then(
    parsers.re(
        r"(?i)(?:the )?(?P<key>R_LD|R_OS|I_f|I_max|n_f|n_max|power|attained power at effect size delta|required effect size \(delta\)) should be (?P<val>[-+]?\d*\.\d+|\d+) with (?P<atol>[-+]?\d*\.\d+|\d+) precision"
    )
)
def then_check_val_exact(
    results: Dict[str, Any], key: str, val: str, atol: str
) -> None:
    lookup = {
        "R_LD": "R_LD",
        "R_OS": "R_OS",
        "I_f": "i_fixed",
        "I_max": "i_max",
        "n_f": "n_f",
        "n_max": "n_max",
        "power": "power",
        "attained power at effect size delta": "power",
        "required effect size (delta)": "delta",
    }
    assert results[lookup[key]] == pytest.approx(float(val), abs=float(atol))


@then(
    parsers.re(
        r"(?i)(?:the )?(?P<key>R_LD|R_OS) should be (?P<val>[-+]?\d*\.\d+|\d+) ± (?P<atol>[-+]?\d*\.\d+|\d+)"
    )
)
def then_check_val_key_pm(
    results: Dict[str, Any], key: str, val: str, atol: str
) -> None:
    lookup = {
        "R_LD": "R_LD_pct",
        "R_OS": "R_OS",
        "alpha": "alpha",
        "power": "power",
        "rho": "rho",
    }
    assert results[lookup[key]] == pytest.approx(float(val), abs=float(atol))


@then(parsers.re(r"(?i)relative ASN should be (?P<vals>.*) ± (?P<atol>[\d.]+)"))
def then_check_asn_list(results: Dict[str, Any], vals: str, atol: str) -> None:
    expected = [float(x.strip().strip('"')) for x in vals.strip("[]").split(",")]
    assert np.allclose(results["relative_asn"], expected, atol=float(atol))


@then(
    parsers.re(
        r"(?i)(?:the )?sequence of standardized log-rank statistics at the interim analyses (?P<stats>.*) should reject H0 at look (?P<look>\d+)"
    )
)
def then_check_rejection(results: Dict[str, Any], stats: str, look: str) -> None:
    bounds, seq = (
        results["boundaries"],
        [float(x.strip().strip('"')) for x in stats.split(",")],
    )
    # Sequence might be shorter than bounds if it only shows stats up to rejection
    found = next((i + 1 for i, (z, b) in enumerate(zip(seq, bounds)) if abs(z) > b), -1)
    assert found == int(look)


@then(
    parsers.re(
        r"(?i)(?:the )?boundaries should be (?P<vals>.*) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_boundary_list(results: Dict[str, Any], vals: str, atol: str) -> None:
    expected = [float(x.strip().strip('"')) for x in vals.strip('"').split(",")]
    assert np.allclose(
        results["boundaries"][: len(expected)], expected, atol=float(atol)
    )


@then(
    parsers.re(
        r"(?i)(?:the )?expected sample size at theta\s*=\s*(?P<condition>[^ ]+) should be (?P<asn>[\d.]+) percent with (?P<atol>[\d.]+) precision"
    )
)
def then_check_asn(
    results: Dict[str, Any], condition: str, asn: str, atol: str
) -> None:
    # Standardize condition string
    c = condition.replace(" ", "").lower()

    if "0.5" in c:
        actual_key = "ASN_0.5"
    elif "1.5" in c:
        actual_key = "ASN_1.5"
    elif "delta" in c or "δ" in c:
        # Check for 1.0 delta vs 1.5 delta etc
        actual_key = "ASN_1.0"
    elif "0" in c:
        actual_key = "ASN_0.0"
    else:
        raise ValueError(f"Unknown ASN condition: {condition}")

    assert results[actual_key] == pytest.approx(float(asn), abs=float(atol))


@then(
    parsers.re(
        r"(?i)(?:the )?maximum sample size relative to fixed design \(R_(?:OS|LD)\) should be (?P<val>[\d.]+) percent with (?P<atol>[\d.]+) precision"
    )
)
def then_check_inflation_pct(results: Dict[str, Any], val: str, atol: str) -> None:
    # Handle both R_OS and R_LD
    if "R_OS" in results:
        actual_val = results["R_OS"]
    else:
        actual_val = results["R_LD_pct"]
    assert actual_val == pytest.approx(float(val), abs=float(atol))


@then(parsers.re(r"(?i)(?:the )?trial should stop at look (?P<look>\d+)"))
def then_check_stop_look(results: Dict[str, Any], look: str) -> None:
    assert results["looks"] == int(look)


@then(
    parsers.re(
        r"(?i)(?:the )?final maximum information should be (?P<val>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_imax(results: Dict[str, Any], val: str, atol: str) -> None:
    assert results["i_max"] == pytest.approx(float(val), abs=float(atol))


@then(
    parsers.re(
        r"(?i)(?:the )?power at mu_A - mu_B = [±\+-]1 should be (?P<val>[\d.]+) with (?P<atol>[\d.]+) precision"
    )
)
def then_check_final_power(results: Dict[str, Any], val: str, atol: str) -> None:
    if "power" not in results:
        print(f"DEBUG: results keys: {list(results.keys())}")
    assert results["power"] == pytest.approx(float(val), abs=float(atol))


# ==============================================================================
# Step definitions for Table 7.2 and Table 7.3 (ported from v0)
# ==============================================================================


@given(
    parsers.re(r"(?i)the sample size is designed to attain this power at theta = ±δ")
)
def given_sample_size_at_delta_ch7(ch7_params: Dict[str, Any]) -> None:
    # This is primarily informational in the Gherkin to match textbook phrasing
    pass


@when(
    "I evaluate the expected sample size relative to fixed design",
    target_fixture="results",
)
def when_evaluate_asn(ch7_params: Dict[str, Any]) -> Dict[str, Any]:
    alpha = ch7_params["alpha"]
    power = ch7_params["power"]
    k = ch7_params["k"]
    rho = ch7_params["rho"]
    n_sims = ch7_params["n_sims"]
    seed = 42

    spending = PowerFamilySpending(budget=alpha, rho=rho)
    info_times = np.linspace(1 / k, 1.0, k)

    config = Config(
        info_times=info_times,
        alpha=alpha,
        power=power,
        efficacy_spending=spending,
        n_sims=n_sims,
        tails=2,
        rng_seed=seed,
    )

    model = CanonicalJointModel(config=config)
    a, _ = model.solve_boundaries(method="simulation")
    if a is None:
        raise ValueError("Failed to solve boundaries")

    # Solve for planned drift that gives target power
    drift_h1 = model.solve_drift(
        info_times=info_times.tolist(),
        boundaries=a.tolist(),
        target_power=power,
        tails=2,
    )

    # Inflation factor R_LD
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    r_ld = (drift_h1 / (z_alpha + z_beta)) ** 2

    # Simulate expected information fraction at various drift points
    sim = OperatingCharacteristicSimulator(n_sims=n_sims, rng_seed=seed)

    # CRN: use fixed H0 samples shifted by drift
    gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=np.random.default_rng(seed))
    samples_h0 = gp_h0.sample(info_times, n_sims)

    def get_asn_percent(drift_val: float) -> float:
        # manual shift to match solve_drift logic
        samples_drift = samples_h0 + drift_val * np.sqrt(info_times)
        oc = sim.simulate_statistical(
            info_times=info_times,
            upper=a,
            lower=-a,
            drift=drift_val,
            n_max=1.0,
            samples=samples_drift,
        )
        return 100.0 * r_ld * oc.asn

    return {
        "R_LD_pct": 100.0 * r_ld,
        "ASN_0.0": get_asn_percent(0.0),
        "ASN_0.5": get_asn_percent(0.5 * drift_h1),
        "ASN_1.0": get_asn_percent(drift_h1),
        "ASN_1.5": get_asn_percent(1.5 * drift_h1),
    }


@when(
    "I evaluate the one-sided expected sample size relative to fixed design",
    target_fixture="results",
)
def when_evaluate_asn_onesided(ch7_params: Dict[str, Any]) -> Dict[str, Any]:
    alpha = ch7_params.get("alpha", 0.05)
    power = ch7_params.get("power", 0.95)
    k = ch7_params.get("k", 5)
    rho = ch7_params.get("rho", 2.0)
    n_sims_final = ch7_params["n_sims"]
    n_sims_iter = n_sims_final // 4
    seed = 42

    eff_spending = PowerFamilySpending(budget=alpha, rho=rho)
    # For Table 7.9, symmetric futility spending means beta spending matches alpha
    fut_spending = PowerFamilySpending(budget=1 - power, rho=rho)
    info_times = np.linspace(1 / k, 1.0, k)

    # Solve for drift and boundaries iteratively (interdependent)
    # Start with fixed design drift as initial guess
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    i_fixed = (z_alpha + z_beta) ** 2
    current_drift = z_alpha + z_beta

    for _ in range(3):  # 3 iterations at lower precision is enough
        config = Config(
            info_times=info_times,
            alpha=alpha,
            power=power,
            efficacy_spending=eff_spending,
            futility_spending=fut_spending,
            n_sims=n_sims_iter,
            tails=1,
            rng_seed=seed,
            efficacy_binding=True,
        )
        model = CanonicalJointModel(config=config)
        a, b = model.solve_boundaries(drift=current_drift, method="simulation")
        current_drift = model.solve_drift(
            info_times.tolist(),
            a.tolist() if a is not None else [],
            target_power=power,
            tails=1,
            futility_boundaries=b.tolist(),
            method="simulation",
        )

    # Final run at high precision
    config_final = Config(
        info_times=info_times,
        alpha=alpha,
        power=power,
        efficacy_spending=eff_spending,
        futility_spending=fut_spending,
        n_sims=n_sims_final,
        tails=1,
        rng_seed=seed,
        efficacy_binding=True,
    )
    model = CanonicalJointModel(config=config_final)
    a, b = model.solve_boundaries(drift=current_drift, method="simulation")
    drift_h1 = model.solve_drift(
        info_times.tolist(),
        a.tolist() if a is not None else [],
        target_power=power,
        tails=1,
        futility_boundaries=b.tolist(),
        method="simulation",
    )
    r_os = (drift_h1**2) / i_fixed

    # OC simulation
    sim = OperatingCharacteristicSimulator(n_sims=n_sims_final, rng_seed=seed)
    gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=np.random.default_rng(seed))
    samples_h0 = gp_h0.sample(info_times, n_sims_final)

    def get_asn_percent(drift_val: float) -> float:
        samples_drift = samples_h0 + drift_val * np.sqrt(info_times)
        oc = sim.simulate_statistical(
            info_times=info_times,
            upper=a,
            lower=b,
            drift=drift_val,
            n_max=1.0,
            samples=samples_drift,
        )
        return 100.0 * r_os * oc.asn

    return {
        "R_OS": 100.0 * r_os,
        "ASN_0.0": get_asn_percent(0.0),
        "ASN_0.5": get_asn_percent(0.5 * drift_h1),
        "ASN_1.0": get_asn_percent(drift_h1),
        "ASN_1.5": get_asn_percent(1.5 * drift_h1),  # Add 1.5 for completeness
    }
