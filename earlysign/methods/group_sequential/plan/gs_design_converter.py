from typing import Any, Dict

import numpy as np

import earlysign.schema.ES3.Base as ES3_BASE
import earlysign.schema.ES3.GST as GST


def convert_gs_design_to_protocol(
    gs_object: Dict[str, Any],
    protocol_name: str = "Imported gsDesign Protocol",
    arm_names: list[str] = ["Control", "Treatment"],
    response_type: GST.ResponseType = GST.ResponseType.BINARY,
) -> GST.Protocol:
    """Converts a gsDesign (R package) output dictionary to an EarlySign GST Protocol.

    Assumption: The input is a dictionary representation of the gsDesign class object.

    Args:
        gs_object: Dictionary containing gsDesign fields (k, alpha, beta, etc.)
        protocol_name: Name for the new protocol.
        arm_names: Names of the arms.
        response_type: Outcome type (binary/continuous).

    Returns:
        A validated GST.Protocol object.
    """

    # 1. Extract Key Parameters
    try:
        k = int(gs_object["k"])
        alpha = float(gs_object["alpha"])
        # gsDesign uses beta (Type II error), EarlySign uses Power (1 - beta)
        beta = float(gs_object.get("beta", 0.0))
        power = 1.0 - beta

        # Timing (Information Fractions)
        if "timing" in gs_object:
            timing = gs_object["timing"]
            # Ensure it's a list
            if hasattr(timing, "tolist"):
                timing = timing.tolist()
        else:
            timing = list(np.linspace(1 / k, 1.0, k))

        # Sample Size
        # gsDesign stores n.I (Information) or n.fix (Sample size)
        # We need the max sample size.
        if "n.I" in gs_object:
            n_I = gs_object["n.I"]
            if hasattr(n_I, "tolist"):
                n_I_vals = n_I.tolist()
            else:
                n_I_vals = n_I
            # Max sample size is the last one if it represents N
            # For Binary, n.I usually is sample size if not dealing with survival
            total_n = int(np.ceil(n_I_vals[-1]))
            # Handle allocation ratio for 2 arms
            ratio = float(gs_object.get("ratio", 1.0))
            if len(arm_names) == 2:
                n_c = int(np.round(total_n / (1.0 + ratio)))
                n_t = total_n - n_c
                max_sample_size = {arm_names[0]: n_c, arm_names[1]: n_t}
            else:
                # Equi-distribute for other cases
                per_arm = total_n // len(arm_names)
                max_sample_size = {name: per_arm for name in arm_names}
                # Fix remainder
                diff = total_n - sum(max_sample_size.values())
                max_sample_size[arm_names[-1]] += diff
        elif "n.fix" in gs_object:
            # Approximate from fixed design inflation
            # Better to look for 'n' or similar
            raise ValueError(
                "Found 'n.fix' but max sample size 'n.I' is missing in gsDesign object."
            )
        else:
            raise ValueError("Cumulative sample size 'n.I' missing in gsDesign object.")
    except KeyError as e:
        raise ValueError(f"Invalid gsDesign object: missing field {e}")

    # 2. Map Spending Functions
    # gsDesign stores spending function info in 'upper' and 'lower' lists usually,
    # or specific 'sf' fields like 'sfFunction'.
    # We will assume standard Lan-DeMets OBF if not specified, or try to parse

    # Simple mapping for now:
    # If test.type is 1 -> One sided
    # If 2 -> Two sided symmetric
    # If 3 -> Two sided asymmetric
    # If 4 -> Two sided asymmetric non-binding

    test_type = int(gs_object.get("test.type", 1))

    # We need to decide if we use AlphaSpendingStrategy (One/Two sided efficacy only)
    # or AlphaBetaSpendingStrategy (Efficacy + Futility).

    strategy_spec: GST.DecisionStrategy

    # Default family mapping
    family_map = {
        "sfLDOF": GST.SpendingFunctionType.OBRIEN_FLEMING,  # Default to OBF-like LD
        "sfLDPocock": GST.SpendingFunctionType.OBRIEN_FLEMING,
        "sfHSD": GST.SpendingFunctionType.HWANG_SHIH_DECANI,
        "sfPower": GST.SpendingFunctionType.POWER_FAMILY,
        "sfKIM": GST.SpendingFunctionType.POWER_FAMILY,  # Kim-DeMets often maps to Power
    }

    # Try to extract family name from 'upper' dictionary
    # gsDesign object usually has 'upper' as a dict with 'sfFunction' or 'name'
    selected_family = GST.SpendingFunctionType.OBRIEN_FLEMING  # Default

    if "upper" in gs_object and isinstance(gs_object["upper"], dict):
        upper_info = gs_object["upper"]
        # R sfFunction might be imported as string or list
        r_sf = upper_info.get("sfFunction")
        if r_sf:
            # Handle list wrapper from R conversion
            if isinstance(r_sf, list) and len(r_sf) > 0:
                r_sf = r_sf[0]

            if isinstance(r_sf, str) and r_sf in family_map:
                selected_family = family_map[r_sf]

    spending_fn = GST.SpendingFunction(family=selected_family)

    if test_type == 1:
        # One-sided efficacy
        strategy_spec = GST.AlphaSpendingStrategy(
            spending_fn=spending_fn,
            budget=alpha,
            sided=GST.Sided.ONE,
            statistical_model=GST.CanonicalGaussianModel(),
        )
    elif test_type == 2:
        # Two-sided symmetric
        strategy_spec = GST.AlphaSpendingStrategy(
            spending_fn=spending_fn,
            budget=alpha,
            sided=GST.Sided.TWO,
            statistical_model=GST.CanonicalGaussianModel(),
        )
    elif test_type in [3, 4]:
        # Efficacy + Futility
        # Type 3 is binding, Type 4 is non-binding
        is_binding = test_type == 3

        strategy_spec = GST.AlphaBetaSpendingStrategy(
            alpha_spending_fn=spending_fn,
            beta_spending_fn=spending_fn,  # Assume same for now
            alpha_budget=alpha,
            beta_budget=beta,
            # 'beta_budget' in strategy corresponds to 'beta' (Type II error) to spend.
            alpha_binding=True,
            beta_binding=is_binding,
            statistical_model=GST.CanonicalGaussianModel(),
        )
    else:
        # Fallback
        strategy_spec = GST.AlphaSpendingStrategy(
            spending_fn=spending_fn,
            budget=alpha,
            sided=GST.Sided.ONE,
            statistical_model=GST.CanonicalGaussianModel(),
        )

    # 3. Construct Protocol

    # Infer Target Effect from delta
    delta_raw = gs_object.get("delta")
    if delta_raw is None:
        delta_raw = gs_object.get("delta1")
    if delta_raw is None:
        raise ValueError(
            "Target effect size 'delta' or 'delta1' missing in gsDesign object."
        )
    delta = float(delta_raw)

    effect_size: GST.EffectSizeUnion
    statistic: GST.TestStatisticSpec

    # Construct Targets based on response type
    if response_type == GST.ResponseType.BINARY:
        # We need the control proportion to reconstruct the binomial variance correctly.
        # This is often 'p_c' or similar in exported objects from R.
        p_c = gs_object.get("p_c")
        if p_c is None:
            raise ValueError(
                "Baseline proportion 'p_c' missing in gsDesign object for binary outcome."
            )
        p_c = float(p_c)
        p_t = p_c + delta
        # Clip
        if p_t > 0.99:
            p_t = 0.99

        effect_size = GST.BinaryEffectSize(
            type="binary", proportions={arm_names[0]: p_c, arm_names[1]: p_t}
        )
        # Stats
        statistic = GST.TwoArmBinomialZ(
            information_unit="fisher_information",
            variance_estimation=GST.VarianceEstimation.POOLED,
        )
    else:
        # Continuous
        effect_size = GST.ContinuousEffectSize(
            type="continuous",
            means={arm_names[0]: 0.0, arm_names[1]: delta},
            standard_deviation=1.0,
        )
        statistic = GST.TwoArmContinuousZ(
            information_unit="fisher_information",
            variance=GST.TwoArmEstimatedVariance(
                kind="estimated", method=GST.MethodModel.POOLED
            ),
        )

    return GST.Protocol(
        name=protocol_name,
        task=GST.TaskSpec(
            kind="group_sequential",
            arms=ES3_BASE.TwoArmComparison(
                control_arm_name=arm_names[0],
                treatment_arm_name=arm_names[1],
            ),
            response_type=response_type,
            hypotheses=GST.HypothesisSpec(
                h_null_description="Difference <= 0",
                h_alt_description=f"Difference > {delta}",
                test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                target_effect=effect_size,
            ),
            efficacy=GST.EfficacyRequirement(alpha=alpha),
            futility=(
                GST.FutilityRequirement(power=power, binding=(test_type == 3))
                if test_type in [3, 4]
                else None
            ),
        ),
        method=GST.MethodSpec(
            kind="group_sequential",
            stopping_policy=GST.StoppingPolicySpec(
                statistic=statistic,
                strategy=strategy_spec,
                timer=GST.SampleSizeTimer(
                    kind="sample_size",
                    unit=GST.Unit.INDIVIDUALS,
                    max_sample_size=max_sample_size,
                ),
                schedule=GST.FixedSchedule(
                    kind="fixed",
                    analyses=timing,
                ),
            ),
        ),
    )
