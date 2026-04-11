import pytest

from earlysign.builtin.group_sequential.controllers.GST_Spending_JennisonTurnbull2000 import (
    JennisonTurnbull2000Controller,
)
from earlysign.builtin.group_sequential.design.protocol_design import ProtocolDesigner


def test_relative_improvement_resolution() -> None:
    """Verify that RelativeImprovement resolves correctly."""
    from earlysign.builtin.group_sequential.controllers.GST_Spending_JennisonTurnbull2000 import (
        _resolve_p_treatment,
    )

    p_c = 0.2
    # 10% relative improvement -> p_t = 0.2 * 1.1 = 0.22
    effect = {"kind": "relative_improvement", "value": 0.1}
    p_t = _resolve_p_treatment(p_c, effect)
    assert pytest.approx(p_t) == 0.22


def test_unequal_allocation_n_max_2_arm() -> None:
    """Verify n_max calculation for unequal 2-arm allocation."""
    designer = ProtocolDesigner()

    # Standard parameters
    alpha = 0.05
    power = 0.8
    p_c = 0.2
    p_t = 0.3
    looks = 1  # Fixed design for simplicity of formula verification

    # 1. Balanced (r=1.0)
    spec_bal, n_bal = designer.design_gs_binomial(
        alpha=alpha,
        power=power,
        p_control=p_c,
        p_treatment=p_t,
        looks=looks,
        spending_function="obrien_fleming",
    )

    # 2. Unequal (r=2.0)
    spec_uneq, n_uneq = designer.design_gs_binomial(
        alpha=alpha,
        power=power,
        p_control=p_c,
        p_treatment=p_t,
        looks=looks,
        spending_function="obrien_fleming",
        allocation_ratios={"treatment": 2.0},
    )

    # n_total(r) = ( (r+1)^2 / (4r) ) * n_total(1)
    # For r=2: (3^2 / (4*2)) = 9/8 = 1.125
    # Minor rounding differences can occur due to per-arm ceils
    assert pytest.approx(n_uneq, abs=2) == 1.125 * n_bal

    # Check that max_sample_size is a dict
    from earlysign.builtin.group_sequential.schema.timers import SampleSizeTimer

    timer = spec_uneq.stopping_policy.timer
    assert isinstance(timer, SampleSizeTimer)
    max_ss = timer.max_sample_size
    assert isinstance(max_ss, dict)
    assert max_ss["treatment"] == pytest.approx(2 * max_ss["control"], abs=1)


def test_multi_arm_allocation_n_max() -> None:
    """Verify n_max calculation for multi-arm allocation."""
    designer = ProtocolDesigner()

    alpha = 0.05
    power = 0.8
    p_c = 0.2
    p_t = 0.3
    looks = 1

    # 1:2:3 allocation
    ratios = {"A": 2.0, "B": 3.0}
    spec, n_total = designer.design_gs_binomial(
        alpha=alpha,
        power=power,
        p_control=p_c,
        p_treatment=p_t,
        looks=looks,
        spending_function="obrien_fleming",
        allocation_ratios=ratios,
        control_arm_name="C",
        treatment_arm_name="A",  # Design against arm A
    )

    from earlysign.builtin.group_sequential.schema.timers import SampleSizeTimer

    timer = spec.stopping_policy.timer
    assert isinstance(timer, SampleSizeTimer)
    max_ss = timer.max_sample_size
    assert isinstance(max_ss, dict)
    assert max_ss["A"] == pytest.approx(2 * max_ss["C"], abs=1)
    assert max_ss["B"] == pytest.approx(3 * max_ss["C"], abs=1)
    assert sum(max_ss.values()) == n_total


def test_template_design_with_relative_improvement() -> None:
    """Verify template design with relative_improvement and unequal allocation."""
    protocol = JennisonTurnbull2000Controller.design(
        p_control=0.2,
        effect_spec={"kind": "relative_improvement", "value": 0.5},  # p_t = 0.3
        alpha=0.05,
        power=0.8,
        looks=1,
        allocation_ratios={"treatment": 2.0},
    )

    from earlysign.builtin.group_sequential.schema.hypotheses import BinaryEffectSize

    hypotheses = protocol.task.hypotheses
    assert isinstance(hypotheses.target_effect, BinaryEffectSize)
    assert hypotheses.target_effect.proportions["treatment"] == pytest.approx(0.3)

    from earlysign.schema.ES3.base import TwoArmComparison

    arms = protocol.task.arms
    assert isinstance(arms, TwoArmComparison)
    assert arms.allocation_ratios == {"treatment": 2.0}

    from earlysign.builtin.group_sequential.schema.timers import SampleSizeTimer

    timer = protocol.method.stopping_policy.timer
    assert isinstance(timer, SampleSizeTimer)
    max_ss = timer.max_sample_size
    assert isinstance(max_ss, dict)
    assert max_ss["treatment"] == pytest.approx(2 * max_ss["control"], abs=1)
