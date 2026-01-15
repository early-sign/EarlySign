import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from earlysign.v0.methods.group_sequential.boundary import (
    BoundaryCalculator,
    BoundaryCalculatorSpec,
    EfficacySpec,
)
from earlysign.v0.stats.primitives.asymptotic_processes import AsymptoticZProcess
from earlysign.v0.tests.util import corresponding_scenario_path

# Load the feature file corresponding to this test runner
scenarios(corresponding_scenario_path(__file__))


@pytest.fixture
def context():
    return {}


@given(
    parsers.parse("a one-sided A/B test design with alpha {alpha:f}"),
    target_fixture="design_params",
)
def given_alpha(alpha):
    return {"alpha": alpha, "tails": 1}


@given(
    parsers.parse("an interim schedule at information fractions {schedule}"),
    target_fixture="schedule",
)
def given_schedule(schedule):
    # Convert "[0.5, 1.0]" string to list of floats
    return [float(x.strip()) for x in schedule.strip("[]").split(",")]


@given(
    parsers.parse('an "{family}" alpha-spending function'),
    target_fixture="spending_family",
)
def given_spending(family):
    return family


@when("I compute the sequential boundaries", target_fixture="results")
def when_compute_boundaries(design_params, schedule, spending_family):
    spec = BoundaryCalculatorSpec(
        alpha=design_params["alpha"],
        tails=design_params["tails"],
        efficacy=EfficacySpec(style="alpha_spending", family=spending_family),
    )
    calc = BoundaryCalculator(spec=spec)
    boundaries = calc.compute_boundaries(np.array(schedule))

    # Also demonstrate the Gaussian process model for Z-values
    # In a two-arm A/B test, Z(t) ~ BrownianMotion(t) / sqrt(t).
    # AsymptoticZProcess can be used to model this.
    gp = AsymptoticZProcess(
        t_grid=schedule,
        drift=0.0,  # null hypothesis
        mean_scale=lambda t: 0.0,
        std_scale=lambda t: 1.0 / np.sqrt(t) if t > 0 else 0.0,
    )

    return {"boundaries": boundaries, "gp": gp}


@then(
    parsers.parse("look {look:d} should have an upper Z-boundary around {threshold:f}")
)
def then_check_boundary(results, look, threshold):
    actual = results["boundaries"]["upper"][look - 1]
    assert actual == pytest.approx(threshold, abs=1e-3)
