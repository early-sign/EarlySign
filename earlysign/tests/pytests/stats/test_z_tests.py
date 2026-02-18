import numpy as np

from earlysign.stats.z_tests import (
    calculate_two_arm_binomial_z,
    calculate_two_arm_continuous_z,
)


def test_calculate_two_arm_binomial_z_pooled() -> None:
    # Case: Equal proportions
    z = calculate_two_arm_binomial_z(
        n_c=100, successes_c=10, n_t=100, successes_t=10, pooled=True
    )
    assert z == 0.0

    # Case: Treatment better
    z = calculate_two_arm_binomial_z(
        n_c=100, successes_c=10, n_t=100, successes_t=20, pooled=True
    )
    assert z is not None
    assert z > 0
    # Expected: p_pool = 30/200 = 0.15. se = sqrt(0.15 * 0.85 * (1/100 + 1/100)) = sqrt(0.15 * 0.85 * 0.02) = 0.0505
    # z = (0.2 - 0.1) / 0.0505 = 1.98
    assert np.allclose(z, 1.980295, atol=1e-5)


def test_calculate_two_arm_binomial_z_unpooled() -> None:
    # Case: Treatment better
    z = calculate_two_arm_binomial_z(
        n_c=100, successes_c=10, n_t=100, successes_t=20, pooled=False
    )
    assert z is not None
    assert z > 0
    # Expected: p_c=0.1, p_t=0.2. se = sqrt(0.1*0.9/100 + 0.2*0.8/100) = sqrt(0.0009 + 0.0016) = sqrt(0.0025) = 0.05
    # z = (0.2 - 0.1) / 0.05 = 2.0
    assert np.allclose(z, 2.0, atol=1e-5)


def test_calculate_two_arm_binomial_z_edge_cases() -> None:
    # n=0
    assert calculate_two_arm_binomial_z(0, 0, 100, 10) is None
    # successes=0
    assert calculate_two_arm_binomial_z(100, 0, 100, 0) == 0.0
    # successes=n
    assert calculate_two_arm_binomial_z(100, 100, 100, 100) == 0.0


def test_calculate_two_arm_continuous_z_unpooled() -> None:
    # Case: Equal means
    z = calculate_two_arm_continuous_z(
        n_c=100, mean_c=5.0, var_c=1.0, n_t=100, mean_t=5.0, var_t=1.0, pooled=False
    )
    assert z == 0.0

    # Case: Treatment better
    z = calculate_two_arm_continuous_z(
        n_c=100, mean_c=5.0, var_c=1.0, n_t=100, mean_t=6.0, var_t=1.0, pooled=False
    )
    assert z is not None
    # se = sqrt(1/100 + 1/100) = sqrt(0.02) = 0.1414
    # z = (6 - 5) / 0.1414 = 7.071
    assert np.allclose(z, 7.0710678, atol=1e-5)


def test_calculate_two_arm_continuous_z_pooled() -> None:
    # Case: Different n and variance
    z = calculate_two_arm_continuous_z(
        n_c=100, mean_c=5.0, var_c=1.0, n_t=50, mean_t=6.0, var_t=2.0, pooled=True
    )
    assert z is not None
    # v_pool = (99*1 + 49*2) / (100+50-2) = (99 + 98) / 148 = 197 / 148 = 1.331
    # se = sqrt(1.331 * (1/100 + 1/50)) = sqrt(1.331 * 0.03) = sqrt(0.03993) = 0.1998
    # z = 1 / 0.1998 = 5.004
    assert np.allclose(z, 5.004223, atol=1e-5)


def test_calculate_two_arm_continuous_z_edge_cases() -> None:
    # n < 2
    assert calculate_two_arm_continuous_z(1, 5.0, 1.0, 100, 5.0, 1.0) is None
    # se = 0
    assert calculate_two_arm_continuous_z(100, 5.0, 0.0, 100, 5.0, 0.0) == 0.0
