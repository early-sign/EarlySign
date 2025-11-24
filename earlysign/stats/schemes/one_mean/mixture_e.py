"""Mixture-evidence e-process helpers for one-mean (Gaussian, known variance)."""

import math

from scipy.stats import norm


def mixture_e_one_mean_known_var(
    n: int, ybar: float, sigma2: float, tau2: float
) -> float:
    """
    Symmetric-alt mixture e-value for one-mean with known variance.

    Under H0: ybar ~ N(0, v0) with v0 = sigma2 / n
    Under H1: ybar ~ N(0, v1) with v1 = sigma2 / n + tau2, tau2 = c * sigma2 / n

    e = f1(ybar) / f0(ybar) = sqrt(v0/v1) * exp(-0.5*ybar^2*(1/v1 - 1/v0))

    Examples
    --------
    >>> round(mixture_e_one_mean_known_var(100, 0.2, 1.0, 1.0), 6)
    0.720823
    """

    if n <= 0 or sigma2 <= 0.0 or tau2 <= 0.0:
        raise ValueError("n>0, sigma2>0, tau2>0 are required.")
    v0 = sigma2 / float(n)
    v1 = v0 + tau2
    return math.sqrt(v0 / v1) * math.exp(-0.5 * (ybar**2) * (1.0 / v1 - 1.0 / v0))


def one_sided_mixture_e_one_mean_known_var(
    n: int, ybar: float, sigma2: float, tau2: float
) -> float:
    """
    One-sided (mu >= 0) mixture e-value.

    The marginal under H1 is the convolution of Normal(mu, v0) with mu~Normal^+(0, tau²).
    Its density at ybar is:
        f1^+(ybar) = 2 * Phi( (tau/sqrt(v1)) * ybar ) * NormalPDF(ybar; 0, v1)
    with v0 = σ²/n, v1 = v0 + τ², τ² = c σ²/n, τ = sqrt(τ²).

    Then e = f1^+(ybar) / f0(ybar).

    Examples
    --------
    >>> round(one_sided_mixture_e_one_mean_known_var(100, 0.2, 1.0, 1.0), 6)
    0.834527
    """

    if n <= 0 or sigma2 <= 0.0 or tau2 <= 0.0:
        raise ValueError("n>0, sigma2>0, tau2>0 are required.")
    v0 = sigma2 / float(n)
    v1 = v0 + tau2
    f1 = (
        2.0
        * norm.cdf((math.sqrt(tau2) / math.sqrt(v1)) * ybar)
        * (1.0 / math.sqrt(2.0 * math.pi * v1))
        * math.exp(-(ybar**2) / (2.0 * v1))
    )
    f0 = (1.0 / math.sqrt(2.0 * math.pi * v0)) * math.exp(-(ybar**2) / (2.0 * v0))
    return float(f1 / max(f0, 1e-300))
