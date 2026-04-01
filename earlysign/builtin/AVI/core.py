import math
from typing import Tuple

import numpy as np
from pydantic import BaseModel, Field


class EValueResult(BaseModel):
    """Result of an e-value calculation."""

    e_value: float
    is_rejected: bool
    alpha: float


class EProcessProtocol(BaseModel):
    """Protocol for continuous monitoring based on E-processes."""

    alpha: float = Field(0.05, description="Type-1 error rate")
    null_p: float = Field(
        ..., description="Success probability hypothesized under the Null (H0)."
    )
    alt_p: float = Field(
        ..., description="Success probability hypothesized under the Alternative (H1)."
    )


class BinomialEValueModel:
    @staticmethod
    def compute(
        n: int, successes: int, null_p: float, alt_p: float, alpha: float = 0.05
    ) -> EValueResult:
        r"""Computes a simple likelihood ratio e-value for a binomial test."""
        if n == 0:
            return EValueResult(e_value=1.0, is_rejected=False, alpha=alpha)

        # Avoid division by zero or log of zero
        null_p = max(min(null_p, 1 - 1e-10), 1e-10)
        alt_p = max(min(alt_p, 1 - 1e-10), 1e-10)

        log_e = successes * np.log(alt_p / null_p) + (n - successes) * np.log(
            (1 - alt_p) / (1 - null_p)
        )

        e_val = float(np.exp(log_e))
        is_rejected = e_val > (1.0 / alpha)

        return EValueResult(e_value=e_val, is_rejected=is_rejected, alpha=alpha)


class GAVIBoundaryModel:
    @staticmethod
    def calculate_boundary(
        n_total: float, alpha: float, phi: float, sigma2: float, n_c: int, n_t: int
    ) -> float:
        """
        Calculates GAVI boundary ci.
        """
        # Variance of the mean difference (two-sample)
        # V = sigma2/n_c + sigma2/n_t
        if n_c <= 0 or n_t <= 0:
            return float("inf")

        V = (sigma2 / n_c) + (sigma2 / n_t)

        # GAVI boundary formula from Waudby-Smith (2021)
        # for standard Brownian motion at time n (where n = n_total/2)
        n = n_total / 2.0

        denom = np.log(np.log(np.exp(1) * alpha ** (-2))) - 2 * np.log(alpha)
        rho = phi / denom
        term_log = np.log((n + rho) / (rho * alpha**2))
        uv = np.sqrt((n + rho) * term_log)

        # Boundary for mean difference: sigma * u(n) / n
        # Note: Baseline GAVI templates in this project use 1-arm equivalent variance scaling (V/2 inside sqrt?).
        # Original code: ci = np.sqrt(V / 2.0) * uv / np.sqrt(n)
        ci = np.sqrt(V / 2.0) * uv / np.sqrt(n)
        return float(ci)


class MSPRTBoundaryModel:
    @staticmethod
    def calculate_boundary(var_diff: float, tau_mde: float, alpha: float) -> float:
        if tau_mde <= 0:
            raise ValueError("A positive Minimum Detectable Effect is required.")

        info = 1.0 / var_diff if var_diff > 0 else 0.0
        if info == 0:
            # If variance is very large (info -> 0), boundary implies no confidence
            return float("inf")

        phi = 1.0 / (tau_mde**2)
        ratio = (info + phi) / phi

        # B = sqrt( 2 * V_diff * (1 + phi/I) * log( sqrt((I+phi)/phi) / alpha ) )
        # Using 1+phi/I = (I+phi)/I
        # And V_diff = 1/I
        # So 2 * (1/I) * ((I+phi)/I) ...
        # Original code used:
        # ci = np.sqrt(2 * var_diff * (1 + phi / info) * np.log(np.sqrt(ratio) / alpha))

        term = 2 * var_diff * (1 + phi / info) * np.log(np.sqrt(ratio) / alpha)
        # Ensure non-negative before sqrt
        if term < 0:
            term = 0

        ci = np.sqrt(term)
        return float(ci)


class QuantileModel:
    @staticmethod
    def get_confidence_interval_ranks(
        n: int, q: float, alpha: float
    ) -> Tuple[int, int]:
        """
        Calculate the 1-indexed ranks (L_t, U_t) for the confidence interval.
        """
        if n == 0:
            return 0, 0

        # Formula: epsilon_t = 0.85 * t^(-1) * (log log(e t) + C)
        # C = log(1612 / alpha) / 1.25
        c = math.log(1612 / alpha) / 1.25
        log_log_et = math.log(math.log(math.e * n))
        epsilon_t = 0.85 * (log_log_et + c) / n

        l_rank = max(1, math.floor(n * (q - epsilon_t)) + 1)
        u_rank = min(n, math.ceil(n * (q + epsilon_t)))

        return l_rank, u_rank
