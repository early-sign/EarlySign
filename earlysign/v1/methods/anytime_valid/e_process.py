import numpy as np
from pydantic import BaseModel


class EValueResult(BaseModel):
    """Result of an e-value calculation."""

    e_value: float
    is_rejected: bool
    alpha: float


def compute_binomial_e_value(
    n: int, successes: int, null_p: float, alt_p: float, alpha: float = 0.05
) -> EValueResult:
    """
    Computes a simple likelihood ratio e-value for a binomial test.
    E = (alt_p / null_p)^S * ((1-alt_p) / (1-null_p))^(n-S)

    >>> res = compute_binomial_e_value(100, 60, 0.5, 0.6)
    >>> res.e_value > 1.0
    True
    """
    # Likelihood under H1 / Likelihood under H0
    log_e = successes * np.log(alt_p / null_p) + (n - successes) * np.log(
        (1 - alt_p) / (1 - null_p)
    )

    e_val = float(np.exp(log_e))
    is_rejected = e_val > (1.0 / alpha)

    return EValueResult(e_value=e_val, is_rejected=is_rejected, alpha=alpha)
