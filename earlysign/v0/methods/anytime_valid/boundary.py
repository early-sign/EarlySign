"""Anytime-valid (safe testing) boundary utilities."""


def ville_threshold(alpha: float) -> float:
    """
    Compute the Ville threshold: 1/alpha.

    Examples
    --------
    >>> ville_threshold(0.05)
    20.0
    """

    if not (0.0 < alpha < 1.0):
        raise ValueError("`alpha` must be in (0,1).")
    return 1.0 / float(alpha)
