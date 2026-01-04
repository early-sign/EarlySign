from typing import Iterator, List, Optional

import numpy as np

from earlysign.v1.methods.binomial import BatchObservation


class BinomialStream:
    """
    Simulates a stream of binomial batch observations for testing/examples.

    Args:
        n_per_batch: Number of samples per variant in each batch.
        p_control: True conversion rate for the control group.
        p_treatment: True conversion rate for the treatment group.
        n_max: Maximum number of samples (per variant) to generate. If None, infinite.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_per_batch: int,
        p_control: float,
        p_treatment: float,
        n_max: Optional[int] = None,
        seed: int = 42,
    ):
        self.n_per_batch = n_per_batch
        self.p_control = p_control
        self.p_treatment = p_treatment
        self.n_max = n_max
        self.current_n = 0
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[List[BatchObservation]]:
        return self

    def __next__(self) -> List[BatchObservation]:
        if self.n_max is not None and self.current_n >= self.n_max:
            raise StopIteration

        # Determine actual batch size (handle remaining samples)
        # We ensure we don't exceed n_max samples per variant
        if self.n_max is not None:
            batch_size = min(self.n_per_batch, self.n_max - self.current_n)
        else:
            batch_size = self.n_per_batch

        if batch_size <= 0:
            raise StopIteration

        # Generate data
        k_c = self.rng.binomial(batch_size, self.p_control)
        k_t = self.rng.binomial(batch_size, self.p_treatment)

        self.current_n += batch_size

        return [
            BatchObservation(n=batch_size, success=k_c, arm="C"),
            BatchObservation(n=batch_size, success=k_t, arm="T"),
        ]
