from pathlib import Path
from typing import Iterator, List, Optional, Union

import numpy as np

from earlysign.schema.ES3.Binomial import ArmData


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

    def __iter__(self) -> Iterator[List[ArmData]]:
        return self

    def __next__(self) -> List[ArmData]:
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
            ArmData(n=batch_size, success=k_c, arm="C"),
            ArmData(n=batch_size, success=k_t, arm="T"),
        ]


def corresponding_scenario_path(caller_file: Union[str, Path]) -> Path:
    """Get the path to the feature file corresponding to the calling test file."""
    import earlysign

    caller_path = Path(caller_file).resolve()
    repo_root = Path(earlysign.__file__).parent.parent
    # We look for features in repo_root/spec/

    # Extract relative path from earlysign/v1/tests/spec_tests/
    # format: earlysign/v1/tests/spec_tests/jennison_turnbull_2000/test_foo.py
    # -> spec/jennison_turnbull_2000/foo.feature

    spec_tests_dir = Path(earlysign.__file__).parent / "v1" / "tests" / "spec_tests"

    try:
        relative_path = caller_path.relative_to(spec_tests_dir)
    except ValueError:
        # Fallback
        relative_path = caller_path.name
        return repo_root / "spec" / relative_path.replace(".py", ".feature")

    # test_feature.py -> feature.feature
    name = relative_path.name
    if name.startswith("test_"):
        name = name[5:]
    feature_name = name.replace(".py", ".feature")

    feature_path = repo_root / "spec" / relative_path.parent / feature_name

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Corresponding feature file not found at: {feature_path}"
        )

    return feature_path
