from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

from earlysign.schema.ES3.Binomial import BinomialArmData


class BinomialStream:
    """
    Simulates a stream of binomial batch observations for testing/examples.

    Args:
        n_per_batch: Number of samples per arm in each batch.
        arms: Dictionary mapping arm name to its true conversion rate.
              Example: {"control": 0.20, "treatment": 0.25}
        n_max: Maximum number of samples (per arm) to generate.
        seed: Random seed for reproducibility.

    Examples:
        >>> # Two-arm A/B test
        >>> stream = BinomialStream(
        ...     n_per_batch=100,
        ...     arms={"control": 0.20, "treatment": 0.25},
        ...     n_max=500,
        ...     seed=42
        ... )
        >>> batch = next(stream)
        >>> len(batch)
        2
    """

    def __init__(
        self,
        n_per_batch: int,
        arms: Dict[str, float],
        n_max: int,
        seed: Optional[int] = None,
    ):
        self.n_per_batch = n_per_batch
        self.arms = arms
        self.n_max = n_max
        self.n_total = 0
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[List[BinomialArmData]]:
        return self

    def __next__(self) -> List[BinomialArmData]:
        if self.n_total >= self.n_max:
            raise StopIteration

        batch = []
        batch_size = min(self.n_per_batch, self.n_max - self.n_total)

        if batch_size <= 0:
            raise StopIteration

        for arm_name, p in self.arms.items():
            k = self.rng.binomial(batch_size, p)
            batch.append(BinomialArmData(total=batch_size, success=k, arm=arm_name))

        self.n_total += batch_size
        return batch


def corresponding_scenario_path(caller_file: Union[str, Path]) -> Path:
    """Get the path to the feature file corresponding to the calling test file."""
    import earlysign

    caller_path = Path(caller_file).resolve()
    repo_root = Path(earlysign.__file__).parent.parent
    # We look for features in repo_root/spec/

    # Extract relative path from earlysign/tests/spec_tests/
    # format: earlysign/tests/spec_tests/jennison_turnbull_2000/test_foo.py
    # -> spec/jennison_turnbull_2000/foo.feature

    spec_tests_dir = Path(earlysign.__file__).parent / "tests" / "spec_tests"

    try:
        relative_path = caller_path.relative_to(spec_tests_dir)
    except ValueError:
        # Fallback
        name_str = caller_path.name
        return repo_root / "spec" / name_str.replace(".py", ".feature")

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
