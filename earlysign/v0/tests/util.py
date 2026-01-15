"""
earlysign.v0.tests.util
===================

Utility functions for testing across the EarlySign package.

This module provides common test utilities including database connection
setup and mock data generation.
"""

from pathlib import Path

import ibis


def create_test_connection() -> ibis.BaseBackend:
    """Create a test database connection for testing purposes.

    Returns
    -------
    ibis connection
        An in-memory DuckDB connection suitable for testing.

    Examples
    --------
    >>> conn = create_test_connection()
    >>> conn is not None
    True
    """
    return ibis.duckdb.connect(":memory:")


def corresponding_scenario_path(caller_file: str | Path) -> Path:
    """Get the path to the feature file corresponding to the calling test file.

    For a test file at ``earlysign/tests/spec_tests/path/to/test_feature.py``,
    this looks for a feature file at ``spec/path/to/feature.feature``.

    Parameters
    ----------
    caller_file : str
        The ``__file__`` of the calling test module.

    Returns
    -------
    Path
        The absolute path to the corresponding .feature file.
    """
    from pathlib import Path

    import earlysign

    caller_path = Path(caller_file).resolve()
    repo_root = Path(earlysign.__file__).parent.parent
    spec_tests_dir = Path(earlysign.__file__).parent / "v0" / "tests" / "spec_tests"
    
    try:
        relative_path = caller_path.relative_to(spec_tests_dir)
    except ValueError:
        # Fallback for tests not in spec_tests
        tests_dir = Path(earlysign.__file__).parent / "v0" / "tests"
        relative_path = caller_path.relative_to(tests_dir)

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
