"""
earlysign.tests.util
===================

Utility functions for testing across the EarlySign package.

This module provides common test utilities including database connection
setup and mock data generation.
"""

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
