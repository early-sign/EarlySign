from typing import Union

import ibis
import ibis.expr.datatypes as dt


def extract_json_scalar(
    col_expr: ibis.Expr, key: str, target_type: Union[str, dt.DataType]
) -> ibis.Expr:
    """Extract a scalar value from a JSON column in a backend-agnostic way.

    Specifically for BigQuery, this avoids strict 'CAST(JSON AS INT64)' which can
    crash on stringified numbers or during non-deterministic evaluation of
    unioned tables. It uses 'JSON_VALUE' to extract as a string first,
    then 'SAFE_CAST' to the target type.

    Args:
        col_expr: The Ibis expression for the JSON column.
        key: The key to extract.
        target_type: The target Ibis type (e.g., 'int', 'float', 'string').

    Returns:
        An Ibis expression for the extracted and casted value.
    """
    # Robust detection: Check for BigQuery specific markers or backend name
    is_bq = False
    try:
        backend = col_expr._find_backend()
        is_bq = backend and "bigquery" in backend.name.lower()
    except Exception:
        # Fallback: check if the expression looks like a BigQuery expression
        pass

    if is_bq:
        from earlysign.core.util.ibis_bigquery import bq_extract_scalar

        # $.key is the JSONPath for BigQuery JSON_VALUE.
        # This returns a STRING. We use try_cast (SAFE_CAST) to yield NULL instead of error
        # if the string is not a valid number.
        return bq_extract_scalar(col_expr, f"$.{key}").try_cast(target_type)
    else:
        # Generic approach: indexing then casting.
        result = col_expr[key].cast(target_type)

        # Robust unquoting for DuckDB if we extracted a string
        # DuckDB's native JSON cast to string preserves quotes: e.g. '"val"'
        if target_type == "string":
            is_duckdb = False
            try:
                backend = col_expr._find_backend()
                is_duckdb = backend and ("duckdb" in backend.name.lower())
            except Exception:
                pass

            if is_duckdb:
                result = result.re_replace('^"|"$', "")
        return result
