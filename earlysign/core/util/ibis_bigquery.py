import ibis
import ibis.expr.datatypes as dt


@ibis.udf.scalar.builtin(name="PARSE_JSON")  # type: ignore
def bq_parse_json(json_str: dt.string) -> dt.json:
    """Parse JSON string to JSON type in BigQuery.

    This maps to the native BigQuery `PARSE_JSON` function.
    """
    ...


@ibis.udf.scalar.builtin(name="JSON_VALUE")  # type: ignore
def bq_extract_scalar(json_val: dt.json, path: str = "$") -> dt.string:
    """Extract a scalar value from a JSON object as a string.

    This maps to the native BigQuery `JSON_VALUE` function.
    Returns a STRING which can be safely cast to numeric types.
    """
    ...
