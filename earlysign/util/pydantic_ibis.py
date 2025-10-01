import types
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterator, Union, get_args, get_origin

import ibis
import ibis.expr.datatypes as dt
from ibis.expr.types import Column, Table
from pydantic import BaseModel, create_model

# --- Type Mapping Definition ---
PY_TO_IBIS_TYPE: dict[type, dt.DataType] = {
    str: dt.string,
    int: dt.int64,
    float: dt.float64,
    bool: dt.boolean,
    Decimal: dt.decimal,
    date: dt.date,
    datetime: dt.timestamp,
}


def explode_json_with_pydantic(
    table: Table,
    model: type[BaseModel],
    *,
    json_col: str = "payload",
    prefix: str | None = None,
) -> Table:
    """
    Explodes a JSON column into typed scalar columns guided by a Pydantic model.

    Examples:
    ---------
    >>> import pandas as pd
    >>>
    >>> con = ibis.connect("duckdb://:memory:")
    >>>
    >>> t = con.sql(
    ...     \"\"\"
    ...     SELECT 1 AS id, '{ "na": 10, "flag": true, "meta": {"z": 1.5, "scale": "bm"} }' AS payload
    ...     UNION ALL
    ...     SELECT 2 AS id, '{ "na": null, "flag": false, "meta": {"z": 0.25, "scale": "z"} }' AS payload
    ...     \"\"\"
    ... )
    >>>
    >>> Inner = create_model("Inner", z=(float, ...), scale=(str, ...))
    >>> Payload = create_model("Payload", na=(int | None, ...), flag=(bool, ...), meta=(Inner, ...))
    >>>
    >>> result_expr = explode_json_with_pydantic(t, Payload, json_col="payload", prefix="p_")
    >>>
    >>> df = result_expr.order_by("id").execute()
    >>> df.dtypes
    id                int32
    payload          object
    p_na            float64
    p_flag             bool
    p_meta_z        float64
    p_meta_scale     object
    dtype: object
    >>> d = df.iloc[0]
    >>> int(d["p_na"]), bool(d["p_flag"]), float(d["p_meta_z"]), str(d["p_meta_scale"])
    (10, True, 1.5, 'bm')
    >>>
    >>> # Use pandas.isna() to robustly check for missing values (NULL/nan)
    >>> pd.isna(df.loc[1, "p_na"]), bool(df.loc[1, "p_flag"]), float(df.loc[1, "p_meta_z"]), str(df.loc[1, "p_meta_scale"])
    (True, False, 0.25, 'z')
    """
    if json_col not in table.columns:
        raise KeyError(f"Column '{json_col}' not found in table.")

    base_expr = table[json_col]
    prefix_str = prefix or ""

    new_cols = {
        prefix_str + _safe_colname(path): _json_get_typed(base_expr, path, dtype)
        for path, dtype in _iter_scalar_paths(model)
    }

    return table.mutate(**new_cols)


# ============================ Private Helper Functions =============================


def _json_get_typed(expr: Column, dotted_path: str, dtype: dt.DataType) -> Column:
    """Builds an Ibis expression to extract a scalar value from a JSON column."""
    node = expr.cast("json")
    for part in dotted_path.split("."):
        node = node[part]
    return node.unwrap_as(dtype)


def _iter_scalar_paths(
    model: type[BaseModel], *, path_prefix: str = ""
) -> Iterator[tuple[str, dt.DataType]]:
    """
    Recursively traverses a Pydantic model and yields (json_path, ibis_type).

    Examples:
    ---------
    >>> InnerModel = create_model("InnerModel", z=(float, ...), scale=(str, ...))
    >>> PayloadModel = create_model(
    ...     "PayloadModel",
    ...     na=(int | None, ...),
    ...     flag=(bool, ...),
    ...     meta=(InnerModel, ...)
    ... )
    >>>
    >>> expected_paths = [
    ...     ("na", dt.Int64(nullable=True)),
    ...     ("flag", dt.Boolean(nullable=False)),
    ...     ("meta.z", dt.Float64(nullable=False)),
    ...     ("meta.scale", dt.String(nullable=False)),
    ... ]
    >>>
    >>> list(_iter_scalar_paths(PayloadModel)) == expected_paths
    True
    """
    for field_name, field in model.model_fields.items():
        current_path = f"{path_prefix}{field_name}"
        annotation = field.annotation
        unwrapped_annotation = _unopt(annotation)

        if isinstance(unwrapped_annotation, type) and issubclass(
            unwrapped_annotation, BaseModel
        ):
            yield from _iter_scalar_paths(
                unwrapped_annotation, path_prefix=f"{current_path}."
            )
        elif unwrapped_annotation in PY_TO_IBIS_TYPE:
            yield (current_path, _ibis_dtype_from_annotation(annotation))


def _ibis_dtype_from_annotation(tp: Any) -> dt.DataType:
    """Maps a Python type annotation to an Ibis DataType."""
    is_optional = get_origin(tp) in (Union, types.UnionType) and type(None) in get_args(
        tp
    )
    ibis_type = PY_TO_IBIS_TYPE.get(_unopt(tp), dt.string)
    return ibis_type.copy(nullable=is_optional)


def _unopt(tp: Any) -> Any:
    """
    Unwraps Optional[T] (or T | None) to T.

    Examples:
    ---------
    >>> _unopt(int | None)
    <class 'int'>
    >>> _unopt(Union[str, None])
    <class 'str'>
    >>> _unopt(bool)
    <class 'bool'>
    >>> _unopt(int | str)  # Not a simple Optional, returns original
    int | str
    """
    if get_origin(tp) in (Union, types.UnionType):
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _safe_colname(dotted_path: str) -> str:
    """Converts a dot-separated path into a safe column name."""
    return dotted_path.replace(".", "_")
