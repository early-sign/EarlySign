"""
Common reporting helpers (Ibis-first).

Design
------
- Primary artifact is `ibis.Table` (Ibis TableExpr). Callers can render them
  directly in notebooks or UIs.
- The same logical tables can be converted to pandas or Markdown.
- SQL emission is supported by compiling Ibis expressions against a backend
  (DuckDB, BigQuery, etc.).
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ----------------------------- markdown helpers ----------------------------- #


def md_h2(title: str) -> str:
    """
    Return a level-2 Markdown header.

    Examples
    --------
    >>> md_h2("Section").strip()
    '## Section'
    """
    return f"## {title}\n"


def md_kv(items: Mapping[str, Any]) -> str:
    """
    Render a small key-value block as Markdown.

    Examples
    --------
    >>> print(md_kv({"A": 1, "B": "x"}))
    - **A**: 1
    - **B**: x
    <BLANKLINE>
    """
    lines = [f"- **{k}**: {items[k]}" for k in items]
    return "\n".join(lines) + "\n"


def md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """
    Render a minimal Markdown table.

    Examples
    --------
    >>> print(md_table([{"k":"a","v":1},{"k":"b","v":2}], ["k","v"]))
    | k | v |
    |---|---|
    | a | 1 |
    | b | 2 |
    <BLANKLINE>
    """
    if not rows:
        return ""
    head = "| " + " | ".join(columns) + " |\n"
    sep = "|" + "|".join(["---"] * len(columns)) + "|\n"
    body = "".join(
        "| " + " | ".join(str(r.get(c, "")) for c in columns) + " |\n" for r in rows
    )
    return head + sep + body + "\n"


# ----------------------------- format converters ---------------------------- #


def to_pandas_tables(tables: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Execute a mapping of Ibis tables (or pandas-like values) to pandas.

    Notes
    -----
    - Values that provide `.execute()` are executed into pandas.
    - A value that is already a pandas.DataFrame is returned as-is.
    - If a value is neither, it is coerced via `pd.DataFrame(value)`.

    Examples
    --------
    >>> import pandas as pd
    >>> to_pandas_tables({"x": pd.DataFrame({"a":[1,2]})})["x"].shape
    (2, 1)
    """
    out: Dict[str, pd.DataFrame] = {}
    for k, v in tables.items():
        if hasattr(v, "execute"):
            df = v.execute()
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            out[k] = df
        elif isinstance(v, pd.DataFrame):
            out[k] = v
        else:
            out[k] = pd.DataFrame(v)
    return out


def compose_markdown(
    sections: Sequence[Tuple[str, Mapping[str, pd.DataFrame]]], max_rows: int = 8
) -> str:
    """
    Compose a Markdown document from (title, dict[pd.DataFrame]) pairs.

    Examples
    --------
    >>> import pandas as pd
    >>> md = compose_markdown([("Sec", {"T": pd.DataFrame([{"a":1},{"a":2}])})], max_rows=1)
    >>> "## Sec" in md and "| a |" in md
    True
    """
    parts: List[str] = []
    for title, tbls in sections:
        parts.append(md_h2(title))
        for name, df in tbls.items():
            pdf: pd.DataFrame = df.head(max_rows)
            cols: List[str] = [str(c) for c in pdf.columns.to_list()]
            rows: List[Dict[str, Any]] = [
                {cols[i]: str(val) for i, val in enumerate(row)}
                for row in pdf.to_numpy()
            ]
            parts.append(f"**{name}**\n\n" + md_table(rows, cols))
        parts.append("")
    return "\n".join(parts)


# ----------------------------- SQL emission --------------------------------- #


def to_sql_dict(con: Any, tables: Mapping[str, Any]) -> Dict[str, str]:
    """
    Compile Ibis tables into SQL strings using the given backend connection.

    Parameters
    ----------
    con : Ibis backend client (e.g., ibis.duckdb.connect(), ibis.bigquery.connect())
    tables : dict[str, ibis.Table]

    Returns
    -------
    dict[str, str]
        Mapping from key to compiled SQL.

    Examples
    --------
    >>> import ibis
    >>> con = ibis.duckdb.connect(":memory:")
    >>> t = ibis.memtable([{"x": 1}, {"x": 2}])
    >>> sqls = to_sql_dict(con, {"T": t})
    >>> "SELECT" in sqls["T"] or "select" in sqls["T"]
    True
    """
    sqls: Dict[str, str] = {}
    for k, v in tables.items():
        if hasattr(con, "compile"):
            sqls[k] = con.compile(v)
        elif hasattr(v, "compile"):
            sqls[k] = v.compile()
        else:
            raise TypeError("Neither backend nor table provides compile().")
    return sqls


# ----------------------------- tidy plotting -------------------------------- #


def plot_tidy(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    series: Optional[str] = None,
    kind: str = "line",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    savepath: Optional[str] = None,
) -> Figure:
    """
    Generic plotting from tidy data (x, y, optional series).

    - kind="line": lines per series
    - kind="bar": grouped bars per series
    - kind="hline_mark": draw hlines using rows where `series` ∈ {"upper","lower"}
                         and mark points where `series` in {"point","wald","stop"}

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x":[1,1,1], "y":[-2,0,2], "series":["lower","point","upper"]})
    >>> fig = plot_tidy(df, x="x", y="y", series="series", kind="hline_mark", title="demo")
    >>> hasattr(fig, "savefig")
    True
    """
    if df is None or len(df) == 0:
        fig, _ = plt.subplots()
        return fig

    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if series is None:
        if kind == "line":
            ax.plot(df[x], df[y])
        elif kind == "bar":
            ax.bar(df[x], df[y])
        else:
            ax.plot(df[x], df[y], marker="o")
    else:
        if kind == "hline_mark":
            lows = df[df[series] == "lower"]
            ups = df[df[series] == "upper"]
            pts = df[df[series].isin(["point", "wald", "stop"])]
            if len(lows) > 0:
                ax.hlines(
                    lows[y].tolist(),
                    xmin=df[x].min(),
                    xmax=df[x].max(),
                    linestyles="dashed",
                    label="Lower",
                )
            if len(ups) > 0:
                ax.hlines(
                    ups[y].tolist(),
                    xmin=df[x].min(),
                    xmax=df[x].max(),
                    linestyles="dashed",
                    label="Upper",
                )
            if len(pts) > 0:
                ax.plot(pts[x], pts[y], marker="o", label="Observed/Stop")
            ax.legend()
        elif kind == "line":
            for g, gdf in df.groupby(series):
                ax.plot(gdf[x], gdf[y], label=str(g))
            ax.legend()
        elif kind == "bar":
            for g, gdf in df.groupby(series):
                ax.bar([str(v) + f"-{g}" for v in gdf[x]], gdf[y], label=str(g))
            ax.legend()
        else:
            for g, gdf in df.groupby(series):
                ax.plot(gdf[x], gdf[y], marker="o", linestyle="none", label=str(g))
            ax.legend()

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# ----------------------------- Reporter base -------------------------------- #


class ReporterBase:
    """
    Uniform surface for scheme-specific reporters.

    Subclasses must implement:
      - report_tables(format="ibis"|"pandas"|"markdown")
      - plot(kind=..., savepath=None)
      - get_sql(kind=..., backend=...)

    Examples
    --------
    >>> class _R(ReporterBase):
    ...     def report_tables(self, format: str = "markdown"):
    ...         return "ok"
    ...     def plot(self, kind: str = "auto", savepath: Optional[str] = None) -> None:
    ...         return None
    ...     def get_sql(self, kind: str = "interim_summary", *, backend: Any = None) -> Dict[str, str]:
    ...         return {}
    >>> _R(scoped=None, ids={}).report_markdown()
    'ok'
    """

    def __init__(self, scoped: Any, ids: Dict[str, str]) -> None:
        self.scoped = scoped
        self.ids = ids

    def report_tables(self, format: str = "ibis") -> Union[Dict[str, Any], str]:
        raise NotImplementedError

    def report_markdown(self) -> str:
        """Return a composed Markdown report from the latest ledger state."""
        out = self.report_tables(format="markdown")
        assert isinstance(out, str)
        return out

    def plot(self, kind: str = "auto", savepath: Optional[str] = None) -> Figure:
        raise NotImplementedError

    def get_sql(
        self, kind: str = "interim_summary", *, backend: Any = None
    ) -> Dict[str, str]:
        raise NotImplementedError
