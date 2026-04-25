from mcp.server.fastmcp import FastMCP
import pandas as pd
import json
import math
from pathlib import Path

mcp = FastMCP("table-agent-toolkit")


@mcp.tool()
def generate_synthetic_data(
    file_path: str,
    num_rows: int,
    backend: str,
    output_path: str | None = None,
) -> str:
    """Generate synthetic tabular data.

    Args:
        file_path: Path to the input data file (CSV, Parquet, Excel, JSON).
        num_rows: Number of synthetic rows to generate.
        backend: One of "tabicl", "ctgan", "tvae".
        output_path: Destination path. Defaults to <input_stem>_synthetic.<same_ext>.

    Returns:
        Path to the saved synthetic file plus a brief summary.
    """
    from .synthetic_generation import (
        load_table,
        save_table,
        default_output_path,
        preprocess_for_generation,
        inverse_transform,
        generate_tabicl,
        generate_ctgan,
        generate_tvae,
    )

    _backends = {"tabicl": generate_tabicl, "ctgan": generate_ctgan, "tvae": generate_tvae}
    if backend not in _backends:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {sorted(_backends)}")

    input_path = Path(file_path)
    data, ext = load_table(input_path)
    processed, discrete_cols, meta = preprocess_for_generation(data)

    synthetic = _backends[backend](processed, discrete_cols, num_rows)
    synthetic = inverse_transform(synthetic, meta)

    out = save_table(
        synthetic,
        Path(output_path) if output_path else default_output_path(input_path, ext),
        ext,
    )
    dropped_info = (
        f" Dropped high-cardinality columns: {meta['dropped_columns']}."
        if meta["dropped_columns"]
        else ""
    )
    return (
        f"Generated {len(synthetic)} rows via {backend}. "
        f"Saved to: {out}. "
        f"Discrete columns detected: {discrete_cols or 'none'}.{dropped_info}"
    )


@mcp.tool()
def summarize_table(file_path: str) -> str:
    """Return a human-readable summary of a tabular dataset.

    For every column reports: dtype, null count, and either the numerical
    distribution (min/max/mean/std/quartiles) or value frequencies for
    categorical/boolean columns.

    Args:
        file_path: Path to the input data file (CSV, Parquet, Excel, JSON).

    Returns:
        Multi-line text summary.
    """
    from .synthetic_generation import load_table

    data, _ = load_table(file_path)

    lines = [f"Shape: {len(data):,} rows × {len(data.columns)} columns", ""]

    col_width = max(len(c) for c in data.columns) + 2

    for col in data.columns:
        s = data[col]
        null_count = int(s.isna().sum())
        null_info = f"{null_count} nulls" if null_count else "no nulls"
        header = f"  {col!r:{col_width}} [{s.dtype}]  {null_info}"

        if pd.api.types.is_bool_dtype(s):
            non_null = s.dropna()
            true_n = int(non_null.sum())
            detail = f"True: {true_n:,}  False: {len(non_null) - true_n:,}"
        elif pd.api.types.is_numeric_dtype(s):
            n = s.dropna()
            detail = (
                f"min: {n.min():.4g}  max: {n.max():.4g}  "
                f"mean: {n.mean():.4g}  std: {n.std():.4g}  "
                f"25%: {n.quantile(0.25):.4g}  "
                f"50%: {n.quantile(0.50):.4g}  "
                f"75%: {n.quantile(0.75):.4g}"
            )
        elif (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        ):
            n_unique = s.nunique()
            top = s.value_counts().head(5)
            top_str = "  ".join(f"{v!r}: {c:,}" for v, c in top.items())
            detail = f"unique: {n_unique:,}  top values — {top_str}"
        else:
            detail = f"unique: {s.nunique():,}"

        lines.append(f"{header}  |  {detail}")

    return "\n".join(lines)


def serve():
    mcp.run()


if __name__ == "__main__":
    serve()
