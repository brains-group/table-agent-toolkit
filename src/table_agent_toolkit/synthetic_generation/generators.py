"""
Tunable defaults (not exposed via MCP tools):
  _DEFAULT_EPOCHS   = 300   — training epochs for CTGAN, TVAE
  _TABICL_ORDER     = None  — column sampling order; options: None (natural), "random",
                              "full_random", or an explicit list of column names
  _TABICL_CARRY_TARGET = False — whether to condition each column on the target from the start
"""

import pandas as pd
from pathlib import Path

_DEFAULT_EPOCHS = 300
_TABICL_ORDER = None
_TABICL_CARRY_TARGET = False

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_LOADERS: dict = {
    ".csv": lambda p: pd.read_csv(p),
    ".parquet": lambda p: pd.read_parquet(p),
    ".xlsx": lambda p: pd.read_excel(p),
    ".xls": lambda p: pd.read_excel(p),
    ".json": lambda p: pd.read_json(p),
}

_SAVERS: dict = {
    ".csv": lambda df, p: df.to_csv(p, index=False),
    ".parquet": lambda df, p: df.to_parquet(p, index=False),
    ".xlsx": lambda df, p: df.to_excel(p, index=False),
    ".xls": lambda df, p: df.to_excel(p, index=False),
    ".json": lambda df, p: df.to_json(p, orient="records", indent=2),
}


def load_table(path: str | Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported: {sorted(_LOADERS)}"
        )
    return _LOADERS[ext](path), ext


def save_table(df: pd.DataFrame, path: Path, ext: str) -> Path:
    if path.suffix.lower() != ext:
        path = path.with_suffix(ext)
    _SAVERS[ext](df, path)
    return path


def default_output_path(input_path: Path, ext: str) -> Path:
    return input_path.with_name(input_path.stem + "_synthetic" + ext)


def detect_discrete_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_string_dtype(df[col])
        or pd.api.types.is_bool_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
    ]


def preprocess_for_generation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Prepare a DataFrame for synthetic generation.

    Drops discrete columns where ``nunique > 0.5 * len(df)`` (too high
    cardinality to generate meaningfully), then ordinal-encodes the remaining
    discrete columns to ``Int64`` integers so every backend receives a uniform
    numeric representation.

    Returns:
        processed_df: copy of *df* with high-cardinality columns removed and
            categorical/object/string/bool columns replaced by integer codes.
        discrete_columns: column names that are discrete in the processed frame.
        metadata: dict with keys ``'dropped_columns'`` (list[str]) and
            ``'encodings'`` ({col: list[str]} mapping ordinal code → original
            category string).
    """
    n = len(df)
    threshold = 0.5 * n
    df = df.copy()

    dropped: list[str] = []
    encodings: dict[str, list[str]] = {}
    discrete_cols: list[str] = []

    for col in list(df.columns):
        s = df[col]
        is_discrete = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_bool_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        )
        if not is_discrete:
            continue

        if s.nunique() > threshold:
            dropped.append(col)
            df.drop(columns=[col], inplace=True)
            continue

        categories = sorted(str(v) for v in s.dropna().unique())
        cat_to_code = {cat: i for i, cat in enumerate(categories)}
        df[col] = (
            s.map(lambda v, m=cat_to_code: m.get(str(v)) if pd.notna(v) else pd.NA)
            .astype("Int64")
        )
        encodings[col] = categories
        discrete_cols.append(col)

    return df, discrete_cols, {"dropped_columns": dropped, "encodings": encodings}


def inverse_transform(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Map ordinal-encoded integer columns back to their original string categories.

    Columns that are not numeric are left untouched (handles the case where a
    backend already returned string values).
    """
    df = df.copy()
    for col, categories in metadata.get("encodings", {}).items():
        if col not in df.columns:
            continue
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        code_to_cat = {i: cat for i, cat in enumerate(categories)}

        def decode(v, m=code_to_cat):
            if pd.isna(v):
                return None
            try:
                return m.get(int(round(float(v))))
            except (ValueError, TypeError):
                return None

        df[col] = s.map(decode)
    return df


# ---------------------------------------------------------------------------
# Non-private generators
# ---------------------------------------------------------------------------


def generate_ctgan(
    data: pd.DataFrame, discrete_columns: list[str], num_rows: int
) -> pd.DataFrame:
    from ctgan import CTGAN

    model = CTGAN(epochs=_DEFAULT_EPOCHS)
    model.fit(data, discrete_columns)
    return model.sample(num_rows)


def generate_tvae(
    data: pd.DataFrame, discrete_columns: list[str], num_rows: int
) -> pd.DataFrame:
    from ctgan import TVAE

    model = TVAE(epochs=_DEFAULT_EPOCHS)
    model.fit(data, discrete_columns)
    return model.sample(num_rows)


def generate_tabicl(
    data: pd.DataFrame, discrete_columns: list[str], num_rows: int
) -> pd.DataFrame:
    from .tabicl_sampler import TabICLSampler

    sampler = TabICLSampler(
        data,
        discrete_columns=discrete_columns,
        order=_TABICL_ORDER,
        carry_target=_TABICL_CARRY_TARGET,
    )
    return sampler.sample(num_rows)

