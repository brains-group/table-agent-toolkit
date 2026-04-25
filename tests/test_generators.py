"""Tests for synthetic_generation helper functions."""

import pandas as pd
import pytest
from pathlib import Path

from table_agent_toolkit.synthetic_generation import (
    load_table,
    save_table,
    default_output_path,
    detect_discrete_columns,
    preprocess_for_generation,
    inverse_transform,
)

ADULT_CSV = Path(__file__).parent.parent / "adult-sampled.csv"


class TestDetectDiscreteColumns:
    def test_detects_object_columns(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [30, 25],
        })
        discrete = detect_discrete_columns(df)
        assert "name" in discrete
        assert "age" not in discrete

    def test_detects_bool_columns(self):
        df = pd.DataFrame({
            "flag": [True, False],
            "value": [1.0, 2.0],
        })
        discrete = detect_discrete_columns(df)
        assert "flag" in discrete
        assert "value" not in discrete

    def test_detects_categorical_columns(self):
        df = pd.DataFrame({
            "cat": pd.Categorical(["a", "b", "a"]),
            "num": [1, 2, 3],
        })
        discrete = detect_discrete_columns(df)
        assert "cat" in discrete
        assert "num" not in discrete

    def test_adult_csv_discrete_columns(self):
        df, _ = load_table(ADULT_CSV)
        discrete = detect_discrete_columns(df)
        expected_discrete = {
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "gender", "native-country", "income",
        }
        assert expected_discrete.issubset(set(discrete))
        for col in ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]:
            assert col not in discrete


class TestPreprocessForGeneration:
    def test_ordinal_encodes_object_column(self):
        # 6 rows, 3 unique = 50% → below threshold, kept
        df = pd.DataFrame({"num": range(6), "cat": ["b", "a", "c", "b", "a", "c"]})
        processed, discrete_cols, meta = preprocess_for_generation(df)
        # categories sorted: a=0, b=1, c=2
        assert processed["cat"].tolist() == [1, 0, 2, 1, 0, 2]
        assert meta["encodings"]["cat"] == ["a", "b", "c"]
        assert "cat" in discrete_cols
        assert "num" not in discrete_cols

    def test_ordinal_encodes_categorical_dtype_column(self):
        # 6 rows, 3 unique = 50% → below threshold, kept
        df = pd.DataFrame({"cat": pd.Categorical(["b", "a", "c", "b", "a", "c"])})
        processed, discrete_cols, meta = preprocess_for_generation(df)
        assert processed["cat"].tolist() == [1, 0, 2, 1, 0, 2]
        assert meta["encodings"]["cat"] == ["a", "b", "c"]
        assert "cat" in discrete_cols

    def test_ordinal_encodes_bool_column(self):
        # 4 rows, 2 unique = 50% → below threshold, kept
        df = pd.DataFrame({"flag": [True, False, True, False]})
        processed, discrete_cols, meta = preprocess_for_generation(df)
        # sorted: False=0, True=1
        assert set(processed["flag"].dropna().tolist()) == {0, 1}
        assert meta["encodings"]["flag"] == ["False", "True"]
        assert "flag" in discrete_cols

    def test_numeric_columns_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [10, 20]})
        processed, discrete_cols, meta = preprocess_for_generation(df)
        assert processed["a"].tolist() == [1.0, 2.0]
        assert processed["b"].tolist() == [10, 20]
        assert discrete_cols == []
        assert meta["encodings"] == {}

    def test_drops_high_cardinality_column(self):
        n = 10
        df = pd.DataFrame({
            "num": range(n),
            "low_card": ["a", "b"] * 5,
            "high_card": [f"x{i}" for i in range(n)],  # n unique out of n rows = 100% > 50%
        })
        processed, discrete_cols, meta = preprocess_for_generation(df)
        assert "high_card" not in processed.columns
        assert "high_card" in meta["dropped_columns"]
        assert "low_card" in processed.columns
        assert "low_card" in discrete_cols

    def test_boundary_cardinality_kept(self):
        # exactly 0.5 * n unique values → kept (not strictly greater)
        n = 10
        df = pd.DataFrame({"cat": list("abcde") * 2})  # 5 unique out of 10 rows = exactly 0.5
        processed, _, meta = preprocess_for_generation(df)
        assert "cat" in processed.columns
        assert "cat" not in meta["dropped_columns"]

    def test_just_over_boundary_dropped(self):
        n = 10
        df = pd.DataFrame({"cat": list("abcdef") + ["a"] * 4})  # 6 unique out of 10 = 60% > 50%
        processed, _, meta = preprocess_for_generation(df)
        assert "cat" not in processed.columns
        assert "cat" in meta["dropped_columns"]

    def test_preserves_na_as_na(self):
        # 4 rows, 2 unique non-null = 50% → below threshold, kept
        df = pd.DataFrame({"cat": ["a", None, "b", "a"]})
        processed, _, _ = preprocess_for_generation(df)
        assert pd.isna(processed["cat"].iloc[1])

    def test_no_mutation_of_input(self):
        df = pd.DataFrame({"cat": ["a", "b"]})
        original_dtype = df["cat"].dtype
        preprocess_for_generation(df)
        assert df["cat"].dtype == original_dtype


class TestInverseTransform:
    def test_maps_integer_codes_to_strings(self):
        meta = {"encodings": {"cat": ["a", "b", "c"]}, "dropped_columns": []}
        df = pd.DataFrame({"num": [1, 2], "cat": [0, 2]})
        result = inverse_transform(df, meta)
        assert result["cat"].tolist() == ["a", "c"]
        assert result["num"].tolist() == [1, 2]

    def test_handles_float_codes_from_model_output(self):
        meta = {"encodings": {"cat": ["a", "b", "c"]}, "dropped_columns": []}
        # values that round unambiguously to 0, 1, 2
        df = pd.DataFrame({"cat": [0.3, 1.1, 2.4]})
        result = inverse_transform(df, meta)
        assert result["cat"].tolist() == ["a", "b", "c"]

    def test_skips_already_string_column(self):
        """If a backend returns strings instead of codes, leave them alone."""
        meta = {"encodings": {"cat": ["a", "b"]}, "dropped_columns": []}
        df = pd.DataFrame({"cat": ["a", "b"]})
        result = inverse_transform(df, meta)
        assert result["cat"].tolist() == ["a", "b"]

    def test_na_values_become_none(self):
        meta = {"encodings": {"cat": ["a", "b"]}, "dropped_columns": []}
        df = pd.DataFrame({"cat": pd.array([0, pd.NA], dtype="Int64")})
        result = inverse_transform(df, meta)
        assert result["cat"].iloc[0] == "a"
        assert pd.isna(result["cat"].iloc[1])

    def test_skips_missing_column(self):
        meta = {"encodings": {"missing_col": ["a", "b"]}, "dropped_columns": []}
        df = pd.DataFrame({"other": [1, 2]})
        result = inverse_transform(df, meta)
        assert "missing_col" not in result.columns

    def test_roundtrip_object_column(self):
        # 6 rows, 3 unique = 50% → below threshold, kept
        df = pd.DataFrame({"cat": ["c", "a", "b", "a", "c", "b"]})
        processed, _, meta = preprocess_for_generation(df)
        recovered = inverse_transform(processed, meta)
        assert recovered["cat"].tolist() == ["c", "a", "b", "a", "c", "b"]

    def test_roundtrip_categorical_dtype_column(self):
        # 6 rows, 3 unique = 50% → below threshold, kept
        df = pd.DataFrame({"cat": pd.Categorical(["x", "z", "y", "x", "z", "y"])})
        processed, _, meta = preprocess_for_generation(df)
        recovered = inverse_transform(processed, meta)
        assert recovered["cat"].tolist() == ["x", "z", "y", "x", "z", "y"]
