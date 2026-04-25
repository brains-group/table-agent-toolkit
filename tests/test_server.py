"""Tests for MCP server tools."""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from table_agent_toolkit.server import (
    summarize_table,
    generate_synthetic_data,
)

ADULT_CSV = Path(__file__).parent.parent / "adult-sampled.csv"
ADULT_CSV_STR = str(ADULT_CSV)


class TestSummarizeTable:
    def test_shape_line(self):
        result = summarize_table(ADULT_CSV_STR)
        assert "1,000 rows × 15 columns" in result

    def test_numeric_column_stats_present(self):
        result = summarize_table(ADULT_CSV_STR)
        assert "min:" in result
        assert "max:" in result
        assert "mean:" in result
        assert "std:" in result
        assert "25%:" in result
        assert "50%:" in result
        assert "75%:" in result

    def test_categorical_column_stats_present(self):
        result = summarize_table(ADULT_CSV_STR)
        assert "unique:" in result
        assert "top values" in result

    def test_all_columns_present(self):
        result = summarize_table(ADULT_CSV_STR)
        for col in ["age", "workclass", "fnlwgt", "education", "income"]:
            assert col in result

    def test_null_reporting(self, tmp_path):
        df = pd.DataFrame({
            "a": [1.0, None, 3.0],
            "b": ["x", "y", None],
        })
        csv_path = tmp_path / "nulls.csv"
        df.to_csv(csv_path, index=False)
        result = summarize_table(str(csv_path))
        assert "1 nulls" in result

    def test_no_nulls_reporting(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        csv_path = tmp_path / "nonulls.csv"
        df.to_csv(csv_path, index=False)
        result = summarize_table(str(csv_path))
        assert "no nulls" in result

    def test_bool_column_reporting(self, tmp_path):
        df = pd.DataFrame({"flag": [True, False, True, True]})
        csv_path = tmp_path / "bools.csv"
        df.to_csv(csv_path, index=False)
        result = summarize_table(str(csv_path))
        assert "True:" in result
        assert "False:" in result


class TestGenerateSyntheticData:
    def test_invalid_backend_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown backend"):
            generate_synthetic_data(
                file_path=ADULT_CSV_STR,
                num_rows=10,
                backend="notabackend",
            )

    @patch("table_agent_toolkit.synthetic_generation.generate_ctgan")
    def test_ctgan_backend_called(self, mock_ctgan, tmp_path):
        fake_synthetic = pd.read_csv(ADULT_CSV).head(10)
        mock_ctgan.return_value = fake_synthetic
        out = tmp_path / "out.csv"

        result = generate_synthetic_data(
            file_path=ADULT_CSV_STR,
            num_rows=10,
            backend="ctgan",
            output_path=str(out),
        )

        mock_ctgan.assert_called_once()
        assert "10" in result
        assert "ctgan" in result
        assert out.exists()

    @patch("table_agent_toolkit.synthetic_generation.generate_tvae")
    def test_tvae_backend_called(self, mock_tvae, tmp_path):
        fake_synthetic = pd.read_csv(ADULT_CSV).head(5)
        mock_tvae.return_value = fake_synthetic
        out = tmp_path / "out.csv"

        result = generate_synthetic_data(
            file_path=ADULT_CSV_STR,
            num_rows=5,
            backend="tvae",
            output_path=str(out),
        )

        mock_tvae.assert_called_once()
        assert "5" in result
        assert "tvae" in result

    @patch("table_agent_toolkit.synthetic_generation.generate_tabicl")
    def test_tabicl_backend_called(self, mock_tabicl, tmp_path):
        fake_synthetic = pd.read_csv(ADULT_CSV).head(8)
        mock_tabicl.return_value = fake_synthetic
        out = tmp_path / "out.csv"

        result = generate_synthetic_data(
            file_path=ADULT_CSV_STR,
            num_rows=8,
            backend="tabicl",
            output_path=str(out),
        )

        mock_tabicl.assert_called_once()
        assert "8" in result
        assert "tabicl" in result

    @patch("table_agent_toolkit.synthetic_generation.generate_ctgan")
    def test_default_output_path(self, mock_ctgan):
        fake_synthetic = pd.read_csv(ADULT_CSV).head(5)
        mock_ctgan.return_value = fake_synthetic

        result = generate_synthetic_data(
            file_path=ADULT_CSV_STR,
            num_rows=5,
            backend="ctgan",
        )

        expected_out = ADULT_CSV.parent / "adult-sampled_synthetic.csv"
        assert str(expected_out) in result
        # clean up
        expected_out.unlink(missing_ok=True)

    @patch("table_agent_toolkit.synthetic_generation.generate_ctgan")
    def test_result_contains_discrete_columns_info(self, mock_ctgan, tmp_path):
        fake_synthetic = pd.read_csv(ADULT_CSV).head(5)
        mock_ctgan.return_value = fake_synthetic
        out = tmp_path / "out.csv"

        result = generate_synthetic_data(
            file_path=ADULT_CSV_STR,
            num_rows=5,
            backend="ctgan",
            output_path=str(out),
        )

        assert "Discrete columns detected" in result


class TestCategoricalHandling:
    """Integration tests: MCP endpoints must handle categorical/object columns
    end-to-end (preprocess → generate → inverse-transform)."""

    def _make_csv(self, tmp_path, df: "pd.DataFrame") -> str:
        p = tmp_path / "input.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_nonprivate_inverse_transforms_categorical_dtype(self, tmp_path):
        """CategoricalDtype columns should come back as strings in the output."""
        n = 20
        df = pd.DataFrame({
            "num": range(n),
            "cat": pd.Categorical(["a", "b"] * 10),
        })
        csv_path = self._make_csv(tmp_path, df)
        out_path = tmp_path / "output.csv"

        def passthrough(data, discrete_cols, num_rows):
            return data.head(num_rows).reset_index(drop=True)

        with patch("table_agent_toolkit.synthetic_generation.generate_ctgan", passthrough):
            generate_synthetic_data(
                file_path=csv_path,
                num_rows=5,
                backend="ctgan",
                output_path=str(out_path),
            )

        output = pd.read_csv(out_path)
        assert pd.api.types.is_string_dtype(output["cat"])
        assert set(output["cat"].dropna()).issubset({"a", "b"})

    def test_nonprivate_inverse_transforms_object_column(self, tmp_path):
        """Plain object/string columns should survive the round-trip as strings."""
        n = 20
        df = pd.DataFrame({
            "num": range(n),
            "city": (["NYC", "LA", "Chicago"] * 7)[:n],
        })
        csv_path = self._make_csv(tmp_path, df)
        out_path = tmp_path / "output.csv"

        def passthrough(data, discrete_cols, num_rows):
            return data.head(num_rows).reset_index(drop=True)

        with patch("table_agent_toolkit.synthetic_generation.generate_ctgan", passthrough):
            generate_synthetic_data(
                file_path=csv_path,
                num_rows=5,
                backend="ctgan",
                output_path=str(out_path),
            )

        output = pd.read_csv(out_path)
        assert pd.api.types.is_string_dtype(output["city"])
        assert set(output["city"].dropna()).issubset({"NYC", "LA", "Chicago"})

    def test_nonprivate_drops_high_cardinality_and_reports(self, tmp_path):
        """Columns with > 0.5*N unique values must be absent from the output
        and mentioned in the result message."""
        n = 20
        df = pd.DataFrame({
            "num": range(n),
            "low_card": ["a", "b"] * 10,
            "high_card": [f"val_{i}" for i in range(n)],  # all unique → dropped
        })
        csv_path = self._make_csv(tmp_path, df)
        out_path = tmp_path / "output.csv"

        def passthrough(data, discrete_cols, num_rows):
            return data.head(num_rows).reset_index(drop=True)

        with patch("table_agent_toolkit.synthetic_generation.generate_ctgan", passthrough):
            result = generate_synthetic_data(
                file_path=csv_path,
                num_rows=5,
                backend="ctgan",
                output_path=str(out_path),
            )

        output = pd.read_csv(out_path)
        assert "high_card" not in output.columns
        assert "high_card" in result

    def test_nonprivate_generator_receives_encoded_data(self, tmp_path):
        """The generator should receive integer-encoded columns, not raw strings."""
        n = 20
        df = pd.DataFrame({
            "num": range(n),
            "cat": (["x", "y", "z"] * 7)[:n],
        })
        csv_path = self._make_csv(tmp_path, df)
        out_path = tmp_path / "output.csv"

        received = {}

        def capturing_generator(data, discrete_cols, num_rows):
            received["data"] = data.copy()
            received["discrete_cols"] = discrete_cols
            return data.head(num_rows).reset_index(drop=True)

        with patch("table_agent_toolkit.synthetic_generation.generate_ctgan", capturing_generator):
            generate_synthetic_data(
                file_path=csv_path,
                num_rows=5,
                backend="ctgan",
                output_path=str(out_path),
            )

        assert "cat" in received["discrete_cols"]
        # encoded values should be integers, not strings
        assert pd.api.types.is_integer_dtype(received["data"]["cat"])

