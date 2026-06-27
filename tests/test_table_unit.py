"""Unit tests for table.py using synthetic parquet files."""

import numpy as np
import pandas
import pytest

from hpmcm.table import TableColumnInfo, TableInterface


class SampleTable(TableInterface):
    """A concrete table subclass for testing."""

    _schema = TableInterface._schema.copy()
    _schema.update(
        x=TableColumnInfo(float, "X coordinate"),
        y=TableColumnInfo(float, "Y coordinate"),
        val=TableColumnInfo(int, "Some value"),
    )


class SampleTableInterface:
    """Tests for TableInterface"""

    def test_validate_success(self):
        """Valid data passes validation."""
        SampleTable.validate(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            val=np.array([5, 6]),
        )

    def test_to_pandas(self):
        """toPandas creates a DataFrame with correct columns."""
        df = SampleTable.toPandas(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            val=np.array([10, 20]),
        )
        assert list(df.columns) == ["x", "y", "val"]
        assert len(df) == 2
        assert df["val"].iloc[1] == 20

    def test_read_parquet(self, tmp_path):
        """read() loads the correct columns from a parquet file."""
        df = pandas.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "val": [7, 8, 9],
            "extra_col": [10, 11, 12],
            "another": [0.1, 0.2, 0.3],
        })
        filepath = str(tmp_path / "test.parquet")
        df.to_parquet(filepath)

        result = SampleTable.read(filepath, extra_cols=["extra_col"])

        assert "x" in result.columns
        assert "y" in result.columns
        assert "val" in result.columns
        assert "extra_col" in result.columns
        assert "another" not in result.columns
        assert len(result) == 3

    def test_read_no_extra_cols(self, tmp_path):
        """read() with no extra columns returns only schema columns."""
        df = pandas.DataFrame({
            "x": [1.0],
            "y": [2.0],
            "val": [3],
            "extra": [99],
        })
        filepath = str(tmp_path / "test.parquet")
        df.to_parquet(filepath)

        result = SampleTable.read(filepath, extra_cols=[])

        assert "extra" not in result.columns
        assert set(result.columns) == {"x", "y", "val"}

    def test_empty_numpy_dict(self):
        """emtpyNumpyDict creates zero-filled arrays of correct types."""
        d = SampleTable.emtpyNumpyDict(5)

        assert d["x"].shape == (5,)
        assert d["x"].dtype == float
        assert d["val"].dtype == int
        assert np.all(d["x"] == 0.0)

    def test_data_property(self):
        """TableInterface wraps a DataFrame accessible via .data."""
        t = SampleTable(
            x=np.array([1.0]),
            y=np.array([2.0]),
            val=np.array([3]),
        )
        assert isinstance(t.data, pandas.DataFrame)
        assert t.data["x"].iloc[0] == 1.0

    def test_construct_from_dataframe(self):
        """TableInterface can be constructed from an existing DataFrame."""
        df = pandas.DataFrame({"x": [1.0], "y": [2.0], "val": [3]})
        t = SampleTable(df=df)
        assert t.data is df


class SampleTableColumnInfo:
    """Tests for TableColumnInfo"""

    def test_repr(self):
        """Repr shows type and message."""
        info = TableColumnInfo(float, "Some description")
        r = repr(info)
        assert "float" in r
        assert "Some description" in r

    def test_validate_correct_type(self):
        """Validates matching dtype."""
        info = TableColumnInfo(float, "desc")
        info.validate(np.array([1.0, 2.0]))

    def test_validate_wrong_type(self):
        """Fails on mismatched dtype."""
        info = TableColumnInfo(int, "desc")
        with pytest.raises(AssertionError):
            info.validate(np.array([1.0, 2.0]))
