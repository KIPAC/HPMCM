"""Unit tests for shear_utils functions using synthetic data."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas
import pytest

from hpmcm import shear_utils
from hpmcm.shear_utils import SHEAR_NAMES


class TestShearStats:
    """Tests for shearStats()"""

    def _make_df(self, i_cats, g1_vals, g2_vals):
        """Helper to build a small DataFrame with i_cat, g_1, g_2 columns."""
        return pandas.DataFrame(
            {"i_cat": i_cats, "g_1": g1_vals, "g_2": g2_vals}
        )

    def test_good_match(self):
        """One source per catalog → good=True, deltas computed."""
        df = self._make_df(
            i_cats=[0, 1, 2, 3, 4],
            g1_vals=[0.0, 0.02, -0.02, 0.01, -0.01],
            g2_vals=[0.0, 0.03, -0.03, 0.005, -0.005],
        )
        result = shear_utils.shearStats(df)

        assert result["good"] is True
        for name in SHEAR_NAMES:
            assert result[f"n_{name}"] == 1

        assert result["delta_g_1_1"] == pytest.approx(0.01 - (-0.01))
        assert result["delta_g_2_2"] == pytest.approx(0.03 - (-0.03))
        assert result["delta_g_1_2"] == pytest.approx(0.02 - (-0.02))
        assert result["delta_g_2_1"] == pytest.approx(0.005 - (-0.005))

    def test_missing_catalog(self):
        """Missing a catalog → good=False, deltas are NaN."""
        df = self._make_df(
            i_cats=[0, 1, 2, 3],
            g1_vals=[0.0, 0.02, -0.02, 0.01],
            g2_vals=[0.0, 0.03, -0.03, 0.005],
        )
        result = shear_utils.shearStats(df)

        assert result["good"] is False
        assert result["n_1m"] == 0
        assert np.isnan(result["g_1_1m"])
        assert np.isnan(result["g_2_1m"])
        assert np.isnan(result["delta_g_1_1"])
        assert np.isnan(result["delta_g_2_2"])

    def test_duplicate_in_catalog(self):
        """Multiple sources in one catalog → good=False, g values are mean."""
        df = self._make_df(
            i_cats=[0, 0, 1, 2, 3, 4],
            g1_vals=[0.01, 0.03, 0.02, -0.02, 0.01, -0.01],
            g2_vals=[0.0, 0.0, 0.03, -0.03, 0.005, -0.005],
        )
        result = shear_utils.shearStats(df)

        assert result["good"] is False
        assert result["n_ns"] == 2
        assert result["g_1_ns"] == pytest.approx(0.02)
        assert np.isnan(result["delta_g_1_1"])

    def test_empty_dataframe(self):
        """Empty DataFrame → good=False, all NaN."""
        df = self._make_df(i_cats=[], g1_vals=[], g2_vals=[])
        result = shear_utils.shearStats(df)

        assert result["good"] is False
        for name in SHEAR_NAMES:
            assert result[f"n_{name}"] == 0
            assert np.isnan(result[f"g_1_{name}"])
            assert np.isnan(result[f"g_2_{name}"])


class TestMergeShearReports:
    """Tests for mergeShearReports()"""

    def test_merge_two_reports(self, tmp_path):
        """Merge two ShearData pickles into a parquet file."""
        dict1 = {"shear": 0.01, "n_good": 100, "effic": 0.95}
        dict2 = {"shear": 0.02, "n_good": 200, "effic": 0.90}

        mock_sd1 = MagicMock()
        mock_sd1.toDict.return_value = dict1
        mock_sd2 = MagicMock()
        mock_sd2.toDict.return_value = dict2

        input_files = [str(tmp_path / "a.pkl"), str(tmp_path / "b.pkl")]
        output_file = str(tmp_path / "merged.pq")

        with patch.object(
            shear_utils.ShearData, "load", side_effect=[mock_sd1, mock_sd2]
        ):
            shear_utils.mergeShearReports(input_files, output_file)

        result = pandas.read_parquet(output_file)
        assert len(result) == 2
        assert list(result["shear"]) == [0.01, 0.02]
        assert list(result["n_good"]) == [100, 200]

    def test_merge_single_report(self, tmp_path):
        """Merge a single ShearData pickle."""
        dict1 = {"shear": 0.01, "n_good": 50}

        mock_sd = MagicMock()
        mock_sd.toDict.return_value = dict1

        input_files = [str(tmp_path / "a.pkl")]
        output_file = str(tmp_path / "merged.pq")

        with patch.object(shear_utils.ShearData, "load", return_value=mock_sd):
            shear_utils.mergeShearReports(input_files, output_file)

        result = pandas.read_parquet(output_file)
        assert len(result) == 1
        assert result["shear"].iloc[0] == 0.01


class TestMakeMatchedShearSourceCatalogs:
    """Tests for makeMatchedShearSourceCatalogs()"""

    def _build_mock_data(self):
        """Build synthetic tables for tables_io.read to return."""
        object_stats = pandas.DataFrame({
            "object_id": [1, 2, 3],
            "ra": [10.0, 20.0, 30.0],
            "dec": [1.0, 2.0, 3.0],
        })
        object_shear = pandas.DataFrame({
            "object_id": [1, 2, 3],
            "good": [True, True, False],
        })
        object_assoc = pandas.DataFrame({
            "object_id": [1, 1, 2, 2, 3, 3, 1, 2, 3, 1],
            "catalog_id": [0, 1, 0, 1, 0, 1, 2, 2, 2, 3],
            "source_id": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        })

        match_tables = {
            "object_stats": object_stats,
            "object_assoc": object_assoc,
            "object_shear": object_shear,
        }

        # Source tables keyed by SHEAR_NAMES
        source_tables = {}
        for i, name in enumerate(SHEAR_NAMES):
            source_tables[name] = pandas.DataFrame({
                "id": list(range(10 + i * 3, 10 + i * 3 + 5)),
                "g_1": np.random.default_rng(i).uniform(-0.1, 0.1, 5),
                "g_2": np.random.default_rng(i + 10).uniform(-0.1, 0.1, 5),
            })

        return match_tables, source_tables

    def test_basic_structure(self):
        """Verify output has expected keys and ns is present."""
        match_tables, source_tables = self._build_mock_data()

        def mock_read(path, keys=None):
            if keys is not None and "object_stats" in keys:
                return match_tables
            return source_tables

        with patch("hpmcm.shear_utils.tables_io.read", side_effect=mock_read):
            result = shear_utils.makeMatchedShearSourceCatalogs("src", "match")

        assert "ns" in result
        for name in SHEAR_NAMES:
            assert name in result

    def test_ns_processed_first(self):
        """Verify non-ns catalogs have columns from ns via left join."""
        match_tables, source_tables = self._build_mock_data()

        def mock_read(path, keys=None):
            if keys is not None and "object_stats" in keys:
                return match_tables
            return source_tables

        with patch("hpmcm.shear_utils.tables_io.read", side_effect=mock_read):
            result = shear_utils.makeMatchedShearSourceCatalogs("src", "match")

        # Non-ns catalogs should have been left-joined with ns
        if len(result["2p"]) > 0:
            assert "object_id" in result["2p"].columns
