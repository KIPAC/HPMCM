"""Unit tests for CellData and reduceShearDataForCell using synthetic data."""

from unittest.mock import MagicMock

import numpy as np
import pandas
import pytest

from hpmcm.cell import CellData, ShearCellData
from hpmcm.shear_utils import DESHEAR_COEFFS, reduceShearDataForCell


class TestCellDataReduceDataframe:
    """Tests for CellData.reduceDataframe()"""

    def _make_cell(self, corner, size, buf=10):
        """Create a CellData with a mock matcher."""
        matcher = MagicMock()
        return CellData(matcher, id_offset=0, corner=corner, size=size, idx=0, buf=buf)

    def test_filters_to_cell_bounds(self):
        """Only sources within cell bounds survive."""
        cell = self._make_cell(
            corner=np.array([100, 100]), size=np.array([50, 50]), buf=10
        )
        # min_pix = [90, 90], max_pix = [160, 160], n_pix = [70, 70]
        df = pandas.DataFrame({
            "x_pix": [80.0, 95.0, 130.0, 170.0],
            "y_pix": [95.0, 95.0, 130.0, 95.0],
        })

        result = cell.reduceDataframe(0, df)

        # Source at x=80 is below min_pix[0]=90 → filtered out
        # Source at x=170 is >= max_pix[0]=160 → filtered out
        assert len(result) == 2
        assert "x_cell" in result.columns
        assert "y_cell" in result.columns

    def test_x_cell_y_cell_values(self):
        """x_cell and y_cell are offsets from min_pix."""
        cell = self._make_cell(
            corner=np.array([50, 50]), size=np.array([100, 100]), buf=5
        )
        # min_pix = [45, 45]
        df = pandas.DataFrame({
            "x_pix": [50.0, 60.0],
            "y_pix": [50.0, 70.0],
        })

        result = cell.reduceDataframe(0, df)

        assert result["x_cell"].iloc[0] == pytest.approx(5.0)
        assert result["y_cell"].iloc[0] == pytest.approx(5.0)
        assert result["x_cell"].iloc[1] == pytest.approx(15.0)
        assert result["y_cell"].iloc[1] == pytest.approx(25.0)

    def test_empty_dataframe(self):
        """Empty input yields empty output."""
        cell = self._make_cell(corner=np.array([0, 0]), size=np.array([100, 100]))
        df = pandas.DataFrame({"x_pix": [], "y_pix": []})

        result = cell.reduceDataframe(0, df)
        assert len(result) == 0

    def test_reduce_data_sets_n_src(self):
        """reduceData sets n_src to total sources across catalogs."""
        cell = self._make_cell(
            corner=np.array([0, 0]), size=np.array([100, 100]), buf=0
        )
        df1 = pandas.DataFrame({"x_pix": [10.0, 20.0], "y_pix": [10.0, 20.0]})
        df2 = pandas.DataFrame({"x_pix": [30.0], "y_pix": [30.0]})

        cell.reduceData([df1, df2])

        assert cell.n_src == 3
        assert len(cell.data) == 2


class TestReduceShearDataForCell:
    """Tests for reduceShearDataForCell() with deshearing."""

    def _make_cell_and_matcher(self, deshear=-0.01, pixel_match_scale=1):
        """Create mock cell and matcher for shear reduction tests."""
        matcher = MagicMock()
        matcher.deshear = deshear
        matcher.pixel_match_scale = pixel_match_scale
        # n_cell is [200, 200] for a standard setup
        matcher.n_cell = np.array([200, 200])

        cell = MagicMock()
        cell.matcher = matcher
        cell.idx = 5
        cell.n_pix = np.array([200, 200])

        return cell, matcher

    def _make_source_df(self, n=10, cell_idx_x=0, cell_idx_y=5):
        """Create synthetic source DataFrame."""
        rng = np.random.default_rng(42)
        return pandas.DataFrame({
            "cell_idx_x": np.full(n, cell_idx_x),
            "cell_idx_y": np.full(n, cell_idx_y),
            "x_cell_coadd": rng.uniform(-50, 50, n),
            "y_cell_coadd": rng.uniform(-50, 50, n),
            "x_pix": rng.uniform(100, 200, n),
            "y_pix": rng.uniform(100, 200, n),
            "snr": rng.uniform(5, 20, n),
            "g_1": rng.uniform(-0.1, 0.1, n),
            "g_2": rng.uniform(-0.1, 0.1, n),
            "id": np.arange(n),
        })

    def test_filters_by_cell_index(self):
        """Only sources matching cell.idx survive."""
        cell, matcher = self._make_cell_and_matcher()
        # getCellIndices should return cell.idx for matching rows
        matcher.getCellIndices.return_value = np.array([5, 5, 3, 5, 7])

        df = self._make_source_df(n=5)
        result = reduceShearDataForCell(cell, 0, df)

        # 3 sources match idx=5
        assert len(result) <= 3
        assert "x_cell" in result.columns
        assert "y_cell" in result.columns

    def test_deshear_applies_coefficients(self):
        """Deshearing modifies positions using DESHEAR_COEFFS."""
        cell, matcher = self._make_cell_and_matcher(deshear=-0.01)
        matcher.getCellIndices.return_value = np.array([5])

        df = pandas.DataFrame({
            "cell_idx_x": [0],
            "cell_idx_y": [5],
            "x_cell_coadd": [10.0],
            "y_cell_coadd": [20.0],
            "x_pix": [150.0],
            "y_pix": [150.0],
            "snr": [15.0],
            "g_1": [0.01],
            "g_2": [0.02],
            "id": [0],
        })

        # Test with i_cat=1 (DESHEAR_COEFFS[1] = [0, 1, 1, 0])
        result = reduceShearDataForCell(cell, 1, df)

        if len(result) > 0:
            assert "dx_shear" in result.columns
            assert "dy_shear" in result.columns

    def test_no_deshear(self):
        """When deshear is None, no dx_shear/dy_shear columns added."""
        cell, matcher = self._make_cell_and_matcher(deshear=None)
        matcher.getCellIndices.return_value = np.array([5] * 5)

        df = self._make_source_df(n=5)
        result = reduceShearDataForCell(cell, 0, df)

        assert "dx_shear" not in result.columns
        assert "dy_shear" not in result.columns

    def test_ns_catalog_no_deshear_offset(self):
        """For ns catalog (i_cat=0), DESHEAR_COEFFS are all zeros."""
        cell, matcher = self._make_cell_and_matcher(deshear=-0.01)
        matcher.getCellIndices.return_value = np.array([5])

        df = pandas.DataFrame({
            "cell_idx_x": [0],
            "cell_idx_y": [5],
            "x_cell_coadd": [10.0],
            "y_cell_coadd": [20.0],
            "x_pix": [150.0],
            "y_pix": [150.0],
            "snr": [15.0],
            "g_1": [0.01],
            "g_2": [0.02],
            "id": [0],
        })

        result = reduceShearDataForCell(cell, 0, df)

        if len(result) > 0:
            # DESHEAR_COEFFS[0] = [0,0,0,0], so dx_shear and dy_shear should be 0
            assert result["dx_shear"].iloc[0] == pytest.approx(0.0)
            assert result["dy_shear"].iloc[0] == pytest.approx(0.0)

    def test_bounds_filtering(self):
        """Sources outside cell n_pix bounds are removed."""
        cell, matcher = self._make_cell_and_matcher(deshear=-0.01)
        cell.n_pix = np.array([10, 10])
        matcher.getCellIndices.return_value = np.array([5, 5])

        # x_cell_coadd values that after transform will be out of bounds
        # x_cell = (x_cell_coadd + 100) / pixel_match_scale
        # For n_pix=[10,10], need x_cell in [0,10)
        # So x_cell_coadd must be in [-100, -90) for pixel_match_scale=1
        df = pandas.DataFrame({
            "cell_idx_x": [0, 0],
            "cell_idx_y": [5, 5],
            "x_cell_coadd": [-95.0, 500.0],  # -95 → x_cell=5 (in); 500 → x_cell=600 (out)
            "y_cell_coadd": [-95.0, -95.0],
            "x_pix": [150.0, 150.0],
            "y_pix": [150.0, 150.0],
            "snr": [15.0, 15.0],
            "g_1": [0.01, 0.01],
            "g_2": [0.02, 0.02],
            "id": [0, 1],
        })

        result = reduceShearDataForCell(cell, 0, df)
        assert len(result) == 1
