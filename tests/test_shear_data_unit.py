"""Unit tests for shear_data classes using synthetic data."""

import os

import numpy as np
import pandas
import pytest

from hpmcm.shear_data import (
    ShearData,
    ShearHistogramStats,
    ShearHistograms,
    ShearProfileHistogramStats,
)


class TestShearHistogramStats:
    """Tests for ShearHistogramStats"""

    def test_basic_stats(self):
        """Verify mean, std, error for a known distribution."""
        bin_centers = np.array([-1.0, 0.0, 1.0])
        weights = np.array([1.0, 2.0, 1.0])

        stats = ShearHistogramStats(weights, bin_centers)

        assert stats.w == pytest.approx(4.0)
        assert stats.mean == pytest.approx(0.0)
        assert stats.std == pytest.approx(np.sqrt(0.5))
        assert stats.error == pytest.approx(np.sqrt(0.5) / 2.0)

    def test_asymmetric(self):
        """Verify mean for asymmetric weights."""
        bin_centers = np.array([0.0, 1.0])
        weights = np.array([1.0, 3.0])

        stats = ShearHistogramStats(weights, bin_centers)

        assert stats.w == pytest.approx(4.0)
        assert stats.mean == pytest.approx(0.75)


class TestShearProfileHistogramStats:
    """Tests for ShearProfileHistogramStats"""

    def test_basic_2d(self):
        """Verify stats from a 2D histogram."""
        # 3 x-bins, 4 y-bins
        weights = np.array([
            [1.0, 2.0, 2.0, 1.0],
            [0.0, 0.0, 4.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        x_edges = np.array([0.0, 1.0, 2.0, 3.0])
        y_edges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        hist_2d = (weights, x_edges, y_edges)
        stats = ShearProfileHistogramStats(hist_2d)

        y_centers = np.array([-1.5, -0.5, 0.5, 1.5])

        # Row 0: w=6, mean = (1*-1.5 + 2*-0.5 + 2*0.5 + 1*1.5)/6 = 0.0
        assert stats.w[0] == pytest.approx(6.0)
        assert stats.mean[0] == pytest.approx(0.0)

        # Row 1: w=4, mean = 4*0.5/4 = 0.5
        assert stats.w[1] == pytest.approx(4.0)
        assert stats.mean[1] == pytest.approx(0.5)

        # Row 2: uniform weights, mean = average of centers
        assert stats.w[2] == pytest.approx(4.0)
        assert stats.mean[2] == pytest.approx(np.mean(y_centers))


class TestShearDataSaveLoad:
    """Tests for ShearData pickle round-trip"""

    def _make_shear_data(self):
        """Create a minimal synthetic ShearData."""
        n = 20
        rng = np.random.default_rng(42)

        # Stats table columns
        stats_table = pandas.DataFrame({
            "x_cent": rng.uniform(50, 150, n),
            "y_cent": rng.uniform(50, 150, n),
            "snr": rng.uniform(5, 20, n),
        })

        # Shear table columns
        shear_cols = {"good": rng.choice([True, False], n, p=[0.8, 0.2])}
        for name in ["ns", "2p", "2m", "1p", "1m"]:
            shear_cols[f"n_{name}"] = np.ones(n)
            shear_cols[f"g_1_{name}"] = rng.uniform(-0.05, 0.05, n)
            shear_cols[f"g_2_{name}"] = rng.uniform(-0.05, 0.05, n)
        for i in [1, 2]:
            for j in [1, 2]:
                shear_cols[f"delta_g_{i}_{j}"] = rng.uniform(-0.01, 0.01, n)
        shear_table = pandas.DataFrame(shear_cols)

        return ShearData(shear_table, stats_table, 0.01, "wmom", 10463, snr_cut=7.5)

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load ShearData, verify key attributes survive."""
        sd = self._make_shear_data()
        filepath = str(tmp_path / "test_shear.pkl")

        sd.save(filepath)
        assert os.path.exists(filepath)

        loaded = ShearData.load(filepath)
        assert loaded.shear == sd.shear
        assert loaded.cat_type == sd.cat_type
        assert loaded.tract == sd.tract
        assert loaded.n_objects == sd.n_objects
        assert loaded.n_good == sd.n_good
        assert loaded.effic == pytest.approx(sd.effic)

    def test_to_dict(self):
        """Verify toDict returns expected keys and types."""
        sd = self._make_shear_data()
        d = sd.toDict()

        assert "shear" in d
        assert d["shear"] == 0.01
        assert "n_objects" in d
        assert "efficiency" in d
        assert "mc_delta_g_1_1" in d
        assert "mc_delta_g_1_1_std" in d
        assert "mc_delta_g_1_1_err" in d
        assert "mc_delta_g_1_1_inv_var" in d
        assert isinstance(d["mc_delta_g_1_1"], float)


class TestShearHistogramsPgauss:
    """Test that pgauss uses wider bin range."""

    def _make_good_bad(self, n=50):
        rng = np.random.default_rng(99)
        cols = {}
        cols["delta_g_1_1"] = rng.uniform(-0.5, 0.5, n)
        cols["delta_g_2_2"] = rng.uniform(-0.5, 0.5, n)
        cols["delta_g_1_2"] = rng.uniform(-0.5, 0.5, n)
        cols["delta_g_2_1"] = rng.uniform(-0.5, 0.5, n)
        for name in ["1p", "1m", "2p", "2m"]:
            cols[f"g_1_{name}"] = rng.uniform(-0.5, 0.5, n)
            cols[f"g_2_{name}"] = rng.uniform(-0.5, 0.5, n)
            cols[f"n_{name}"] = np.ones(n)
        return pandas.DataFrame(cols)

    def test_pgauss_bins(self):
        """pgauss uses [-10, 10] range."""
        good = self._make_good_bad()
        bad = self._make_good_bad()
        hists = ShearHistograms(good, bad, "pgauss")

        assert hists.bin_edges[0] == pytest.approx(-10.0)
        assert hists.bin_edges[-1] == pytest.approx(10.0)
        assert len(hists.bin_edges) == 20001

    def test_wmom_bins(self):
        """Non-pgauss uses [-1, 1] range."""
        good = self._make_good_bad()
        bad = self._make_good_bad()
        hists = ShearHistograms(good, bad, "wmom")

        assert hists.bin_edges[0] == pytest.approx(-1.0)
        assert hists.bin_edges[-1] == pytest.approx(1.0)
        assert len(hists.bin_edges) == 2001
