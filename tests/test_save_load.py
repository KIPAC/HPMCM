"""Round-trip tests for Match.save() / WcsMatch.load() / ShearMatch.load().

Requires the external test data (setup_data fixture downloads it).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hpmcm

DATADIR = "examples/test_data"
SHEAR_ST = "0p01"
SHEAR = 0.01
CATALOG_TYPE = "wmom"
TRACT = 10463

REF_DIR = (37.9, 7.0)
REGION_SIZE = (0.375, 0.375)
PIXEL_SIZE = 0.5 / 3600.0


def _assert_stats_equal(
    orig: pd.DataFrame, loaded: pd.DataFrame, id_col: str
) -> None:
    """Sort both DataFrames by id_col and compare with floating-point tolerance."""
    a = orig.sort_values(id_col).reset_index(drop=True)
    b = loaded.sort_values(id_col).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-5, atol=1e-10)


# ---------------------------------------------------------------------------
# Shared fixture: run WcsMatch analysis once, save to disk, reuse across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", name="wcs_save_dir")
def wcsSaveDir(setup_data: int, tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """Run a 1×2 WcsMatch analysis once per module, save, and return (path, stats)."""
    assert setup_data == 0

    source_files = sorted(
        glob.glob(os.path.join(DATADIR, f"shear_*_{SHEAR_ST}_cleaned_{TRACT}_ns.pq"))
    )
    source_files.append(os.path.join(DATADIR, f"object_{TRACT}.pq"))
    source_files.reverse()
    catalog_ids = list(np.arange(2))

    matcher = hpmcm.WcsMatch.create(REF_DIR, REGION_SIZE, pixel_size=PIXEL_SIZE)
    matcher.reduceData(source_files[:2], catalog_ids)
    matcher.analysisLoop(range(1, 2), range(1, 2))
    stats_orig = matcher.extractStats()

    save_dir = tmp_path_factory.mktemp("wcs_save")
    matcher.save(save_dir)
    return save_dir, stats_orig


# ---------------------------------------------------------------------------
# WcsMatch save / load
# ---------------------------------------------------------------------------


def testWcsMatchSaveLoad(wcs_save_dir: tuple) -> None:
    """Reload a saved WcsMatch; extractStats() must be unchanged."""
    save_dir, stats_orig = wcs_save_dir

    assert (save_dir / "geometry.json").exists()
    assert (save_dir / "object_stats.parquet").exists()
    assert (save_dir / "cluster_stats.parquet").exists()
    assert (save_dir / "cat_0.parquet").exists()
    assert (save_dir / "cat_1.parquet").exists()

    loaded = hpmcm.WcsMatch.load(save_dir)
    stats_loaded = loaded.extractStats()

    _assert_stats_equal(stats_orig["object_stats"], stats_loaded["object_stats"], "object_id")
    _assert_stats_equal(stats_orig["cluster_stats"], stats_loaded["cluster_stats"], "cluster_id")
    # Association row counts must agree (distances round-trip through arcsec conversion)
    assert len(stats_orig["object_assoc"]) == len(stats_loaded["object_assoc"])
    assert len(stats_orig["cluster_assoc"]) == len(stats_loaded["cluster_assoc"])


def testWcsMatchLoadWithRange(wcs_save_dir: tuple) -> None:
    """Loading with x_range / y_range restricts which cells are reconstructed."""
    save_dir, _ = wcs_save_dir

    # The saved data has only cell (ix=1, iy=1). An out-of-range load gives nothing.
    loaded_empty = hpmcm.WcsMatch.load(save_dir, x_range=(2, 3), y_range=(1, 1))
    assert len(loaded_empty.cell_dict) == 0

    # An in-range load gives the cell with the correct coordinates.
    loaded = hpmcm.WcsMatch.load(save_dir, x_range=(1, 1), y_range=(1, 1))
    assert loaded.cell_dict, "expected at least one cell after load"
    for cell_idx in loaded.cell_dict:
        ix, iy = loaded.getCellXY(cell_idx)
        assert ix == 1
        assert iy == 1


# ---------------------------------------------------------------------------
# ShearMatch save / load
# ---------------------------------------------------------------------------


def testShearMatchSaveLoad(setup_data: int, tmp_path: Path) -> None:
    """Save a ShearMatch then reload it; extractStats() must be unchanged."""
    assert setup_data == 0

    source_files = sorted(
        glob.glob(
            os.path.join(DATADIR, f"shear_{CATALOG_TYPE}_{SHEAR_ST}_uncleaned_{TRACT}_*.pq")
        )
    )
    source_files.reverse()
    catalog_ids = list(np.arange(len(source_files)))

    geom = hpmcm.shear_utils.ShearCellGeometry(ref_dir=REF_DIR)
    matcher = hpmcm.ShearMatch.createShearMatch(geometry=geom, deshear=-SHEAR)
    matcher.reduceData(source_files, catalog_ids)
    matcher.analysisLoop(range(60, 61), range(175, 176))
    stats_orig = matcher.extractStats()

    save_dir = tmp_path / "shear_save"
    matcher.save(save_dir)

    assert (save_dir / "geometry.json").exists()
    assert (save_dir / "object_stats.parquet").exists()

    loaded = hpmcm.ShearMatch.load(save_dir)
    stats_loaded = loaded.extractStats()

    _assert_stats_equal(stats_orig["object_stats"], stats_loaded["object_stats"], "object_id")
    _assert_stats_equal(stats_orig["cluster_stats"], stats_loaded["cluster_stats"], "cluster_id")
    assert len(stats_orig["object_assoc"]) == len(stats_loaded["object_assoc"])
    assert len(stats_orig["cluster_assoc"]) == len(stats_loaded["cluster_assoc"])


# ---------------------------------------------------------------------------
# ShearMatch construction guard
# ---------------------------------------------------------------------------


def testShearMatchRequiresRefDir() -> None:
    """ShearMatch must raise RuntimeError when geometry.ref_dir is None."""
    geom = hpmcm.shear_utils.ShearCellGeometry(ref_dir=None)
    with pytest.raises(RuntimeError, match="ref_dir"):
        hpmcm.ShearMatch.createShearMatch(geometry=geom)
