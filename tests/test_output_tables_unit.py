"""Unit tests for output_tables joined-table helpers.

Uses synthetic in-memory data; no internet access required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpmcm.output_tables import (
    _resolve_cols,
    buildJoinedClusterTable,
    buildJoinedObjectTable,
)


@pytest.fixture
def source_parquets(tmp_path):
    """Write two tiny source catalog parquets and return (paths, ids, cat0, cat1)."""
    cat0 = pd.DataFrame(
        {"id": [10, 11, 12], "flux": [1.0, 2.0, 3.0], "size": [0.5, 0.6, 0.7]}
    )
    cat1 = pd.DataFrame(
        {"id": [20, 21], "flux": [4.0, 5.0], "color": [0.1, 0.2]}
    )
    f0 = str(tmp_path / "cat0.parquet")
    f1 = str(tmp_path / "cat1.parquet")
    cat0.to_parquet(f0)
    cat1.to_parquet(f1)
    return [f0, f1], [0, 1], cat0, cat1


@pytest.fixture
def object_tables():
    """Minimal object_stats and object_assoc DataFrames for testing."""
    object_stats = pd.DataFrame(
        {
            "object_id": [100, 101, 102],
            "ra": [0.1, 0.2, 0.3],
            "dec": [0.1, 0.2, 0.3],
            "catalog_mask": [3, 1, 2],
        }
    )
    # object 100 matches sources in both catalogs; 101 only cat 0; 102 only cat 1
    object_assoc = pd.DataFrame(
        {
            "object_id": [100, 100, 101, 102],
            "source_id": [10, 20, 11, 21],
            "catalog_id": [0, 1, 0, 1],
            "distance": [0.1, 0.2, 0.15, 0.05],
            "cell_idx": [0, 0, 0, 0],
        }
    )
    return object_stats, object_assoc


# ---------------------------------------------------------------------------
# _resolve_cols
# ---------------------------------------------------------------------------


def test_resolve_cols_none(source_parquets):
    _, _, cat0, _ = source_parquets
    result = _resolve_cols(None, 0, cat0)
    pd.testing.assert_frame_equal(result, cat0)


def test_resolve_cols_list_keeps_existing(source_parquets):
    _, _, cat0, _ = source_parquets
    result = _resolve_cols(["flux", "nonexistent"], 0, cat0)
    assert list(result.columns) == ["flux"]


def test_resolve_cols_dict_present(source_parquets):
    _, _, cat0, _ = source_parquets
    result = _resolve_cols({0: ["size"]}, 0, cat0)
    assert list(result.columns) == ["size"]


def test_resolve_cols_dict_absent_returns_full(source_parquets):
    """When cat_id is not in the dict, return the full DataFrame unchanged."""
    _, _, cat0, _ = source_parquets
    result = _resolve_cols({99: ["flux"]}, 0, cat0)
    pd.testing.assert_frame_equal(result, cat0)


# ---------------------------------------------------------------------------
# buildJoinedObjectTable
# ---------------------------------------------------------------------------


def test_buildJoinedObjectTable_basic(source_parquets, object_tables):
    input_files, catalog_ids, _, _ = source_parquets
    object_stats, object_assoc = object_tables

    result = buildJoinedObjectTable(object_stats, object_assoc, input_files, catalog_ids)

    assert len(result) == 3
    assert "object_id" in result.columns
    assert "ra" in result.columns
    assert "flux_0" in result.columns
    assert "size_0" in result.columns
    assert "flux_1" in result.columns

    row100 = result.loc[result.object_id == 100].iloc[0]
    assert row100["flux_0"] == pytest.approx(1.0)  # cat0[id=10]
    assert row100["flux_1"] == pytest.approx(4.0)  # cat1[id=20]

    # object 101 has only a cat-0 source; cat-1 cols should be NaN
    row101 = result.loc[result.object_id == 101].iloc[0]
    assert row101["flux_0"] == pytest.approx(2.0)  # cat0[id=11]
    assert np.isnan(row101["flux_1"])


def test_buildJoinedObjectTable_preserves_all_stats_rows(source_parquets, object_tables):
    """Every row in object_stats appears exactly once in the output."""
    input_files, catalog_ids, _, _ = source_parquets
    object_stats, object_assoc = object_tables

    result = buildJoinedObjectTable(object_stats, object_assoc, input_files, catalog_ids)

    assert set(result.object_id) == set(object_stats.object_id)


def test_buildJoinedObjectTable_source_cols_list(source_parquets, object_tables):
    input_files, catalog_ids, _, _ = source_parquets
    object_stats, object_assoc = object_tables

    result = buildJoinedObjectTable(
        object_stats, object_assoc, input_files, catalog_ids, source_cols=["flux"]
    )
    assert "flux_0" in result.columns
    assert "size_0" not in result.columns
    # id is used internally for the join but not emitted when source_cols filters it out
    assert "id_0" not in result.columns


def test_buildJoinedObjectTable_source_cols_dict(source_parquets, object_tables):
    input_files, catalog_ids, _, _ = source_parquets
    object_stats, object_assoc = object_tables

    result = buildJoinedObjectTable(
        object_stats,
        object_assoc,
        input_files,
        catalog_ids,
        source_cols={0: ["size"], 1: ["color"]},
    )
    assert "size_0" in result.columns
    assert "flux_0" not in result.columns
    assert "color_1" in result.columns
    assert "id_0" not in result.columns


def test_buildJoinedObjectTable_no_assoc_for_catalog(source_parquets, object_tables):
    """Catalog with no association rows is skipped entirely (no NaN columns added)."""
    input_files, _, _, _ = source_parquets
    object_stats, object_assoc = object_tables
    assoc_cat0_only = object_assoc[object_assoc.catalog_id == 0].copy()

    result = buildJoinedObjectTable(object_stats, assoc_cat0_only, input_files, [0, 1])

    assert "flux_0" in result.columns
    assert "flux_1" not in result.columns


# ---------------------------------------------------------------------------
# buildJoinedClusterTable
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_tables():
    cluster_stats = pd.DataFrame(
        {"cluster_id": [200, 201], "ra": [0.1, 0.2], "dec": [0.1, 0.2]}
    )
    # cluster 200 has one source from cat 0 (source_idx=0) and one from cat 1 (source_idx=0)
    # cluster 201 has one source from cat 0 (source_idx=1)
    cluster_assoc = pd.DataFrame(
        {
            "cluster_id": [200, 200, 201],
            "source_idx": [0, 0, 1],
            "catalog_id": [0, 1, 0],
            "distance": [0.1, 0.2, 0.05],
            "cell_idx": [0, 0, 0],
        }
    )
    return cluster_stats, cluster_assoc


def test_buildJoinedClusterTable_basic(source_parquets, cluster_tables):
    input_files, catalog_ids, _, _ = source_parquets
    cluster_stats, cluster_assoc = cluster_tables

    result = buildJoinedClusterTable(cluster_stats, cluster_assoc, input_files, catalog_ids)

    assert len(result) == 2
    assert "cluster_id" in result.columns
    assert "flux_0" in result.columns

    # cluster 200: cat0.iloc[0] -> flux=1.0, cat1.iloc[0] -> flux=4.0
    row200 = result.loc[result.cluster_id == 200].iloc[0]
    assert row200["flux_0"] == pytest.approx(1.0)
    assert row200["flux_1"] == pytest.approx(4.0)

    # cluster 201: cat0.iloc[1] -> flux=2.0; no cat-1 source
    row201 = result.loc[result.cluster_id == 201].iloc[0]
    assert row201["flux_0"] == pytest.approx(2.0)
    assert np.isnan(row201["flux_1"])


def test_buildJoinedClusterTable_source_cols(source_parquets, cluster_tables):
    input_files, catalog_ids, _, _ = source_parquets
    cluster_stats, cluster_assoc = cluster_tables

    result = buildJoinedClusterTable(
        cluster_stats, cluster_assoc, input_files, catalog_ids, source_cols=["size"]
    )
    assert "size_0" in result.columns
    assert "flux_0" not in result.columns


def test_buildJoinedClusterTable_preserves_all_stats_rows(source_parquets, cluster_tables):
    input_files, catalog_ids, _, _ = source_parquets
    cluster_stats, cluster_assoc = cluster_tables

    result = buildJoinedClusterTable(cluster_stats, cluster_assoc, input_files, catalog_ids)

    assert set(result.cluster_id) == set(cluster_stats.cluster_id)
