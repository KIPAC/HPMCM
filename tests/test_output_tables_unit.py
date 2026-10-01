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
    computeColumnStats,
    reduceJoinedTable,
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
            "cluster_id": [200, 200, 201],
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


# ---------------------------------------------------------------------------
# object_shear / cluster_shear optional join
# ---------------------------------------------------------------------------


@pytest.fixture
def shear_parquets(tmp_path):
    """Write minimal shear parquets matching the object/cluster fixtures."""
    object_shear = pd.DataFrame(
        {
            "object_id": [100, 101, 102],
            "cluster_id": [200, 200, 201],  # should be dropped before join
            "good": [True, True, False],
            "g_1_ns": [0.01, 0.02, 0.03],
        }
    )
    cluster_shear = pd.DataFrame(
        {
            "cluster_id": [200, 201],
            "good": [True, False],
            "g_1_ns": [0.015, 0.025],
        }
    )
    obj_f = str(tmp_path / "object_shear.parquet")
    clust_f = str(tmp_path / "cluster_shear.parquet")
    object_shear.to_parquet(obj_f)
    cluster_shear.to_parquet(clust_f)
    return obj_f, clust_f, object_shear, cluster_shear


def test_buildJoinedObjectTable_with_shear(source_parquets, object_tables, shear_parquets):
    input_files, catalog_ids, _, _ = source_parquets
    object_stats, object_assoc = object_tables
    obj_shear_f, _, object_shear, _ = shear_parquets

    result = buildJoinedObjectTable(
        object_stats, object_assoc, input_files, catalog_ids,
        object_shear=obj_shear_f,
    )

    assert "g_1_ns" in result.columns
    assert "good" in result.columns
    # cluster_id from shear table is dropped; only one copy (from object_stats) remains
    assert result.columns.tolist().count("cluster_id") == 1
    assert result.loc[result.object_id == 100, "cluster_id"].iloc[0] == 200
    # values match the shear fixture
    row100 = result.loc[result.object_id == 100].iloc[0]
    assert row100["g_1_ns"] == pytest.approx(0.01)
    row102 = result.loc[result.object_id == 102].iloc[0]
    assert row102["good"] is False or row102["good"] == False  # noqa: E712


def test_buildJoinedClusterTable_with_shear(source_parquets, cluster_tables, shear_parquets):
    input_files, catalog_ids, _, _ = source_parquets
    cluster_stats, cluster_assoc = cluster_tables
    _, clust_shear_f, _, cluster_shear = shear_parquets

    result = buildJoinedClusterTable(
        cluster_stats, cluster_assoc, input_files, catalog_ids,
        cluster_shear=clust_shear_f,
    )

    assert "g_1_ns" in result.columns
    assert "good" in result.columns
    row200 = result.loc[result.cluster_id == 200].iloc[0]
    assert row200["g_1_ns"] == pytest.approx(0.015)
    row201 = result.loc[result.cluster_id == 201].iloc[0]
    assert row201["good"] is False or row201["good"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# computeColumnStats
# ---------------------------------------------------------------------------


@pytest.fixture
def joined_df():
    """Minimal joined DataFrame with flux columns from three catalogs."""
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4],
            "flux_0": [1.0, 2.0, np.nan, np.nan],
            "flux_1": [3.0, np.nan, 5.0, np.nan],
            "flux_2": [5.0, 4.0, 7.0, np.nan],
        }
    )


def test_computeColumnStats_basic(joined_df):
    result = computeColumnStats(joined_df, "flux", [0, 1, 2])

    assert list(result.columns) == ["flux_mean", "flux_std"]
    assert len(result) == len(joined_df)

    # row 0: values [1, 3, 5] → mean=3, std=2
    assert result["flux_mean"].iloc[0] == pytest.approx(3.0)
    assert result["flux_std"].iloc[0] == pytest.approx(2.0)

    # row 1: values [2, NaN, 4] → mean=3, std≈1.414
    assert result["flux_mean"].iloc[1] == pytest.approx(3.0)
    assert result["flux_std"].iloc[1] == pytest.approx(np.sqrt(2))

    # row 2: values [NaN, 5, 7] → mean=6
    assert result["flux_mean"].iloc[2] == pytest.approx(6.0)


def test_computeColumnStats_single_value_has_nan_std(joined_df):
    """With only one non-NaN value, sample std (ddof=1) is NaN."""
    result = computeColumnStats(joined_df, "flux", [1])
    assert np.isnan(result["flux_std"].iloc[2])


def test_computeColumnStats_all_nan_row(joined_df):
    """Row with all-NaN values gets NaN mean and std."""
    result = computeColumnStats(joined_df, "flux", [0, 1, 2])
    assert np.isnan(result["flux_mean"].iloc[3])
    assert np.isnan(result["flux_std"].iloc[3])


def test_computeColumnStats_missing_catalog_skipped(joined_df):
    """Catalog IDs with no matching column are silently skipped."""
    result = computeColumnStats(joined_df, "flux", [0, 99])  # 99 doesn't exist
    assert result["flux_mean"].iloc[0] == pytest.approx(1.0)


def test_computeColumnStats_no_matching_columns(joined_df):
    """When no column matches the prefix, returns NaN columns of correct length."""
    result = computeColumnStats(joined_df, "nonexistent", [0, 1, 2])
    assert len(result) == len(joined_df)
    assert np.all(np.isnan(result["nonexistent_mean"]))


# ---------------------------------------------------------------------------
# reduceJoinedTable
# ---------------------------------------------------------------------------


@pytest.fixture
def wide_df():
    """Joined DataFrame with several per-catalog columns and metadata columns."""
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "ra": [0.1, 0.2, 0.3],
            "dec": [0.1, 0.2, 0.3],
            "cluster_id": [10, 10, 20],
            "flux_0": [1.0, 2.0, np.nan],
            "flux_1": [3.0, np.nan, 5.0],
            "flux_2": [5.0, 4.0, 7.0],
            "size_0": [0.5, 0.6, 0.7],
            "size_1": [0.4, np.nan, 0.8],
            "size_2": [0.6, 0.5, np.nan],
        }
    )


def test_reduceJoinedTable_keep_cols(wide_df):
    """keep_cols selects only the named columns."""
    result = reduceJoinedTable(wide_df, [0, 1, 2], keep_cols=["object_id", "ra"])
    assert list(result.columns) == ["object_id", "ra"]
    assert len(result) == len(wide_df)


def test_reduceJoinedTable_drop_cols(wide_df):
    """drop_cols removes columns when keep_cols is None."""
    result = reduceJoinedTable(wide_df, [0, 1, 2], drop_cols=["cluster_id", "dec"])
    assert "cluster_id" not in result.columns
    assert "dec" not in result.columns
    assert "object_id" in result.columns


def test_reduceJoinedTable_drop_cols_ignored_with_keep(wide_df):
    """drop_cols is ignored when keep_cols is provided."""
    result = reduceJoinedTable(
        wide_df, [0, 1, 2],
        keep_cols=["object_id", "cluster_id"],
        drop_cols=["cluster_id"],
    )
    assert "cluster_id" in result.columns


def test_reduceJoinedTable_stats_cols(wide_df):
    """stats_cols replaces per-catalog columns with mean and std (no _n)."""
    result = reduceJoinedTable(wide_df, [0, 1, 2], stats_cols=["flux"])
    assert "flux_mean" in result.columns
    assert "flux_std" in result.columns
    assert "flux_n" not in result.columns
    # raw per-catalog columns are gone
    for cid in [0, 1, 2]:
        assert f"flux_{cid}" not in result.columns
    # other columns untouched
    assert "size_0" in result.columns
    assert "object_id" in result.columns


def test_reduceJoinedTable_stats_values(wide_df):
    """Stats in the reduced table match direct computeColumnStats output."""
    result = reduceJoinedTable(wide_df, [0, 1, 2], stats_cols=["flux"])
    expected = computeColumnStats(wide_df, "flux", [0, 1, 2])
    np.testing.assert_array_almost_equal(result["flux_mean"], expected["flux_mean"])
    np.testing.assert_array_almost_equal(result["flux_std"], expected["flux_std"])
    assert "flux_n" not in result.columns


def test_reduceJoinedTable_multiple_stats_cols(wide_df):
    """Multiple stats_cols are all aggregated."""
    result = reduceJoinedTable(wide_df, [0, 1, 2], stats_cols=["flux", "size"])
    for prefix in ["flux", "size"]:
        assert f"{prefix}_mean" in result.columns
        for cid in [0, 1, 2]:
            assert f"{prefix}_{cid}" not in result.columns


def test_reduceJoinedTable_keep_and_stats(wide_df):
    """keep_cols combined with stats_cols: raw per-catalog cols are still removed."""
    result = reduceJoinedTable(
        wide_df, [0, 1, 2],
        keep_cols=["object_id", "flux_0", "flux_1", "flux_2"],
        stats_cols=["flux"],
    )
    assert "flux_mean" in result.columns
    for cid in [0, 1, 2]:
        assert f"flux_{cid}" not in result.columns
    assert "object_id" in result.columns


def test_reduceJoinedTable_bare_prefix_col_dropped(wide_df):
    """A column named exactly after a stats prefix is also dropped."""
    df_with_bare = wide_df.copy()
    df_with_bare["flux"] = 99.0  # bare column with the same name as the prefix
    result = reduceJoinedTable(df_with_bare, [0, 1, 2], stats_cols=["flux"])
    assert "flux" not in result.columns
    assert "flux_mean" in result.columns


def test_reduceJoinedTable_no_reduction(wide_df):
    """With no stats_cols and no drops, the DataFrame is returned unchanged."""
    result = reduceJoinedTable(wide_df, [0, 1, 2])
    assert list(result.columns) == list(wide_df.columns)
