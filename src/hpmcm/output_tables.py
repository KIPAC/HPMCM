"""Schema for various output tables produced by hpmcm"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas

from .cluster import ShearClusterData
from .object import ShearObjectData
from .shear_utils import SHEAR_NAMES
from .table import TableColumnInfo, TableInterface

if TYPE_CHECKING:
    from .cell import CellData


class ObjectAssocTable(TableInterface):
    """Interface of table with associations between objects and sources"""

    _schema = TableInterface._schema.copy()
    _schema.update(
        object_id=TableColumnInfo(int, "Unique Object ID"),
        cluster_id=TableColumnInfo(int, "Parent Cluster Unique ID"),
        source_id=TableColumnInfo(int, "Source id in input catalog"),
        source_idx=TableColumnInfo(int, "Source index in input catalog"),
        catalog_id=TableColumnInfo(int, "Associated catalog ID"),
        distance=TableColumnInfo(float, "Distance from sources to object centroid"),
        cell_idx=TableColumnInfo(int, "Index of associated cell"),
    )

    @staticmethod
    def buildFromCellData(cell_data: CellData) -> ObjectAssocTable:
        """Create object association table

        Parameters
        ----------
        cell_data:
            Cell we are making table for

        Returns
        -------
        Object Association table
        """
        cluster_ids = []
        object_ids = []
        source_ids = []
        source_idxs = []
        cat_idxs = []
        distances_list: list[np.ndarray] = []

        for obj in cell_data.object_dict.values():
            cluster_ids.append(
                np.full((obj.n_src), obj.parent_cluster.i_cluster, dtype=int)
            )
            object_ids.append(np.full((obj.n_src), obj.object_id, dtype=int))
            source_ids.append(obj.sourceIds())
            source_idxs.append(obj.sourceIdxs())
            cat_idxs.append(obj.catalog_id)
            assert obj.dist_2.size
            distances_list.append(obj.dist_2)
        if not distances_list:
            return ObjectAssocTable(
                object_id=np.array([], int),
                cluster_id=np.array([], int),
                source_id=np.array([], int),
                source_idx=np.array([], int),
                catalog_id=np.array([], int),
                distance=np.array([], float),
                cell_idx=np.array([], int),
            )
        distances = np.hstack(distances_list)
        distances = cell_data.matcher.pixToArcsec() * np.sqrt(distances)
        return ObjectAssocTable(
            object_id=np.hstack(object_ids),
            cluster_id=np.hstack(cluster_ids),
            source_id=np.hstack(source_ids),
            source_idx=np.hstack(source_idxs),
            catalog_id=np.hstack(cat_idxs),
            distance=distances,
            cell_idx=np.repeat(cell_data.idx, len(distances)).astype(int),
        )


class ObjectStatsTable(TableInterface):
    """Interface of table of object statistics"""

    _schema = TableInterface._schema.copy()
    _schema.update(
        object_id=TableColumnInfo(int, "Unique Object ID"),
        cluster_id=TableColumnInfo(int, "Parent Cluster Unique ID"),
        n_unique=TableColumnInfo(int, "Number of unique catalogs represented"),
        n_src=TableColumnInfo(int, "Number of sources"),
        dist_rms=TableColumnInfo(
            float, "RMS of distance from sources to object centroid"
        ),
        ra=TableColumnInfo(float, "RA of object centroid"),
        dec=TableColumnInfo(float, "DEC of object centroid"),
        x_cent=TableColumnInfo(float, "X-value of object centroid in cell pixels"),
        y_cent=TableColumnInfo(float, "Y-value of object centroid in cell pixels"),
        x_pix=TableColumnInfo(float, "X-value of object centroid in global WCS pixels"),
        y_pix=TableColumnInfo(float, "Y-value of object centroid in global WCS pixels"),
        snr=TableColumnInfo(float, "Mean signal-to-noise ratio"),
        snr_rms=TableColumnInfo(float, "RMS signal-to-noise ratio"),
        cell_idx=TableColumnInfo(int, "Index of associated cell"),
        has_ref_cat=TableColumnInfo(bool, "Has source from the reference catalog"),
        catalog_mask=TableColumnInfo(int, "Mask of which catalogs are in object"),
    )

    @staticmethod
    def buildFromCellData(cell_data: CellData) -> ObjectStatsTable:
        """Create object stats table

        Parameters
        ----------
        cell_data:
            Cell we are making table for

        Returns
        -------
        Object stats table
        """
        n_obj = cell_data.n_objects
        catalog_id_map = cell_data.matcher.catalog_id_map
        cluster_ids = np.zeros((n_obj), dtype=int)
        object_ids = np.zeros((n_obj), dtype=int)
        n_srcs = np.zeros((n_obj), dtype=int)
        n_uniques = np.zeros((n_obj), dtype=int)
        dist_rms = np.zeros((n_obj), dtype=float)
        x_cents = np.zeros((n_obj), dtype=float)
        y_cents = np.zeros((n_obj), dtype=float)
        snrs = np.zeros((n_obj), dtype=float)
        snr_rms = np.zeros((n_obj), dtype=float)
        has_ref_cat = np.zeros((n_obj), dtype=bool)
        catalog_mask = np.zeros((n_obj), dtype=int)

        for idx, obj in enumerate(cell_data.object_dict.values()):
            cluster_ids[idx] = obj.parent_cluster.i_cluster
            object_ids[idx] = obj.object_id
            n_srcs[idx] = obj.n_src
            n_uniques[idx] = obj.n_unique
            dist_rms[idx] = obj.rms_dist
            x_cents[idx] = obj.x_cent
            y_cents[idx] = obj.y_cent
            snrs[idx] = obj.snr_mean
            snr_rms[idx] = obj.snr_rms
            has_ref_cat[idx] = obj.hasRefCatalog()
            catalog_mask[idx] = obj.catalogMask(catalog_id_map)

        ra, dec = cell_data.getRaDec(x_cents, y_cents)
        dist_rms *= cell_data.matcher.pixToArcsec()
        x_pix = x_cents + cell_data.min_pix[0]
        y_pix = y_cents + cell_data.min_pix[1]

        return ObjectStatsTable(
            cluster_id=cluster_ids,
            object_id=object_ids,
            n_unique=n_uniques,
            n_src=n_srcs,
            dist_rms=dist_rms,
            ra=ra,
            dec=dec,
            x_cent=x_cents,
            y_cent=y_cents,
            x_pix=x_pix,
            y_pix=y_pix,
            snr=snrs,
            snr_rms=snr_rms,
            cell_idx=np.repeat(cell_data.idx, len(dist_rms)).astype(int),
            has_ref_cat=has_ref_cat,
            catalog_mask=catalog_mask,
        )


class ClusterAssocTable(TableInterface):
    """Interface of table with associations between clusters and sources"""

    _schema = TableInterface._schema.copy()
    _schema.update(
        cluster_id=TableColumnInfo(int, "Unique cluster ID"),
        source_id=TableColumnInfo(int, "Source id in input catalog"),
        source_idx=TableColumnInfo(int, "Source index in input catalog"),
        catalog_id=TableColumnInfo(int, "Associated catalog ID"),
        distance=TableColumnInfo(float, "Distance from sources to cluster centroid"),
        cell_idx=TableColumnInfo(int, "Index of associated cell"),
    )

    @staticmethod
    def buildFromCellData(cell_data: CellData) -> ClusterAssocTable:
        """Create object association table

        Parameters
        ----------
        cell_data:
            Cell we are making table for

        Returns
        -------
        Cluster Association table
        """
        cluster_ids = []
        source_ids = []
        source_idxs = []
        cat_idxs = []
        distances_list: list[np.ndarray] = []
        for cluster in cell_data.cluster_dict.values():
            cluster_ids.append(np.full((cluster.n_src), cluster.i_cluster, dtype=int))
            source_ids.append(cluster.src_id)
            source_idxs.append(cluster.src_idx)
            cat_idxs.append(cluster.catalog_id)
            assert cluster.dist_2.size
            distances_list.append(cluster.dist_2)
        if not distances_list:
            return ClusterAssocTable(
                distance=np.array([], float),
                source_id=np.array([], int),
                source_idx=np.array([], int),
                catalog_id=np.array([], int),
                cluster_id=np.array([], int),
                cell_idx=np.array([], int),
            )
        distances = np.hstack(distances_list)
        distances = cell_data.matcher.pixToArcsec() * np.sqrt(distances)
        return ClusterAssocTable(
            cluster_id=np.hstack(cluster_ids),
            source_id=np.hstack(source_ids),
            source_idx=np.hstack(source_idxs),
            catalog_id=np.hstack(cat_idxs),
            distance=distances,
            cell_idx=np.repeat(cell_data.idx, len(distances)).astype(int),
        )


class ClusterStatsTable(TableInterface):
    """Interface of table of cluster statistics"""

    _schema = TableInterface._schema.copy()
    _schema.update(
        cluster_id=TableColumnInfo(int, "Parent Cluster Unique ID"),
        n_object=TableColumnInfo(int, "Number of objects in cluster"),
        n_unique=TableColumnInfo(int, "Number of unique catalogs represented"),
        n_src=TableColumnInfo(int, "Number of sources"),
        dist_rms=TableColumnInfo(
            float, "RMS of distance from sources to object centroid"
        ),
        ra=TableColumnInfo(float, "RA of cluster centroid"),
        dec=TableColumnInfo(float, "DEC of cluster centroid"),
        x_cent=TableColumnInfo(float, "X-value of cluster centroid in cell pixels"),
        y_cent=TableColumnInfo(float, "Y-value of cluster centroid in cell pixels"),
        x_pix=TableColumnInfo(float, "X-value of cluster centroid in global WCS pixels"),
        y_pix=TableColumnInfo(float, "Y-value of cluster centroid in global WCS pixels"),
        snr=TableColumnInfo(float, "Mean signal-to-noise ratio"),
        snr_rms=TableColumnInfo(float, "RMS signal-to-noise ratio"),
        cell_idx=TableColumnInfo(int, "Index of associated cell"),
        has_ref_cat=TableColumnInfo(bool, "Has source from reference catalog"),
        catalog_mask=TableColumnInfo(int, "Mask of which catalogs are in cluster"),
        fp_x_min=TableColumnInfo(int, "Footprint min x in cell pixels"),
        fp_x_max=TableColumnInfo(int, "Footprint max x in cell pixels"),
        fp_y_min=TableColumnInfo(int, "Footprint min y in cell pixels"),
        fp_y_max=TableColumnInfo(int, "Footprint max y in cell pixels"),
    )

    @staticmethod
    def buildFromCellData(cell_data: CellData) -> ClusterStatsTable:
        """Create object stats table

        Parameters
        ----------
        cell_data:
            Cell we are making table for

        Returns
        -------
        Object stats table
        """
        n_clust = cell_data.n_clusters
        catalog_id_map = cell_data.matcher.catalog_id_map
        cluster_ids = np.zeros((n_clust), dtype=int)
        n_srcs = np.zeros((n_clust), dtype=int)
        n_uniques = np.zeros((n_clust), dtype=int)
        n_objects = np.zeros((n_clust), dtype=int)
        dist_rms = np.zeros((n_clust), dtype=float)
        x_cents = np.zeros((n_clust), dtype=float)
        y_cents = np.zeros((n_clust), dtype=float)
        snrs = np.zeros((n_clust), dtype=float)
        snr_rms = np.zeros((n_clust), dtype=float)
        has_ref_cat = np.zeros((n_clust), dtype=bool)
        catalog_mask = np.zeros((n_clust), dtype=int)
        fp_x_mins = np.zeros((n_clust), dtype=int)
        fp_x_maxs = np.zeros((n_clust), dtype=int)
        fp_y_mins = np.zeros((n_clust), dtype=int)
        fp_y_maxs = np.zeros((n_clust), dtype=int)

        for idx, cluster in enumerate(cell_data.cluster_dict.values()):
            cluster_ids[idx] = cluster.i_cluster
            n_srcs[idx] = cluster.n_src
            n_uniques[idx] = cluster.n_unique
            n_objects[idx] = len(cluster.objects)
            dist_rms[idx] = cluster.rms_dist
            x_cents[idx] = cluster.x_cent
            y_cents[idx] = cluster.y_cent
            snrs[idx] = cluster.snr_mean
            snr_rms[idx] = cluster.snr_rms
            has_ref_cat[idx] = cluster.hasRefCatalog()
            catalog_mask[idx] = cluster.catalogMask(catalog_id_map)
            fp_x_mins[idx] = cluster.footprint.slice_x.start
            fp_x_maxs[idx] = cluster.footprint.slice_x.stop
            fp_y_mins[idx] = cluster.footprint.slice_y.start
            fp_y_maxs[idx] = cluster.footprint.slice_y.stop

        ra, dec = cell_data.getRaDec(x_cents, y_cents)
        dist_rms *= cell_data.matcher.pixToArcsec()
        x_pix = x_cents + cell_data.min_pix[0]
        y_pix = y_cents + cell_data.min_pix[1]

        return ClusterStatsTable(
            cluster_id=cluster_ids,
            n_src=n_srcs,
            n_object=n_objects,
            n_unique=n_uniques,
            dist_rms=dist_rms,
            ra=ra,
            dec=dec,
            x_cent=x_cents,
            y_cent=y_cents,
            x_pix=x_pix,
            y_pix=y_pix,
            snr=snrs,
            snr_rms=snr_rms,
            cell_idx=np.repeat(cell_data.idx, len(dist_rms)).astype(int),
            has_ref_cat=has_ref_cat,
            catalog_mask=catalog_mask,
            fp_x_min=fp_x_mins,
            fp_x_max=fp_x_maxs,
            fp_y_min=fp_y_mins,
            fp_y_max=fp_y_maxs,
        )


class ShearTable(TableInterface):
    """Base interface of table with shear information (no id columns)."""

    _schema = TableInterface._schema.copy()
    _schema["good"] = TableColumnInfo(bool, "Has unique match")
    for _name in SHEAR_NAMES:
        _schema[f"n_{_name}"] = TableColumnInfo(
            float, f"number of sources from catalog {_name}"
        )
        for _i in [1, 2]:
            _schema[f"g_{_i}_{_name}"] = TableColumnInfo(
                float, f"g {_i} for catalog {_name}"
            )
    for _i in [1, 2]:
        for _j in [1, 2]:
            _schema[f"delta_g_{_i}_{_j}"] = TableColumnInfo(
                float, f"delta g {_i} for {_j}p - {_j}m"
            )

    @classmethod
    def buildObjectShearStats(cls, cell_data: CellData) -> ObjectShearTable:
        """Create shear stats table for objects in a cell

        Parameters
        ----------
        cell_data:
            Cell we are making table for

        Returns
        -------
        Object shear stats table (includes ``object_id`` and ``cluster_id``)
        """
        n_obj = cell_data.n_objects
        out_dict = ObjectShearTable.emtpyNumpyDict(n_obj)
        for idx, obj in enumerate(cell_data.object_dict.values()):
            assert isinstance(obj, ShearObjectData)
            out_dict["object_id"][idx] = obj.object_id
            out_dict["cluster_id"][idx] = obj.parent_cluster.i_cluster
            for key, val in obj.shearStats().items():
                out_dict[key][idx] = val
        return ObjectShearTable(**out_dict)

    @classmethod
    def buildClusterShearStats(cls, cell_data: CellData) -> ClusterShearTable:
        """Create shear stats table for clusters in a cell

        Parameters
        ----------
        cell_data:
            Cell we are making table for

        Returns
        -------
        Cluster shear stats table (includes ``cluster_id``)
        """
        n_clusters = cell_data.n_clusters
        out_dict = ClusterShearTable.emtpyNumpyDict(n_clusters)
        for idx, clus in enumerate(cell_data.cluster_dict.values()):
            assert isinstance(clus, ShearClusterData)
            out_dict["cluster_id"][idx] = clus.i_cluster
            for key, val in clus.shearStats().items():
                out_dict[key][idx] = val
        return ClusterShearTable(**out_dict)


class ObjectShearTable(ShearTable):
    """Shear stats table for objects — adds ``object_id`` and ``cluster_id``."""

    _schema = {
        "object_id": TableColumnInfo(int, "Unique Object ID"),
        "cluster_id": TableColumnInfo(int, "Parent Cluster Unique ID"),
        **ShearTable._schema,
    }


class ClusterShearTable(ShearTable):
    """Shear stats table for clusters — adds ``cluster_id``."""

    _schema = {
        "cluster_id": TableColumnInfo(int, "Unique Cluster ID"),
        **ShearTable._schema,
    }


SourceColsType = list[str] | dict[int, list[str]] | None



def _resolve_cols(
    source_cols: SourceColsType,
    cat_id: int,
    src_df: pandas.DataFrame,
) -> pandas.DataFrame:
    """Return src_df filtered to the requested columns for cat_id."""
    if source_cols is None:
        return src_df
    if isinstance(source_cols, dict):
        if cat_id not in source_cols:
            return src_df
        cols = source_cols[cat_id]
    else:
        cols = source_cols
    return src_df[[c for c in cols if c in src_df.columns]]


def buildJoinedObjectTable(
    object_stats: pandas.DataFrame,
    object_assoc: pandas.DataFrame,
    input_files: list[str],
    catalog_ids: list[int],
    source_cols: SourceColsType = None,
    object_shear: str | Path | None = None,
) -> pandas.DataFrame:
    """Build a wide joined table from an ObjectStatsTable and source catalogs.

    Produces one row per object: all stats columns from ``object_stats`` plus
    source-level data from each catalog joined in as additional columns,
    suffixed by ``_{catalog_id}``.  Optionally joins shear statistics from an
    ``ObjectShearTable`` parquet file.

    Rows are matched using ``object_assoc.object_id == object_stats.object_id``
    and ``object_assoc.source_id == input_catalog.id``.

    Parameters
    ----------
    object_stats:
        Stats DataFrame (``ObjectStatsTable.data``) with one row per object.
    object_assoc:
        Association DataFrame (``ObjectAssocTable.data``) mapping objects to
        individual sources.
    input_files:
        List of input catalog file paths (parquet).
    catalog_ids:
        Catalog IDs corresponding to each entry in ``input_files``.
    source_cols:
        Columns to pull from source catalogs.

        - ``None``: all columns from every catalog.
        - ``list[str]``: same column list applied to every catalog (missing
          columns in a given file are silently skipped).
        - ``dict[int, list[str]]``: per-catalog column lists, keyed by
          ``catalog_id``.  Catalogs absent from the dict get no columns
          (only ``distance`` is added for them).
    object_shear:
        Optional path to an ``ObjectShearTable`` parquet file.  When provided,
        shear statistics are joined on ``object_id``.  The ``cluster_id``
        column is dropped before joining because it is already present in
        ``object_stats``.

    Returns
    -------
    DataFrame with one row per ``object_id``.  Columns from each catalog are
    renamed ``{col}_{catalog_id}``.  Objects that have no source in a
    given catalog will have ``NaN`` for that catalog's columns.
    """
    catalog_file_map = dict(zip(catalog_ids, input_files))
    base = object_stats.set_index("object_id")

    if object_shear is not None:
        shear_df = pandas.read_parquet(object_shear).drop(columns=["cluster_id"], errors="ignore")
        base = base.join(shear_df.set_index("object_id"), how="left")

    for cat_id, f_name in catalog_file_map.items():
        mask = object_assoc["catalog_id"] == cat_id
        if not mask.any():
            continue

        assoc_sub = object_assoc.loc[mask, ["object_id", "source_id"]]

        full_src = pandas.read_parquet(f_name)
        # Normalize: catalogs written from shear matches use object_id instead of id
        if "id" not in full_src.columns and "object_id" in full_src.columns:
            full_src = full_src.rename(columns={"object_id": "id"})

        src_df = _resolve_cols(source_cols, cat_id, full_src)

        # Re-inject the join key when column filtering stripped it
        id_injected = "id" not in src_df.columns
        if id_injected:
            src_df = src_df.copy()
            src_df["id"] = full_src["id"]

        rows = assoc_sub.merge(src_df, left_on="source_id", right_on="id", how="left")
        drop_cols = ["source_id", "id"] if id_injected else ["source_id"]
        rows = rows.drop(columns=drop_cols).set_index("object_id")
        rows = rows.rename(columns={c: f"{c}_{cat_id}" for c in rows.columns})
        base = base.join(rows, how="left")

    return base.reset_index()


def buildJoinedClusterTable(

    cluster_stats: pandas.DataFrame,
    cluster_assoc: pandas.DataFrame,
    input_files: list[str],
    catalog_ids: list[int],
    source_cols: SourceColsType = None,
    cluster_shear: str | Path | None = None,
) -> pandas.DataFrame:
    """Build a wide joined table from a ClusterStatsTable and source catalogs.

    Produces one row per cluster: all stats columns from ``cluster_stats`` plus
    source-level data from each catalog joined in as additional columns,
    suffixed by ``_{catalog_id}``.  Optionally joins shear statistics from a
    ``ClusterShearTable`` parquet file.

    Parameters
    ----------
    cluster_stats:
        Stats DataFrame (``ClusterStatsTable.data``) with one row per cluster.
    cluster_assoc:
        Association DataFrame (``ClusterAssocTable.data``) mapping clusters to
        individual sources.
    input_files:
        List of input catalog file paths (parquet).
    catalog_ids:
        Catalog IDs corresponding to each entry in ``input_files``.
    source_cols:
        Columns to pull from source catalogs.

        - ``None``: all columns from every catalog.
        - ``list[str]``: same column list applied to every catalog (missing
          columns in a given file are silently skipped).
        - ``dict[int, list[str]]``: per-catalog column lists, keyed by
          ``catalog_id``.  Catalogs absent from the dict get no columns
          (only ``distance`` is added for them).
    cluster_shear:
        Optional path to a ``ClusterShearTable`` parquet file.  When provided,
        shear statistics are joined on ``cluster_id``.

    Returns
    -------
    DataFrame with one row per ``cluster_id``.  Columns from each catalog are
    renamed ``{col}_{catalog_id}``.  Clusters that have no source in a
    given catalog will have ``NaN`` for that catalog's columns.
    """
    catalog_file_map = dict(zip(catalog_ids, input_files))
    base = cluster_stats.set_index("cluster_id")

    if cluster_shear is not None:
        shear_df = pandas.read_parquet(cluster_shear)
        base = base.join(shear_df.set_index("cluster_id"), how="left")

    for cat_id, f_name in catalog_file_map.items():
        mask = cluster_assoc["catalog_id"] == cat_id
        if not mask.any():
            continue

        assoc_sub = cluster_assoc.loc[mask, ["cluster_id", "source_idx", "distance"]]

        src_df = _resolve_cols(source_cols, cat_id, pandas.read_parquet(f_name))

        rows = src_df.iloc[assoc_sub["source_idx"].values].copy()
        rows.index = assoc_sub["cluster_id"].values
        rows["distance"] = assoc_sub["distance"].values

        rows = rows.rename(columns={c: f"{c}_{cat_id}" for c in rows.columns})
        base = base.join(rows, how="left")

    return base.reset_index()


def computeColumnStats(
    df: pandas.DataFrame,
    col_prefix: str,
    catalog_ids: list[int],
) -> pandas.DataFrame:
    """Compute per-row mean and std across per-catalog columns.

    For each row in ``df``, collects the values of all columns named
    ``{col_prefix}_{cat_id}`` for each ``cat_id`` in ``catalog_ids`` that
    exists in ``df``, then computes the mean and sample standard deviation
    while ignoring NaN values (missing matches).

    Parameters
    ----------
    df:
        DataFrame produced by :func:`buildJoinedObjectTable` or
        :func:`buildJoinedClusterTable`.
    col_prefix:
        Prefix shared by the per-catalog columns to aggregate, e.g. ``"flux"``
        to operate on ``flux_0``, ``flux_1``, ...
    catalog_ids:
        Catalog IDs whose columns to include.  Any catalog ID whose column
        ``{col_prefix}_{cat_id}`` is absent from ``df`` is silently skipped.

    Returns
    -------
    DataFrame with the same index as ``df`` and two columns:

    * ``{col_prefix}_mean`` -- row-wise mean of non-NaN values.
    * ``{col_prefix}_std``  -- row-wise sample std (ddof=1); NaN when fewer
      than two non-NaN values are available.
    """
    cols = [
        f"{col_prefix}_{cat_id}"
        for cat_id in catalog_ids
        if f"{col_prefix}_{cat_id}" in df.columns
    ]
    if not cols:
        return pandas.DataFrame(
            {
                f"{col_prefix}_mean": np.full(len(df), np.nan),
                f"{col_prefix}_std": np.full(len(df), np.nan),
            },
            index=df.index,
        )

    arr = df[cols].to_numpy(dtype=float)
    n = np.sum(~np.isnan(arr), axis=1)

    mean = np.full(len(df), np.nan)
    std = np.full(len(df), np.nan)

    has_any = n >= 1
    has_two = n >= 2
    if has_any.any():
        mean[has_any] = np.nanmean(arr[has_any], axis=1)
    if has_two.any():
        std[has_two] = np.nanstd(arr[has_two], axis=1, ddof=1)

    return pandas.DataFrame(
        {
            f"{col_prefix}_mean": mean,
            f"{col_prefix}_std": std,
        },
        index=df.index,
    )


def reduceJoinedTable(
    df: pandas.DataFrame,
    catalog_ids: list[int],
    keep_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
    stats_cols: list[str] | None = None,
) -> pandas.DataFrame:
    """Reduce a joined table by selecting columns and aggregating per-catalog ones.

    Applies three independent operations in sequence:

    1. **Column selection** — keep only the columns in ``keep_cols``; or, if
       ``keep_cols`` is ``None``, start with all columns and remove any in
       ``drop_cols``.  (``drop_cols`` is ignored when ``keep_cols`` is given.)
    2. **Per-catalog column removal** — for each prefix in ``stats_cols``,
       the individual per-catalog columns ``{prefix}_{cat_id}`` are removed
       from the selected set.
    3. **Stats aggregation** — for each prefix in ``stats_cols``,
       :func:`computeColumnStats` is called on the *original* ``df`` and the
       three summary columns (``{prefix}_mean``, ``{prefix}_std``,
       ``{prefix}_n``) are appended to the result.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`buildJoinedObjectTable` or
        :func:`buildJoinedClusterTable`.
    catalog_ids:
        Catalog IDs used to identify per-catalog columns and to pass to
        :func:`computeColumnStats`.
    keep_cols:
        Columns to retain as-is.  ``None`` keeps all columns (subject to
        ``drop_cols``).
    drop_cols:
        Columns to remove.  Only used when ``keep_cols`` is ``None``; silently
        ignored otherwise.
    stats_cols:
        Column-name prefixes whose per-catalog variants should be replaced by
        aggregated statistics.  Each prefix ``p`` causes ``p_0``, ``p_1``, …
        **and any column named exactly** ``p`` to be dropped, and ``p_mean``,
        ``p_std``, ``p_n`` to be added.

    Returns
    -------
    Reduced DataFrame with the same row order as ``df``.
    """
    # 1. Build the base column list
    if keep_cols is not None:
        base_cols = [c for c in keep_cols if c in df.columns]
    else:
        base_cols = list(df.columns)
        if drop_cols:
            excluded = set(drop_cols)
            base_cols = [c for c in base_cols if c not in excluded]

    # 2. Remove columns that will be replaced by stats:
    #    both the bare prefix names and all per-catalog variants {prefix}_{cat_id}
    if stats_cols:
        to_drop = set(stats_cols) | {f"{p}_{cid}" for p in stats_cols for cid in catalog_ids}
        base_cols = [c for c in base_cols if c not in to_drop]

    result = df[base_cols].copy()

    # 3. Append aggregated stats columns
    if stats_cols:
        stats_frames = [computeColumnStats(df, p, catalog_ids) for p in stats_cols]
        result = pandas.concat([result, *stats_frames], axis=1)

    return result
