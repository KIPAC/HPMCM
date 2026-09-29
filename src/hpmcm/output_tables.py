"""Schema for various output tables produced by hpmcm"""

from __future__ import annotations

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
    """Interface of table with shear information"""

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
    def buildObjectShearStats(cls, cell_data: CellData) -> ShearTable:
        """Create shear stats table for objects in a cell

        Parameters
        ----------
        cell_data:
            Cell we are making table for

        Returns
        -------
        Shear stats table
        """
        n_obj = cell_data.n_objects
        out_dict = ShearTable.emtpyNumpyDict(n_obj)
        for idx, obj in enumerate(cell_data.object_dict.values()):
            assert isinstance(obj, ShearObjectData)
            obj_stats = obj.shearStats()
            for key, val in obj_stats.items():
                out_dict[key][idx] = val
        return ShearTable(**out_dict)

    @classmethod
    def buildClusterShearStats(cls, cell_data: CellData) -> ShearTable:
        """Create shear stats table for clusters in a cell

        Parameters
        ----------
        cell_data:
            Cell we are making table for

        Returns
        -------
        Shear stats table
        """
        n_clusters = cell_data.n_clusters
        out_dict = ShearTable.emtpyNumpyDict(n_clusters)
        for idx, clus in enumerate(cell_data.cluster_dict.values()):
            assert isinstance(clus, ShearClusterData)
            clus_stats = clus.shearStats()
            for key, val in clus_stats.items():
                out_dict[key][idx] = val
        return ShearTable(**out_dict)


SourceColsType = list[str] | dict[int, list[str]] | None


def _catalog_mask_series(
    assoc: pandas.DataFrame,
    id_col: str,
    catalog_ids: list[int],
) -> pandas.Series:
    """Return an integer bitmask series indexed by id_col.

    Bit ``i`` is set when ``catalog_ids[i]`` contributed at least one source
    to that object or cluster.
    """
    bit = {cat_id: 1 << i for i, cat_id in enumerate(catalog_ids)}
    return (
        assoc.groupby(id_col)["catalog_id"]
        .apply(lambda cats: sum(bit.get(c, 0) for c in cats.unique()))
        .astype(int)
    )


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
    object_assoc: pandas.DataFrame,
    input_files: list[str],
    catalog_ids: list[int],
    source_cols: SourceColsType = None,
) -> pandas.DataFrame:
    """Build a wide joined table from an ObjectAssocTable and source catalogs.

    Produces one row per object with source-level data from each catalog
    joined in as additional columns, suffixed by ``_{catalog_id}``.

    Parameters
    ----------
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

    Returns
    -------
    DataFrame with one row per ``object_id``.  Columns from each catalog are
    renamed ``{col}_{catalog_id}``.  Objects that have no source in a
    given catalog will have ``NaN`` for that catalog's columns.
    """
    catalog_file_map = dict(zip(catalog_ids, input_files))

    base = (
        object_assoc[["object_id", "cluster_id", "cell_idx"]]
        .drop_duplicates("object_id")
        .set_index("object_id")
    )

    for cat_id, f_name in catalog_file_map.items():
        mask = object_assoc["catalog_id"] == cat_id
        if not mask.any():
            continue

        assoc_sub = object_assoc.loc[mask, ["object_id", "source_idx", "distance"]]

        src_df = _resolve_cols(source_cols, cat_id, pandas.read_parquet(f_name))

        rows = src_df.iloc[assoc_sub["source_idx"].values].copy()
        rows.index = assoc_sub["object_id"].values
        rows["distance"] = assoc_sub["distance"].values

        rows = rows.rename(columns={c: f"{c}_{cat_id}" for c in rows.columns})
        base = base.join(rows, how="left")

    base["catalog_mask"] = _catalog_mask_series(object_assoc, "object_id", catalog_ids)
    return base.reset_index()


def buildJoinedClusterTable(
    cluster_assoc: pandas.DataFrame,
    input_files: list[str],
    catalog_ids: list[int],
    source_cols: SourceColsType = None,
) -> pandas.DataFrame:
    """Build a wide joined table from a ClusterAssocTable and source catalogs.

    Produces one row per cluster with source-level data from each catalog
    joined in as additional columns, suffixed by ``_{catalog_id}``.

    Parameters
    ----------
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

    Returns
    -------
    DataFrame with one row per ``cluster_id``.  Columns from each catalog are
    renamed ``{col}_{catalog_id}``.  Clusters that have no source in a
    given catalog will have ``NaN`` for that catalog's columns.
    """
    catalog_file_map = dict(zip(catalog_ids, input_files))

    base = (
        cluster_assoc[["cluster_id", "cell_idx"]]
        .drop_duplicates("cluster_id")
        .set_index("cluster_id")
    )

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

    base["catalog_mask"] = _catalog_mask_series(cluster_assoc, "cluster_id", catalog_ids)
    return base.reset_index()
