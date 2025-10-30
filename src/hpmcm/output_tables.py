"""Schema for various output tables produced by hpmcm"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .cluster import ShearClusterData
from .object import ShearObjectData
from .table import TableColumnInfo, TableInterface

if TYPE_CHECKING:
    from .cell import CellData


SHEAR_NAMES = ["ns", "2p", "2m", "1p", "1m"]

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
        x_cent=TableColumnInfo(float, "X-value of cluster centroid in WCS pixels"),
        y_cent=TableColumnInfo(float, "Y-value of cluster centroid in WCS pixels"),
        snr=TableColumnInfo(float, "Mean signal-to-noise ratio"),
        snr_rms=TableColumnInfo(float, "RMS signal-to-noise ratio"),
        cell_idx=TableColumnInfo(int, "Index of associated cell"),
        has_ref_cat=TableColumnInfo(bool, "Has source from the reference catalog"),
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

        for idx, obj in enumerate(cell_data.object_dict.values()):
            cluster_ids[idx] = obj.parent_cluster.i_cluster
            object_ids[idx] = obj.object_id
            n_srcs[idx] = obj.n_src
            n_uniques[idx] = obj.n_unique
            dist_rms[idx] = obj.rms_dist
            x_cents[idx] = obj.x_cent
            y_cents[idx] = obj.y_cent
            assert obj.data is not None
            sum_snr = obj.data.snr.sum()
            x_cents[idx] = np.sum(obj.data.snr * obj.data.x_cell) / sum_snr
            y_cents[idx] = np.sum(obj.data.snr * obj.data.y_cell) / sum_snr
            snrs[idx] = obj.snr_mean
            snr_rms[idx] = obj.snr_rms
            has_ref_cat[idx] = obj.hasRefCatalog()

        ra, dec = cell_data.getRaDec(x_cents, y_cents)
        dist_rms *= cell_data.matcher.pixToArcsec()

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
            snr=snrs,
            snr_rms=snr_rms,
            cell_idx=np.repeat(cell_data.idx, len(dist_rms)).astype(int),
            has_ref_cat=has_ref_cat,
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
        ra=TableColumnInfo(float, "RA of object centroid"),
        dec=TableColumnInfo(float, "DEC of object centroid"),
        x_cent=TableColumnInfo(float, "X-value of cluster centroid in WCS pixels"),
        y_cent=TableColumnInfo(float, "Y-value of cluster centroid in WCS pixels"),
        snr=TableColumnInfo(float, "Mean signal-to-noise ratio"),
        snr_rms=TableColumnInfo(float, "RMS signal-to-noise ratio"),
        cell_idx=TableColumnInfo(int, "Index of associated cell"),
        has_ref_cat=TableColumnInfo(bool, "Has source from reference catalog"),
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
        cluster_ids = np.zeros((n_clust), dtype=int)
        n_srcs = np.zeros((n_clust), dtype=int)
        n_uniques = np.zeros((n_clust), dtype=int)
        n_objects = np.zeros((n_clust), dtype=int)
        n_uniques = np.zeros((n_clust), dtype=int)
        dist_rms = np.zeros((n_clust), dtype=float)
        x_cents = np.zeros((n_clust), dtype=float)
        y_cents = np.zeros((n_clust), dtype=float)
        snrs = np.zeros((n_clust), dtype=float)
        snr_rms = np.zeros((n_clust), dtype=float)
        has_ref_cat = np.zeros((n_clust), dtype=bool)

        for idx, cluster in enumerate(cell_data.cluster_dict.values()):
            cluster_ids[idx] = cluster.i_cluster
            n_srcs[idx] = cluster.n_src
            n_uniques[idx] = cluster.n_unique
            n_objects[idx] = len(cluster.objects)
            n_uniques[idx] = cluster.n_unique
            dist_rms[idx] = cluster.rms_dist
            assert cluster.data is not None
            sum_snr = cluster.data.snr.sum()
            x_cents[idx] = np.sum(cluster.data.snr * cluster.data.x_cell) / sum_snr
            y_cents[idx] = np.sum(cluster.data.snr * cluster.data.y_cell) / sum_snr
            snrs[idx] = cluster.snr_mean
            snr_rms[idx] = cluster.snr_rms
            has_ref_cat[idx] = cluster.hasRefCatalog()

        ra, dec = cell_data.getRaDec(x_cents, y_cents)
        dist_rms *= cell_data.matcher.pixToArcsec()

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
            snr=snrs,
            snr_rms=snr_rms,
            cell_idx=np.repeat(cell_data.idx, len(dist_rms)).astype(int),
            has_ref_cat=has_ref_cat,
        )


class ShearTable(TableInterface):
    """Interface of table with shear information"""

    _schema = TableInterface._schema.copy()
    _schema["good"] = TableColumnInfo(bool, "Has unique match")
    for _name in SHEAR_NAMES:
        _schema[f"n_{_name}"] = TableColumnInfo(
            float, f"number of sourcrs from catalog {_name}"
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
