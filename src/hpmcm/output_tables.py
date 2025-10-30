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

xque
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
    def buildFromCellData(cellData: CellData) -> ObjectAssocTable:
        """Create object association table

        Parameters
        ----------
        cellData:
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
        distancesList: list[np.ndarray] = []

        for obj in cellData.objectDict.values():
            cluster_ids.append(
                np.full((obj.n_src), obj.parentCluster.iCluster, dtype=int)
            )
            object_ids.append(np.full((obj.n_src), obj.object_id, dtype=int))
            source_ids.append(obj.source_ids())
            source_idxs.append(obj.source_idxs())
            cat_idxs.append(obj.catIndices)
            assert obj.dist2.size
            distancesList.append(obj.dist2)
        if not distancesList:
            return ObjectAssocTable(
                object_id=np.array([], int),
                cluster_id=np.array([], int),
                source_id=np.array([], int),
                source_idx=np.array([], int),
                catalog_id=np.array([], int),
                distance=np.array([], float),
                cell_idx=np.array([], int),
            )
        distances = np.hstack(distancesList)
        distances = cellData.matcher.pixToArcsec() * np.sqrt(distances)
        return ObjectAssocTable(
            object_id=np.hstack(object_ids),
            cluster_id=np.hstack(cluster_ids),
            source_id=np.hstack(source_ids),
            source_idx=np.hstack(source_idxs),
            catalog_id=np.hstack(cat_idxs),
            distance=distances,
            cell_idx=np.repeat(cellData.idx, len(distances)).astype(int),
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
        SNR=TableColumnInfo(float, "Mean signal-to-noise ratio"),
        SNRRms=TableColumnInfo(float, "RMS signal-to-noise ratio"),
        cell_idx=TableColumnInfo(int, "Index of associated cell"),
        hasRefCat=TableColumnInfo(bool, "Has source from the reference catalog"),
    )

    @staticmethod
    def buildFromCellData(cellData: CellData) -> ObjectStatsTable:
        """Create object stats table

        Parameters
        ----------
        cellData:
            Cell we are making table for

        Returns
        -------
        Object stats table
        """
        nObj = cellData.n_objects
        cluster_ids = np.zeros((nObj), dtype=int)
        object_ids = np.zeros((nObj), dtype=int)
        n_srcs = np.zeros((nObj), dtype=int)
        n_uniques = np.zeros((nObj), dtype=int)
        dist_rms = np.zeros((nObj), dtype=float)
        x_cents = np.zeros((nObj), dtype=float)
        y_cents = np.zeros((nObj), dtype=float)
        SNRs = np.zeros((nObj), dtype=float)
        SNRRms = np.zeros((nObj), dtype=float)
        hasRefCat = np.zeros((nObj), dtype=bool)

        for idx, obj in enumerate(cellData.objectDict.values()):
            cluster_ids[idx] = obj.parentCluster.iCluster
            object_ids[idx] = obj.object_id
            n_srcs[idx] = obj.n_src
            n_uniques[idx] = obj.n_unique
            dist_rms[idx] = obj.rmsDist
            x_cents[idx] = obj.x_cent
            y_cents[idx] = obj.y_cent
            assert obj.data is not None
            sumSNR = obj.data.SNR.sum()
            x_cents[idx] = np.sum(obj.data.SNR * obj.data.xCell) / sumSNR
            y_cents[idx] = np.sum(obj.data.SNR * obj.data.yCell) / sumSNR
            SNRs[idx] = obj.snrMean
            SNRRms[idx] = obj.snrRms
            hasRefCat[idx] = obj.hasRefCatalog()

        ra, dec = cellData.getRaDec(x_cents, y_cents)
        dist_rms *= cellData.matcher.pixToArcsec()

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
            SNR=SNRs,
            SNRRms=SNRRms,
            cell_idx=np.repeat(cellData.idx, len(dist_rms)).astype(int),
            hasRefCat=hasRefCat,
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
    def buildFromCellData(cellData: CellData) -> ClusterAssocTable:
        """Create object association table

        Parameters
        ----------
        cellData:
            Cell we are making table for

        Returns
        -------
        Cluster Association table
        """
        cluster_ids = []
        source_ids = []
        source_idxs = []
        cat_idxs = []
        distancesList: list[np.ndarray] = []
        for cluster in cellData.clusterDict.values():
            cluster_ids.append(np.full((cluster.n_src), cluster.iCluster, dtype=int))
            source_ids.append(cluster.src_ids)
            source_idxs.append(cluster.src_idxs)
            cat_idxs.append(cluster.catIndices)
            assert cluster.dist2.size
            distancesList.append(cluster.dist2)
        if not distancesList:
            return ClusterAssocTable(
                distance=np.array([], float),
                source_id=np.array([], int),
                source_idx=np.array([], int),
                catalog_id=np.array([], int),
                cluster_id=np.array([], int),
                cell_idx=np.array([], int),
            )
        distances = np.hstack(distancesList)
        distances = cellData.matcher.pixToArcsec() * np.sqrt(distances)
        return ClusterAssocTable(
            cluster_id=np.hstack(cluster_ids),
            source_id=np.hstack(source_ids),
            source_idx=np.hstack(source_idxs),
            catalog_id=np.hstack(cat_idxs),
            distance=distances,
            cell_idx=np.repeat(cellData.idx, len(distances)).astype(int),
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
        SNR=TableColumnInfo(float, "Mean signal-to-noise ratio"),
        SNRRms=TableColumnInfo(float, "RMS signal-to-noise ratio"),
        cell_idx=TableColumnInfo(int, "Index of associated cell"),
        hasRefCat=TableColumnInfo(bool, "Has source from reference catalog"),
    )

    @staticmethod
    def buildFromCellData(cellData: CellData) -> ClusterStatsTable:
        """Create object stats table

        Parameters
        ----------
        cellData:
            Cell we are making table for

        Returns
        -------
        Object stats table
        """
        nClust = cellData.nClusters
        cluster_ids = np.zeros((nClust), dtype=int)
        n_srcs = np.zeros((nClust), dtype=int)
        n_uniques = np.zeros((nClust), dtype=int)
        n_objects = np.zeros((nClust), dtype=int)
        n_uniques = np.zeros((nClust), dtype=int)
        dist_rms = np.zeros((nClust), dtype=float)
        x_cents = np.zeros((nClust), dtype=float)
        y_cents = np.zeros((nClust), dtype=float)
        SNRs = np.zeros((nClust), dtype=float)
        SNRRms = np.zeros((nClust), dtype=float)
        hasRefCat = np.zeros((nClust), dtype=bool)

        for idx, cluster in enumerate(cellData.clusterDict.values()):
            cluster_ids[idx] = cluster.iCluster
            n_srcs[idx] = cluster.n_src
            n_uniques[idx] = cluster.n_unique
            n_objects[idx] = len(cluster.objects)
            n_uniques[idx] = cluster.n_unique
            dist_rms[idx] = cluster.rmsDist
            assert cluster.data is not None
            sumSNR = cluster.data.SNR.sum()
            x_cents[idx] = np.sum(cluster.data.SNR * cluster.data.xCell) / sumSNR
            y_cents[idx] = np.sum(cluster.data.SNR * cluster.data.yCell) / sumSNR
            SNRs[idx] = cluster.snrMean
            SNRRms[idx] = cluster.snrRms
            hasRefCat[idx] = cluster.hasRefCatalog()

        ra, dec = cellData.getRaDec(x_cents, y_cents)
        dist_rms *= cellData.matcher.pixToArcsec()

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
            SNR=SNRs,
            SNRRms=SNRRms,
            cell_idx=np.repeat(cellData.idx, len(dist_rms)).astype(int),
            hasRefCat=hasRefCat,
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
    def buildObjectShearStats(cls, cellData: CellData) -> ShearTable:
        """Create shear stats table for objects in a cell

        Parameters
        ----------
        cellData:
            Cell we are making table for

        Returns
        -------
        Shear stats table
        """
        nObj = cellData.n_objects
        outDict = ShearTable.emtpyNumpyDict(nObj)
        for idx, obj in enumerate(cellData.objectDict.values()):
            assert isinstance(obj, ShearObjectData)
            objStats = obj.shearStats()
            for key, val in objStats.items():
                outDict[key][idx] = val
        return ShearTable(**outDict)

    @classmethod
    def buildClusterShearStats(cls, cellData: CellData) -> ShearTable:
        """Create shear stats table for clusters in a cell

        Parameters
        ----------
        cellData:
            Cell we are making table for

        Returns
        -------
        Shear stats table
        """
        nClusters = cellData.nClusters
        outDict = ShearTable.emtpyNumpyDict(nClusters)
        for idx, obj in enumerate(cellData.clusterDict.values()):
            assert isinstance(obj, ShearClusterData)
            objStats = obj.shearStats()
            for key, val in objStats.items():
                outDict[key][idx] = val
        return ShearTable(**outDict)
