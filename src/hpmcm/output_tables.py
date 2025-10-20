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
        objectId=TableColumnInfo(int, "Unique Object ID"),
        clusterId=TableColumnInfo(int, "Parent Cluster Unique ID"),
        sourceId=TableColumnInfo(int, "Source id in input catalog"),
        sourceIdx=TableColumnInfo(int, "Source index in input catalog"),
        catalogId=TableColumnInfo(int, "Associated catalog ID"),
        distance=TableColumnInfo(float, "Distance from sources to object centroid"),
        cellIdx=TableColumnInfo(int, "Index of associated cell"),
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
        clusterIds = []
        objectIds = []
        sourceIds = []
        sourceIdxs = []
        catIdxs = []
        distancesList: list[np.ndarray] = []

        for obj in cellData.objectDict.values():
            clusterIds.append(
                np.full((obj.nSrc), obj.parentCluster.iCluster, dtype=int)
            )
            objectIds.append(np.full((obj.nSrc), obj.objectId, dtype=int))
            sourceIds.append(obj.sourceIds())
            sourceIdxs.append(obj.sourceIdxs())
            catIdxs.append(obj.catIndices)
            assert obj.dist2.size
            distancesList.append(obj.dist2)
        if not distancesList:
            return ObjectAssocTable(
                objectId=np.array([], int),
                clusterId=np.array([], int),
                sourceId=np.array([], int),
                sourceIdx=np.array([], int),
                catalogId=np.array([], int),
                distance=np.array([], float),
                cellIdx=np.array([], int),
            )
        distances = np.hstack(distancesList)
        distances = cellData.matcher.pixToArcsec() * np.sqrt(distances)
        return ObjectAssocTable(
            objectId=np.hstack(objectIds),
            clusterId=np.hstack(clusterIds),
            sourceId=np.hstack(sourceIds),
            sourceIdx=np.hstack(sourceIdxs),
            catalogId=np.hstack(catIdxs),
            distance=distances,
            cellIdx=np.repeat(cellData.idx, len(distances)).astype(int),
        )


class ObjectStatsTable(TableInterface):
    """Interface of table of object statistics"""

    _schema = TableInterface._schema.copy()
    _schema.update(
        objectId=TableColumnInfo(int, "Unique Object ID"),
        clusterId=TableColumnInfo(int, "Parent Cluster Unique ID"),
        nUnique=TableColumnInfo(int, "Number of unique catalogs represented"),
        nSrc=TableColumnInfo(int, "Number of sources"),
        distRms=TableColumnInfo(
            float, "RMS of distance from sources to object centroid"
        ),
        ra=TableColumnInfo(float, "RA of object centroid"),
        dec=TableColumnInfo(float, "DEC of object centroid"),
        xCent=TableColumnInfo(float, "X-value of cluster centroid in WCS pixels"),
        yCent=TableColumnInfo(float, "Y-value of cluster centroid in WCS pixels"),
        SNR=TableColumnInfo(float, "Mean signal-to-noise ratio"),
        SNRRms=TableColumnInfo(float, "RMS signal-to-noise ratio"),
        cellIdx=TableColumnInfo(int, "Index of associated cell"),
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
        nObj = cellData.nObjects
        clusterIds = np.zeros((nObj), dtype=int)
        objectIds = np.zeros((nObj), dtype=int)
        nSrcs = np.zeros((nObj), dtype=int)
        nUniques = np.zeros((nObj), dtype=int)
        distRms = np.zeros((nObj), dtype=float)
        xCents = np.zeros((nObj), dtype=float)
        yCents = np.zeros((nObj), dtype=float)
        SNRs = np.zeros((nObj), dtype=float)
        SNRRms = np.zeros((nObj), dtype=float)

        for idx, obj in enumerate(cellData.objectDict.values()):
            clusterIds[idx] = obj.parentCluster.iCluster
            objectIds[idx] = obj.objectId
            nSrcs[idx] = obj.nSrc
            nUniques[idx] = obj.nUnique
            distRms[idx] = obj.rmsDist
            xCents[idx] = obj.xCent
            yCents[idx] = obj.yCent
            assert obj.data is not None
            sumSNR = obj.data.SNR.sum()
            xCents[idx] = np.sum(obj.data.SNR * obj.data.xCell) / sumSNR
            yCents[idx] = np.sum(obj.data.SNR * obj.data.yCell) / sumSNR
            SNRs[idx] = obj.snrMean
            SNRRms[idx] = obj.snrRms

        ra, dec = cellData.getRaDec(xCents, yCents)
        distRms *= cellData.matcher.pixToArcsec()

        return ObjectStatsTable(
            clusterId=clusterIds,
            objectId=objectIds,
            nUnique=nUniques,
            nSrc=nSrcs,
            distRms=distRms,
            ra=ra,
            dec=dec,
            xCent=xCents,
            yCent=yCents,
            SNR=SNRs,
            SNRRms=SNRRms,
            cellIdx=np.repeat(cellData.idx, len(distRms)).astype(int),
        )


class ClusterAssocTable(TableInterface):
    """Interface of table with associations between clusters and sources"""

    _schema = TableInterface._schema.copy()
    _schema.update(
        clusterId=TableColumnInfo(int, "Unique cluster ID"),
        sourceId=TableColumnInfo(int, "Source id in input catalog"),
        sourceIdx=TableColumnInfo(int, "Source index in input catalog"),
        catalogId=TableColumnInfo(int, "Associated catalog ID"),
        distance=TableColumnInfo(float, "Distance from sources to cluster centroid"),
        cellIdx=TableColumnInfo(int, "Index of associated cell"),
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
        clusterIds = []
        sourceIds = []
        sourceIdxs = []
        catIdxs = []
        distancesList: list[np.ndarray] = []
        for cluster in cellData.clusterDict.values():
            clusterIds.append(np.full((cluster.nSrc), cluster.iCluster, dtype=int))
            sourceIds.append(cluster.srcIds)
            sourceIdxs.append(cluster.srcIdxs)
            catIdxs.append(cluster.catIndices)
            assert cluster.dist2.size
            distancesList.append(cluster.dist2)
        if not distancesList:
            return ClusterAssocTable(
                distance=np.array([], float),
                sourceId=np.array([], int),
                sourceIdx=np.array([], int),
                catalogId=np.array([], int),
                clusterId=np.array([], int),
                cellIdx=np.array([], int),
            )
        distances = np.hstack(distancesList)
        distances = cellData.matcher.pixToArcsec() * np.sqrt(distances)
        return ClusterAssocTable(
            clusterId=np.hstack(clusterIds),
            sourceId=np.hstack(sourceIds),
            sourceIdx=np.hstack(sourceIdxs),
            catalogId=np.hstack(catIdxs),
            distance=distances,
            cellIdx=np.repeat(cellData.idx, len(distances)).astype(int),
        )


class ClusterStatsTable(TableInterface):
    """Interface of table of cluster statistics"""

    _schema = TableInterface._schema.copy()
    _schema.update(
        clusterId=TableColumnInfo(int, "Parent Cluster Unique ID"),
        nObject=TableColumnInfo(int, "Number of objects in cluster"),
        nUnique=TableColumnInfo(int, "Number of unique catalogs represented"),
        nSrc=TableColumnInfo(int, "Number of sources"),
        distRms=TableColumnInfo(
            float, "RMS of distance from sources to object centroid"
        ),
        ra=TableColumnInfo(float, "RA of object centroid"),
        dec=TableColumnInfo(float, "DEC of object centroid"),
        xCent=TableColumnInfo(float, "X-value of cluster centroid in WCS pixels"),
        yCent=TableColumnInfo(float, "Y-value of cluster centroid in WCS pixels"),
        SNR=TableColumnInfo(float, "Mean signal-to-noise ratio"),
        SNRRms=TableColumnInfo(float, "RMS signal-to-noise ratio"),
        cellIdx=TableColumnInfo(int, "Index of associated cell"),
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
        clusterIds = np.zeros((nClust), dtype=int)
        nSrcs = np.zeros((nClust), dtype=int)
        nUniques = np.zeros((nClust), dtype=int)
        nObjects = np.zeros((nClust), dtype=int)
        nUniques = np.zeros((nClust), dtype=int)
        distRms = np.zeros((nClust), dtype=float)
        xCents = np.zeros((nClust), dtype=float)
        yCents = np.zeros((nClust), dtype=float)
        SNRs = np.zeros((nClust), dtype=float)
        SNRRms = np.zeros((nClust), dtype=float)

        for idx, cluster in enumerate(cellData.clusterDict.values()):
            clusterIds[idx] = cluster.iCluster
            nSrcs[idx] = cluster.nSrc
            nUniques[idx] = cluster.nUnique
            nObjects[idx] = len(cluster.objects)
            nUniques[idx] = cluster.nUnique
            distRms[idx] = cluster.rmsDist
            assert cluster.data is not None
            sumSNR = cluster.data.SNR.sum()
            xCents[idx] = np.sum(cluster.data.SNR * cluster.data.xCell) / sumSNR
            yCents[idx] = np.sum(cluster.data.SNR * cluster.data.yCell) / sumSNR
            SNRs[idx] = cluster.snrMean
            SNRRms[idx] = cluster.snrRms

        ra, dec = cellData.getRaDec(xCents, yCents)
        distRms *= cellData.matcher.pixToArcsec()

        return ClusterStatsTable(
            clusterId=clusterIds,
            nSrc=nSrcs,
            nObject=nObjects,
            nUnique=nUniques,
            distRms=distRms,
            ra=ra,
            dec=dec,
            xCent=xCents,
            yCent=yCents,
            SNR=SNRs,
            SNRRms=SNRRms,
            cellIdx=np.repeat(cellData.idx, len(distRms)).astype(int),
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
        nObj = cellData.nObjects
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
