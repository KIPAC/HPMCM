from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas

from . import shear_utils
from .object import ObjectData
from .table import TableColumnInfo, TableInterface

if TYPE_CHECKING:
    try:
        import lsst.afw.detection as afwDetect
    except (ImportError, ModuleNotFoundError):
        pass

    from .cell import CellData


class ClusterAssocTable(TableInterface):
    """Interface of table with associations between clusters and sources"""

    schema = TableInterface.schema.copy()
    schema.update(
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

    schema = TableInterface.schema.copy()
    schema.update(
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


class ClusterData:
    """Class that defines clusters

    A cluster is a set of sources within a footprint of
    adjacent pixels.

    Attributes
    ----------
    iCluster : int
        Cluster ID

    footprint: afwDetect.Footprint
        Footprint of this cluster in the CellData counts map

    origCluster : int
        Id of the original cluster this cluster was made from

    sources: np.ndarray
        Data about the sources in this cluster

    nSrc : int
        Number of sources in this cluster

    nUnique : int
        Number of catalogs contributing sources to this cluster

    objects: list[ObjectData]
        Data about the objects in this cluster

    xCent : float
        X-pixel value of cluster centroid (in WCS used to do matching)

    yCent : float
        Y-pixel value of cluster centroid (in WCS used to do matching)

    rmsDist: float
        RMS distance of sources to the centroid

    snrMean: float
        mean signal-to-noise

    snrRms: float
        RMS of signal-to-noise

    dist2: np.ndarray
        Distances of sources to the centroid
    """

    def __init__(
        self,
        iCluster: int,
        footprint: afwDetect.Footprint,
        sources: np.ndarray,
        origCluster: int | None = None,
    ):
        """Build from a Footprint and data about the
        sources in that Footprint"""
        self.iCluster: int = iCluster
        self.footprint: afwDetect.Footprint = footprint
        self.origCluster: int = iCluster
        if origCluster is not None:
            self.origCluster = origCluster
        self.sources: np.ndarray = sources

        self.nSrc: int = self.sources[0].size
        self.nUnique: int = len(np.unique(self.sources[0]))
        self.objects: list[ObjectData] = []
        self.data: pandas.DataFrame | None = None

        self.xCent: float = np.nan
        self.yCent: float = np.nan
        self.rmsDist: float = np.nan
        self.snrMean: float = np.nan
        self.snrRms: float = np.nan
        self.dist2: np.ndarray = np.array([])
        self.pixelMatchScale: int = 1

    def extract(self, cellData: CellData) -> None:
        """Extract the xPix, yPix and snr data from
        the sources in this cluster
        """
        bbox = self.footprint.getBBox()

        xOffset = bbox.getBeginY() * self.pixelMatchScale
        yOffset = bbox.getBeginX() * self.pixelMatchScale

        seriesList = []

        for _i, (iCat, srcIdx) in enumerate(zip(self.sources[0], self.sources[2])):
            seriesList.append(cellData.data[iCat].iloc[srcIdx])

        self.data = pandas.DataFrame(seriesList)
        self.data["iCat"] = self.sources[0]
        self.data["srcId"] = self.sources[1]
        self.data["srcIdx"] = self.sources[2]
        self.data["xCluster"] = self.data.xCell - xOffset
        self.data["yCluster"] = self.data.yCell - yOffset

    @property
    def catIndices(self) -> np.ndarray:
        """Return the catalog indices for all the sources in the cluster"""
        assert self.data is not None
        return self.data.iCat

    @property
    def srcIds(self) -> np.ndarray:
        """Return the ids for all the sources in the cluster"""
        assert self.data is not None
        return self.data.srcId

    @property
    def srcIdxs(self) -> np.ndarray:
        """Return the indices for all the sources in the cluster"""
        assert self.data is not None
        return self.data.srcIdx

    @property
    def xCluster(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the footprint"""
        assert self.data is not None
        return self.data.xCluster

    @property
    def yCluster(self) -> np.ndarray:
        """Return the y-positions of the soures w.r.t. the footprint"""
        assert self.data is not None
        return self.data.yCluster

    @property
    def xPix(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the WCS"""
        assert self.data is not None
        return self.data.xPix

    @property
    def yPix(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the WCS"""
        assert self.data is not None
        return self.data.yPix

    def addObject(
        self, cellData: CellData, mask: np.ndarray | None = None
    ) -> ObjectData:
        """Add a new object to this cluster"""
        newObject = cellData.addObject(self, mask)
        self.objects.append(newObject)
        return newObject


class ShearClusterData(ClusterData):
    """Subclass of ClusterData that can compute shear statisitics

    Attributes
    ----------
    pixelMatchScale: int
        Number of pixel merged in the original counts map
    """

    def __init__(
        self,
        iCluster: int,
        footprint: afwDetect.Footprint,
        sources: np.ndarray,
        origCluster: int | None = None,
        pixelMatchScale: int = 1,
    ):
        ClusterData.__init__(self, iCluster, footprint, sources, origCluster)
        self.pixelMatchScale = pixelMatchScale

    def shearStats(self) -> dict:
        """Return the shear statistics"""
        assert self.data is not None
        return shear_utils.shearStats(self.data)
