from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import lsst.afw.detection as afwDetect
import numpy as np
import pandas

from . import utils
from .cluster import ClusterData, ShearClusterData
from .object import ObjectData, ShearObjectData

if TYPE_CHECKING:
    from .match import Match
    from .shear_match import ShearMatch


class CellData:
    """Class to analyze data for a cell

    Includes cell boundries, reduced data tables
    and clustering results

    Does not store sky maps

    Cells are square sub-regions of the analysis region
    that are extracted from the ranges of pixels in the WCS

    The cell covers corner:corner+size

    The sources are projected into an array that extends `buf` pixels
    beyond the cell

    The used the afwDetect.FootprintSet to identify pixels which contain
    source, and builds those into clusters

    Attributes
    ----------
    _matcher: Match
        Parent Match object

    _idOffset: int
        Offset used for the Object and Cluster IDs for this cell

    _corner: np.ndarray
        pixX, pixY for corner of cell

    _size: np.ndarray
        size of the cell (in pixels)

    _idx: np.ndarray
        Id of the cell (ix, iy)

    _buf: int
        Number of buffer pixels around the edge of the cell

    _minPix: np.ndarray
        Lowest number pixel in center of cell

    _maxPix: np.ndarray
        Highest number pixel in center of cell

    _nPix: np.ndarray
        Number of pixels in center of cell

    _data : list[pandas.DataFrame]
        Reduced dataframes with only sources for this cell

    _nSrc : int
        Number of sources in this cell

    _footprintIds : list[np.ndarray]
        Matched arrays with the index of the cluster associated to each
        source.  I.e., these could added to the Dataframes as
        additional columns

    _clusterDict : OrderedDict[int, ClusterData]
        Dictionary with cluster membership data

    _objectDict : OrderedDict[int, ObjectData]
        Dictionary with object membership data

    """

    def __init__(
        self,
        matcher: Match,
        idOffset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: np.ndarray,
        buf: int = 10,
    ):
        self._matcher: Match = matcher
        # Offset used for the Object and Cluster IDs for this cell
        self._idOffset: int = idOffset
        self._corner: np.ndarray = corner  # pixX, pixY for corner of cell
        self._size: np.ndarray = size  # size of cell
        self._idx: np.ndarray = idx
        self._buf: int = buf
        self._minPix: np.ndarray = corner - buf
        self._maxPix: np.ndarray = corner + size + buf
        self._nPix: np.ndarray = self._maxPix - self._minPix

        self._data: list[pandas.DataFrame] = []
        self._nSrc: int = 0
        self._footprintIds: list[np.ndarray] = []
        self._clusterDict: OrderedDict[int, ClusterData] = OrderedDict()
        self._objectDict: OrderedDict[int, ObjectData] = OrderedDict()

    def reduceData(self, data: list[pandas.DataFrame]) -> None:
        """Pull out only the data needed for this cell"""
        self._data = [self.reduceDataframe(i, val) for i, val in enumerate(data)]
        self._nSrc = int(np.sum([len(df) for df in self._data]))

    @property
    def nClusters(self) -> int:
        """Return the number of clusters in this cell"""
        return len(self._clusterDict)

    @property
    def nObjects(self) -> int:
        """Return the number of objects in this cell"""
        return len(self._objectDict)

    @property
    def objectDict(self) -> OrderedDict[int, ObjectData]:
        """Return the object dictionary"""
        return self._objectDict

    @property
    def data(self) -> pandas.DataFrame:
        """Return the data associated to this cell"""
        return self._data

    @property
    def clusterDict(self) -> OrderedDict[int, ClusterData]:
        """Return a dictionary mapping clusters Ids to clusters"""
        return self._clusterDict

    @property
    def minPix(self) -> np.ndarray:
        """Return the location of the min corner"""
        return self._minPix

    @property
    def maxPix(self) -> np.ndarray:
        """Return the location of the max corner"""
        return self._maxPix

    def reduceDataframe(
        self, iCat: int, dataframe: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Filters dataframe to keep only source in the cell"""

        # WCS is defined, use it
        xCell = dataframe["xPix"] - self._minPix[0]
        yCell = dataframe["yPix"] - self._minPix[1]
        filtered = (
            (xCell >= 0)
            & (xCell < self._nPix[0])
            & (yCell >= 0)
            & (yCell < self._nPix[1])
        )
        red = dataframe[filtered].copy(deep=True)
        red["xCell"] = xCell[filtered]
        red["yCell"] = yCell[filtered]
        return red

    def countsMap(self, weightName: str | None = None) -> np.ndarray:
        """Fill a map that counts the number of source per cell"""
        toFill = self._emtpyCountsMaps()
        assert self._data is not None
        for df in self._data:
            toFill += self._singleCatalogCountsMap(df, weightName)
        return toFill

    def buildClusterData(
        self,
        fpSet: afwDetect.FootprintSet,
        pixelR2Cut: float = 4.0,
    ) -> None:
        """Loop through cluster ids and collect sources into
        the ClusterData objects"""
        footprints = fpSet.getFootprints()
        footprintDict: dict[int, list[tuple[int, int, int]]] = {}
        nMissing = 0
        nFound = 0
        assert self._data is not None
        assert self._footprintIds
        for iCat, (df, footprintIds) in enumerate(zip(self._data, self._footprintIds)):
            for srcIdx, (srcId, footprintId) in enumerate(zip(df["id"], footprintIds)):
                if footprintId < 0:
                    nMissing += 1
                    continue
                if footprintId not in footprintDict:
                    footprintDict[footprintId] = [(iCat, srcId, srcIdx)]
                else:
                    footprintDict[footprintId].append((iCat, srcId, srcIdx))
                nFound += 1
        for footprintId, sources in footprintDict.items():
            footprint = footprints[footprintId]
            iCluster = footprintId + self._idOffset
            cluster = self._buildClusterData(iCluster, footprint, np.array(sources).T)
            self._clusterDict[iCluster] = cluster
            cluster.processCluster(self, pixelR2Cut)

    def analyze(
        self, weightName: str | None = None, pixelR2Cut: float = 2.0
    ) -> dict | None:
        """Analyze this cell

        Note that this returns the counts maps and clustering info,
        which can be helpful for debugging.
        """
        if self._nSrc == 0:
            return None
        countsMap = self.countsMap(weightName)
        oDict = self._getFootprints(countsMap)
        oDict["countsMap"] = countsMap
        assert self._data is not None
        self._footprintIds = self._associateSourcesToFootprints(
            self._data, oDict["footprintKey"]
        )
        self.buildClusterData(oDict["footprints"], pixelR2Cut)
        return oDict

    def getClusterAssociations(self) -> pandas.DataFrame:
        """Convert the clusters to a set of associations"""
        clusterIds = []
        sourceIds = []
        sourceIdxs = []
        catIdxs = []
        distancesList: list[np.ndarray] = []
        for cluster in self._clusterDict.values():
            clusterIds.append(np.full((cluster.nSrc), cluster.iCluster, dtype=int))
            sourceIds.append(cluster.srcIds)
            sourceIdxs.append(cluster.srcIdxs)
            catIdxs.append(cluster.catIndices)
            assert cluster.dist2.size
            distancesList.append(cluster.dist2)
        if not distancesList:
            return pandas.DataFrame(
                dict(
                    distance=[],
                    id=np.array([], int),
                    idx=np.array([], int),
                    cat=np.array([], int),
                    object=np.array([], int),
                )
            )
        distances = np.hstack(distancesList)
        distances = self._matcher.pixToArcsec() * np.sqrt(distances)
        data = dict(
            object=np.hstack(clusterIds),
            id=np.hstack(sourceIds),
            idx=np.hstack(sourceIdxs),
            cat=np.hstack(catIdxs),
            distance=distances,
        )
        return pandas.DataFrame(data)

    def getObjectAssociations(self) -> pandas.DataFrame:
        """Convert the objects to a set of associations"""
        clusterIds = []
        objectIds = []
        sourceIds = []
        sourceIdxs = []
        catIdxs = []
        distancesList: list[np.ndarray] = []

        for obj in self._objectDict.values():
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
            return pandas.DataFrame(
                dict(
                    object=np.array([], int),
                    parent=np.array([], int),
                    id=np.array([], int),
                    idx=np.array([], int),
                    cat=np.array([], int),
                    distance=[],
                )
            )
        distances = np.hstack(distancesList)
        distances = self._matcher.pixToArcsec() * np.sqrt(distances)
        data = dict(
            object=np.hstack(objectIds),
            parent=np.hstack(clusterIds),
            id=np.hstack(sourceIds),
            idx=np.hstack(sourceIdxs),
            cat=np.hstack(catIdxs),
            distance=distances,
        )
        return pandas.DataFrame(data)

    def getClusterStats(self) -> pandas.DataFrame:
        """Get the stats for all the clusters"""
        nClust = self.nClusters
        clusterIds = np.zeros((nClust), dtype=int)
        nSrcs = np.zeros((nClust), dtype=int)
        nUniques = np.zeros((nClust), dtype=int)
        nObjects = np.zeros((nClust), dtype=int)
        nUniques = np.zeros((nClust), dtype=int)
        distRms = np.zeros((nClust), dtype=float)
        xCents = np.zeros((nClust), dtype=float)
        yCents = np.zeros((nClust), dtype=float)
        SNRs = np.zeros((nClust), dtype=float)
        for idx, cluster in enumerate(self._clusterDict.values()):
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
            SNRs[idx] = cluster.data.SNR.min()
        ra, dec = self._getRaDec(xCents, yCents)
        distRms *= self._matcher.pixToArcsec()

        data = dict(
            clusterIds=clusterIds,
            nSrcs=nSrcs,
            nObject=nObjects,
            nUniques=nUniques,
            distRms=distRms,
            ra=ra,
            dec=dec,
            xCents=xCents,
            yCents=yCents,
            SNRs=SNRs,
        )

        return pandas.DataFrame(data)

    def getObjectStats(self) -> pandas.DataFrame:
        """Get the stats for all the objects"""
        nObj = self.nObjects
        clusterIds = np.zeros((nObj), dtype=int)
        objectIds = np.zeros((nObj), dtype=int)
        nSrcs = np.zeros((nObj), dtype=int)
        nUniques = np.zeros((nObj), dtype=int)
        distRms = np.zeros((nObj), dtype=float)
        xCents = np.zeros((nObj), dtype=float)
        yCents = np.zeros((nObj), dtype=float)
        SNRs = np.zeros((nObj), dtype=float)

        for idx, obj in enumerate(self._objectDict.values()):
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
            SNRs[idx] = obj.data.SNR.min()

        ra, dec = self._getRaDec(xCents, yCents)
        distRms *= self._matcher.pixToArcsec()

        data = dict(
            clusterIds=clusterIds,
            objectIds=objectIds,
            nUniques=nUniques,
            nSrcs=nSrcs,
            distRms=distRms,
            ra=ra,
            dec=dec,
            xCents=xCents,
            yCents=yCents,
            SNRs=SNRs,
        )

        return pandas.DataFrame(data)

    def addObject(
        self, cluster: ClusterData, mask: np.ndarray | None = None
    ) -> ObjectData:
        """Add an object to this cell"""
        objectId = self.nObjects + self._idOffset
        newObject = self._newObject(cluster, objectId, mask)
        self._objectDict[objectId] = newObject
        return newObject

    @classmethod
    def _newObject(
        cls, cluster: ClusterData, objectId: int, mask: np.ndarray | None
    ) -> ObjectData:
        return ObjectData(cluster, objectId, mask)

    def _emtpyCountsMaps(self) -> np.ndarray:
        toFill = np.zeros(np.ceil(self._nPix).astype(int))
        return toFill

    def _singleCatalogCountsMap(
        self, df: pandas.DataFrame, weightName: str | None = None
    ) -> np.ndarray:
        return utils.fillCountsMapFromDf(
            df,
            nPix=self._nPix,
            weightName=weightName,
        )

    def _buildClusterData(
        self, iCluster: int, footprint: afwDetect.FootPrint, sources: np.ndarray
    ) -> ClusterData:
        return ClusterData(iCluster, footprint, sources)

    def _getFootprints(self, countsMap: np.ndarray) -> dict:
        return utils.getFootprints(countsMap, buf=self._buf)

    def _associateSourcesToFootprints(
        self,
        data: list[pandas.DataFrame],
        clusterKey: np.ndarray,
    ) -> list[np.ndarray]:
        return utils.associateSourcesToFootprints(
            data,
            clusterKey,
        )

    def _getRaDec(
        self, xCents: np.ndarray, yCents: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._matcher.pixToWorld(xCents, yCents)


class ShearCellData(CellData):
    """Subclass of CellData that can compute shear statisitics

    Attributes
    ----------
    _pixelMatchScale: int
        Number of pixel merged in the original counts map
    """

    def __init__(
        self,
        matcher: ShearMatch,
        idOffset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: np.ndarray,
        buf: int = 10,
    ):
        CellData.__init__(self, matcher, idOffset, corner, size, idx, buf)
        self._pixelMatchScale = matcher._pixelMatchScale

    def reduceDataframe(
        self, iCat: int, dataframe: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Filters dataframe to keep only source in the cell"""

        # These are the coeffs for the various shear catalogs
        deshear_coeffs = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, -1, -1, 0],
                [1, 0, 0, -1],
                [-1, 0, 0, 1],
            ]
        )
        # No WCS, use the original cells
        xCellOrig = dataframe["xCell_coadd"].values
        yCellOrig = dataframe["yCell_coadd"].values
        if TYPE_CHECKING:
            assert isinstance(self._matcher, ShearMatch)
        if self._matcher.deshear is not None:
            # De-shear in the cell frame to do matching
            dxShear = self._matcher.deshear * (
                xCellOrig * deshear_coeffs[iCat][0]
                + yCellOrig * deshear_coeffs[iCat][2]
            )
            dyShear = self._matcher.deshear * (
                xCellOrig * deshear_coeffs[iCat][1]
                + yCellOrig * deshear_coeffs[iCat][3]
            )
            xCell = xCellOrig + dxShear
            yCell = yCellOrig + dyShear
        else:
            dxShear = np.zeros(len(dataframe))
            dyShear = np.zeros(len(dataframe))
            xCell = xCellOrig
            yCell = yCellOrig

        xCell = (xCellOrig + 100) / self._pixelMatchScale
        yCell = (xCellOrig + 100) / self._pixelMatchScale
        filteredIdx = np.bitwise_and(
            dataframe["idx_x"] - self._idx[0] == 1,
            dataframe["idx_y"] - self._idx[1] == 1,
        )
        filteredX = np.bitwise_and(xCell >= 0, xCell < self._nPix[0])
        filteredY = np.bitwise_and(yCell >= 0, yCell < self._nPix[1])
        filteredBounds = np.bitwise_and(filteredX, filteredY)
        filtered = np.bitwise_and(filteredIdx, filteredBounds)
        red = dataframe[filtered].copy(deep=True)
        red["xCell"] = xCell[filtered]
        red["yCell"] = yCell[filtered]
        if self._matcher.deshear is not None:
            red["dxShear"] = dxShear[filtered]
            red["dyShear"] = dyShear[filtered]
            red["xPix"] += dxShear[filtered]
            red["yPix"] += dyShear[filtered]
        return red

    def getObjectShearStats(self) -> pandas.DataFrame:
        """Get the shear stats for all the objects"""
        nObj = self.nObjects
        outDict: dict[str, np.ndarray] = {}
        names = ["ns", "2p", "2m", "1p", "1m"]

        for name_ in names:
            outDict[f"n_{name_}"] = np.zeros((nObj), dtype=int)
            outDict[f"g1_{name_}"] = np.zeros((nObj), dtype=float)
            outDict[f"g2_{name_}"] = np.zeros((nObj), dtype=float)
        outDict["good"] = np.zeros((nObj), dtype=bool)
        outDict["delta_g_1"] = np.zeros((nObj), dtype=float)
        outDict["delta_g_2"] = np.zeros((nObj), dtype=float)
        for idx, obj in enumerate(self._objectDict.values()):
            assert isinstance(obj, ShearObjectData)
            objStats = obj.shearStats()
            for key, val in objStats.items():
                outDict[key][idx] = val
        return pandas.DataFrame(outDict)

    def getClusterShearStats(self) -> pandas.DataFrame:
        """Get the shear stats for all the objects"""
        nClusters = self.nClusters
        outDict: dict[str, np.ndarray] = {}
        names = ["ns", "2p", "2m", "1p", "1m"]

        for name_ in names:
            outDict[f"n_{name_}"] = np.zeros((nClusters), dtype=int)
            outDict[f"g1_{name_}"] = np.zeros((nClusters), dtype=float)
            outDict[f"g2_{name_}"] = np.zeros((nClusters), dtype=float)
        outDict["delta_g_1"] = np.zeros((nClusters), dtype=float)
        outDict["delta_g_2"] = np.zeros((nClusters), dtype=float)
        outDict["good"] = np.zeros((nClusters), dtype=bool)
        for idx, cluster in enumerate(self._clusterDict.values()):
            assert isinstance(cluster, ShearClusterData)
            clusterStats = cluster.shearStats()
            for key, val in clusterStats.items():
                outDict[key][idx] = val
        return pandas.DataFrame(outDict)

    @classmethod
    def _newObject(
        cls, cluster: ClusterData, objectId: int, mask: np.ndarray | None
    ) -> ObjectData:
        return ShearObjectData(cluster, objectId, mask)

    def _emtpyCountsMaps(self) -> np.ndarray:
        pixelMatchScale = self._pixelMatchScale
        toFill = np.zeros(np.ceil(self._nPix / pixelMatchScale).astype(int))
        return toFill

    def _singleCatalogCountsMap(
        self, df: pandas.DataFrame, weightName: str | None = None
    ) -> np.ndarray:
        return utils.fillCountsMapFromDf(
            df,
            nPix=self._nPix,
            weightName=weightName,
            pixelMatchScale=self._pixelMatchScale,
        )

    def _buildClusterData(
        self, iCluster: int, footprint: afwDetect.FootPrint, sources: np.ndarray
    ) -> ClusterData:
        return ShearClusterData(
            iCluster, footprint, sources, pixelMatchScale=self._pixelMatchScale
        )

    def _getFootprints(self, countsMap: np.ndarray) -> dict:
        return utils.getFootprints(
            countsMap, buf=0, pixelMatchScale=self._pixelMatchScale
        )

    def _associateSourcesToFootprints(
        self,
        data: list[pandas.DataFrame],
        clusterKey: np.ndarray,
    ) -> list[np.ndarray]:
        return utils.associateSourcesToFootprints(
            data,
            clusterKey,
            pixelMatchScale=self._pixelMatchScale,
        )

    def _getRaDec(
        self, xCents: np.ndarray, yCents: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.repeat(np.nan, len(xCents)), np.repeat(np.nan, len(yCents))
