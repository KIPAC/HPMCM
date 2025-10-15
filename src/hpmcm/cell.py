from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import pandas

from . import utils
from .cluster import ClusterData, ShearClusterData
from .object import ObjectData, ShearObjectData

if TYPE_CHECKING:
    try:
        import lsst.afw.detection as afwDetect
    except (ImportError, ModuleNotFoundError):
        pass

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
    matcher: Match
        Parent Match object

    idOffset: int
        Offset used for the Object and Cluster IDs for this cell

    corner: np.ndarray
        pixX, pixY for corner of cell

    size: np.ndarray
        size of the cell (in pixels)

    idx: int
        Id of the cell

    buf: int
        Number of buffer pixels around the edge of the cell

    minPix: np.ndarray
        Lowest number pixel in center of cell

    maxPix: np.ndarray
        Highest number pixel in center of cell

    nPix: np.ndarray
        Number of pixels in center of cell

    data : list[pandas.DataFrame]
        Reduced dataframes with only sources for this cell

    nSrc : int
        Number of sources in this cell

    footprintIds : list[np.ndarray]
        Matched arrays with the index of the cluster associated to each
        source.  I.e., these could added to the Dataframes as
        additional columns

    clusterDict : OrderedDict[int, ClusterData]
        Dictionary with cluster membership data

    objectDict : OrderedDict[int, ObjectData]
        Dictionary with object membership data

    """

    def __init__(
        self,
        matcher: Match,
        idOffset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
        buf: int = 10,
    ):
        self.matcher: Match = matcher
        # Offset used for the Object and Cluster IDs for this cell
        self.idOffset: int = idOffset
        self.corner: np.ndarray = corner  # pixX, pixY for corner of cell
        self.size: np.ndarray = size  # size of cell
        self.idx: int = idx  # cell index
        self.buf: int = buf
        self.minPix: np.ndarray = corner - buf
        self.maxPix: np.ndarray = corner + size + buf
        self.nPix: np.ndarray = self.maxPix - self.minPix

        self.data: list[pandas.DataFrame] = []
        self.nSrc: int = 0
        self.footprintIds: list[np.ndarray] = []
        self.clusterDict: OrderedDict[int, ClusterData] = OrderedDict()
        self.objectDict: OrderedDict[int, ObjectData] = OrderedDict()

    def reduceData(self, data: list[pandas.DataFrame]) -> None:
        """Pull out only the data needed for this cell"""
        self.data = [self.reduceDataframe(i, val) for i, val in enumerate(data)]
        self.nSrc = int(np.sum([len(df) for df in self.data]))

    @property
    def nClusters(self) -> int:
        """Return the number of clusters in this cell"""
        return len(self.clusterDict)

    @property
    def nObjects(self) -> int:
        """Return the number of objects in this cell"""
        return len(self.objectDict)

    def reduceDataframe(
        self, iCat: int, dataframe: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Filters dataframe to keep only source in the cell"""

        # WCS is defined, use it
        xCell = dataframe["xPix"] - self.minPix[0]
        yCell = dataframe["yPix"] - self.minPix[1]
        filtered = (
            (xCell >= 0)
            & (xCell < self.nPix[0])
            & (yCell >= 0)
            & (yCell < self.nPix[1])
        )
        red = dataframe[filtered].copy(deep=True)
        red["xCell"] = xCell[filtered]
        red["yCell"] = yCell[filtered]
        return red

    def countsMap(self, weightName: str | None = None) -> np.ndarray:
        """Fill a map that counts the number of source per cell"""
        toFill = self._emtpyCountsMaps()
        assert self.data is not None
        for df in self.data:
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
        assert self.data is not None
        assert self.footprintIds
        for iCat, (df, footprintIds) in enumerate(zip(self.data, self.footprintIds)):
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
            iCluster = footprintId + self.idOffset
            cluster = self._buildClusterData(iCluster, footprint, np.array(sources).T)
            self.clusterDict[iCluster] = cluster
            heirarchicalProcessCluster(cluster, pixelR2Cut)

    def analyze(
        self, weightName: str | None = None, pixelR2Cut: float = 2.0
    ) -> dict | None:
        """Analyze this cell

        Note that this returns the counts maps and clustering info,
        which can be helpful for debugging.
        """
        if self.nSrc == 0:
            return None
        countsMap = self.countsMap(weightName)
        oDict = self._getFootprints(countsMap)
        oDict["countsMap"] = countsMap
        assert self.data is not None
        self.footprintIds = self._associateSourcesToFootprints(
            self.data, oDict["footprintKey"]
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
        for cluster in self.clusterDict.values():
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
                    cluster=np.array([], int),
                )
            )
        distances = np.hstack(distancesList)
        distances = self.matcher.pixToArcsec() * np.sqrt(distances)
        data = dict(
            cluster=np.hstack(clusterIds),
            id=np.hstack(sourceIds),
            idx=np.hstack(sourceIdxs),
            cat=np.hstack(catIdxs),
            distance=distances,
            cellIdx=np.repeat(self.idx, len(distances)).astype(int),
        )
        return pandas.DataFrame(data)

    def getObjectAssociations(self) -> pandas.DataFrame:
        """Convert the objects to a set of associations"""
        return makeObjectAssocTable(self)

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
        SNRRms = np.zeros((nClust), dtype=float)
        
        for idx, cluster in enumerate(self.clusterDict.values()):
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

        ra, dec = self._getRaDec(xCents, yCents)
        distRms *= self.matcher.pixToArcsec()

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
            SNRRms=SNRRms,            
            cellIdx=np.repeat(self.idx, len(distRms)).astype(int),
        )

        return pandas.DataFrame(data)

    def getObjectStats(self) -> pandas.DataFrame:
        """Get the stats for all the objects"""
        return makeObjectStatsTable(self)

    def addObject(
        self, cluster: ClusterData, mask: np.ndarray | None = None
    ) -> ObjectData:
        """Add an object to this cell"""
        objectId = self.nObjects + self.idOffset
        newObject = self._newObject(cluster, objectId, mask)
        self.objectDict[objectId] = newObject
        return newObject

    @classmethod
    def _newObject(
        cls, cluster: ClusterData, objectId: int, mask: np.ndarray | None
    ) -> ObjectData:
        return ObjectData(cluster, objectId, mask)

    def _emtpyCountsMaps(self) -> np.ndarray:
        toFill = np.zeros(np.ceil(self.nPix).astype(int))
        return toFill

    def _singleCatalogCountsMap(
        self, df: pandas.DataFrame, weightName: str | None = None
    ) -> np.ndarray:
        return utils.fillCountsMapFromDf(
            df,
            nPix=self.nPix,
            weightName=weightName,
        )

    def _buildClusterData(
        self, iCluster: int, footprint: afwDetect.FootPrint, sources: np.ndarray
    ) -> ClusterData:
        return ClusterData(iCluster, footprint, sources)

    def _getFootprints(self, countsMap: np.ndarray) -> dict:
        return utils.getFootprints(countsMap, buf=self.buf)

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
        return self.matcher.pixToWorld(xCents, yCents)


class ShearCellData(CellData):
    """Subclass of CellData that can compute shear statisitics

    Attributes
    ----------
    pixelMatchScale: int
        Number of pixel merged in the original counts map
    """

    def __init__(
        self,
        matcher: ShearMatch,
        idOffset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
        buf: int = 10,
    ):
        CellData.__init__(self, matcher, idOffset, corner, size, idx, buf)
        self.pixelMatchScale = matcher.pixelMatchScale

    def reduceDataframe(
        self, iCat: int, dataframe: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Filters dataframe to keep only source in the cell"""

        if TYPE_CHECKING:
            assert isinstance(self.matcher, ShearMatch)

        filteredIdx = self.matcher.getCellIndices(dataframe) == self.idx
        reduced = dataframe[filteredIdx].copy(deep=True)

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

        xCellOrig = reduced["xCellCoadd"]
        yCellOrig = reduced["yCellCoadd"]
        xPixOrig = reduced["xPix"]
        yPixOrig = reduced["yPix"]
        if TYPE_CHECKING:
            assert isinstance(self.matcher, ShearMatch)
        if self.matcher.deshear is not None:
            # De-shear in the cell frame to do matching
            dxShear = self.matcher.deshear * (
                xCellOrig * deshear_coeffs[iCat][0]
                + yCellOrig * deshear_coeffs[iCat][2]
            )
            dyShear = self.matcher.deshear * (
                xCellOrig * deshear_coeffs[iCat][1]
                + yCellOrig * deshear_coeffs[iCat][3]
            )
            xCell = xCellOrig + dxShear
            yCell = yCellOrig + dyShear
            xPix = xPixOrig + dxShear
            yPix = yPixOrig + dyShear
        else:
            dxShear = np.zeros(len(dataframe))
            dyShear = np.zeros(len(dataframe))
            xCell = xCellOrig
            yCell = yCellOrig
            xPix = xPixOrig
            yPix = yPixOrig

        xCell = (xCell + 100) / self.pixelMatchScale
        yCell = (yCell + 100) / self.pixelMatchScale
        filteredX = np.bitwise_and(xCell >= 0, xCell < self.nPix[0])
        filteredY = np.bitwise_and(yCell >= 0, yCell < self.nPix[1])
        filteredBounds = np.bitwise_and(filteredX, filteredY)
        red = reduced[filteredBounds].copy(deep=True)
        red["xCell"] = xCell[filteredBounds]
        red["yCell"] = yCell[filteredBounds]
        red["xPix"] = xPix[filteredBounds]
        red["yPix"] = yPix[filteredBounds]
        if self.matcher.deshear is not None:
            red["dxShear"] = dxShear[filteredBounds]
            red["dyShear"] = dyShear[filteredBounds]
        return red

    def getObjectShearStats(self) -> pandas.DataFrame:
        """Get the shear stats for all the objects"""
        nObj = self.nObjects
        outDict: dict[str, np.ndarray] = {}
        names = ["ns", "2p", "2m", "1p", "1m"]

        for name_ in names:
            outDict[f"n_{name_}"] = np.zeros((nObj), dtype=int)
            outDict[f"g_1_{name_}"] = np.zeros((nObj), dtype=float)
            outDict[f"g_2_{name_}"] = np.zeros((nObj), dtype=float)
        outDict["good"] = np.zeros((nObj), dtype=bool)
        outDict["delta_g_1_1"] = np.zeros((nObj), dtype=float)
        outDict["delta_g_2_2"] = np.zeros((nObj), dtype=float)
        outDict["delta_g_1_2"] = np.zeros((nObj), dtype=float)
        outDict["delta_g_2_1"] = np.zeros((nObj), dtype=float)
        for idx, obj in enumerate(self.objectDict.values()):
            assert isinstance(obj, ShearObjectData)
            objStats = obj.shearStats()
            for key, val in objStats.items():
                outDict[key][idx] = val
        return pandas.DataFrame(outDict)

    def getClusterShearStats(self) -> pandas.DataFrame:
        """Get the shear stats for all the clusters"""
        nClusters = self.nClusters
        outDict: dict[str, np.ndarray] = {}
        names = ["ns", "2p", "2m", "1p", "1m"]

        for name_ in names:
            outDict[f"n_{name_}"] = np.zeros((nClusters), dtype=int)
            outDict[f"g_1_{name_}"] = np.zeros((nClusters), dtype=float)
            outDict[f"g_2_{name_}"] = np.zeros((nClusters), dtype=float)
        outDict["delta_g_1_1"] = np.zeros((nClusters), dtype=float)
        outDict["delta_g_2_2"] = np.zeros((nClusters), dtype=float)
        outDict["delta_g_1_2"] = np.zeros((nClusters), dtype=float)
        outDict["delta_g_2_1"] = np.zeros((nClusters), dtype=float)
        outDict["good"] = np.zeros((nClusters), dtype=bool)
        for idx, cluster in enumerate(self.clusterDict.values()):
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
        pixelMatchScale = self.pixelMatchScale
        toFill = np.zeros(np.ceil(self.nPix / pixelMatchScale).astype(int))
        return toFill

    def _singleCatalogCountsMap(
        self, df: pandas.DataFrame, weightName: str | None = None
    ) -> np.ndarray:
        return utils.fillCountsMapFromDf(
            df,
            nPix=self.nPix,
            weightName=weightName,
            pixelMatchScale=self.pixelMatchScale,
        )

    def _buildClusterData(
        self, iCluster: int, footprint: afwDetect.FootPrint, sources: np.ndarray
    ) -> ClusterData:
        return ShearClusterData(
            iCluster, footprint, sources, pixelMatchScale=self.pixelMatchScale
        )

    def _getFootprints(self, countsMap: np.ndarray) -> dict:
        return utils.getFootprints(
            countsMap, buf=0, pixelMatchScale=self.pixelMatchScale
        )

    def _associateSourcesToFootprints(
        self,
        data: list[pandas.DataFrame],
        clusterKey: np.ndarray,
    ) -> list[np.ndarray]:
        return utils.associateSourcesToFootprints(
            data,
            clusterKey,
            pixelMatchScale=self.pixelMatchScale,
        )

    def _getRaDec(
        self, xCents: np.ndarray, yCents: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.repeat(np.nan, len(xCents)), np.repeat(np.nan, len(yCents))
