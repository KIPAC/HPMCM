from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import lsst.afw.detection as afwDetect
import numpy as np
import pandas
from astropy.table import Table

from . import utils
from .cluster import ClusterData
from .object import ObjectData

if TYPE_CHECKING:
    from .match import Match


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

    Parameters
    ----------
    _data : `list`, [`Dataframe`]
        Reduced dataframes with only sources for this cell

    _clusterIds : `list`, [`np.array`]
        Matched arrays with the index of the cluster associated to each
        source.  I.e., these could added to the Dataframes as
        additional columns

    _clusterDict : `dict`, [`int` : `ClusterData`]
        Dictionary with cluster membership data

    _objectDict : `dict`, [`int` : `ObjectData`]
        Dictionary with object membership data

    TODO:  Add code to filter out clusters centered in the buffer
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
        self._data = [self.reduceDataframe(val) for val in data]
        self._nSrc = int(np.sum([len(df) for df in self._data]))

    @property
    def nClusters(self) -> int:
        """Return the number of clusters in this cell"""
        return len(self._clusterDict)

    @property
    def clusterDict(self) -> OrderedDict[int, ClusterData]:
        """Return the cluster dictionary"""
        return self._clusterDict

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

    def reduceDataframe(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """Filters dataframe to keep only source in the cell"""
        if self._matcher.wcs is not None:
            xCell = dataframe["xPix"] - self._minPix[0]
            yCell = dataframe["yPix"] - self._minPix[1]
            filtered = (
                (xCell >= 0)
                & (xCell < self._nPix[0])
                & (yCell >= 0)
                & (yCell < self._nPix[1])
            )
        else:

            xCell = (dataframe["xCell_coadd"] + 100) / self._matcher._pixelMatchScale
            yCell = (dataframe["yCell_coadd"] + 100) / self._matcher._pixelMatchScale
            filtered = np.bitwise_and(
                dataframe["idx_x"] - self._idx[0] == 1,
                dataframe["idx_y"] - self._idx[1] == 1,
            )

        red = dataframe[filtered].copy(deep=True)
        red["xCell"] = xCell[filtered]
        red["yCell"] = yCell[filtered]
        return red

    def countsMap(self, weightName: str | None = None) -> np.ndarray:
        """Fill a map that counts the number of source per cell"""
        toFill = np.zeros((self._nPix))
        assert self._data is not None
        for df in self._data:
            toFill += utils.fillCountsMapFromDf(
                df, nPix=self._nPix, weightName=weightName
            )
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
            cluster = ClusterData(iCluster, footprint, np.array(sources).T)
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
        oDict = utils.getFootprints(countsMap, buf=self._buf)
        oDict["countsMap"] = countsMap
        assert self._data is not None
        self._footprintIds = utils.associateSourcesToFootprints(
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
        nObjects = np.zeros((nClust), dtype=int)
        nUniques = np.zeros((nClust), dtype=int)
        distRms = np.zeros((nClust), dtype=float)
        xCents = np.zeros((nClust), dtype=float)
        yCents = np.zeros((nClust), dtype=float)
        for idx, cluster in enumerate(self._clusterDict.values()):
            clusterIds[idx] = cluster.iCluster
            nSrcs[idx] = cluster.nSrc
            nObjects[idx] = len(cluster.objects)
            nUniques[idx] = cluster.nUnique
            distRms[idx] = cluster.rmsDist
            xCents[idx] = cluster.xCent
            yCents[idx] = cluster.yCent
        if self._matcher._wcs is not None:
            ra, dec = self._matcher.pixToWorld(xCents, yCents)
        else:
            ra, dec = np.repeat(np.nan, len(nSrcs)), np.repeat(np.nan, len(nSrcs))
            
        distRms *= self._matcher.pixToArcsec()

        data = dict(
            clusterIds=clusterIds,
            nSrcs=nSrcs,
            nObject=nObjects,
            nUnique=nUniques,
            distRms=distRms,
            ra=ra,
            decl=dec,
        )

        return pandas.DataFrame(data)

    def getObjectStats(self) -> pandas.DataFrame:
        """Get the stats for all the objects"""
        nObj = self.nObjects
        clusterIds = np.zeros((nObj), dtype=int)
        objectIds = np.zeros((nObj), dtype=int)
        nSrcs = np.zeros((nObj), dtype=int)
        distRms = np.zeros((nObj), dtype=float)
        xCents = np.zeros((nObj), dtype=float)
        yCents = np.zeros((nObj), dtype=float)
        for idx, obj in enumerate(self._objectDict.values()):
            clusterIds[idx] = obj.parentCluster.iCluster
            objectIds[idx] = obj.objectId
            nSrcs[idx] = obj.nSrc
            distRms[idx] = obj.rmsDist
            xCents[idx] = obj.xCent
            yCents[idx] = obj.yCent

        if self._matcher._wcs is not None:
            ra, dec = self._matcher.pixToWorld(xCents, yCents)
        else:
            ra, dec = np.repeat(np.nan, len(nSrcs)), np.repeat(np.nan, len(nSrcs))
        distRms *= self._matcher.pixToArcsec()

        data = dict(
            clusterIds=clusterIds,
            objectIds=objectIds,
            nSrcs=nSrcs,
            distRms=distRms,
            ra=ra,
            decl=dec,
        )

        return pandas.DataFrame(data)

    def getObjectShearStats(self)  -> pandas.DataFrame:
        """Get the shear stats for all the objects"""
        nObj = self.nObjects
        out_dict: dict[str, np.ndarray] = {}
        names = ['ns', '2p', '2m', '1p', '1m']

        for name_ in names:
            out_dict[f"n_{name_}"] = np.zeros((nObj), dtype=int)
            out_dict[f"g1_{name_}"] = np.zeros((nObj), dtype=float)
            out_dict[f"g2_{name_}"] = np.zeros((nObj), dtype=float)
        out_dict[f"good"] = np.zeros((nObj), dtype=bool)
        out_dict['delta_g_1'] = np.zeros((nObj), dtype=float)
        out_dict['delta_g_2'] = np.zeros((nObj), dtype=float)        
        for idx, obj in enumerate(self._objectDict.values()):
            objStats = obj.shearStats()
            for key, val in objStats.items():
                out_dict[key][idx] = val
        return pandas.DataFrame(out_dict)

    def getClusterShearStats(self)  -> pandas.DataFrame:
        """Get the shear stats for all the objects"""
        nClusters = self.nClusters
        out_dict: dict[str, np.ndarray] = {}
        names = ['ns', '2p', '2m', '1p', '1m']

        for name_ in names:
            out_dict[f"n_{name_}"] = np.zeros((nClusters), dtype=int)
            out_dict[f"g1_{name_}"] = np.zeros((nClusters), dtype=float)
            out_dict[f"g2_{name_}"] = np.zeros((nClusters), dtype=float)
        out_dict['delta_g_1'] = np.zeros((nClusters), dtype=float)
        out_dict['delta_g_2'] = np.zeros((nClusters), dtype=float)
        out_dict[f"good"] = np.zeros((nClusters), dtype=bool)
        for idx, cluster in enumerate(self._clusterDict.values()):
            clusterStats = cluster.shearStats()
            for key, val in clusterStats.items():
                out_dict[key][idx] = val
        return pandas.DataFrame(out_dict)

    def addObject(
        self, cluster: ClusterData, mask: np.ndarray | None = None
    ) -> ObjectData:
        """Add an object to this cell"""
        objectId = self.nObjects + self._idOffset
        newObject = ObjectData(cluster, objectId, mask)
        self._objectDict[objectId] = newObject
        return newObject
