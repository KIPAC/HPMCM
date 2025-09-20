from __future__ import annotations

from typing import TYPE_CHECKING
from collections import OrderedDict

import np as np

from astropy.table import Table
import pandas

import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage

from .object import ObjectData
from .cluser import ClusterData

if TYPE_CHECKING:
    from .match import Match


class CellData:
    """ Class to analyze data for a cell

    Includes cell boundries, reduced data tables
    and clustering results

    Does not store sky maps

    Cells are square sub-regions of the Skymap
    constructed with the WCS

    The cell covers corner:corner+size

    The sources are projected into an array that extends `buf` pixels
    beyond the cell

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

    TODO:  Add code to filter out clusters centered in the buffer
    """
    def __init__(
        self,
        matcher: Match,
        idOffset: int,
        corner: np.ndarray,
        size: np.ndarray,
        buf: int=10,
    ):
        self._matcher = matcher
        self._idOffset = idOffset # Offset used for the Object and Cluster IDs for this cell
        self._corner = corner # cellX, cellY for corner of cell
        self._size = size # size of cell
        self._buf = buf
        self._minPix = corner - buf
        self._maxPix = corner + size + buf
        self._nPix = self._maxPix - self._minPix
        self._data: list[pandas.DataFrame]|None = None
        self._nSrc: int|None = None
        self._footprintIds: list[np.ndarray]|None = None
        self._clusterDict: OrderedDict[int, ClusterData] = OrderedDict()
        self._objectDict: OrderedDict[int, ObjectData] = OrderedDict()

    def reduceData(self, data: pandas.DataFrame) -> None:
        """ Pull out only the data needed for this cell """
        self._data = [self.reduceDataframe(val) for val in data]
        self._nSrc = sum([len(df) for df in self._data])
        
    @property
    def nClusters(self) -> int:
        """ Return the number of clusters in this cell """
        return len(self._clusterDict)

    @property
    def nObjects(self) -> int:
        """ Return the number of objects in this cell """
        return len(self._objectDict)

    @property
    def data(self) -> pandas.DataFrame:
        """ Return the data associated to this cell """
        return self._data

    @property
    def clusterDist(self) -> OrderedDict[int, ClusterData]:
        """ Return a dictionary mapping clusters Ids to clusters """
        return self._clusterDict

    def reduceDataframe(self, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """ Filters dataframe to keep only source in the cell """
        xLocal = dataframe['xcell'] - self._minPix[0]
        yLocal = dataframe['ycell'] - self._minPix[1]
        filtered = (xLocal >= 0) & (xLocal < self._nPix[0]) & (yLocal >= 0) & (yLocal < self._nPix[1])
        red = dataframe[filtered].copy(deep=True)
        red['xlocal'] = xLocal[filtered]
        red['ylocal'] = yLocal[filtered]
        return red

    def countsMap(self, weightName: str|None=None) -> np.ndarray:
        """ Fill a map that counts the number of source per cell """
        toFill = np.zeros((self._nPix))
        assert self._data
        for df in self._data:
            toFill += self.fillCellFromDf(df, weightName=weightName)
        return toFill

    def associateSourcesToFootprints(self, clusterKey: int) -> None:
        """ Loop through data and associate sources to clusters """
        assert self._data
        self._footprintIds = [self.findClusterIds(df, clusterKey) for df in self._data]

    def buildClusterData(
        self,
        fpSet: afwDetect.FootprintSet,
        pixelR2Cut: float=4.,
    ) -> None:
        """ Loop through cluster ids and collect sources into
        the ClusterData objects """
        footprints = fpSet.getFootprints()
        footprintDict = {}
        nMissing = 0
        nFound = 0
        assert self._data
        assert self._footprintIds
        for iCat, (df, footprintIds) in enumerate(zip(self._data, self._footprintIds)):
            for srcIdx, (srcId, footprintId) in enumerate(zip(df['id'], footprintIds)):
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
            iCluster = footprintId+self._idOffset
            cluster = ClusterData(iCluster, footprint, np.array(sources).T)
            self._clusterDict[iCluster] = cluster
            cluster.processCluster(self, pixelR2Cut)

    def analyze(self, weightName: str|None=None, pixelR2Cut: float=4.) -> dict | None:
        """ Analyze this cell

        Note that this returns the counts maps and clustering info,
        which can be helpful for debugging.
        """
        if self._nSrc == 0:
            return None
        countsMap = self.countsMap(weightName)
        oDict = self.getFootprints(countsMap)
        oDict['countsMap'] = countsMap
        self.associateSourcesToFootprints(oDict['footprintKey'])
        self.buildClusterData(oDict['footprints'], pixelR2Cut)
        return oDict

    @staticmethod
    def findClusterIds(df: pandas.DataFrame, clusterKey: np.ndarray) -> np.ndarray:
        """ Associate sources to clusters using `clusterkey`
        which is a map where any pixel associated to a cluster
        has the cluster index as its value """
        return np.array([clusterKey[yLocal,xLocal] for xLocal, yLocal in zip(df['xlocal'], df['ylocal'])]).astype(np.int32)

    def fillCellFromDf(self, df: pandas.DataFrame, weightName: str|None=None) -> np.ndarray:
        """ Fill a source counts map from a reduced dataframe for one input
        catalog """
        if weightName is None:
            weights = None
        else:
            weights = df[weightName].values
        hist = np.histogram2d(df['xlocal'], df['ylocal'], bins=self._nPix,
                              range=((0, self._nPix[0]),
                                     (0, self._nPix[1])),
                              weights=weights)
        return hist[0]

    @staticmethod
    def filterFootprints(fpSet: afwDetect.FootprintSet, buf: int) -> afwDetect.FootprintSet:
        """ Remove footprints within `buf` pixels of the celll edge """
        region = fpSet.getRegion()
        width, height = region.getWidth(), region.getHeight()
        outList = []
        maxX = width - buf
        maxY = height - buf
        for fp in fpSet.getFootprints():
            cent = fp.getCentroid()
            xC = cent.getX()
            yC = cent.getY()
            if xC < buf or xC > maxX or yC < buf or yC > maxY:
                continue
            outList.append(fp)
        fpSetOut = afwDetect.FootprintSet(fpSet.getRegion())
        fpSetOut.setFootprints(outList)
        return fpSetOut

    def getFootprints(self, countsMap: np.ndarray) -> dict:
        """ Take a source counts map and do clustering using Footprint detection
        """
        image = afwImage.ImageF(countsMap.astype(np.float32))
        footprintsOrig = afwDetect.FootprintSet(image, afwDetect.Threshold(0.5))
        footprints = self.filterFootprints(footprintsOrig, self._buf)
        footprintKey = afwImage.ImageI(np.full(countsMap.shape, -1, dtype=np.int32))
        for i, footprint in enumerate(footprints.getFootprints()):
            footprint.spans.setImage(footprintKey, i, doClip=True)
        return dict(image=image, footprints=footprints, footprintKey=footprintKey)

    def getClusterAssociations(self) -> Table:
        """ Convert the clusters to a set of associations """
        clusterIds = []
        sourceIds = []
        distances = []
        for cluster in self._clusterDict.values():
            clusterIds.append(np.full((cluster.nSrc), cluster.iCluster, dtype=int))
            sourceIds.append(cluster.sourceIds)
            distances.append(cluster.dist2)
        if not distances:
            return Table(dict(distance=[], id=np.array([], int), object=np.array([], int)))
        distances = np.hstack(distances)
        distances = self._matcher.cellToArcsec() * np.sqrt(distances)
        data = dict(object=np.hstack(clusterIds),
                    id=np.hstack(sourceIds),
                    distance=distances)
        return Table(data)

    def getObjectAssociations(self) -> Table:
        clusterIds = []
        objectIds = []
        sourceIds = []
        distances = []
        for obj in self._objectDict.values():
            clusterIds.append(np.full((obj._nSrc), obj._parentCluster.iCluster, dtype=int))
            objectIds.append(np.full((obj._nSrc), obj._objectId, dtype=int))
            sourceIds.append(obj.sourceIds())
            distances.append(obj.dist2)
        if not distances:
            return Table(dict(object=np.array([], int),
                              parent=np.array([], int),
                              id=np.array([], int),
                              distance=[]))
        distances = np.hstack(distances)
        distances = self._matcher.cellToArcsec() * np.sqrt(distances)            
        data = dict(object=np.hstack(objectIds),
                    parent=np.hstack(clusterIds),
                    id=np.hstack(sourceIds),
                    distance=distances)
        return Table(data)

    def getClusterStats(self) -> Table:
        """ Convert the clusters to a set of associations """
        nClust = self.nClusters
        clusterIds = np.zeros((nClust), dtype=int)
        nSrcs = np.zeros((nClust), dtype=int)
        nObjects = np.zeros((nClust), dtype=int)
        nUniques = np.zeros((nClust), dtype=int)
        distRms = np.zeros((nClust), dtype=float)
        xCents = np.zeros((nClust), dtype=float)
        yCents = np.zeros((nClust), dtype=float)
        for idx, cluster in enumerate(self._clusterDict.values()):
            clusterIds[idx] = cluster._iCluster
            nSrcs[idx] = cluster.nSrc
            nObjects[idx] = len(cluster._objects)
            nUniques[idx] = cluster.nUnique
            distRms[idx] = cluster._rmsDist
            xCents[idx] = cluster._xCent
            yCents[idx] = cluster._yCent
        ra, decl = self._matcher.cellToWorld(xCents, yCents)
        distRms *= self._matcher.cellToArcsec()

        data = dict(clusterIds=clusterIds,
                    nSrcs=nSrcs,
                    nObject=nObjects,
                    nUnique=nUniques,
                    distRms=distRms,
                    ra=ra,
                    decl=decl)

        return Table(data)

    def getObjectStats(self) -> Table:
        """ Convert the clusters to a set of associations """
        nObj = self.nObjects
        clusterIds = np.zeros((nObj), dtype=int)
        objectIds = np.zeros((nObj), dtype=int)
        nSrcs = np.zeros((nObj), dtype=int)
        distRms = np.zeros((nObj), dtype=float)
        xCents = np.zeros((nObj), dtype=float)
        yCents = np.zeros((nObj), dtype=float)
        for idx, obj in enumerate(self._objectDict.values()):
            clusterIds[idx] = obj._parentCluster._iCluster
            objectIds[idx] = obj._objectId
            nSrcs[idx] = obj.nSrc
            distRms[idx] = obj._rmsDist
            xCents[idx] = obj._xCent
            yCents[idx] = obj._yCent

        ra, decl = self._matcher.cellToWorld(xCents, yCents)
        distRms *= self._matcher.cellToArcsec()
        
        data = dict(clusterIds=clusterIds,
                    objectIds=objectIds,
                    nSrcs=nSrcs,
                    distRms=distRms,
                    ra=ra,
                    decl=decl)

        return Table(data)

    def addObject(self, cluster: ClusterData, mask: np.ndarray|None=None) -> ObjectData:
        """ Add an object to this cell """
        objectId = self.nObjects + self._idOffset
        newObject = ObjectData(cluster, objectId, mask)
        self._objectDict[objectId] = newObject
        return newObject
