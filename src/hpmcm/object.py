from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas

from . import shear_utils, utils

if TYPE_CHECKING:
    from .cell import CellData
    from .cluster import ClusterData


RECURSE_MAX = 4


class ObjectData:
    """Small class to define 'Objects', i.e., sets of associated sources

    Attributes
    ----------
    _parentCluster : ClusterData
        Parent cluster that this object is build from

    _objectId : int
        Unique object identifier

    _mask : np.ndarray[bool]
        Mask of which source in parent cluster are in this object

    _catIndices: np.ndarray
        Indicies showing which catalog each source belongs to

    _nSrc: int
        Number of sources in this object

    _nUnique: int
        Number of catalogs contributing to this object

    _data: pandas.DataFrame
        Data for the sources in this object

    _recurse: int
        Recursion level needed to make this object

    _xCent: float
        X Centeroid of this object in global pixel coordinates

    _yCent: float
        Y Centeroid of this object in global pixel coordinates

    _rmsDist: float
        RMS distance of sources to the centroid

    _dist2: np.ndarray
        Distances of sources to the centroid
    """

    def __init__(
        self,
        cluster: ClusterData,
        objectId: int,
        mask: np.ndarray | None,
        recurse: int = 0,
    ):
        """Build from `ClusterData`, an objectId and mask specifying with sources
        in the cluster are part of the object"""
        self._parentCluster: ClusterData = cluster
        self._objectId: int = objectId
        if mask is None:
            self._mask = np.ones((self._parentCluster.nSrc), dtype=bool)
        else:
            self._mask = mask
        self._catIndices: np.ndarray = self._parentCluster.catIndices[self._mask]
        self._nSrc: int = self._catIndices.size
        self._nUnique: int = np.unique(self._catIndices).size
        self._data: pandas.DataFrame | None = None

        self._recurse: int = recurse
        self._xCent: float = np.nan
        self._yCent: float = np.nan
        self._rmsDist: float = np.nan
        self._dist2: np.ndarray = np.array([])
        self._extract()

    @property
    def parentCluster(self) -> ClusterData:
        """Return the parent cluster"""
        return self._parentCluster

    @property
    def objectId(self) -> int:
        """Return the object id"""
        return self._objectId

    @property
    def catIndices(self) -> np.ndarray:
        """Return the catalog indcies associated to this object"""
        return self._catIndices

    @property
    def data(self) -> pandas.DataFrame | None:
        """Return the data for this object"""
        return self._data

    @property
    def mask(self) -> np.ndarray:
        """Return the mask of object membership in the cluster"""
        return self._mask

    @property
    def nSrc(self) -> int:
        """Return the number of sources associated to the object"""
        return self._nSrc

    @property
    def nUnique(self) -> int:
        """Return the number of catalogs contributing sources to the object"""
        return self._nUnique

    @property
    def xCent(self) -> float:
        """Return the x-position of the centroid of the object"""
        return self._xCent

    @property
    def yCent(self) -> float:
        """Return the y-position of the centroid of the object"""
        return self._yCent

    @property
    def rmsDist(self) -> float:
        """Return the RMS distance of the sources in the object"""
        return self._rmsDist

    @property
    def dist2(self) -> np.ndarray:
        """Return an array with the distance squared (in cells)
        between each source and the object centroid"""
        return self._dist2

    @property
    def xCluster(self) -> np.ndarray:
        """Return the x-position of the sources within the footprint"""
        assert self._data is not None
        return self._data.xCluster

    @property
    def yCluster(self) -> np.ndarray:
        """Return the y-position of the sources within the footprint"""
        assert self._data is not None
        return self._data.yCluster

    @property
    def xPix(self) -> np.ndarray:
        """Return the x-position of the sources within the cell"""
        assert self._data is not None
        return self._data.xPix

    @property
    def yPix(self) -> np.ndarray:
        """Return the y-position of the sources within the cell"""
        assert self._data is not None
        return self._data.yPix

    def sourceIds(self) -> np.ndarray:
        """Return the source ids for the sources in the object"""
        return self._parentCluster.srcIds[self._mask]

    def sourceIdxs(self) -> np.ndarray:
        """Return the source indices for the sources in the object"""
        return self._parentCluster.srcIdxs[self._mask]

    def _updateCatIndices(self) -> None:
        self._catIndices = self._parentCluster.catIndices[self._mask]
        self._nSrc = self._catIndices.size
        self._nUnique = np.unique(self._catIndices).size

    def _extract(self) -> None:
        assert self._parentCluster.data is not None
        self._data = self._parentCluster.data.iloc[self._mask]
        self._updateCatIndices()

    def processObject(
        self, cellData: CellData, pixelR2Cut: float, recurse: int = 0
    ) -> None:
        """Recursively process an object and make sub-objects"""
        if recurse > RECURSE_MAX:
            return
        self._recurse = recurse

        if self._nSrc == 0:
            print("Empty object", self._nSrc, self._nUnique, recurse)
            return

        assert self._data is not None
        if self._mask.sum() == 1:
            self._xCent = self._data.xPix.values[0]
            self._yCent = self._data.yPix.values[0]
            self._dist2 = np.zeros((1), float)
            self._rmsDist = 0.0
            return

        sumSnr = np.sum(self._data.SNR)
        self._xCent = np.sum(self._data.xPix * self._data.SNR) / sumSnr
        self._yCent = np.sum(self._data.yPix * self._data.SNR) / sumSnr
        self._dist2 = np.array(
            (self._xCent - self._data.xPix) ** 2 + (self._yCent - self._data.yPix) ** 2
        )
        self._rmsDist = np.sqrt(np.mean(self._dist2))

        subMask = self._dist2 < pixelR2Cut
        if subMask.all():
            return

        if recurse >= RECURSE_MAX:
            return

        self.splitObject(cellData, pixelR2Cut, recurse=recurse + 1)
        return

    def splitObject(
        self, cellData: CellData, pixelR2Cut: float, recurse: int = 0
    ) -> None:
        """Split up a cluster keeping only one source per input
        catalog
        """
        if recurse > RECURSE_MAX:
            return
        self._recurse = recurse
        self._extract()

        assert self._data is not None

        bbox = self._parentCluster.footprint.getBBox()
        zoomFactors = [1, 2, 4, 8, 16, 32]
        zoomFactor = zoomFactors[recurse]

        nPix = zoomFactor * np.array([bbox.getHeight(), bbox.getWidth()])
        zoomX = zoomFactor * self._data.xCluster / self._parentCluster.pixelMatchScale()
        zoomY = zoomFactor * self._data.yCluster / self._parentCluster.pixelMatchScale()

        countsMap = utils.fillCountsMapFromArrays(zoomX, zoomY, nPix)

        fpDict = utils.getFootprints(countsMap, buf=0)
        footprints = fpDict["footprints"]
        nFootprints = len(footprints.getFootprints())
        if nFootprints == 1:
            if recurse >= RECURSE_MAX:
                return
            self.splitObject(cellData, pixelR2Cut, recurse=recurse + 1)
            return

        footprintKey = fpDict["footprintKey"]
        footprintIds = utils.findClusterIdsFromArrays(zoomX, zoomY, footprintKey)

        biggest = np.argmax(np.bincount(footprintIds))
        biggestMask = np.zeros(self._mask.shape, dtype=bool)

        for iFp in range(nFootprints):
            subMask = footprintIds == iFp
            count = 0
            newMask = np.zeros(self._mask.shape, dtype=bool)
            for i, val in enumerate(self._mask):
                if val:
                    newMask[i] = subMask[count]
                    count += 1
            assert subMask.sum() == newMask.sum()

            if iFp == biggest:
                biggestMask = newMask
                continue

            newObject = self._parentCluster.addObject(cellData, newMask)
            newObject.processObject(cellData, pixelR2Cut, recurse=recurse)

        self._mask = biggestMask
        self._extract()
        self.processObject(cellData, pixelR2Cut, recurse=recurse)
        return


class ShearObjectData(ObjectData):
    """Subclass of ObjectData that can compute shear statisitics"""

    def shearStats(self) -> dict:
        """Return the shear statistics"""
        assert self._data is not None
        return shear_utils.shearStats(self._data)
