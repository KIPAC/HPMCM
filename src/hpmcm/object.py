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
    parentCluster : ClusterData
        Parent cluster that this object is build from

    objectId : int
        Unique object identifier

    mask : np.ndarray[bool]
        Mask of which source in parent cluster are in this object

    catIndices: np.ndarray[int]
        Indicies showing which catalog each source belongs to

    nSrc: int
        Number of sources in this object

    nUnique: int
        Number of catalogs contributing to this object

    data: pandas.DataFrame
        Data for the sources in this object

    recurse: int
        Recursion level needed to make this object

    xCent: float
        X Centeroid of this object in global pixel coordinates

    yCent: float
        Y Centeroid of this object in global pixel coordinates

    rmsDist: float
        RMS distance of sources to the centroid

    dist2: np.ndarray
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
        self.parentCluster: ClusterData = cluster
        self.objectId: int = objectId
        if mask is None:
            self.mask = np.ones((self.parentCluster.nSrc), dtype=bool)
        else:
            self.mask = mask
        self.catIndices: np.ndarray = self.parentCluster.catIndices[self.mask]
        self.nSrc: int = self.catIndices.size
        self.nUnique: int = np.unique(self.catIndices).size
        self.data: pandas.DataFrame | None = None

        self.recurse: int = recurse
        self.xCent: float = np.nan
        self.yCent: float = np.nan
        self.rmsDist: float = np.nan
        self.dist2: np.ndarray = np.array([])
        self._extract()

    @property
    def xCluster(self) -> np.ndarray:
        """Return the x-position of the sources within the footprint"""
        assert self.data is not None
        return self.data.xCluster

    @property
    def yCluster(self) -> np.ndarray:
        """Return the y-position of the sources within the footprint"""
        assert self.data is not None
        return self.data.yCluster

    @property
    def xPix(self) -> np.ndarray:
        """Return the x-position of the sources within the cell"""
        assert self.data is not None
        return self.data.xPix

    @property
    def yPix(self) -> np.ndarray:
        """Return the y-position of the sources within the cell"""
        assert self.data is not None
        return self.data.yPix

    def sourceIds(self) -> np.ndarray:
        """Return the source ids for the sources in the object"""
        return self.parentCluster.srcIds[self.mask]

    def sourceIdxs(self) -> np.ndarray:
        """Return the source indices for the sources in the object"""
        return self.parentCluster.srcIdxs[self.mask]

    def _updateCatIndices(self) -> None:
        self.catIndices = self.parentCluster.catIndices[self.mask]
        self.nSrc = self.catIndices.size
        self.nUnique = np.unique(self.catIndices).size

    def _extract(self) -> None:
        assert self.parentCluster.data is not None
        self.data = self.parentCluster.data.iloc[self.mask]
        self._updateCatIndices()

    def processObject(
        self, cellData: CellData, pixelR2Cut: float, recurse: int = 0
    ) -> None:
        """Recursively process an object and make sub-objects"""
        if recurse > RECURSE_MAX:
            return
        self.recurse = recurse

        if self.nSrc == 0:
            print("Empty object", self.nSrc, self.nUnique, recurse)
            return

        assert self.data is not None
        if self.mask.sum() == 1:
            self.xCent = self.data.xPix.values[0]
            self.yCent = self.data.yPix.values[0]
            self.dist2 = np.zeros((1), float)
            self.rmsDist = 0.0
            self.snrMean = self.data.SNR.values[0]
            self.snrRms = 0.0            
            return

        sumSnr = np.sum(self.data.SNR)
        self.xCent = np.sum(self.data.xPix * self.data.SNR) / sumSnr
        self.yCent = np.sum(self.data.yPix * self.data.SNR) / sumSnr
        self.dist2 = np.array(
            (self.xCent - self.data.xPix) ** 2 + (self.yCent - self.data.yPix) ** 2
        )
        self.rmsDist = np.sqrt(np.mean(self.dist2))
        self.snrMean = np.mean(self.data.SNR.values)
        self.snrRms = np.std(self.data.SNR.values)

        subMask = self.dist2 < pixelR2Cut
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
        self.recurse = recurse
        self._extract()

        assert self.data is not None

        bbox = self.parentCluster.footprint.getBBox()
        zoomFactors = [1, 2, 4, 8, 16, 32]
        zoomFactor = zoomFactors[recurse]

        nPix = zoomFactor * np.array([bbox.getHeight(), bbox.getWidth()])
        zoomX = zoomFactor * self.data.xCluster / self.parentCluster.pixelMatchScale
        zoomY = zoomFactor * self.data.yCluster / self.parentCluster.pixelMatchScale

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
        biggestMask = np.zeros(self.mask.shape, dtype=bool)

        for iFp in range(nFootprints):
            subMask = footprintIds == iFp
            count = 0
            newMask = np.zeros(self.mask.shape, dtype=bool)
            for i, val in enumerate(self.mask):
                if val:
                    newMask[i] = subMask[count]
                    count += 1
            assert subMask.sum() == newMask.sum()

            if iFp == biggest:
                biggestMask = newMask
                continue

            newObject = self.parentCluster.addObject(cellData, newMask)
            newObject.processObject(cellData, pixelR2Cut, recurse=recurse)

        self.mask = biggestMask
        self._extract()
        self.processObject(cellData, pixelR2Cut, recurse=recurse)
        return


class ShearObjectData(ObjectData):
    """Subclass of ObjectData that can compute shear statisitics"""

    def shearStats(self) -> dict:
        """Return the shear statistics"""
        assert self.data is not None
        return shear_utils.shearStats(self.data)
