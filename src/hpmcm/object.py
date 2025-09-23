from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas

from . import utils

if TYPE_CHECKING:
    from .cell import CellData
    from .cluster import ClusterData

RECURSE_MAX = 4


class ObjectData:
    """Small class to define 'Objects', i.e., sets of associated sources"""

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
        self._catIndices: np.ndarray = self._parentCluster._catIndices[self._mask]
        self._nSrc: int = self._catIndices.size
        self._nUnique: int = np.unique(self._catIndices).size
        self._data: pandas.DataFrame | None = None

        self._recurse: int = recurse
        self._xCent: float = np.nan
        self._yCent: float = np.nan
        self._rmsDist: float = np.nan
        self._xCell: np.ndarray = np.array([])
        self._yCell: np.ndarray = np.array([])
        self._xCluster: np.ndarray = np.array([])
        self._yCluster: np.ndarray = np.array([])
        self._dist2: np.ndarray = np.array([])
        self._snr: np.ndarray = np.array([])
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
    def xCluster(self) -> np.ndarray:
        """Return the x-position of the sources within the footprint"""
        return self._xCluster

    @property
    def yCluster(self) -> np.ndarray:
        """Return the y-position of the sources within the footprint"""
        return self._yCluster

    @property
    def rmsDist(self) -> float:
        """Return the RMS distance of the sources in the object"""
        return self._rmsDist

    @property
    def xCell(self) -> np.ndarray:
        """Return the x-position of the sources within the cell"""
        return self._xCell

    @property
    def yCell(self) -> np.ndarray:
        """Return the y-position of the sources within the cell"""
        return self._yCell

    @property
    def snr(self) -> np.ndarray:
        """Return the signal to noise of the sources in the object"""
        return self._snr

    @property
    def dist2(self) -> np.ndarray:
        """Return an array with the distance squared (in cells)
        between each source and the object centroid"""
        return self._dist2

    def _updateCatIndices(self) -> None:
        self._catIndices = self._parentCluster.catIndices[self._mask]
        self._nSrc = self._catIndices.size
        self._nUnique = np.unique(self._catIndices).size

    def sourceIds(self) -> np.ndarray:
        return self._parentCluster.sourceIds[self._mask]

    def _extract(self) -> None:
        assert (
            self._parentCluster.xCell.size
            and self._parentCluster.yCell.size
            and self._parentCluster.snr.size
        )
        assert self._parentCluster.data is not None
        self._data = self._parentCluster.data.iloc[self._mask]
        self._xCell = self._parentCluster.xCell[self._mask]
        self._yCell = self._parentCluster.yCell[self._mask]
        self._xCluster = self._parentCluster.xCluster[self._mask]
        self._yCluster = self._parentCluster.yCluster[self._mask]
        self._snr = self._parentCluster.snr[self._mask]
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

        if self._mask.sum() == 1:
            self._xCent = self._xCell[0]
            self._yCent = self._yCell[0]
            self._dist2 = np.zeros((1), float)
            self._rmsDist = 0.0
            return

        assert self._snr.size
        sumSnr = np.sum(self._snr)
        self._xCent = np.sum(self._xCell * self._snr) / sumSnr
        self._yCent = np.sum(self._yCell * self._snr) / sumSnr
        self._dist2 = np.array(
            (self._xCent - self._xCell) ** 2 + (self._yCent - self._yCell) ** 2
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

        bbox = self._parentCluster.footprint.getBBox()
        zoom_factors = [1, 2, 4, 8, 16, 32]
        zoom_factor = zoom_factors[recurse]

        nPix = zoom_factor * np.array([bbox.getHeight(), bbox.getWidth()])
        zoom_x = zoom_factor * self._xCluster
        zoom_y = zoom_factor * self._yCluster

        countsMap = utils.fillCountsMapFromArrays(zoom_x, zoom_y, nPix)

        fpDict = utils.getFootprints(countsMap, buf=0)
        footprints = fpDict["footprints"]
        n_footprints = len(footprints.getFootprints())
        if n_footprints == 1:
            if recurse >= RECURSE_MAX:
                return
            self.splitObject(cellData, pixelR2Cut, recurse=recurse + 1)
            return

        footprintKey = fpDict["footprintKey"]
        footprintIds = utils.findClusterIdsFromArrays(zoom_x, zoom_y, footprintKey)

        biggest = np.argmax(np.bincount(footprintIds))
        biggest_mask = np.zeros(self._mask.shape, dtype=bool)

        for i_fp in range(n_footprints):
            subMask = footprintIds == i_fp
            count = 0
            newMask = np.zeros(self._mask.shape, dtype=bool)
            for i, val in enumerate(self._mask):
                if val:
                    newMask[i] = subMask[count]
                    count += 1
            assert subMask.sum() == newMask.sum()

            if i_fp == biggest:
                biggest_mask = newMask
                continue

            newObject = self._parentCluster.addObject(cellData, newMask)
            newObject.processObject(cellData, pixelR2Cut, recurse=recurse)

        self._mask = biggest_mask
        self._extract()
        self.processObject(cellData, pixelR2Cut, recurse=recurse)
        return
