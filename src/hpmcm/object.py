from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas

if TYPE_CHECKING:
    from .cell import CellData
    from .cluster import ClusterData

RECURSE_MAX = 20


class ObjectData:
    """Small class to define 'Objects', i.e., sets of associated sources"""

    def __init__(self, cluster: ClusterData, objectId: int, mask: np.ndarray | None):
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

        self._xCent: float | None = None
        self._yCent: float | None = None
        self._rmsDist: float | None = None
        self._xCell: np.ndarray | None = None
        self._yCell: np.ndarray | None = None
        self._dist2: np.ndarray | None = None
        self._snr: np.ndarray | None = None

    @property
    def parentCluster(self) -> ClusterData:
        """Return the parent cluster"""
        return self._parentCluster

    @property
    def objectId(self) -> int:
        """Return the object id"""
        return self._objectId

    @property
    def nSrc(self) -> int:
        """Return the number of sources associated to the object"""
        return self._nSrc

    @property
    def nUnique(self) -> int:
        """Return the number of catalogs contributing sources to the object"""
        return self._nUnique

    @property
    def xCent(self) -> float | None:
        """Return the x-position of the centroid of the object"""
        return self._xCent

    @property
    def yCent(self) -> float | None:
        """Return the y-position of the centroid of the object"""
        return self._yCent

    @property
    def rmsDist(self) -> float | None:
        """Return the RMS distance of the sources in the object"""
        return self._rmsDist

    @property
    def xCell(self) -> np.ndarray | None:
        """Return the x-position of the sources within the cell"""
        return self._xCell

    @property
    def yCell(self) -> np.ndarray | None:
        """Return the y-position of the sources within the cell"""
        return self._yCell

    @property
    def snr(self) -> np.ndarray | None:
        """Return the signal to noise of the sources in the object"""
        return self._snr

    @property
    def dist2(self) -> np.ndarray | None:
        """Return an array with the distance squared (in cells)
        between each source and the object centroid"""
        return self._dist2

    def _updateCatIndices(self) -> None:
        self._catIndices = self._parentCluster.catIndices[self._mask]
        self._nSrc = self._catIndices.size
        self._nUnique = np.unique(self._catIndices).size

    def sourceIds(self) -> np.ndarray:
        return self._parentCluster.sourceIds[self._mask]

    def processObject(
        self, cellData: CellData, pixelR2Cut: float, recurse: int = 0
    ) -> None:
        """Recursively process an object and make sub-objects"""
        if recurse > RECURSE_MAX:
            print("Recursion limit: ", self._nSrc, self._nUnique)
            return
        if self._nSrc == 0:
            print("Empty object", self._nSrc, self._nUnique, recurse)
            return
        assert (
            self._parentCluster.xCell is not None
            and self._parentCluster.yCell is not None
            and self._parentCluster.snr is not None
        )

        self._data = self._parentCluster.data.iloc[self._mask]
        self._xCell = self._parentCluster.xCell[self._mask]
        self._yCell = self._parentCluster.yCell[self._mask]
        self._snr = self._parentCluster.snr[self._mask]

        if self._mask.sum() == 1:
            self._xCent = self._xCell[0]
            self._yCent = self._yCell[0]
            self._dist2 = np.zeros((1), float)
            self._rmsDist = 0.0
            self._updateCatIndices()
            return

        assert self._snr is not None
        sumSnr = np.sum(self._snr)
        self._xCent = np.sum(self._xCell * self._snr) / sumSnr
        self._yCent = np.sum(self._yCell * self._snr) / sumSnr
        self._dist2 = np.array((self._xCent - self._xCell) ** 2 + (self._yCent - self._yCell) ** 2)
        self._rmsDist = np.sqrt(np.mean(self._dist2))

        subMask = self._dist2 < pixelR2Cut
        if subMask.all() and self._nSrc == self._nUnique:
            return

        self.splitObject(cellData, pixelR2Cut, recurse=recurse + 1)
        self._updateCatIndices()
        return
        
    def splitObject(
        self, cellData: CellData, pixelR2Cut: float, recurse: int = 0
    ) -> None:
        """Split up a cluster keeping only one source per input
        catalog
        """
        pass
