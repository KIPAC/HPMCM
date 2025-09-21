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

        self._xCent: np.ndarray | None = None
        self._yCent: np.ndarray | None = None
        self._dist2: np.ndarray | None = None
        self._rmsDist: np.ndarray | None = None
        self._xCell: np.ndarray | None = None
        self._yCell: np.ndarray | None = None
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
            self._parentCluster.xCell
            and self._parentCluster.yCell
            and self._parentCluster.snr
        )

        # FIXME, update self._data

        xCell = self._parentCluster.xCell[self._mask]
        yCell = self._parentCluster.yCell[self._mask]
        snr = self._parentCluster.snr[self._mask]
        self._parentCluster.extract(cellData)
        xCell = self._parentCluster.xCell[self._mask]
        yCell = self._parentCluster.yCell[self._mask]

        if self._mask.sum() == 1:
            self._xCent = np.array([xCell[0]])
            self._yCent = np.array([yCell[0]])
            self._dist2 = np.zeros((1), float)
            self._rmsDist = np.array([0.0])
            self._xCell = np.array([xCell[0]])
            self._yCell = np.array([yCell[0]])
            self._snr = np.array([snr[0]])
            self._updateCatIndices()
            return

        sumSnr = np.sum(snr)
        self._xCent = np.sum(xCell * snr) / sumSnr
        self._yCent = np.sum(yCell * snr) / sumSnr
        self._dist2 = np.array((self._xCent - xCell) ** 2 + (self._yCent - yCell) ** 2)
        self._xCell = xCell
        self._yCell = yCell
        self._snr = snr

        self._rmsDist = np.sqrt(np.mean(self._dist2))
        subMask = self._dist2 < pixelR2Cut
        if subMask.all():
            if self._nSrc != self._nUnique:
                self.splitObject(cellData, pixelR2Cut, recurse=recurse + 1)
            self._updateCatIndices()
            return

        if not subMask.any():
            idx = np.argmax(snr)
            self._xCent = np.array([xCell[idx]])
            self._yCent = np.array([yCell[idx]])
            self._dist2 = np.array(
                (self._xCent - xCell) ** 2 + (self._yCent - yCell) ** 2
            )
            self._rmsDist = np.array([np.sqrt(np.mean(self._dist2))])
            self._xCell = np.array([xCell[idx]])
            self._yCell = np.array([yCell[idx]])
            self._snr = np.array([snr[idx]])

            subMask = self._dist2 < pixelR2Cut

        newObjMask = self._mask.copy()
        newObjMask[newObjMask] *= subMask

        newObject = self._parentCluster.addObject(cellData, newObjMask)
        newObject.processObject(cellData, pixelR2Cut)

        self._mask[self._mask] *= ~subMask
        self._updateCatIndices()
        self.processObject(cellData, pixelR2Cut, recurse=recurse + 1)

    def splitObject(
        self, cellData: CellData, pixelR2Cut: float, recurse: int = 0
    ) -> None:
        """Split up a cluster keeping only one source per input
        catalog, choosing the one closest to the cluster center"""
        assert self._dist2
        sortIdx = np.argsort(self._dist2)
        mask = np.ones((self._nSrc), dtype=bool)
        usedCats = {}
        for iSrc, catIdx in zip(sortIdx, self._catIndices[sortIdx]):
            if catIdx not in usedCats:
                usedCats[catIdx] = 1
                continue
            usedCats[catIdx] += 1
            mask[iSrc] = False

        newObjMask = self._mask.copy()
        newObjMask[newObjMask] *= mask

        newObject = self._parentCluster.addObject(cellData, newObjMask)
        newObject.processObject(cellData, pixelR2Cut)

        self._mask[self._mask] *= ~mask
        self._updateCatIndices()
        self.processObject(cellData, pixelR2Cut, recurse=recurse + 1)
