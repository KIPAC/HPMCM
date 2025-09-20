from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cell import CellData
    from .cluster import ClusterData


RECURSE_MAX = 20


class ObjectData:
    """Small class to define 'Objects', i.e., sets of associated sources"""

    def __init__(self, cluster: ClusterData, objectId: int, mask: np.ndarray):
        """Build from `ClusterData`, an objectId and mask specifying with sources
        in the cluster are part of the object"""
        self._parentCluster = cluster
        self._objectId = objectId
        if mask is None:
            self._mask = np.ones((self._parentCluster.nSrc), dtype=bool)
        else:
            self._mask = mask
        self._catIndices = self._parentCluster._catIndices[self._mask]
        self._nSrc = self._catIndices.size
        self._nUnique = np.unique(self._catIndices).size

        self._dist2: np.ndarray | None = None
        self._xCent = None
        self._yCent = None
        self._local_x = None
        self._local_y = None
        self._g1 = None
        self._g2 = None
        self._rmsDist = None
        self.xCell = None
        self.yCell = None
        self.snr = None

    @property
    def nSrc(self) -> int:
        """Return the number of sources associated to the object"""
        return self._nSrc

    @property
    def nUnique(self) -> int:
        """Return the number of catalogs contributing sources to the object"""
        return self._nUnique

    @property
    def dist2(self) -> np.ndarray:
        """Return an array with the distance squared (in cells)
        between each source and the object centroid"""
        return self._dist2

    def _updateCatIndices(self) -> None:
        self._catIndices = self._parentCluster._catIndices[self._mask]
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

        xCell = self._parentCluster.xCell[self._mask]
        yCell = self._parentCluster.yCell[self._mask]
        snr = self._parentCluster.snr[self._mask]
        self._parentCluster.extract(cellData)
        local_x = self._parentCluster._local_x[self._mask]
        local_y = self._parentCluster._local_y[self._mask]
        g1 = self._parentCluster._g1[self._mask]
        g2 = self._parentCluster._g2[self._mask]
        xCell = self._parentCluster.xCell[self._mask]
        yCell = self._parentCluster.yCell[self._mask]

        if self._mask.sum() == 1:
            self._xCent = np.array([xCell[0]])
            self._yCent = np.array([yCell[0]])
            self._dist2 = np.zeros((1), float)
            self._local_x = np.array([local_x[0]])
            self._local_y = np.array([local_y[0]])
            self._g1 = np.array([g1[0]])
            self._g2 = np.array([g2[0]])
            self._rmsDist = np.array([0.0])
            self.xCell = np.array([xCell[0]])
            self.yCell = np.array([yCell[0]])
            self.snr = np.array([snr[0]])
            self._updateCatIndices()
            return

        sumSnr = np.sum(snr)
        self._xCent = np.sum(xCell * snr) / sumSnr
        self._yCent = np.sum(yCell * snr) / sumSnr
        self._dist2 = np.array((self._xCent - xCell) ** 2 + (self._yCent - yCell) ** 2)
        self._local_x = local_x
        self._local_y = local_y
        self._g1 = g1
        self._g2 = g2
        self.xCell = xCell
        self.yCell = yCell
        self.snr = snr

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
            self._local_x = np.array([local_x[idx]])
            self._local_y = np.array([local_y[idx]])
            self._g1 = np.array([g1[idx]])
            self._g2 = np.array([g2[idx]])
            self.xCell = np.array([xCell[idx]])
            self.yCell = np.array([yCell[idx]])
            self.snr = np.array([snr[idx]])

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
