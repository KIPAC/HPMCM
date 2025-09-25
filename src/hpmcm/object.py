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

    def _updateCatIndices(self) -> None:
        self._catIndices = self._parentCluster.catIndices[self._mask]
        self._nSrc = self._catIndices.size
        self._nUnique = np.unique(self._catIndices).size

    def sourceIds(self) -> np.ndarray:
        return self._parentCluster.srcIds[self._mask]

    def sourceIdxs(self) -> np.ndarray:
        return self._parentCluster.srcIdxs[self._mask]
    
    def _extract(self) -> None:
        assert self._parentCluster.data is not None
        self._data = self._parentCluster.data.iloc[self._mask]
        self._updateCatIndices()

    def shearStats(self) -> dict:
        out_dict = {}
        names = ['ns', '2p', '2m', '1p', '1m']
        all_good = True
        for i, name_ in enumerate(names):
            mask = self._data.iCat == i
            n_cat =  mask.sum()
            if n_cat != 1:
                all_good = False
            out_dict[f"n_{name_}"] = n_cat
            if n_cat:
                out_dict[f"g1_{name_}"] = self._data.g_1[mask].mean()
                out_dict[f"g2_{name_}"] = self._data.g_2[mask].mean()
            else:
                out_dict[f"g1_{name_}"] = np.nan
                out_dict[f"g2_{name_}"] = np.nan
        out_dict['delta_g_1'] = out_dict['g1_1p'] - out_dict['g1_1m'] 
        out_dict['delta_g_2'] = out_dict['g2_2p'] - out_dict['g1_2m'] 
        out_dict['good'] = all_good
        return out_dict

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
        zoom_factors = [1, 2, 4, 8, 16, 32]
        zoom_factor = zoom_factors[recurse]

        nPix = zoom_factor * np.array([bbox.getHeight(), bbox.getWidth()])
        zoom_x = zoom_factor * self._data.xCluster
        zoom_y = zoom_factor * self._data.yCluster

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
