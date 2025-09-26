from __future__ import annotations

from typing import TYPE_CHECKING

import lsst.afw.detection as afwDetect
import numpy as np
import pandas

from .object import ObjectData

if TYPE_CHECKING:
    from .cell import CellData


class ClusterData:
    """Class that defines clusters

    A cluster is a set of sources within a footprint of
    adjacent pixels.


    Parameters
    ----------
    iCluster : `int`
        Cluster ID
    origCluster : `int`
        Id of the original cluster this cluster was made from
    nSrc : `int`
        Number of sources in this cluster
    nUnique : `int`
        Number of catalogs contributing sources to this cluster
    catIndices : `np.array`, [`int`]
        Indices of the catalogs of sources associated to this cluster
    sourcdIds : `np.array`, [`int`]
        Sources IDs of the sources associated to this cluster
    sourcdIdxs : `np.array`, [`int`]
        Indices of the sources with their respective catalogs
    xCent : `float`
        X-pixel value of cluster centroid (in WCS used to do matching)
    yCent : `float`
        Y-pixel value of cluster centroid (in WCS used to do matching)
    """

    def __init__(
        self,
        iCluster: int,
        footprint: afwDetect.Footprint,
        sources: np.ndarray,
        origCluster: int | None = None,
        pixelMatchScale: int =1,
    ):
        self._iCluster: int = iCluster
        self._footprint: afwDetect.Footprint = footprint
        self._origCluster: int = iCluster
        if origCluster is not None:
            self._origCluster = origCluster
        self._sources: np.ndarray = sources
        self._pixelMatchScale = pixelMatchScale
        
        self._nSrc: int = self._sources[0].size
        self._nUnique: int = len(np.unique(self._sources[0]))
        self._objects: list[ObjectData] = []
        self._data: pandas.DataFrame | None = None

        self._xCent: float = np.nan
        self._yCent: float = np.nan
        self._rmsDist: float = np.nan
        self._dist2: np.ndarray = np.array([])

    def extract(self, cellData: CellData) -> None:
        """Extract the xPix, yPix and snr data from
        the sources in this cluster
        """
        bbox = self._footprint.getBBox()
        # xOffset = cellData.minPix[0] + bbox.getBeginY()
        # yOffset = cellData.minPix[1] + bbox.getBeginX()

        xOffset = bbox.getBeginY()*self._pixelMatchScale
        yOffset = bbox.getBeginX()*self._pixelMatchScale

        series_list = []

        for _i, (iCat, srcIdx) in enumerate(zip(self._sources[0], self._sources[2])):
            series_list.append(cellData.data[iCat].iloc[srcIdx])

        self._data = pandas.DataFrame(series_list)
        self._data["iCat"] = self._sources[0]
        self._data["srcId"] = self._sources[1]
        self._data["srcIdx"] = self._sources[2]
        # self._data["xCluster"] = self._data.xPix - xOffset
        # self._data["yCluster"] = self._data.yPix - yOffset
        self._data["xCluster"] = self._data.xCell - xOffset
        self._data["yCluster"] = self._data.yCell - yOffset

    @property
    def data(self) -> pandas.DataFrame | None:
        """Return the data for this cluster"""
        return self._data

    @property
    def footprint(self) -> afwDetect.Footprint:
        """Return the footprint associated to this cluster"""
        return self._footprint

    @property
    def iCluster(self) -> int:
        """Return the cluster ID"""
        return self._iCluster

    @property
    def nSrc(self) -> int:
        """Return the number of sources associated to the cluster"""
        return self._nSrc

    @property
    def nUnique(self) -> int:
        """Return the number of catalogs contributing sources to the cluster"""
        return self._nUnique

    @property
    def catIndices(self) -> np.ndarray:
        """Return the catalog indices for all the sources in the cluster"""
        assert self._data is not None
        return self._data.iCat

    @property
    def srcIds(self) -> np.ndarray:
        """Return the ids for all the sources in the cluster"""
        assert self._data is not None
        return self._data.srcId

    @property
    def srcIdxs(self) -> np.ndarray:
        """Return the indices for all the sources in the cluster"""
        assert self._data is not None
        return self._data.srcIdx

    @property
    def xCluster(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the footprint"""
        assert self._data is not None
        return self._data.xCluster

    @property
    def yCluster(self) -> np.ndarray:
        """Return the y-positions of the soures w.r.t. the footprint"""
        assert self._data is not None
        return self._data.yCluster

    @property
    def xPix(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the WCS"""
        assert self._data is not None
        return self._data.xPix

    @property
    def yPix(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the WCS"""
        assert self._data is not None
        return self._data.yPix

    @property
    def xCent(self) -> float:
        """Return the x-position of the centroid of the cluster in the WCS"""
        return self._xCent

    @property
    def yCent(self) -> float:
        """Return the y-position of the centroid of the cluster in the WCS"""
        return self._yCent

    @property
    def rmsDist(self) -> float:
        """Return the RMS distance of the sources in the cluster"""
        return self._rmsDist

    @property
    def dist2(self) -> np.ndarray:
        """Return an array with the distance squared (in cells)
        between each source and the cluster centroid"""
        return self._dist2

    @property
    def objects(self) -> list[ObjectData]:
        """Return the objects associated with this cluster"""
        return self._objects

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
        out_dict['delta_g_2'] = out_dict['g2_2p'] - out_dict['g2_2m']
        out_dict['good'] = all_good
        return out_dict


    def processCluster(self, cellData: CellData, pixelR2Cut: float) -> list[ObjectData]:
        """Function that is called recursively to
        split clusters until they consist only of sources within
        the match radius of the cluster centroid.
        """
        self.extract(cellData)
        assert self._data is not None

        self._nSrc = len(self._data.iCat)
        self._nUnique = len(np.unique(self._data.iCat.values))
        if self._nSrc == 0:
            print("Empty cluster", self._nSrc, self._nUnique)
            return self._objects

        if self._nSrc == 1:
            self._xCent = self._data.xPix.values[0]
            self._yCent = self._data.yPix.values[0]
            self._dist2 = np.zeros((1))
            self._rmsDist = 0.0
            initialObject = self.addObject(cellData)
            initialObject.processObject(cellData, pixelR2Cut)
            return self._objects

        sumSnr = np.sum(self._data.SNR)
        self._xCent = np.sum(self._data.xPix * self._data.SNR) / sumSnr
        self._yCent = np.sum(self._data.yPix * self._data.SNR) / sumSnr
        self._dist2 = (self._xCent - self._data.xPix) ** 2 + (
            self._yCent - self._data.yPix
        ) ** 2
        self._rmsDist = np.sqrt(np.mean(self._dist2))

        initialObject = self.addObject(cellData)
        initialObject.processObject(cellData, pixelR2Cut)
        return self._objects

    def addObject(
        self, cellData: CellData, mask: np.ndarray | None = None
    ) -> ObjectData:
        """Add a new object to this cluster"""
        newObject = cellData.addObject(self, mask)
        self._objects.append(newObject)
        return newObject
