from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections import OrderedDict

import np as np

from astropy import wcs
from astropy.table import Table
from astropy.table import vstack
from astropy.io import fits

import pandas

import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage

from .object import ObjectData

if TYPE_CHECKING:
    from .cell import CellData
    from .match import Match


class ClusterData:
    """ Class to store data about clusters

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
        sources: tuple,
        origCluster: int|None=None,
    ):
        self._iCluster = iCluster
        self._footprint = footprint
        if origCluster is None:
            self._origCluster = self._iCluster
        else:
            self._origCluster = origCluster
        self._catIndices = sources[0]
        self._sourceIds = sources[1]
        self._sourceIdxs = sources[2]
        self._nSrc =  self._catIndices.size
        self._nUnique = len(np.unique(self._catIndices))
        self._objects: list[ObjectData] = []
        self._data: pandas.DataFrame|None = None
        self._xCent: np.ndarray|None = None
        self._yCent: np.ndarray|None = None
        self._dist2: np.ndarray|None = None
        self._rmsDist: np.ndarray|None = None
        self._xCell: np.ndarray|None = None
        self._yCell: np.ndarray|None = None
        self._snr: np.ndarray|None = None
        
    def extract(self, cellData: CellData) -> None:
        """ Extract the xCell, yCell and snr data from
        the sources in this cluster
        """

        series_list = []
        iCat_list = []
        src_idx_list = []
        
        for i, (iCat, srcIdx) in enumerate(zip(self._catIndices, self._sourceIdxs)):

            series_list.append(cellData.data[iCat].iloc[srcIdx])
            iCat_list.append(iCat)
            src_idx_list.append(srcIdx)

        self._data = pandas.DataFrame(series_list)
        self._data['iCat'] = iCat_list
        self._data['idx'] = src_idx_list

        self._xCell = self._data['xcell'].values()
        self._yCell = self._data['ycell'].values()
        self._snr = self._data['snr'].values()
        
    def clearTempData(self) -> None:
        """ Remove temporary data only used when making objects """
        self._data = None
        self._xCell = None
        self._yCell = None
        self._snr = None

    @property
    def iCluster(self) -> int:
        """ Return the cluster ID """
        return self._iCluster

    @property
    def nSrc(self) -> int:
        """ Return the number of sources associated to the cluster """
        return self._nSrc

    @property
    def nUnique(self) -> int:
        """ Return the number of catalogs contributing sources to the cluster """
        return self._nUnique

    @property
    def sourceIds(self) -> list[int]:
        """ Return the source IDs associated to this cluster """
        return self._sourceIds

    @property
    def dist2(self) -> np.ndarray:
        """ Return an array with the distance squared (in cells)
        between each source and the cluster centroid """
        return self._dist2

    @property
    def objects(self) -> list[ObjectData]:
        """ Return the objects associated with this cluster """
        return self._objects

    def processCluster(self, cellData: CellData, pixelR2Cut: float) -> list[ObjectData]:
        """ Function that is called recursively to
        split clusters until they:

        1.  Consist only of sources with the match radius of the cluster
        centroid.

        2.  Have at most one source per input catalog
        """
        self._nSrc =  self._catIndices.size
        self._nUnique = len(np.unique(self._catIndices))
        if self._nSrc == 0:
            print("Empty cluster", self._nSrc, self._nUnique)
            return self._objects
        self.extract(cellData)
        assert self._xCell and self._yCell
        
        if self._nSrc == 1:
            self._xCent = self._xCell[0]
            self._yCent = self._yCell[0]
            self._dist2 = np.zeros((1))
            self._rmsDist = 0.
            initialObject = self.addObject(cellData)
            initialObject.processObject(cellData, pixelR2Cut)
            self.clearTempData()
            return self._objects

        sumSnr = np.sum(self._snr)
        self._xCent = np.sum(self._xCell*self._snr) / sumSnr
        self._yCent = np.sum(self._yCell*self._snr) / sumSnr
        self._dist2 = (self._xCent - self._xCell)**2 + (self._yCent - self._yCell)**2
        self._rmsDist = np.sqrt(np.mean(self._dist2))
        
        initialObject = self.addObject(cellData)
        initialObject.processObject(cellData, pixelR2Cut)
        self.clearTempData()
        return self._objects

    def addObject(self, cellData: CellData, mask: np.ndarray|None=None) -> ObjectData:
        """ Add a new object to this cluster """
        newObject = cellData.addObject(self, mask)
        self._objects.append(newObject)
        return newObject
