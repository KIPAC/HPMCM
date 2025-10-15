from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas

from . import shear_utils
from .object import ObjectData

if TYPE_CHECKING:
    try:
        import lsst.afw.detection as afwDetect
    except ImportError:
        pass
    import lsst.afw.detection as afwDetect

    from .cell import CellData


class ClusterData:
    """Class that defines clusters

    A cluster is a set of sources within a footprint of
    adjacent pixels.

    Attributes
    ----------
    iCluster : int
        Cluster ID

    footprint: afwDetect.Footprint
        Footprint of this cluster in the CellData counts map

    origCluster : int
        Id of the original cluster this cluster was made from

    sources: np.ndarray
        Data about the sources in this cluster

    nSrc : int
        Number of sources in this cluster

    nUnique : int
        Number of catalogs contributing sources to this cluster

    objects: list[ObjectData]
        Data about the objects in this cluster

    xCent : float
        X-pixel value of cluster centroid (in WCS used to do matching)

    yCent : float
        Y-pixel value of cluster centroid (in WCS used to do matching)

    rmsDist: float
        RMS distance of sources to the centroid

    snrMean: float
        mean signal-to-noise

    snrRms: float
        RMS of signal-to-noise

    dist2: np.ndarray
        Distances of sources to the centroid
    """

    def __init__(
        self,
        iCluster: int,
        footprint: afwDetect.Footprint,
        sources: np.ndarray,
        origCluster: int | None = None,
    ):
        """Build from a Footprint and data about the
        sources in that Footprint"""
        self.iCluster: int = iCluster
        self.footprint: afwDetect.Footprint = footprint
        self.origCluster: int = iCluster
        if origCluster is not None:
            self.origCluster = origCluster
        self.sources: np.ndarray = sources

        self.nSrc: int = self.sources[0].size
        self.nUnique: int = len(np.unique(self.sources[0]))
        self.objects: list[ObjectData] = []
        self.data: pandas.DataFrame | None = None

        self.xCent: float = np.nan
        self.yCent: float = np.nan
        self.rmsDist: float = np.nan
        self.snrMean: float = np.nan        
        self.snrRms: float = np.nan
        self.dist2: np.ndarray = np.array([])
        self.pixelMatchScale: int = 1

    def extract(self, cellData: CellData) -> None:
        """Extract the xPix, yPix and snr data from
        the sources in this cluster
        """
        bbox = self.footprint.getBBox()

        xOffset = bbox.getBeginY() * self.pixelMatchScale
        yOffset = bbox.getBeginX() * self.pixelMatchScale

        seriesList = []

        for _i, (iCat, srcIdx) in enumerate(zip(self.sources[0], self.sources[2])):
            seriesList.append(cellData.data[iCat].iloc[srcIdx])

        self.data = pandas.DataFrame(seriesList)
        self.data["iCat"] = self.sources[0]
        self.data["srcId"] = self.sources[1]
        self.data["srcIdx"] = self.sources[2]
        self.data["xCluster"] = self.data.xCell - xOffset
        self.data["yCluster"] = self.data.yCell - yOffset

    @property
    def catIndices(self) -> np.ndarray:
        """Return the catalog indices for all the sources in the cluster"""
        assert self.data is not None
        return self.data.iCat

    @property
    def srcIds(self) -> np.ndarray:
        """Return the ids for all the sources in the cluster"""
        assert self.data is not None
        return self.data.srcId

    @property
    def srcIdxs(self) -> np.ndarray:
        """Return the indices for all the sources in the cluster"""
        assert self.data is not None
        return self.data.srcIdx

    @property
    def xCluster(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the footprint"""
        assert self.data is not None
        return self.data.xCluster

    @property
    def yCluster(self) -> np.ndarray:
        """Return the y-positions of the soures w.r.t. the footprint"""
        assert self.data is not None
        return self.data.yCluster

    @property
    def xPix(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the WCS"""
        assert self.data is not None
        return self.data.xPix

    @property
    def yPix(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the WCS"""
        assert self.data is not None
        return self.data.yPix

    def processCluster(self, cellData: CellData, pixelR2Cut: float) -> list[ObjectData]:
        """Function that is called recursively to
        split clusters until they consist only of sources within
        the match radius of the cluster centroid.
        """
        self.extract(cellData)
        assert self.data is not None

        self.nSrc = len(self.data.iCat)
        self.nUnique = len(np.unique(self.data.iCat.values))
        if self.nSrc == 0:
            print("Empty cluster", self.nSrc, self.nUnique)
            return self.objects

        if self.nSrc == 1:
            self.xCent = self.data.xPix.values[0]
            self.yCent = self.data.yPix.values[0]
            self.dist2 = np.zeros((1))
            self.rmsDist = 0.0
            self.snrMean = self.data.SNR.values[0]
            self.snrRms = 0.0
            initialObject = self.addObject(cellData)
            initialObject.processObject(cellData, pixelR2Cut)
            return self.objects

        sumSnr = np.sum(self.data.SNR)
        self.xCent = np.sum(self.data.xPix * self.data.SNR) / sumSnr
        self.yCent = np.sum(self.data.yPix * self.data.SNR) / sumSnr
        self.dist2 = (self.xCent - self.data.xPix) ** 2 + (
            self.yCent - self.data.yPix
        ) ** 2
        self.rmsDist = np.sqrt(np.mean(self.dist2))
        self.snrMean = np.mean(self.data.SNR.values)
        self.snrRms = np.std(self.data.SNR.values)

        initialObject = self.addObject(cellData)
        initialObject.processObject(cellData, pixelR2Cut)
        return self.objects

    def addObject(
        self, cellData: CellData, mask: np.ndarray | None = None
    ) -> ObjectData:
        """Add a new object to this cluster"""
        newObject = cellData.addObject(self, mask)
        self.objects.append(newObject)
        return newObject


class ShearClusterData(ClusterData):
    """Subclass of ClusterData that can compute shear statisitics

    Attributes
    ----------
    pixelMatchScale: int
        Number of pixel merged in the original counts map
    """

    def __init__(
        self,
        iCluster: int,
        footprint: afwDetect.Footprint,
        sources: np.ndarray,
        origCluster: int | None = None,
        pixelMatchScale: int = 1,
    ):
        ClusterData.__init__(self, iCluster, footprint, sources, origCluster)
        self.pixelMatchScale = pixelMatchScale

    def shearStats(self) -> dict:
        """Return the shear statistics"""
        assert self.data is not None
        return shear_utils.shearStats(self.data)

    def addObject(
        self, cellData: CellData, mask: np.ndarray | None = None
    ) -> ObjectData:
        """Add a new object to this cluster"""
        newObject = cellData.addObject(self, mask)
        self.objects.append(newObject)
        return newObject
