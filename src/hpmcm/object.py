from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas

from . import shear_utils

if TYPE_CHECKING:
    from .cluster import ClusterData


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
        self.snrMean: float = np.nan
        self.snrRms: float = np.nan
        self.dist2: np.ndarray = np.array([])
        self.extract()

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

    def hasRefCatalog(self, refCatId: int = 0) -> bool:
        """Is there a source from the reference catalog"""
        return refCatId in self.catIndices

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

    def extract(self) -> None:
        """Extract data from parent cluster"""
        assert self.parentCluster.data is not None
        self.data = self.parentCluster.data.iloc[self.mask]
        self._updateCatIndices()


class ShearObjectData(ObjectData):
    """Subclass of ObjectData that can compute shear statisitics"""

    def shearStats(self) -> dict:
        """Return the shear statistics"""
        assert self.data is not None
        return shear_utils.shearStats(self.data)
