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
    parent_cluster : ClusterData
        Parent cluster that this object is build from

    object_id : int
        Unique object identifier

    mask : np.ndarray[bool]
        Mask of which source in parent cluster are in this object

    catalog_id: np.ndarray[int]
        Indicies showing which catalog each source belongs to

    n_src: int
        Number of sources in this object

    n_unique: int
        Number of catalogs contributing to this object

    data: pandas.DataFrame
        Data the sources in this object

    recurse: int
        Recursion level needed to make this object

    x_cent: float
        X centroid of this object in cell pixel coordinates

    y_cent: float
        Y centroid of this object in cell pixel coordinates

    rms_dist: float
        RMS distance of sources to the centroid

    dist_2: np.ndarray
        Distances of sources to the centroid
    """

    def __init__(
        self,
        cluster: ClusterData,
        object_id: int,
        mask: np.ndarray | None,
        recurse: int = 0,
    ):
        """Build from `ClusterData`, an object_id and mask specifying with sources
        in the cluster are part of the object"""
        self.parent_cluster: ClusterData = cluster
        self.object_id: int = object_id
        if mask is None:
            self.mask = np.ones((self.parent_cluster.n_src), dtype=bool)
        else:
            self.mask = mask
        self.catalog_id: np.ndarray = self.parent_cluster.catalog_id[self.mask]
        self.n_src: int = self.catalog_id.size
        self.n_unique: int = np.unique(self.catalog_id).size
        self.data: pandas.DataFrame | None = None

        self.recurse: int = recurse
        self.x_cent: float = np.nan
        self.y_cent: float = np.nan
        self.rms_dist: float = np.nan
        self.snr_mean: float = np.nan
        self.snr_rms: float = np.nan
        self.dist_2: np.ndarray = np.array([])
        self.extract()

    @property
    def footprint_offset(self) -> tuple[float, float]:
        """Return the (x, y) pixel offset of the footprint origin within the cell"""
        return self.parent_cluster.footprint_offset

    @property
    def x_pix(self) -> np.ndarray:
        """Return the x-position of the sources in global WCS pixel coordinates"""
        assert self.data is not None
        return self.data.x_pix

    @property
    def y_pix(self) -> np.ndarray:
        """Return the y-position of the sources in global WCS pixel coordinates"""
        assert self.data is not None
        return self.data.y_pix

    def hasRefCatalog(self, ref_cat_id: int = 0) -> bool:
        """Is there a source from the reference catalog"""
        return ref_cat_id in self.catalog_id

    def catalogMask(self, cat_map: dict[int, int]) -> int:
        """Make a bit mask of which catalogs are in this object"""
        unmapped = np.unique(self.catalog_id)
        mapped_bits = set(cat_map[val] for val in unmapped)
        ret_val = 0
        for set_bit in mapped_bits:
            ret_val += 1 << set_bit
        return ret_val

    def sourceIds(self) -> np.ndarray:
        """Return the source ids for the sources in the object"""
        return self.parent_cluster.src_id[self.mask]

    def sourceIdxs(self) -> np.ndarray:
        """Return the source indices for the sources in the object"""
        return self.parent_cluster.src_idx[self.mask]

    def _updateCatIndices(self) -> None:
        self.catalog_id = self.parent_cluster.catalog_id[self.mask]
        self.n_src = self.catalog_id.size
        self.n_unique = np.unique(self.catalog_id).size

    def extract(self) -> None:
        """Extract data from parent cluster"""
        assert self.parent_cluster.data is not None
        self.data = self.parent_cluster.data.iloc[self.mask]
        self._updateCatIndices()


class ShearObjectData(ObjectData):
    """Subclass of ObjectData that can compute shear statistics"""

    def shearStats(self) -> dict:
        """Return the shear statistics"""
        assert self.data is not None
        return shear_utils.shearStats(self.data)
