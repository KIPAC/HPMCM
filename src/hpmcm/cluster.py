from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas

from . import shear_utils
from .footprint import Footprint
from .object import ObjectData

if TYPE_CHECKING:
    from .cell import CellData


class ClusterData:
    """Class that defines clusters

    A cluster is a set of sources within a footprint of
    adjacent pixels.

    Attributes
    ----------
    i_cluster : int
        Cluster ID

    footprint: Footprint
        Footprint of this cluster in the CellData counts map

    orig_cluster : int
        Id of the original cluster this cluster was made from

    sources: np.ndarray
        Data about the sources in this cluster

    n_src : int
        Number of sources in this cluster

    n_unique : int
        Number of catalogs contributing sources to this cluster

    objects: list[ObjectData]
        Data about the objects in this cluster

    x_cent : float
        X-pixel value of cluster centroid (in WCS used to do matching)

    y_cent : float
        Y-pixel value of cluster centroid (in WCS used to do matching)

    rms_dist: float
        RMS distance of sources to the centroid

    snr_mean: float
        mean signal-to-noise

    snr_rms: float
        RMS of signal-to-noise

    dist_2: np.ndarray
        Distances of sources to the centroid
    """

    def __init__(
        self,
        i_cluster: int,
        footprint: Footprint,
        sources: np.ndarray,
        orig_cluster: int | None = None,
    ):
        """Build from a Footprint and data about the
        sources in that Footprint"""
        self.i_cluster: int = i_cluster
        self.footprint: Footprint = footprint
        self.orig_cluster: int = i_cluster
        if orig_cluster is not None:
            self.orig_cluster = orig_cluster
        self.sources: np.ndarray = sources

        self.n_src: int = self.sources[0].size
        self.n_unique: int = len(np.unique(self.sources[0]))
        self.objects: list[ObjectData] = []
        self.data: pandas.DataFrame | None = None

        self.x_cent: float = np.nan
        self.y_cent: float = np.nan
        self.rms_dist: float = np.nan
        self.snr_mean: float = np.nan
        self.snr_rms: float = np.nan
        self.dist_2: np.ndarray = np.array([])
        self.pixel_match_scale: int = 1

    def extract(self, cell_data: CellData) -> None:
        """Extract the x_pix, y_pix and snr data from
        the sources in this cluster
        """
        x_offset = self.footprint.slice_x.start * self.pixel_match_scale
        y_offset = self.footprint.slice_y.start * self.pixel_match_scale

        series_list = []

        for _i, (i_cat_, src_idx_) in enumerate(zip(self.sources[0], self.sources[2])):
            series_list.append(cell_data.data[i_cat_].iloc[src_idx_])

        self.data = pandas.DataFrame(series_list)
        self.data["i_cat"] = self.sources[0]
        self.data["src_id"] = self.sources[1]
        self.data["src_idx"] = self.sources[2]
        self.data["x_cluster"] = self.data.x_cell - x_offset
        self.data["y_cluster"] = self.data.y_cell - y_offset

    @property
    def catalog_id(self) -> np.ndarray:
        """Return the catalog indices for all the sources in the cluster"""
        assert self.data is not None
        return self.data.i_cat

    @property
    def src_id(self) -> np.ndarray:
        """Return the ids for all the sources in the cluster"""
        assert self.data is not None
        return self.data.src_id

    @property
    def src_idx(self) -> np.ndarray:
        """Return the indices for all the sources in the cluster"""
        assert self.data is not None
        return self.data.src_idx

    def hasRefCatalog(self, ref_cat_id: int = 0) -> bool:
        """Is there a source from the reference catalog"""
        return ref_cat_id in self.sources[0]

    @property
    def x_cluster(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the footprint"""
        assert self.data is not None
        return self.data.x_cluster

    @property
    def y_cluster(self) -> np.ndarray:
        """Return the y-positions of the soures w.r.t. the footprint"""
        assert self.data is not None
        return self.data.y_cluster

    @property
    def x_pix(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the WCS"""
        assert self.data is not None
        return self.data.x_pix

    @property
    def y_pix(self) -> np.ndarray:
        """Return the x-positions of the soures w.r.t. the WCS"""
        assert self.data is not None
        return self.data.y_pix

    def addObject(
        self, cell_data: CellData, mask: np.ndarray | None = None
    ) -> ObjectData:
        """Add a new object to this cluster"""
        new_object = cell_data.addObject(self, mask)
        self.objects.append(new_object)
        return new_object


class ShearClusterData(ClusterData):
    """Subclass of ClusterData that can compute shear statisitics

    Attributes
    ----------
    pixel_match_scale: int
        Number of pixel merged in the original counts map
    """

    def __init__(
        self,
        i_cluster: int,
        footprint: Footprint,
        sources: np.ndarray,
        orig_cluster: int | None = None,
        pixel_match_scale: int = 1,
    ):
        ClusterData.__init__(self, i_cluster, footprint, sources, orig_cluster)
        self.pixel_match_scale = pixel_match_scale

    def shearStats(self) -> dict:
        """Return the shear statistics"""
        assert self.data is not None
        return shear_utils.shearStats(self.data)
