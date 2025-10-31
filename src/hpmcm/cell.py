from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import pandas

from . import match_utils, shear_utils, utils
from .cluster import ClusterData, ShearClusterData
from .footprint import Footprint, FootprintSet
from .object import ObjectData, ShearObjectData

if TYPE_CHECKING:
    from .match import Match
    from .shear_match import ShearMatch


class CellData:
    """Class to store analyze data for a cell

    Includes cell boundries, reduced data tables
    and clustering results

    Does not store sky maps

    Cells are square sub-regions of the analysis region
    that are extracted from the ranges of pixels in the WCS

    The cell covers corner:corner+size

    The sources are projected into an array that extends `buf` pixels
    beyond the cell

    This uses the FootprintSet to identify pixels which contain
    source, and builds those into clusters

    Attributes
    ----------
    matcher: Match
        Parent Match object

    id_offset: int
        Offset used for the Object and Cluster IDs for this cell

    corner: np.ndarray
        pixX, pixY for corner of cell

    size: np.ndarray
        size of the cell (in pixels)

    idx: int
        Id of the cell

    buf: int
        Number of buffer pixels around the edge of the cell

    min_pix: np.ndarray
        Lowest number pixel in center of cell

    max_pix: np.ndarray
        Highest number pixel in center of cell

    n_pix: np.ndarray
        Number of pixels in center of cell

    data : list[pandas.DataFrame]
        Reduced dataframes with only sources for this cell

    n_src : int
        Number of sources in this cell

    footprint_ids : list[np.ndarray]
        Matched arrays with the index of the cluster associated to each
        source.  I.e., these could added to the Dataframes as
        additional columns

    cluster_dict : OrderedDict[int, ClusterData]
        Dictionary with cluster membership data

    object_dict : OrderedDict[int, ObjectData]
        Dictionary with object membership data

    """

    def __init__(
        self,
        matcher: Match,
        id_offset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
        buf: int = 10,
    ):
        self.matcher: Match = matcher
        # Offset used for the Object and Cluster IDs for this cell
        self.id_offset: int = id_offset
        self.corner: np.ndarray = corner  # pixX, pixY for corner of cell
        self.size: np.ndarray = size  # size of cell
        self.idx: int = idx  # cell index
        self.buf: int = buf
        self.min_pix: np.ndarray = corner - buf
        self.max_pix: np.ndarray = corner + size + buf
        self.n_pix: np.ndarray = self.max_pix - self.min_pix

        self.data: list[pandas.DataFrame] = []
        self.n_src: int = 0
        self.footprint_ids: list[np.ndarray] = []
        self.cluster_dict: OrderedDict[int, ClusterData] = OrderedDict()
        self.object_dict: OrderedDict[int, ObjectData] = OrderedDict()

    def reduceData(self, data: list[pandas.DataFrame]) -> None:
        """Pull out only the data needed for this cell"""
        self.data = [self.reduceDataframe(i, val) for i, val in enumerate(data)]
        self.n_src = int(np.sum([len(df) for df in self.data]))

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters in this cell"""
        return len(self.cluster_dict)

    @property
    def n_objects(self) -> int:
        """Return the number of objects in this cell"""
        return len(self.object_dict)

    def reduceDataframe(
        self, i_cat: int, dataframe: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Filters dataframe to keep only source in the cell"""
        assert i_cat is not None

        # WCS is defined, use it
        x_cell = dataframe["x_pix"] - self.min_pix[0]
        y_cell = dataframe["y_pix"] - self.min_pix[1]
        filtered = (
            (x_cell >= 0)
            & (x_cell < self.n_pix[0])
            & (y_cell >= 0)
            & (y_cell < self.n_pix[1])
        )
        red = dataframe[filtered].copy(deep=True)
        red["x_cell"] = x_cell[filtered]
        red["y_cell"] = y_cell[filtered]
        return red

    def countsMap(self, weight_name: str | None = None) -> np.ndarray:
        """Fill a map that counts the number of source per cell"""
        to_fill = self._emtpyCountsMaps()
        assert self.data is not None
        for df in self.data:
            to_fill += self._singleCatalogCountsMap(df, weight_name)
        return to_fill

    def buildClusterData(
        self,
        fp_set: FootprintSet,
        pixel_r2_cut: float = 4.0,
    ) -> None:
        """Loop through cluster ids and collect sources into
        the ClusterData objects"""
        footprint_dict: dict[int, list[tuple[int, int, int]]] = {}
        n_missing = 0
        n_found = 0
        assert self.data is not None
        assert self.footprint_ids
        for i_cat, (df, footprint_ids) in enumerate(zip(self.data, self.footprint_ids)):
            for src_idx, (src_id, footprint_id) in enumerate(
                zip(df["id"], footprint_ids)
            ):
                if footprint_id < 0:
                    n_missing += 1
                    continue
                if footprint_id not in footprint_dict:
                    footprint_dict[footprint_id] = [(i_cat, src_id, src_idx)]
                else:
                    footprint_dict[footprint_id].append((i_cat, src_id, src_idx))
                n_found += 1
        for footprint_id, sources in footprint_dict.items():
            footprint = fp_set.footprints[footprint_id]
            i_cluster = footprint_id + self.id_offset
            cluster = self._buildClusterData(i_cluster, footprint, np.array(sources).T)
            self.cluster_dict[i_cluster] = cluster
            match_utils.heirarchicalProcessCluster(cluster, self, pixel_r2_cut)

    def analyze(
        self, weight_name: str | None = None, pixel_r2_cut: float = 2.0
    ) -> dict | None:
        """Analyze this cell

        Note that this returns the counts maps and clustering info,
        which can be helpful for debugging.
        """
        if self.n_src == 0:
            return None
        counts_map = self.countsMap(weight_name)
        o_dict = self._getFootprints(counts_map)
        o_dict["counts_map"] = counts_map
        assert self.data is not None
        self.footprint_ids = self._associateSourcesToFootprints(
            self.data, o_dict["footprint_key"]
        )
        self.buildClusterData(o_dict["footprints"], pixel_r2_cut)
        return o_dict

    def addObject(
        self, cluster: ClusterData, mask: np.ndarray | None = None
    ) -> ObjectData:
        """Add an object to this cell

        Parameters
        ----------
        cluster:
            Parent cluster for the object

        mask:
            Mask of which sources in the cluster to include in the object

        Returns
        -------
        Newly created ObjectData
        """
        object_id = self.n_objects + self.id_offset
        new_object = self._newObject(cluster, object_id, mask)
        self.object_dict[object_id] = new_object
        return new_object

    @classmethod
    def _newObject(
        cls, cluster: ClusterData, object_id: int, mask: np.ndarray | None
    ) -> ObjectData:
        return ObjectData(cluster, object_id, mask)

    def _emtpyCountsMaps(self) -> np.ndarray:
        to_fill = np.zeros(np.ceil(self.n_pix).astype(int))
        return to_fill

    def _singleCatalogCountsMap(
        self, df: pandas.DataFrame, weight_name: str | None = None
    ) -> np.ndarray:
        return utils.fillCountsMapFromDf(
            df,
            n_pix=self.n_pix,
            weight_name=weight_name,
        )

    def _buildClusterData(
        self, i_cluster: int, footprint: Footprint, sources: np.ndarray
    ) -> ClusterData:
        return ClusterData(i_cluster, footprint, sources)

    def _getFootprints(self, counts_map: np.ndarray) -> dict:
        return utils.getFootprints(counts_map, buf=self.buf)

    def _associateSourcesToFootprints(
        self,
        data: list[pandas.DataFrame],
        cluster_key: np.ndarray,
    ) -> list[np.ndarray]:
        return utils.associateSourcesToFootprints(
            data,
            cluster_key,
        )

    def getRaDec(
        self, x_cents: np.ndarray, y_cents: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the RA, DEC of based from pixel coords"""
        return self.matcher.pixToWorld(x_cents, y_cents)


class ShearCellData(CellData):
    """Subclass of CellData that can compute shear statisitics

    Attributes
    ----------
    pixel_match_scale: int
        Number of pixel merged in the original counts map
    """

    def __init__(
        self,
        matcher: ShearMatch,
        id_offset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
        buf: int = 10,
    ):
        CellData.__init__(self, matcher, id_offset, corner, size, idx, buf)
        self.pixel_match_scale = matcher.pixel_match_scale

    def reduceDataframe(
        self, i_cat: int, dataframe: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Filters dataframe to keep only source in the cell"""
        return shear_utils.reduceShearDataForCell(self, i_cat, dataframe)

    @classmethod
    def _newObject(
        cls, cluster: ClusterData, object_id: int, mask: np.ndarray | None
    ) -> ObjectData:
        return ShearObjectData(cluster, object_id, mask)

    def _emtpyCountsMaps(self) -> np.ndarray:
        pixel_match_scale = self.pixel_match_scale
        to_fill = np.zeros(np.ceil(self.n_pix / pixel_match_scale).astype(int))
        return to_fill

    def _singleCatalogCountsMap(
        self, df: pandas.DataFrame, weight_name: str | None = None
    ) -> np.ndarray:
        return utils.fillCountsMapFromDf(
            df,
            n_pix=self.n_pix,
            weight_name=weight_name,
            pixel_match_scale=self.pixel_match_scale,
        )

    def _buildClusterData(
        self, i_cluster: int, footprint: Footprint, sources: np.ndarray
    ) -> ClusterData:
        return ShearClusterData(
            i_cluster, footprint, sources, pixel_match_scale=self.pixel_match_scale
        )

    def _getFootprints(self, counts_map: np.ndarray) -> dict:
        return utils.getFootprints(
            counts_map, buf=0, pixel_match_scale=self.pixel_match_scale
        )

    def _associateSourcesToFootprints(
        self,
        data: list[pandas.DataFrame],
        cluster_key: np.ndarray,
    ) -> list[np.ndarray]:
        return utils.associateSourcesToFootprints(
            data,
            cluster_key,
            pixel_match_scale=self.pixel_match_scale,
        )

    def getRaDec(
        self, x_cents: np.ndarray, y_cents: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.repeat(np.nan, len(x_cents)), np.repeat(np.nan, len(y_cents))
