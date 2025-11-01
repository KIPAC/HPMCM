from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas
import pyarrow.parquet as pq

from . import input_tables, output_tables
from .cell import CellData
from .cluster import ClusterData
from .object import ObjectData


class Match:
    """Class to do N-way matching

    Uses a provided WCS to define a Skymap that covers the full region
    begin matched.

    Uses that WCS to assign pixel locations to all sources in the input catalogs

    Iterates over cells and does source clustering in each cell
    using Footprint detection on a Skymap of source counts per pixel.

    Assigns each input source to a cluster.

    At that stage the clusters are not the final product as they can include
    more than one soruce from a given catalog.

    Loops over clusters and processes each cluster to resolve confusion.

    If there is not a unqiue source per-catalog redo the clustering with
    half-size pixels to try to split the cluster (down to minimum pixel scale)

    Attributes
    ----------
    pix_size : float
        Pixel size in arcseconds

    n_pix_side: int
        Number of pixels in the match region

    cell_size: int
        Number of pixels in a Cell

    cell_buffer: int
        Number of overlapping pixel in a Cell

    cell_max_object: int
        Max number of objects in a cell, used to make unique IDs

    max_sub_division: int
        Maximum number of cell sub-divisions

    pixel_r2_cut: float
        Distance cut for Object membership, in pixels**2

    n_cell: np.ndarray
        Number of cells in match region

    full_data: list[DataFrame]
        Full input DataFrames

    red_data : list[DataFrame]
        Reduced DataFrames with only the columns needed for matching

    cell_dict : OrderedDict[int, CellData]
        Dictionary providing access to cell data

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames.
    The expected columns depend on which sub-class of `Match` is being used.

    Four output tables are produced:

    +----------------+---------------------------------------------------+
    | Key            | Class                                             |
    +================+===================================================+
    | _cluster_assoc | :py:class:`hpmcm.output_tables.ClusterAssocTable` |
    +----------------+---------------------------------------------------+
    | _cluster_stats | :py:class:`hpmcm.output_tables.ClusterStatsTable` |
    +----------------+---------------------------------------------------+
    | _object_assoc  | :py:class:`hpmcm.output_tables.ObjectAssocTable`  |
    +----------------+---------------------------------------------------+
    | _object_stats  | :py:class:`hpmcm.output_tables.ObjectStatsTable`  |
    +----------------+---------------------------------------------------+

    """

    inputTableClass: type = input_tables.SourceTable
    extraCols: list[str] = []

    def __init__(
        self,
        **kwargs: Any,
    ):
        self.pix_size = kwargs["pixel_size"]
        self.n_pix_side = kwargs["n_pixels"]
        self.cell_size: int = kwargs.get("cell_size", 1000)
        self.cell_buffer: int = kwargs.get("cell_buffer", 10)
        self.cell_max_object: int = kwargs.get("cell_max_object", 100000)
        self.max_sub_division: int = kwargs.get("max_sub_division", 3)
        self.pixel_r2_cut: float = kwargs.get("pixel_r2_cut", 1.0)
        self.n_cell: np.ndarray = np.ceil(self.n_pix_side / self.cell_size)

        self.full_data: OrderedDict[int, pandas.DataFrame] = OrderedDict()
        self.red_data: OrderedDict[int, pandas.DataFrame] = OrderedDict()
        self.cell_dict: OrderedDict[int, CellData] = OrderedDict()

    def pixToArcsec(self) -> float:
        """Convert pixel size (in degrees) to arcseconds"""
        return 3600.0 * self.pix_size

    def pixToWorld(
        self,
        x_pix: np.ndarray,
        y_pix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert local coords in pixels to world coordinates (RA, DEC)"""
        return np.repeat(np.nan, len(x_pix)), np.repeat(np.nan, len(y_pix))

    def getCellIdx(
        self,
        ix: int,
        iy: int,
    ) -> int:
        """Get the Index to use for a given cell"""
        return int(self.n_cell[1] * ix + iy)

    def getIdOffset(
        self,
        ix: int,
        iy: int,
    ) -> int:
        """Get the ID offset to use for a given cell"""
        cell_idx = self.getCellIdx(ix, iy)
        return int(self.cell_max_object * cell_idx)

    def reduceData(
        self,
        input_files: list[str],
        catalog_id: list[int],
    ) -> None:
        """Read input files and filter out only the columns we need

        Each input file should have an associated catalog_id.
        This is used to test if we have more than one-source
        per input catalog.

        If the inputs files have a pre-defined ID associated with them
        that can be used.   Otherwise it is fine just to give a range from
        0 to nInputs.
        """
        for f_name, cid in zip(input_files, catalog_id):
            self.full_data[cid] = self._readDataFrame(f_name)
            self.red_data[cid] = self.reduceDataFrame(self.full_data[cid])
            self.full_data[cid].set_index("id", inplace=True)

    def _buildCellData(
        self,
        id_offset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
    ) -> CellData:
        return CellData(self, id_offset, corner, size, idx, self.cell_buffer)

    def analyzeCell(
        self,
        ix: int,
        iy: int,
        full_data: bool = False,
    ) -> dict | None:
        """Analyze a single cell

        Parameters
        ----------
        ix:
            Cell index in x-coord

        iy:
            Cell index in y-coord

        Returns
        -------
        Output of cell analysis


        Notes
        -----
        cell_data : CellData : The analysis data for the Cell

        image : np.ndarray : Image of cell source counts map

        countsMap : np.ndarray : Numpy array with cell source counts

        clusters : FootprintSet : Clusters as dectected by finding FootprintSet on source counts map

        clusterKey : np.ndarray : Map of cell with pixels filled with index of associated Footprints

        Notes
        -----
        If full_data is False, only cell_data will be returned
        """
        i_cell = self.getCellIdx(ix, iy)
        cell_step = np.array([self.cell_size, self.cell_size])
        corner = np.array([ix - 1, iy - 1]) * cell_step
        id_offset = self.getIdOffset(ix, iy)
        cell_data = self._buildCellData(id_offset, corner, cell_step, i_cell)
        cell_data.reduceData(list(self.red_data.values()))
        o_dict = cell_data.analyze(pixel_r2_cut=self.pixel_r2_cut)
        if cell_data.n_objects >= self.cell_max_object:  # pragma: no cover
            print(
                "Too many object in a cell", cell_data.n_objects, self.cell_max_object
            )

        self.cell_dict[i_cell] = cell_data
        if o_dict is None:
            return None
        if full_data:  # pragma: no cover
            o_dict["cell_data"] = cell_data
            return o_dict

        return dict(cell_data=cell_data)

    def analysisLoop(
        self, x_range: Iterable | None = None, y_range: Iterable | None = None
    ) -> None:
        """Does matching for all cells.

        This stores the results, but does not write or return them.

        Parameters
        ----------
        x_range:
            Range of cells to analysze in X.  None -> Entire range.

        y_range:
            Range of cells to analysis in Y.  None -> Entire range.
        """
        self.cell_dict.clear()

        if x_range is None:
            x_range = range(int(self.n_cell[0]))
        if y_range is None:
            y_range = range(int(self.n_cell[1]))

        for ix in x_range:
            for iy in y_range:
                odict = self.analyzeCell(ix, iy)
                if odict is None:
                    continue
            if ix == 0:
                pass
            elif ix % 10 == 0:
                sys.stdout.write(f" {ix}!\n")
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()

        sys.stdout.write(" Done!\n")
        sys.stdout.flush()

    def extractStats(self) -> list[pandas.DataFrame]:
        """Extracts cluster statisistics

        Returns
        -------
        DataFrames with matching info,
        :py:class:`hpmcm.output_tables.ClusterAssocTable`,
        :py:class:`hpmcm.output_tables.ObjectAssocTable`.
        :py:class:`hpmcm.output_tables.ClusterStatsTable`.
        :py:class:`hpmcm.output_tables.ObjectStatsTable`.

        """
        cluster_assoc_tables = []
        object_assoc_tables = []
        cluster_stats_tables = []
        object_stats_tables = []

        for ix in range(int(self.n_cell[0])):
            for iy in range(int(self.n_cell[1])):
                i_cell = self.getCellIdx(ix, iy)
                if i_cell not in self.cell_dict:
                    continue
                cell_data = self.cell_dict[i_cell]
                cluster_assoc_tables.append(
                    output_tables.ClusterAssocTable.buildFromCellData(cell_data).data,
                )
                object_assoc_tables.append(
                    output_tables.ObjectAssocTable.buildFromCellData(cell_data).data,
                )
                cluster_stats_tables.append(
                    output_tables.ClusterStatsTable.buildFromCellData(cell_data).data,
                )
                object_stats_tables.append(
                    output_tables.ObjectStatsTable.buildFromCellData(cell_data).data,
                )
        return [
            pandas.concat(cluster_assoc_tables),
            pandas.concat(object_assoc_tables),
            pandas.concat(cluster_stats_tables),
            pandas.concat(object_stats_tables),
        ]

    def _readDataFrame(
        self,
        f_name: str,
    ) -> pandas.DataFrame:
        """Read a single input file"""
        # FIXME, we want to use this function
        # return self.inputTableClass.read(f_name, self.extraCols)
        parq = pq.read_pandas(f_name)
        df = parq.to_pandas()
        return df

    def _getPixValues(self, df: pandas.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame

        Parameters
        ----------
        df:
            Input data frame

        Returns
        -------
        Reduced DataFrame
        """
        raise NotImplementedError()

    def getCluster(self, i_k: tuple[int, int]) -> ClusterData:
        """Get a particular cluster

        Parameters
        ----------
        i_k:
            CellId, ClusterId

        Returns
        -------
        Requested cluster
        """
        cell_data = self.cell_dict[i_k[0]]
        cluster = cell_data.cluster_dict[i_k[1]]
        return cluster

    def getObject(self, i_k: tuple[int, int]) -> ObjectData:
        """Get a particular object

        Parameters
        ----------
        i_k:
            CellId, ObjectId

        Returns
        -------
        Requested object
        """
        cell_data = self.cell_dict[i_k[0]]
        the_obj = cell_data.object_dict[i_k[1]]
        return the_obj
