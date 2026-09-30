from __future__ import annotations

import json
import sys
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas
import pyarrow.parquet as pq

from . import input_tables, output_tables
from .cell import CellData
from .cluster import ClusterData
from .footprint import Footprint
from .object import ObjectData


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_json_safe(x) for x in obj]
        return converted if isinstance(obj, list) else converted
    return obj


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
        self.n_cell_buffer: int =  kwargs.get("n_cell_buffer", 0)
        self.n_cell: np.ndarray = np.ceil(self.n_pix_side / self.cell_size) + self.n_cell_buffer

        self.full_data: OrderedDict[int, pandas.DataFrame] = OrderedDict()
        self.red_data: OrderedDict[int, pandas.DataFrame] = OrderedDict()
        self.cell_dict: OrderedDict[int, CellData] = OrderedDict()

        self.catalog_id_map: dict[int, int] = {}

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
        for idx, (f_name, cid) in enumerate(zip(input_files, catalog_id)):
            self.catalog_id_map[cid] = idx
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
        corner = np.array([ix - self.n_cell_buffer, iy - self.n_cell_buffer]) * cell_step
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

    def extractStats(self) -> dict[str, pandas.DataFrame]:
        """Extracts cluster statisistics

        Returns
        -------
        Dict with keys:
        ``cluster_assoc`` (:py:class:`hpmcm.output_tables.ClusterAssocTable`),
        ``object_assoc`` (:py:class:`hpmcm.output_tables.ObjectAssocTable`),
        ``cluster_stats`` (:py:class:`hpmcm.output_tables.ClusterStatsTable`),
        ``object_stats`` (:py:class:`hpmcm.output_tables.ObjectStatsTable`).

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

        return {
            "cluster_assoc": pandas.concat(cluster_assoc_tables),
            "object_assoc": pandas.concat(object_assoc_tables),
            "cluster_stats": pandas.concat(cluster_stats_tables),
            "object_stats": pandas.concat(object_stats_tables),
        }

    def getCellXY(self, cell_idx: int) -> tuple[int, int]:
        """Inverse of getCellIdx: return (ix, iy) for a flat cell index."""
        ix = cell_idx // int(self.n_cell[1])
        iy = cell_idx % int(self.n_cell[1])
        return ix, iy

    def _geometryDict(self) -> dict[str, Any]:
        """Serialize geometry and match parameters for save/load.

        Subclasses override this to add their own parameters.
        """
        return {
            "class": type(self).__name__,
            "pixel_size": float(self.pix_size),
            "n_pixels": [int(x) for x in np.asarray(self.n_pix_side).flat],
            "cell_size": int(self.cell_size),
            "cell_buffer": int(self.cell_buffer),
            "cell_max_object": int(self.cell_max_object),
            "max_sub_division": int(self.max_sub_division),
            "pixel_r2_cut": float(self.pixel_r2_cut),
            "n_cell_buffer": int(self.n_cell_buffer),
            "catalog_id_map": {str(k): int(v) for k, v in self.catalog_id_map.items()},
        }

    def save(self, save_dir: str | Path) -> None:
        """Persist the full match state to a directory.

        Saves geometry, association/stats tables, and one parquet per catalog
        (the already-reduced source DataFrames).  Call this after
        ``analysisLoop()`` to enable later restoration via ``load()``.

        Parameters
        ----------
        save_dir:
            Directory to write into (created if absent).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "geometry.json", "w") as fh:
            json.dump(_to_json_safe(self._geometryDict()), fh, indent=2)

        stats = self.extractStats()
        for name, df in stats.items():
            df.to_parquet(save_dir / f"{name}.parquet")

        for cat_id, df in self.red_data.items():
            df.to_parquet(save_dir / f"cat_{cat_id}.parquet")

    def _loadReducedData(self, save_dir: Path) -> None:
        """Read per-catalog parquets written by save() into self.red_data."""
        for cat_id, i_cat in self.catalog_id_map.items():
            self.red_data[cat_id] = pandas.read_parquet(save_dir / f"cat_{cat_id}.parquet")

    def _filterAssocByRange(
        self,
        cluster_assoc: pandas.DataFrame,
        object_assoc: pandas.DataFrame,
        x_range: tuple[int, int] | None,
        y_range: tuple[int, int] | None,
    ) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        """Filter assoc tables to only cells whose (ix, iy) fall in the given ranges.

        Parameters
        ----------
        x_range:
            ``(x_min, x_max)`` inclusive bounds on the cell x-index.
            ``None`` means no filtering on x.
        y_range:
            ``(y_min, y_max)`` inclusive bounds on the cell y-index.
            ``None`` means no filtering on y.
        """
        if x_range is None and y_range is None:
            return cluster_assoc, object_assoc
        keep = []
        for cell_idx_raw in cluster_assoc["cell_idx"].unique():
            ix, iy = self.getCellXY(int(cell_idx_raw))
            if x_range is not None and not (x_range[0] <= ix <= x_range[1]):
                continue
            if y_range is not None and not (y_range[0] <= iy <= y_range[1]):
                continue
            keep.append(int(cell_idx_raw))
        cluster_assoc = cluster_assoc[cluster_assoc["cell_idx"].isin(keep)]
        object_assoc = object_assoc[object_assoc["cell_idx"].isin(keep)]
        return cluster_assoc, object_assoc

    def _buildPerCellData(
        self, cluster_assoc: pandas.DataFrame
    ) -> dict[int, list[pandas.DataFrame]]:
        """Filter self.red_data to each cell's bounds.

        Returns a dict mapping cell_idx to a list of DataFrames in i_cat order.
        """
        per_cell_data: dict[int, list[pandas.DataFrame]] = {}
        unique_cells = cluster_assoc["cell_idx"].unique()
        for counter, cell_idx_raw in enumerate(unique_cells):
            cell_idx = int(cell_idx_raw)
            ix, iy = self.getCellXY(cell_idx)
            if counter > 0 and counter % 100 == 0:
                sys.stdout.write(f" {counter}/{len(unique_cells)}\n")
                sys.stdout.flush()
            cell_step = np.array([self.cell_size, self.cell_size])
            corner = (
                np.array([ix - self.n_cell_buffer, iy - self.n_cell_buffer]) * cell_step
            )
            id_offset = int(self.cell_max_object * cell_idx)
            cell_tmp = self._buildCellData(id_offset, corner, cell_step, cell_idx)
            cell_tmp.reduceData(list(self.red_data.values()))
            per_cell_data[cell_idx] = cell_tmp.data
        return per_cell_data

    def populateFromTables(
        self,
        cluster_assoc: pandas.DataFrame,
        object_assoc: pandas.DataFrame,
        cluster_stats: pandas.DataFrame,
        object_stats: pandas.DataFrame,
        input_files: list[str],
        catalog_ids: list[int],
    ) -> None:
        """Populate cell_dict from saved output tables and original input files.

        Use as an alternative to ``reduceData()`` + ``analysisLoop()`` when you
        already have the output tables from a prior run.  The original input
        files are re-read and filtered to each cell's bounds; no analysis is
        re-run.

        Parameters
        ----------
        cluster_assoc:
            ``ClusterAssocTable`` DataFrame from ``extractStats()``.
        object_assoc:
            ``ObjectAssocTable`` DataFrame from ``extractStats()``.
        cluster_stats:
            ``ClusterStatsTable`` DataFrame from ``extractStats()`` — must
            include ``fp_x_min/max``, ``fp_y_min/max`` columns.
        object_stats:
            ``ObjectStatsTable`` DataFrame from ``extractStats()``.
        input_files:
            Original input catalog file paths (parquet).
        catalog_ids:
            Catalog IDs corresponding to each entry in ``input_files``.
        """
        self.reduceData(input_files, catalog_ids)
        per_cell_data = self._buildPerCellData(cluster_assoc)
        self._reconstructCells(cluster_assoc, object_assoc, cluster_stats, object_stats, per_cell_data)

    def _reconstructCells(
        self,
        cluster_assoc: pandas.DataFrame,
        object_assoc: pandas.DataFrame,
        cluster_stats: pandas.DataFrame,
        object_stats: pandas.DataFrame,
        per_cell_data: dict[int, list[pandas.DataFrame]],
    ) -> None:
        """Shared reconstruction core for load() and populateFromTables().

        Rebuilds the cell/cluster/object hierarchy from flat tables and
        pre-built per-cell source DataFrames.
        """
        pix_to_arcsec = self.pixToArcsec()
        cs_idx = cluster_stats.set_index("cluster_id")
        os_idx = object_stats.set_index("object_id")

        for counter, cell_idx_raw in enumerate(cluster_assoc["cell_idx"].unique()):
            cell_idx = int(cell_idx_raw)
            ix, iy = self.getCellXY(cell_idx)

            if counter == 0:
                pass
            elif counter % 1000 == 0:
                sys.stdout.write(f" {counter}!\n")
                sys.stdout.flush()
            elif counter % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            
            cell_step = np.array([self.cell_size, self.cell_size])
            corner = (
                np.array([ix - self.n_cell_buffer, iy - self.n_cell_buffer]) * cell_step
            )
            id_offset = int(self.cell_max_object * cell_idx)
            cell = self._buildCellData(id_offset, corner, cell_step, cell_idx)
            cell.data = per_cell_data[cell_idx]
            cell.n_src = sum(len(df) for df in cell.data)

            ca_cell = cluster_assoc[cluster_assoc["cell_idx"] == cell_idx]
            oa_cell = object_assoc[object_assoc["cell_idx"] == cell_idx]

            for cluster_id_raw in ca_cell["cluster_id"].unique():
                cluster_id = int(cluster_id_raw)
                ca_rows = ca_cell[ca_cell["cluster_id"] == cluster_id]

                i_cat_arr = np.array(
                    [self.catalog_id_map[int(cid)] for cid in ca_rows["catalog_id"]]
                )
                sources = np.array(
                    [
                        i_cat_arr,
                        ca_rows["source_id"].values,
                        ca_rows["source_idx"].values,
                    ]
                )

                sr = cs_idx.loc[cluster_id]
                fp = Footprint.from_bounds(
                    int(sr["fp_x_min"]),
                    int(sr["fp_x_max"]),
                    int(sr["fp_y_min"]),
                    int(sr["fp_y_max"]),
                )
                cluster = cell._buildClusterData(cluster_id, fp, sources)
                cluster.extract(cell)

                cluster.x_cent = float(sr["x_cent"])
                cluster.y_cent = float(sr["y_cent"])
                cluster.dist_2 = (ca_rows["distance"].values / pix_to_arcsec) ** 2
                cluster.rms_dist = float(sr["dist_rms"]) / pix_to_arcsec
                cluster.snr_mean = float(sr["snr"])
                cluster.snr_rms = float(sr["snr_rms"])

                oa_clust = oa_cell[oa_cell["cluster_id"] == cluster_id]
                for obj_id_raw in oa_clust["object_id"].unique():
                    obj_id = int(obj_id_raw)
                    obj_rows = oa_clust[oa_clust["object_id"] == obj_id]
                    mask = self._buildObjectMask(cluster, obj_rows)
                    obj = cell._newObject(cluster, obj_id, mask)
                    or_ = os_idx.loc[obj_id]
                    obj.dist_2 = (obj_rows["distance"].values / pix_to_arcsec) ** 2
                    obj.x_cent = float(or_["x_cent"])
                    obj.y_cent = float(or_["y_cent"])
                    obj.rms_dist = float(or_["dist_rms"]) / pix_to_arcsec
                    obj.snr_mean = float(or_["snr"])
                    obj.snr_rms = float(or_["snr_rms"])
                    cell.object_dict[obj_id] = obj
                    cluster.objects.append(obj)

                cell.cluster_dict[cluster_id] = cluster

            self.cell_dict[cell_idx] = cell

        sys.stdout.write("Done!\n")
        sys.stdout.flush()

            
    def _buildObjectMask(
        self,
        cluster: ClusterData,
        obj_rows: pandas.DataFrame,
    ) -> np.ndarray:
        """Build boolean mask into cluster.sources for one object's sources."""
        mask = np.zeros(cluster.n_src, dtype=bool)
        i_cat_vals = np.array(
            [self.catalog_id_map[int(cid)] for cid in obj_rows["catalog_id"]]
        )
        src_idx_vals = obj_rows["source_idx"].values
        for i_cat_v, src_idx_v in zip(i_cat_vals, src_idx_vals):
            hits = (cluster.sources[0] == i_cat_v) & (cluster.sources[2] == src_idx_v)
            mask |= hits
        return mask

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
