from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas

from . import input_tables, output_tables, shear_utils
from .cell import CellData, ShearCellData
from .match import Match
from .shear_utils import DEFAULT_GEOMETRY, ShearCellGeometry
from .wcs_match import createGlobalWcs


class ShearMatch(Match):
    """Class to do N-way matching for shear calibration.

    Uses pre-assigned pixel locations from cell-based coadd WCS.

    Since the pixel locations and cells are pre-assigned, the only
    configurable parameters this class takes are the attributres listed
    here.

    The pixel_match_scale can we used to allow for matching sources
    that are seperate by more that 1 pixel.

    By default expects 5 input catalogs (reference catalog ``ns`` and
    counterfactual shear catalogs ``1p``, ``1m``, ``2p``, ``2m``).  The active
    set is configurable via the ``shear_names`` constructor parameter, e.g.
    pass ``shear_names=["ns", "1p", "1m"]`` for 3-catalog mode.

    Attributes
    ----------
    pixel_match_scale: int
        Number of pixels to merge in original counts map

    cat_type: str
        Shear catalog type

    deshear: float | None
        Deshearing parameter, -1*applied shear.  None -> deshearing is not done.

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames
    that contain the following columns.

    +--------------+---------------------------------------------------------------+
    | Column name  | Description                                                   |
    +==============+===============================================================+
    | id           | source ID                                                     |
    +--------------+---------------------------------------------------------------+
    | tract        | Tract being matched                                           |
    +--------------+---------------------------------------------------------------+
    | x_cell_coadd | X-position in cell-based coadd used for metadetect             |
    +--------------+---------------------------------------------------------------+
    | y_cell_coadd | Y-position in cell-based coadd used for metadetect             |
    +--------------+---------------------------------------------------------------+
    | snr          | Signal-to-Noise of source, used for filtering and centroiding |
    +--------------+---------------------------------------------------------------+
    | cell_idx_x   | Cell x-index within Tract                                     |
    +--------------+---------------------------------------------------------------+
    | cell_idx_y   | Cell y-index within Tract                                     |
    +--------------+---------------------------------------------------------------+
    | g_1          | Shear g1 component                                            |
    +--------------+---------------------------------------------------------------+
    | g_2          | Shear g2 component                                            |
    +--------------+---------------------------------------------------------------+

    (see :py:class:`hpmcm.input_tables.ShearCoaddSourceTable`)


    These parquet files can be generated from files with the following
    columns using the ShearMatch.splitByTypeAndClean() function.

    +---------------------------------+---------------------------------------+
    | Column name                     | Description                           |
    +=================================+=======================================+
    | id                              | source ID                             |
    +---------------------------------+---------------------------------------+
    | shear_type                      | one of "ns", "1p", "1m", "2p" "2m"    |
    +---------------------------------+---------------------------------------+
    | patch_{x,y}                     | id of the patch within the tract      |
    +---------------------------------+---------------------------------------+
    | cell_{x,y}                      | id of the cell withing the patch      |
    +---------------------------------+---------------------------------------+
    | snr                             | Signal-to-Noise of source             |
    +---------------------------------+---------------------------------------+
    | {cat_type}_band_flux_{band}     | Flux measuremnt in the reference band |
    +---------------------------------+---------------------------------------+
    | {cat_type}_band_flux_err_{band} | Flux error in the reference band      |
    +---------------------------------+---------------------------------------+
    | {cat_type}_g_{i}                | Shear measurements                    |
    +---------------------------------+---------------------------------------+


    Two additional tables are produced beyond the tables produced by
    the base :py:class:`hpmcm.Match` class

    +----------------+---------------------------------------------------+
    | Key            | Class                                             |
    +================+===================================================+
    | _object_shear  | :py:class:`hpmcm.output_tables.ShearTable`        |
    +----------------+---------------------------------------------------+
    | _cluster_shear | :py:class:`hpmcm.output_tables.ShearTable`        |
    +----------------+---------------------------------------------------+
    """

    inputTableClass: type = input_tables.ShearCoaddSourceTable
    extraCols: list[str] = ["ra", "dec", "x_pix", "y_pix", "g_1", "g_2"]

    def __init__(
        self,
        **kwargs: Any,
    ):
        self.cat_type: str = kwargs.get("catalogType", "wmom")
        self.deshear: float | None = kwargs.get("deshear", None)
        shear_names = list(kwargs.get("shear_names", shear_utils.SHEAR_NAMES))
        if not all(n in shear_utils.SHEAR_NAMES for n in shear_names):
            raise ValueError(
                f"All shear_names must be in {shear_utils.SHEAR_NAMES}, got {shear_names}"
            )
        self.shear_names: list[str] = shear_names
        Match.__init__(self, **kwargs)
        geometry: ShearCellGeometry = kwargs.get("geometry", DEFAULT_GEOMETRY)
        self.geometry = geometry
        if geometry.ref_dir is None:
            raise RuntimeError("ShearMatch requires geometry.ref_dir to be set")
        self._wcs = createGlobalWcs(
            geometry.ref_dir,
            geometry.pixel_size,
            geometry.tract_size,
            ctype=geometry.wcs_ctype,
        )

    @classmethod
    def createShearMatch(
        cls,
        geometry: ShearCellGeometry = DEFAULT_GEOMETRY,
        **kwargs: Any,
    ) -> ShearMatch:
        """Helper function to create a `ShearMatch` object

        This will use the use pixel-coordinates read from
        the input shear tables.

        Parameters
        ----------
        geometry:
            Cell geometry and WCS matching configuration.
        kwargs:
            Additional keyword arguments passed to the `ShearMatch` constructor,
            overriding values derived from geometry.

        Returns
        -------
        Object to create matches for the requested region
        """
        kw = dict(
            pixel_size=geometry.pixel_size,
            n_pixels=geometry.tract_size,
            cell_size=geometry.cell_inner_size,
            cell_buffer=geometry.match_buffer,
            cell_max_object=1000,
            n_cell_buffer=1,
            geometry=geometry,
        )
        kw.update(kwargs)
        return cls(**kw)

    def pixToWorld(
        self,
        x_pix: np.ndarray,
        y_pix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert pixel coordinates to RA/Dec using the geometry WCS.

        """
        ra, dec = self._wcs.wcs_pix2world(x_pix, y_pix, 0)
        return ra, dec

    def _geometryDict(self) -> dict:
        d = super()._geometryDict()
        d.update(
            shear_names=list(self.shear_names),
            deshear=self.deshear,
            cat_type=self.cat_type,
            shear_geometry=dataclasses.asdict(self.geometry),
        )
        return d

    @classmethod
    def load(
        cls,
        save_dir: str | Path,
        x_range: tuple[int, int] | None = None,
        y_range: tuple[int, int] | None = None,
    ) -> ShearMatch:
        """Restore a ShearMatch from a directory written by save().

        Parameters
        ----------
        save_dir:
            Directory written by ``save()``.
        x_range:
            Optional ``(x_min, x_max)`` inclusive bounds on the cell x-index.
            Cells outside this range are skipped.
        y_range:
            Optional ``(y_min, y_max)`` inclusive bounds on the cell y-index.
            Cells outside this range are skipped.

        Returns
        -------
        Restored ShearMatch with ``cell_dict`` populated.
        """
        save_dir = Path(save_dir)
        with open(save_dir / "geometry.json") as fh:
            geo = json.load(fh)

        sg = geo["shear_geometry"]
        geometry = ShearCellGeometry(**sg)

        matcher = cls(
            pixel_size=geo["pixel_size"],
            n_pixels=np.array(geo["n_pixels"]),
            cell_size=geo["cell_size"],
            cell_buffer=geo["cell_buffer"],
            cell_max_object=geo["cell_max_object"],
            max_sub_division=geo["max_sub_division"],
            pixel_r2_cut=geo["pixel_r2_cut"],
            n_cell_buffer=geo["n_cell_buffer"],
            shear_names=geo["shear_names"],
            deshear=geo["deshear"],
            catalogType=geo["cat_type"],
            pixel_match_scale=geo["pixel_match_scale"],
            geometry=geometry,
        )
        matcher.catalog_id_map = {int(k): int(v) for k, v in geo["catalog_id_map"].items()}

        cluster_assoc = pandas.read_parquet(save_dir / "cluster_assoc.parquet")
        object_assoc = pandas.read_parquet(save_dir / "object_assoc.parquet")
        cluster_stats = pandas.read_parquet(save_dir / "cluster_stats.parquet")
        object_stats = pandas.read_parquet(save_dir / "object_stats.parquet")

        cluster_assoc, object_assoc = matcher._filterAssocByRange(
            cluster_assoc, object_assoc, x_range, y_range
        )

        matcher._loadReducedData(save_dir)
        per_cell_data = matcher._buildPerCellData(cluster_assoc)
        matcher._reconstructCells(cluster_assoc, object_assoc, cluster_stats, object_stats, per_cell_data)
        return matcher

    def getCellIndices(
        self,
        df: pandas.DataFrame,
    ) -> np.ndarray:
        """Get the cell index assocatiated to each source"""
        return (self.n_cell[1] * df["cell_idx_x"] + df["cell_idx_y"]).astype(int)

    def _buildCellData(
        self,
        id_offset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
    ) -> CellData:
        return ShearCellData(self, id_offset, corner, size, idx, self.cell_buffer)

    def extractShearStats(self, central_only: bool = True) -> dict[str, pandas.DataFrame]:
        """Extract shear stats

        Parameters
        ----------
        central_only:
            When ``True`` (default), exclude objects and clusters whose centroid
            falls outside the inner cell region.  Mirrors the behaviour of
            :meth:`~hpmcm.match.Match.extractStats`.

        Returns
        -------
        Dict with keys:
        ``cluster_shear`` and ``object_shear``
        (:py:class:`hpmcm.output_tables.ShearTable`).
        """
        cluster_shear_stats_tables = []
        object_shear_stats_tables = []

        for ix in range(int(self.n_cell[0])):
            for iy in range(int(self.n_cell[1])):
                i_cell = self.getCellIdx(ix, iy)
                if i_cell not in self.cell_dict:
                    continue
                cell_data = self.cell_dict[i_cell]
                assert isinstance(cell_data, ShearCellData)

                cs = output_tables.ShearTable.buildClusterShearStats(cell_data).data
                os_ = output_tables.ShearTable.buildObjectShearStats(cell_data).data

                if central_only:
                    c_obj, c_clust = self._getCentralIds(cell_data)
                    cs = cs[cs["cluster_id"].isin(c_clust)]
                    os_ = os_[os_["object_id"].isin(c_obj)]

                cluster_shear_stats_tables.append(cs)
                object_shear_stats_tables.append(os_)

        return {
            "cluster_shear": pandas.concat(cluster_shear_stats_tables),
            "object_shear": pandas.concat(object_shear_stats_tables),
        }

    def _getPixValues(self, df: pandas.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_pix, y_pix = (
            df["x_pix"].values,
            df["y_pix"].values,
        )
        return x_pix, y_pix

    def reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame

        Notes
        -----

        This applies a trivial cut on signal-to-noise (snr>1).

        This will add these columns to the output dataframes

        +--------------+-------------------------------------+
        | Column       | Description                         |
        +==============+=====================================+
        | id           | Index of object inside catalog      |
        +--------------+-------------------------------------+
        | ra           | Source RA                           |
        +--------------+-------------------------------------+
        | dec          | Source DEC                          |
        +--------------+-------------------------------------+
        | cell_idx_x   | X-index of Cell                     |
        +--------------+-------------------------------------+
        | cell_idx_y   | Y-index of Cell                     |
        +--------------+-------------------------------------+
        | x_cell_coadd | X-coordinate in cell frame          |
        +--------------+-------------------------------------+
        | y_cell_coadd | Y-coordinate in cell frame          |
        +--------------+-------------------------------------+
        | x_pix        | X-coordinate in global WCS frame    |
        +--------------+-------------------------------------+
        | y_pix        | Y-coordinate in global WCS frame    |
        +--------------+-------------------------------------+
        | g_1          | Shear g_1 component estimate        |
        +--------------+-------------------------------------+
        | g_2          | Shear g_2 component estimate        |
        +--------------+-------------------------------------+
        | snr          | Signal-to-noise ratio               |
        +--------------+-------------------------------------+

        """
        df_clean = df[(df.snr > 1)]
        df_red = df_clean.copy(deep=True)

        return df_red[
            [
                "id",
                "ra",
                "dec",
                "x_pix",
                "y_pix",
                "x_cell_coadd",
                "y_cell_coadd",
                "snr",
                "g_1",
                "g_2",
                "cell_idx_x",
                "cell_idx_y",
            ]
        ]
