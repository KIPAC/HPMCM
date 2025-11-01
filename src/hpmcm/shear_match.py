from __future__ import annotations

from typing import Any

import numpy as np
import pandas

from . import input_tables, output_tables
from .cell import CellData, ShearCellData
from .match import Match


TRACT_SIZE = np.array([30000, 30000])
PIXEL_SIZE = 0.2 / 3600.0
CELL_INNER_SIZE = 150
CELL_BUFFER = 25


class ShearMatch(Match):
    """Class to do N-way matching for shear calibration.

    Uses pre-assigned pixel locations from cell-based coadd WCS.

    Since the pixel locations and cells are pre-assigned, the only
    configurable parameters this class takes are the attributres listed
    here.

    The pixel_match_scale can we used to allow for matching sources
    that are seperate by more that 1 pixel.

    Expects 5 input catalogs: a reference catalog and 4 counterfactual
    shear catalogs.

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
    | x_cell_coadd | X-postion in cell-based coadd used for metadetect             |
    +--------------+---------------------------------------------------------------+
    | y_cell_coadd | Y-postion in cell-based coadd used for metadetect             |
    +--------------+---------------------------------------------------------------+
    | snr          | Signal-to-Noise of source, used for filtering and centroiding |
    +--------------+---------------------------------------------------------------+
    | cell_idx_x   | Cell x-index within Tract                                     |
    +--------------+---------------------------------------------------------------+
    | cell_idx_y   | Cell y-index within Tract                                     |
    +--------------+---------------------------------------------------------------+
    | g_1          | Shear g1 component                                            |
    +--------------+---------------------------------------------------------------+
    | g_2          | Shear g1 component                                            |
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
        self.pixel_match_scale: int = kwargs.get("pixel_match_scale", 1)
        self.cat_type: str = kwargs.get("catalogType", "wmom")
        self.deshear: float | None = kwargs.get("deshear", None)
        Match.__init__(self, **kwargs)

    @classmethod
    def createShearMatch(
        cls,
        **kwargs: Any,
    ) -> ShearMatch:
        """Helper function to create a `ShearMatch` object

        This will use the use pixel-coordinates read from
        the input shear tables.

        Parameters
        ----------
        kwargs:
            Passed directly to `ShearMatch` constructor.

        Returns
        -------
        Object to create matches for the requested region
        """
        n_pix = TRACT_SIZE
        kw = dict(
            pixel_size=PIXEL_SIZE,
            n_pixels=n_pix,
            cell_size=CELL_INNER_SIZE,
            cell_buffer=CELL_BUFFER,
            cell_max_object=1000,
        )
        return cls(**kw, **kwargs)

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

    def extractShearStats(self) -> list[pandas.DataFrame]:
        """Extract shear stats

        Theis will produce two :py:class:`hpmcm.output_tables.ShearTable`,
        one for the objects, and the other for the clusters.
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
                cluster_shear_stats_tables.append(
                    output_tables.ShearTable.buildClusterShearStats(cell_data).data
                )
                object_shear_stats_tables.append(
                    output_tables.ShearTable.buildObjectShearStats(cell_data).data
                )

        return [
            pandas.concat(cluster_shear_stats_tables),
            pandas.concat(object_shear_stats_tables),
        ]

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
