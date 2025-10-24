from __future__ import annotations

from typing import Any

import numpy as np
import pandas

from . import input_tables, output_tables
from .cell import CellData, ShearCellData
from .match import Match


class ShearMatch(Match):
    """Class to do N-way matching for shear calibration.

    Uses pre-assigned pixel locations from cell-based coadd WCS.

    Expects 5 input catalogs: a reference catalog and 4 counterfactual
    shear catalogs.

    Attributes
    ----------
    pixelMatchScale: int
        Number of cells to merge in original counts map

    catType: str
        Shear catalog type

    deshear: float | None
        Deshearing parameter, -1*applied shear.  None -> dshearing is not done.

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames
    that contain the following columns.

    +-------------+---------------------------------------------------------------+
    | Column name | Description                                                   |
    +=============+===============================================================+
    | id          | source ID                                                     |
    +-------------+---------------------------------------------------------------+
    | tract       | Tract being matched                                           |
    +-------------+---------------------------------------------------------------+
    | xCellCoadd  | X-postion in cell-based coadd used for metadetect             |
    +-------------+---------------------------------------------------------------+
    | yCellCoadd  | Y-postion in cell-based coadd used for metadetect             |
    +-------------+---------------------------------------------------------------+
    | SNR         | Signal-to-Noise of source, used for filtering and centroiding |
    +-------------+---------------------------------------------------------------+
    | cellIdxX    | Cell x-index within Tract                                     |
    +-------------+---------------------------------------------------------------+
    | cellIdxY    | Cell y-index within Tract                                     |
    +-------------+---------------------------------------------------------------+
    | g_1         | Shear g1 component                                            |
    +-------------+---------------------------------------------------------------+
    | g_2         | Shear g1 component                                            |
    +-------------+---------------------------------------------------------------+

    (see :py:class:`hpmcm.input_tables.ShearCoaddSourceTable`)


    These parquet files can be generated from files with the following
    columns using the ShearMatch.splitByTypeAndClean() function.

    +--------------------------------+---------------------------------------+
    | Column name                    | Description                           |
    +================================+=======================================+
    | id                             | source ID                             |
    +--------------------------------+---------------------------------------+
    | shear_type                     | one of "ns", "1p", "1m", "2p" "2m"    |
    +--------------------------------+---------------------------------------+
    | patch_{x,y}                    | id of the patch within the tract      |
    +--------------------------------+---------------------------------------+
    | cell_{x,y}                     | id of the cell withing the patch      |
    +--------------------------------+---------------------------------------+
    | SNR                            | Signal-to-Noise of source             |
    +--------------------------------+---------------------------------------+
    | {catType}_band_flux_{band}     | Flux measuremnt in the reference band |
    +--------------------------------+---------------------------------------+
    | {catType}_band_flux_err_{band} | Flux error in the reference band      |
    +--------------------------------+---------------------------------------+
    | {catType}_g_{i}                | Shear measurements                    |
    +--------------------------------+---------------------------------------+


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
    extraCols: list[str] = ["ra", "dec", "xPix", "yPix", "g_1", "g_2"]

    def __init__(
        self,
        **kwargs: Any,
    ):
        self.pixelMatchScale: int = kwargs.get("pixelMatchScale", 1)
        self.catType: str = kwargs.get("catalogType", "wmom")
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
        nPix = np.array([30000, 30000])
        kw = dict(
            pixelSize=0.2 / 3600.0,
            nPixels=nPix,
            cellSize=150,
            cellBuffer=25,
            cellMaxObject=1000,
        )
        return cls(**kw, **kwargs)

    def getCellIndices(
        self,
        df: pandas.DataFrame,
    ) -> np.ndarray:
        """Get the cell index assocatiated to each source"""
        return (self.nCell[1] * df["cellIdxX"] + df["cellIdxY"]).astype(int)

    def _buildCellData(
        self,
        idOffset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
    ) -> CellData:
        return ShearCellData(self, idOffset, corner, size, idx, self.cellBuffer)

    def extractShearStats(self) -> list[pandas.DataFrame]:
        """Extract shear stats

        Theis will produce two :py:class:`hpmcm.output_tables.ShearTable`, 
        one for the objects, and the other for the clusters.
        """
        clusterShearStatsTables = []
        objectShearStatsTables = []

        for ix in range(int(self.nCell[0])):
            for iy in range(int(self.nCell[1])):
                iCell = self.getCellIdx(ix, iy)
                if iCell not in self.cellDict:
                    continue
                cellData = self.cellDict[iCell]
                assert isinstance(cellData, ShearCellData)
                clusterShearStatsTables.append(
                    output_tables.ShearTable.buildClusterShearStats(cellData).data
                )
                objectShearStatsTables.append(
                    output_tables.ShearTable.buildObjectShearStats(cellData).data
                )

        return [
            pandas.concat(clusterShearStatsTables),
            pandas.concat(objectShearStatsTables),
        ]

    def _getPixValues(self, df: pandas.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xPix, yPix = (
            df["xPix"].values,
            df["yPix"].values,
        )
        return xPix, yPix

    def _reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame

        This applies a trivial cut on signal-to-noise (SNR>1).
        """
        df_clean = df[(df.SNR > 1)]
        df_red = df_clean.copy(deep=True)

        return df_red[
            [
                "id",
                "ra",
                "dec",
                "xPix",
                "yPix",
                "xCellCoadd",
                "yCellCoadd",
                "SNR",
                "g_1",
                "g_2",
                "cellIdxX",
                "cellIdxY",
            ]
        ]
