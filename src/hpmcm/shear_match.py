from __future__ import annotations

from typing import Any

import numpy as np
import pandas

from . import input_tables, output_tables
from .cell import CellData, ShearCellData
from .match import Match


class ShearMatch(Match):
    """Class to do N-way matching for shear calibration.

    Uses cell-based coadds toassign pixel locations to all sources in
    the input catalogs

    Expects 5 input catalogs.

    Attributes
    ----------
    maxSubDivision: int
        Maximum number of cell sub-diviions

    pixelMatchScale: int
        Number of cells to merge in original counts map

    catType: str
        Shear catalog type

    deshear: float
        Deshearing parameter

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames
    that contain the following columns.

     :py:class:`hpmcm.ShearCoaddSourceTable`

    These parquet files can be generated from files with the following
    columns using the ShearMatch.splitByTypeAndClean() function.

    "id": source ID
    "shear_type": one of "ns", "1p", "1m", "2p" "2m"
    "patch_{x,y}": id of the patch within the tract
    "cell_{x,y}": id of the cell withing the patch
    "row, col": global row/col in the tract WCS
    "{catType}_band_flux_{band}": flux measuremnt in the reference band
    "{catType}_band_flux_err_{band}": flux measuremnt error in the reference band
    "{catType}_g_{i}": shear measurements

    Two additional tables are produced beyond the tables produeced by
    the base :py:class:`hpmcm.Match` class

    _object_shear: :py:class:`hpmcm.ShearTable`

    _cluster_shear: :py:class:`hpmcm.ShearTable`
    """

    inputTableClass: type = input_tables.ShearCoaddSourceTable
    extraCols: list[str] = ["ra", "dec", "xPix", "yPix", "g_1", "g_2"]

    def __init__(
        self,
        **kwargs: Any,
    ):
        self.maxSubDivision: int = kwargs.get("maxSubDivision", 3)
        self.pixelMatchScale: int = kwargs.get("pixelMatchScale", 1)
        self.catType: str = kwargs.get("catalogType", "wmom")
        self.deshear: float | None = kwargs.get("deshear", None)
        Match.__init__(self, **kwargs)

    @classmethod
    def createShearMatch(
        cls,
        **kwargs: Any,
    ) -> ShearMatch:
        """Helper function to create a Match object

        Parameters
        ----------
        refDir:
            Reference Direction (RA, DEC) in degrees

        pixSize:
            Pixel size in degrees

        Returns
        -------
        Match:
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
        """Get the Index to use for a given cell"""
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
        """Extract shear stats"""
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
        """Reduce a single input DataFrame"""
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
