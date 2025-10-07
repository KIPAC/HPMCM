from __future__ import annotations

from typing import Any

import numpy as np
import pandas

from .cell import CellData, ShearCellData
from .match import Match

COLUMNS = ["ra", "dec", "id", "patch_x", "patch_y", "cell_x", "cell_y", "row", "col"]


class ShearMatch(Match):
    """Class to do N-way matching for shear calibration.

    Uses cell-based coadds toassign pixel locations to all sources in
    the input catalogs

    Expects 5 input catalogs.

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames
    that contain the following columns.

    "id" : source ID
    "xCell_coadd": X-postion in cell-based coadd used for metadetect
    "yCell_coadd": Y-postion in cell-based coadd used for metadetect
    "g_1": shape measurement
    "g_2": shape measurement
    "SNR": Signal-to-Noise of source, used for filtering and centroiding

    """

    def __init__(
        self,
        **kwargs: Any,
    ):
        self._pixSize = kwargs.get("pixelSize", 1.0)
        self._nPixSide = kwargs["nPixels"]
        self._maxSubDivision: int = kwargs.get("maxSubDivision", 3)
        self._pixelMatchScale: int = kwargs.get("pixelMatchScale", 1)
        self._catType: str = kwargs.get("catalogType", "wmom")
        self._deshear: float | None = kwargs.get("deshear", None)
        Match.__init__(self, **kwargs)

    @property
    def deshear(self) -> float | None:
        return self._deshear

    @classmethod
    def createShearMatch(
        cls,
        **kwargs: Any,
    ) -> Match:
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
            pixelSize=0.2 * 3600.0,
            nPixels=nPix,
            cellSize=150,
            cellBuffer=25,
            cellMaxObject=1000,
        )
        return cls(**kw, **kwargs)

    def analyzeCell(
        self,
        ix: int,
        iy: int,
        fullData: bool = False,
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
        dict:
            Dict with the information listed below

        'cellData' : `CellData`
            The analysis data for the Cell

        if fullData is True the return dict will include

        'image' : `afwImage.ImageI`
            Image of cell source counts map
        'countsMap' : `np.array`
            Numpy array with same
        'clusters' : `afwDetect.FootprintSet`
            Clusters as dectected by finding FootprintSet on source counts map
        'clusterKey' : `afwImage.ImageI`
            Map of cell with pixels filled with index of
            associated Footprints
        """
        cellKey = (ix, iy)
        iCell = np.array(cellKey).astype(int)
        cellStep = np.array([self._cellSize, self._cellSize])
        corner = iCell * cellStep
        idOffset = self.getIdOffset(ix, iy)
        cellData = CellData(self, idOffset, corner, cellStep, iCell, self._cellBuffer)
        cellData.reduceData(list(self._redData.values()))
        oDict = cellData.analyze(pixelR2Cut=self._pixelR2Cut)
        if cellData.nObjects >= self._cellMaxObject:
            print("Too many object in a cell", cellData.nObjects, self._cellMaxObject)

        self._cellDict[cellKey] = cellData
        if oDict is None:
            return None
        if fullData:
            oDict["cellData"] = cellData
            return oDict

        return dict(cellData=cellData)

    def extractShearStats(self) -> list[pandas.DataFrame]:
        clusterShearStatsTables = []
        objectShearStatsTables = []

        for ix in range(int(self._nCell[0])):
            for iy in range(int(self._nCell[1])):
                iCell = (ix, iy)
                if iCell not in self._cellDict:
                    continue
                cellData = self._cellDict[iCell]
                assert isinstance(cellData, ShearCellData)
                clusterShearStatsTables.append(cellData.getClusterShearStats())
                objectShearStatsTables.append(cellData.getObjectShearStats())

        return [
            pandas.concat(clusterShearStatsTables),
            pandas.concat(objectShearStatsTables),
        ]

    def _reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame"""
        df_clean = df[(df.SNR > 1)]
        if self._wcs is not None:
            xPix, yPix = self._wcs.wcs_world2pix(
                df_clean["ra"].values, df_clean["dec"].values, 0
            )
        else:
            xPix, yPix = (
                df_clean["col"].values + 25,
                df_clean["row"].values + 25,
            )
        df_red = df_clean.copy(deep=True)

        df_red["xPix"] = xPix
        df_red["yPix"] = yPix

        return df_red[
            [
                "id",
                "ra",
                "dec",
                "xPix",
                "yPix",
                "xCell_coadd",
                "yCell_coadd",
                "SNR",
                "g_1",
                "g_2",
                "idx_x",
                "idx_y",
            ]
        ]
