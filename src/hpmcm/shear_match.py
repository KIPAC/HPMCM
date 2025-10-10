from __future__ import annotations

from typing import Any

import numpy as np
import pandas
import tables_io

from .cell import CellData, ShearCellData
from .match import Match
from .shear_data import ShearData

COLUMNS = ["ra", "dec", "id", "patch_x", "patch_y", "cell_x", "cell_y", "row", "col"]


class ShearMatch(Match):
    """Class to do N-way matching for shear calibration.

    Uses cell-based coadds toassign pixel locations to all sources in
    the input catalogs

    Expects 5 input catalogs.

    Attributes
    ----------
    _maxSubDivision: int
        Maximum number of cell sub-diviions

    _pixelMatchScale: int
        Number of cells to merge in original counts map

    _catType: str
        Shear catalog type

    _deshear: float
        Deshearing parameter

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
        self._maxSubDivision: int = kwargs.get("maxSubDivision", 3)
        self._pixelMatchScale: int = kwargs.get("pixelMatchScale", 1)
        self._catType: str = kwargs.get("catalogType", "wmom")
        self._deshear: float | None = kwargs.get("deshear", None)
        Match.__init__(self, **kwargs)

    @property
    def deshear(self) -> float | None:
        """Return the deshearing parameter"""
        return self._deshear

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
            pixelSize=0.2 * 3600.0,
            nPixels=nPix,
            cellSize=150,
            cellBuffer=25,
            cellMaxObject=1000,
        )
        return cls(**kw, **kwargs)

    @classmethod
    def splitByTypeAndClean(
        cls,
        basefile: str,
        tract: int,
        shear: float,
    ) -> None:
        """Split a parquet file by shear catalog type"""
        TYPES = ["ns", "1m", "2m", "1p", "2p"]
        p = tables_io.read(basefile)
        for type_ in TYPES:
            mask = p["shear_type"] == type_
            sub = p[mask].copy(deep=True)
            idx_x = (20 * sub["patch_x"].values + sub["cell_x"].values).astype(int)
            idx_y = (20 * sub["patch_y"].values + sub["cell_y"].values).astype(int)
            cent_x = 150 * idx_x - 75
            cent_y = 150 * idx_y - 75
            xCell_coadd = sub["col"] - cent_x
            yCell_coadd = sub["row"] - cent_y
            sub["xCell_coadd"] = xCell_coadd
            sub["yCell_coadd"] = yCell_coadd
            sub["SNR"] = sub["wmom_band_flux_r"] / sub["wmom_band_flux_err_r"]
            sub["g_1"] = sub["wmom_g_1"]
            sub["g_2"] = sub["wmom_g_2"]
            sub["idx_x"] = idx_x
            sub["idx_y"] = idx_y
            sub["orig_id"] = sub.id
            sub["id"] = np.arange(len(sub))
            central_to_cell = np.bitwise_and(
                np.fabs(xCell_coadd) < 80, np.fabs(yCell_coadd) < 80
            )
            central_to_patch = np.bitwise_and(
                np.fabs(sub["cell_x"].values - 10.5) < 10,
                np.fabs(sub["cell_y"].values - 10.5) < 10,
            )
            right_tract = sub["tract"] == tract
            central = np.bitwise_and(central_to_cell, central_to_patch)
            selected = np.bitwise_and(right_tract, central)
            cleaned = sub[selected].copy(deep=True)
            cleaned["shear"] = np.repeat(shear, len(cleaned))
            cleaned.to_parquet(
                basefile.replace(".parq", f"_uncleaned_{tract}_{type_}.pq")
            )

    def _buildCellData(
        self,
        idOffset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: np.ndarray,
    ) -> CellData:
        return ShearCellData(self, idOffset, corner, size, idx, self._cellBuffer)

    def extractShearStats(self) -> list[pandas.DataFrame]:
        """Extract shear stats"""
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

    def _getPixValues(self, df: pandas.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xPix, yPix = (
            df["col"].values + 25,
            df["row"].values + 25,
        )
        return xPix, yPix

    def _reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame"""
        df_clean = df[(df.SNR > 1)]
        xPix, yPix = self._getPixValues(df_clean)
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

    @staticmethod
    def stats(
        weights: np.ndarray,
        bin_centers: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Compute the stats from a histogram"""
        w = np.sum(weights)
        mean = np.sum(weights * bin_centers) / w
        deltas = bin_centers - mean
        var = np.sum(weights * deltas * deltas) / w
        std = np.sqrt(var)
        error = std / np.sqrt(w)
        inv_var = 1.0 / (error * error)
        return float(mean), float(std), float(error), float(inv_var)

    @classmethod
    def shear_report(
        cls,
        basefile: str,
        outputFileBase: str | None,
        shear: float,
        catType: str,
        tract: int,
    ) -> None:
        """Report on the shear calibration"""
        t = tables_io.read(f"{basefile}_object_shear.pq")
        t2 = tables_io.read(f"{basefile}_object_stats.pq")

        shearData = ShearData(t, t2, shear, catType, tract, snrCut=7.5)

        if outputFileBase is not None:
            shearData.save(f"{outputFileBase}.pkl")
            shearData.savefigs(outputFileBase)

    @classmethod
    def merge_shear_reports(
        cls,
        inputs: list[str],
        outputFile: str,
    ) -> None:
        """Merge report on the shear calibration"""

        outDict: dict[str, Any] = {}
        for input_ in inputs:
            shearData = ShearData.load(input_)
            inputDict = shearData.toDict()
            for key, val in inputDict.items():
                if key in outDict:
                    outDict[key].append(val)
                else:
                    outDict[key] = [val]

        outDF = pandas.DataFrame(outDict)
        outDF.to_parquet(outputFile)
