from __future__ import annotations

from typing import Any

import numpy as np
import pandas
import tables_io

from .cell import CellData, ShearCellData
from .match import Match
from .shear_data import ShearData


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

    "id" : source ID
    "xCellCoadd": X-postion in cell-based coadd used for metadetect
    "yCellCoadd": Y-postion in cell-based coadd used for metadetect
    "g_1": shape measurement
    "g_2": shape measurement
    "SNR": Signal-to-Noise of source, used for filtering and centroiding

    These parquet files can be generate from files with the following
    columns using the ShearMatch.splitByTypeAndClean() function.

    "id": source ID
    "shear_type": one of "ns", "1p", "1m", "2p" "2m"
    "patch_{x,y}": id of the patch within the tract
    "cell_{x,y}": id of the cell withing the patch
    "row, col": global row/col in the tract WCS
    "{catType}_band_flux_{band}": flux measuremnt in the reference band
    "{catType}_band_flux_err_{band}": flux measuremnt error in the reference band
    "{catType}_g_{i}": shear measurements


    Two additional tables are produced

    object_shear:
    good              : True if assoicated object is well matched
    n_{name}          : Number of sources from catalog {name}
    g_{i}_{name}      : Shear measurement {i} from catalog {name}
    delta_g_{i}_{j}   : g_{i}_{j}p - g_{i}_{j}m for good objects only

    cluster_shear:
    good              : True if assoicated cluster is well matched
    n_{name}          : Number of sources from catalog {name}
    g_{i}_{name}      : Shear measurement {i} from catalog {name}
    delta_g_{i}_{j}   : g_{i}_{j}p - g_{i}_{j}m for good objects only
    """

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

    @classmethod
    def splitByTypeAndClean(
        cls,
        basefile: str,
        tract: int,
        shear: float,
        catType: str,
    ) -> None:
        """Split a parquet file by shear catalog type"""
        TYPES = ["ns", "1m", "2m", "1p", "2p"]
        p = tables_io.read(basefile)
        for type_ in TYPES:
            mask = p["shear_type"] == type_
            sub = p[mask].copy(deep=True)
            cellIdxX = (20 * sub["patch_x"].values + sub["cell_x"].values).astype(int)
            cellIdxY = (20 * sub["patch_y"].values + sub["cell_y"].values).astype(int)
            cent_x = 150 * cellIdxX - 75
            cent_y = 150 * cellIdxY - 75
            xCellCoadd = sub["col"] - cent_x
            yCellCoadd = sub["row"] - cent_y
            sub["xCellCoadd"] = xCellCoadd
            sub["yCellCoadd"] = yCellCoadd
            sub["SNR"] = (
                sub[f"{catType}_band_flux_r"] / sub[f"{catType}_band_flux_err_r"]
            )
            sub["g_1"] = sub[f"{catType}_g_1"]
            sub["g_2"] = sub[f"{catType}_g_2"]
            sub["cellIdxX"] = cellIdxX
            sub["cellIdxY"] = cellIdxY
            sub["orig_id"] = sub.id
            sub["id"] = np.arange(len(sub))
            central_to_cell = np.bitwise_and(
                np.fabs(xCellCoadd) < 80, np.fabs(yCellCoadd) < 80
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
                "xCellCoadd",
                "yCellCoadd",
                "SNR",
                "g_1",
                "g_2",
                "cellIdxX",
                "cellIdxY",
            ]
        ]

    @classmethod
    def shear_report(
        cls,
        basefile: str,
        outputFileBase: str | None,
        shear: float,
        catType: str,
        tract: int,
        snrCut: float = 7.5,
    ) -> None:
        """Report on the shear calibration"""
        t = tables_io.read(f"{basefile}_object_shear.pq")
        t2 = tables_io.read(f"{basefile}_object_stats.pq")

        shearData = ShearData(t, t2, shear, catType, tract, snrCut=snrCut)

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
