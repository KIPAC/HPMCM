from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas
import tables_io

from .shear_data import ShearData

if TYPE_CHECKING:
    from .cell import CellData
    from .shear_match import ShearMatch


SHEAR_NAMES = ["ns", "2p", "2m", "1p", "1m"]


def shearStats(df: pandas.DataFrame) -> dict:
    """Return the shear statistics

    {st} is the shear type, one of "gauss", "pgauss", "wmom"

    {i}, {j} index the shear parameters 1, 2

    Parameters
    ----------
    df:
        Input DataFrame, must have :py:class:`hpmcm.ShearTable` schema

    Returns
    -------
    n_{st} : int
        Number of sources from that catalog

    g_{i}_{st} : float
        g_{i} shear parameter for that catalog

    delta_g_{i}_{j} : float
        g_{i,j} shear measurment: g_{i}_{j}p - g_{i}_{j}m

    good: bool
        True if every catalog has one source in this object

    Notes
    -----
    If the matching is not good, then delta_g_1 = delta_g_2 = np.nan
    """
    outDict: dict[str, float | int] = {}
    allGood = True
    for i, name_ in enumerate(SHEAR_NAMES):
        mask = df.iCat == i
        nCat = mask.sum()
        if nCat != 1:
            allGood = False
        outDict[f"n_{name_}"] = int(nCat)
        if nCat:
            outDict[f"g_1_{name_}"] = df[mask].g_1.values.mean()
            outDict[f"g_2_{name_}"] = df[mask].g_2.values.mean()
        else:
            outDict[f"g_1_{name_}"] = np.nan
            outDict[f"g_2_{name_}"] = np.nan
    if allGood:
        outDict["delta_g_1_1"] = outDict["g_1_1p"] - outDict["g_1_1m"]
        outDict["delta_g_2_2"] = outDict["g_2_2p"] - outDict["g_2_2m"]
        outDict["delta_g_1_2"] = outDict["g_1_2p"] - outDict["g_1_2m"]
        outDict["delta_g_2_1"] = outDict["g_2_1p"] - outDict["g_2_1m"]
    else:
        outDict["delta_g_1_1"] = np.nan
        outDict["delta_g_2_2"] = np.nan
        outDict["delta_g_1_2"] = np.nan
        outDict["delta_g_2_1"] = np.nan
    outDict["good"] = allGood
    return outDict


def shearReport(
    basefile: str,
    outputFileBase: str | None,
    shear: float,
    catType: str,
    tract: int,
    snrCut: float = 7.5,
) -> None:
    """Report on the shear calibration

    Parameters
    ----------
    basefile
        Input base file name (see notes)

    outputFileBase:
        Output file name (see notes)

    shear:
        Applied shear

    catType:
        Catalog type (one of ["pgauss", "gauss", "wmom"]

    tract:
        Tract, written to outout data

    snrCut:
        Signal-to-noise cut.

    Notes
    -----
    This will read the object shear data from "{basefile}_object_shear.pq"
    This will read the object statisticis from "{basefile}_object_stats.pq"

    This will write the shear stats to "{outputFileBase}.pkl"
    This will write the figures to "{outputFileBase}_{figure}.png"
    """
    t = tables_io.read(f"{basefile}_object_shear.pq")
    t2 = tables_io.read(f"{basefile}_object_stats.pq")

    shearData = ShearData(t, t2, shear, catType, tract, snrCut=snrCut)

    if outputFileBase is not None:
        shearData.save(f"{outputFileBase}.pkl")
        shearData.savefigs(outputFileBase)


def mergeShearReports(
    inputs: list[str],
    outputFile: str,
) -> None:
    """Merge reports on the shear calibration

    Parameters
    ----------
    inputs:
        List of input ShearData pickle files

    outputFile:
        Where to write the merged file
    """
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


def splitByTypeAndClean(
    basefile: str,
    tract: int,
    shear: float,
    catType: str,
    *,
    clean: bool = False,
) -> None:
    """Split a parquet file by shear catalog type and tract

    Parameters
    ----------
    basefile:
        Original file name

    tract:
        Tract to select

    shear:
        Applied shear, saved to output

    catType:
        Catalog type to select

    clean:
        Remove duplicates

    Notes
    -----
    This will create 5 files with the pattern:
    "{basefile}_uncleaned_{tract}_{type}.pq"
    """
    p = tables_io.read(basefile)
    if clean:
        clean_st = "cleaned"
        cellCut = 75
    else:
        clean_st = "uncleaned"
        cellCut = 80
    for type_ in SHEAR_NAMES:
        mask = p["shear_type"] == type_
        sub = p[mask].copy(deep=True)
        cellIdxX = (20 * sub["patch_x"].values + sub["cell_x"].values).astype(int)
        cellIdxY = (20 * sub["patch_y"].values + sub["cell_y"].values).astype(int)
        cent_x = 150 * cellIdxX - 75
        cent_y = 150 * cellIdxY - 75
        xCellCoadd = sub["col"] - cent_x
        yCellCoadd = sub["row"] - cent_y
        sub["xPix"] = sub["col"] + 25
        sub["yPix"] = sub["row"] + 25
        sub["xCellCoadd"] = xCellCoadd
        sub["yCellCoadd"] = yCellCoadd
        sub["SNR"] = sub[f"{catType}_band_flux_r"] / sub[f"{catType}_band_flux_err_r"]
        sub["g_1"] = sub[f"{catType}_g_1"]
        sub["g_2"] = sub[f"{catType}_g_2"]
        sub["cellIdxX"] = cellIdxX
        sub["cellIdxY"] = cellIdxY
        sub["orig_id"] = sub.id
        sub["id"] = np.arange(len(sub))
        central_to_cell = np.bitwise_and(
            np.fabs(xCellCoadd) < cellCut, np.fabs(yCellCoadd) < cellCut
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
        cleaned.to_parquet(basefile.replace(".parq", f"_{clean_st}_{tract}_{type_}.pq"))


def reduceShearDataForCell(
    cell: CellData, iCat: int, dataframe: pandas.DataFrame
) -> pandas.DataFrame:
    """Filters dataframe to keep only source in the cell"""

    matcher = cell.matcher

    if TYPE_CHECKING:
        assert isinstance(matcher, ShearMatch)

    filteredIdx = matcher.getCellIndices(dataframe) == cell.idx
    reduced = dataframe[filteredIdx].copy(deep=True)

    # These are the coeffs for the various shear catalogs
    deshear_coeffs = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, -1, -1, 0],
            [1, 0, 0, -1],
            [-1, 0, 0, 1],
        ]
    )

    xCellOrig = reduced["xCellCoadd"]
    yCellOrig = reduced["yCellCoadd"]
    xPixOrig = reduced["xPix"]
    yPixOrig = reduced["yPix"]

    if matcher.deshear is not None:
        # De-shear in the cell frame to do matching
        dxShear = matcher.deshear * (
            xCellOrig * deshear_coeffs[iCat][0] + yCellOrig * deshear_coeffs[iCat][2]
        )
        dyShear = matcher.deshear * (
            xCellOrig * deshear_coeffs[iCat][1] + yCellOrig * deshear_coeffs[iCat][3]
        )
        xCell = xCellOrig + dxShear
        yCell = yCellOrig + dyShear
        xPix = xPixOrig + dxShear
        yPix = yPixOrig + dyShear
    else:
        dxShear = np.zeros(len(dataframe))
        dyShear = np.zeros(len(dataframe))
        xCell = xCellOrig
        yCell = yCellOrig
        xPix = xPixOrig
        yPix = yPixOrig

    xCell = (xCell + 100) / matcher.pixelMatchScale
    yCell = (yCell + 100) / matcher.pixelMatchScale
    filteredX = np.bitwise_and(xCell >= 0, xCell < cell.nPix[0])
    filteredY = np.bitwise_and(yCell >= 0, yCell < cell.nPix[1])
    filteredBounds = np.bitwise_and(filteredX, filteredY)
    red = reduced[filteredBounds].copy(deep=True)
    red["xCell"] = xCell[filteredBounds]
    red["yCell"] = yCell[filteredBounds]
    red["xPix"] = xPix[filteredBounds]
    red["yPix"] = yPix[filteredBounds]
    if matcher.deshear is not None:
        red["dxShear"] = dxShear[filteredBounds]
        red["dyShear"] = dyShear[filteredBounds]
    return red
