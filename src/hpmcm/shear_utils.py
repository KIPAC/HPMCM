from __future__ import annotations

import numpy as np
import pandas

SHEAR_NAMES = ["ns", "2p", "2m", "1p", "1m"]


def shearStats(df: pandas.DataFrame) -> dict:
    """Return the shear statistics

    {st} is the shear type, one of "gauss", "pgauss", "wmom"

    {i}, {j} index the shear parameters 1, 2

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
            outDict[f"g_1_{name_}"] = df[mask].values.mean()
            outDict[f"g_2_{name_}"] = df[mask].values.mean()
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
