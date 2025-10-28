from __future__ import annotations


import numpy as np
import pandas

from .footprint import FootprintSet


def findClusterIdsFromArrays(
    xLocals: np.ndarray,
    yLocals: np.ndarray,
    clusterKey: np.ndarray,
) -> np.ndarray:
    """Associate sources to clusters using `clusterkey`
    which is a map where any pixel associated to a cluster
    has the cluster index as its value

    Parameters
    ----------
    xLocals:
        Local pixel x-positions

    yLocals:
        Local pixel y-positions

    clusterKey:
        2D-map of cluster Ids by pixel position

    Returns
    -------
    Ids of associated clusters
    """
    return np.array(
        [clusterKey[xLocal, yLocal] for xLocal, yLocal in zip(xLocals, yLocals)]
    ).astype(np.int32)


def findClusterIds(
    df: pandas.DataFrame,
    clusterKey: np.ndarray,
    pixelMatchScale: int = 1,
) -> np.ndarray:
    """Associate sources to clusters using `clusterkey`
    which is a map where any pixel associated to a cluster
    has the cluster index as its value

    Parameters
    ----------
    df:
        DataFrame with local pixel positions (xCell, yCell)

    clusterKey:
        2D-map of cluster Ids by pixel position

    pixelMatchScale:
        Scale-factor to use in making cluster map

    Returns
    -------
    Ids of associated clusters
    """
    return findClusterIdsFromArrays(
        np.floor(df["xCell"] / pixelMatchScale).astype(int),
        np.floor(df["yCell"] / pixelMatchScale).astype(int),
        clusterKey,
    )


def fillCountsMapFromArrays(
    xLocals: np.ndarray,
    yLocals: np.ndarray,
    nPix: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fill a source counts map

    Parameters
    ----------
    xLocals:
        Local pixel x-positions

    yLocals:
        Local pixel y-positions

    nPix:
        Number of pixels in x,y for counts map

    weights:
        If provided, weights to apply for each entry in counts map

    Returns
    -------
    Counts map of source in cell, projected into nPix,nPix grid
    """
    hist = np.histogram2d(
        xLocals,
        yLocals,
        bins=(nPix[0], nPix[1]),
        range=((0, nPix[0]), (0, nPix[1])),
        weights=weights,
    )
    return hist[0]


def fillCountsMapFromDf(
    df: pandas.DataFrame,
    nPix: np.ndarray,
    weightName: str | None = None,
    pixelMatchScale: int = 1,
) -> np.ndarray:
    """Fill a source counts map from a reduced dataframe for one input
    catalog

    Parameters
    ----------
    df:
        DataFrame with local pixel positions (xCell, yCell)

    nPix:
        Number of pixels in x,y for counts map

    weightName:
        If provided column to use for weights

    pixelMatchScale:
        Scale-factor to use in making cluster map

    Returns
    -------
    Counts map of source in cell, projected into nPix,nPix grid
    """
    if weightName is None:
        weights = None
    else:  # pragma: no cover
        weights = df[weightName].values
    return fillCountsMapFromArrays(
        df["xCell"] / pixelMatchScale,
        df["yCell"] / pixelMatchScale,
        nPix=np.ceil(nPix / pixelMatchScale).astype(int),
        weights=weights,
    )


def getFootprints(
    countsMap: np.ndarray,
    buf: int,
    pixelMatchScale: int = 1,
) -> dict:
    """Take a source counts map and do clustering using Footprint detection

    Parameters
    ----------
    countsMap:
        Map of source counts

    buf:
        Number of pixels in cell-edge buffer

    pixelMatchScale:
        Scale-factor used in making cluster map

    Returns
    -------
    Footprint data


    Notes
    -----
    image: np.ndarray : countsMap

    footprints: FootprintSet : Clustering FootprintSet

    footprintKey: np.ndarray : Array with same shape as countsMap, with cluster associations
    """
    footprintsOrig = FootprintSet.detect(countsMap)
    footprints = footprintsOrig.filter(buf, pixelMatchScale)
    footprintKey = footprints.fpKey
    return dict(image=countsMap, footprints=footprints, footprintKey=footprintKey)


def associateSourcesToFootprints(
    data: list[pandas.DataFrame],
    clusterKey: np.ndarray,
    pixelMatchScale: int = 1,
) -> list[np.ndarray]:
    """Loop through data and associate sources to clusters

    Parameters
    ----------
    data:
        Input DataFrames

    clusterKey:
        2D-map of cluster Ids by pixel position

    pixelMatchScale:
        Scale-factor used in making cluster map

    Returns
    -------
    Lists of clusters associated to each source
    output[i][j] will give the id of the cluster associated
    to source j in input catalog i.
    """
    return [findClusterIds(df, clusterKey, pixelMatchScale) for df in data]
