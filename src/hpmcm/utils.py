from __future__ import annotations

import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage
import numpy as np
import pandas


def findClusterIdsFromArrays(
    xLocals: np.ndarray,
    yLocals: np.ndarray,
    clusterKey: np.ndarray,
) -> np.ndarray:
    """Associate sources to clusters using `clusterkey`
    which is a map where any pixel associated to a cluster
    has the cluster index as its value"""
    return np.array(
        [clusterKey[yLocal, xLocal] for xLocal, yLocal in zip(xLocals, yLocals)]
    ).astype(np.int32)


def findClusterIds(df: pandas.DataFrame, clusterKey: np.ndarray) -> np.ndarray:
    """Associate sources to clusters using `clusterkey`
    which is a map where any pixel associated to a cluster
    has the cluster index as its value"""
    return findClusterIdsFromArrays(df["xlocal"], df["ylocal"], clusterKey)


def fillCountsMapFromArrays(
    xLocals: np.ndarray,
    yLocals: np.ndarray,
    nPix: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fill a source counts map"""
    hist = np.histogram2d(
        xLocals,
        yLocals,
        bins=(nPix[0], nPix[1]),
        range=((0, nPix[0]), (0, nPix[1])),
        weights=weights,
    )
    return hist[0]


def fillCountsMapFromDf(
    df: pandas.DataFrame, nPix: np.ndarray, weightName: str | None = None
) -> np.ndarray:
    """Fill a source counts map from a reduced dataframe for one input
    catalog"""
    if weightName is None:
        weights = None
    else:
        weights = df[weightName].values
    return fillCountsMapFromArrays(
        df["xlocal"],
        df["ylocal"],
        nPix=nPix,
        weights=weights,
    )


def filterFootprints(fpSet: afwDetect.FootprintSet, buf: int) -> afwDetect.FootprintSet:
    """Remove footprints within `buf` pixels of the celll edge"""
    region = fpSet.getRegion()
    width, height = region.getWidth(), region.getHeight()
    outList = []
    maxX = width - buf
    maxY = height - buf
    for fp in fpSet.getFootprints():
        cent = fp.getCentroid()
        xC = cent.getX()
        yC = cent.getY()
        if xC < buf or xC > maxX or yC < buf or yC > maxY:
            continue
        outList.append(fp)
    fpSetOut = afwDetect.FootprintSet(fpSet.getRegion())
    fpSetOut.setFootprints(outList)
    return fpSetOut


def getFootprints(countsMap: np.ndarray, buf: int) -> dict:
    """Take a source counts map and do clustering using Footprint detection"""
    image = afwImage.ImageF(countsMap.astype(np.float32))
    footprintsOrig = afwDetect.FootprintSet(image, afwDetect.Threshold(0.5))
    if buf == 0:
        footprints = footprintsOrig
    else:
        footprints = filterFootprints(footprintsOrig, buf)
    footprintKey = afwImage.ImageI(np.full(countsMap.shape, -1, dtype=np.int32))
    for i, footprint in enumerate(footprints.getFootprints()):
        footprint.spans.setImage(footprintKey, i, doClip=True)
    return dict(image=image, footprints=footprints, footprintKey=footprintKey)


def associateSourcesToFootprints(
    data: list[pandas.DataFrame], clusterKey: np.ndarray
) -> list[np.ndarray]:
    """Loop through data and associate sources to clusters"""
    return [findClusterIds(df, clusterKey) for df in data]

