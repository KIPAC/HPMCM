from __future__ import annotations

# This is to simplify sphinx builds
try:
    import lsst.afw.detection as afwDetect
    import lsst.afw.image as afwImage
except ImportError:
    pass
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


def findClusterIds(
    df: pandas.DataFrame,
    clusterKey: np.ndarray,
    pixelMatchScale: int = 1,
) -> np.ndarray:
    """Associate sources to clusters using `clusterkey`
    which is a map where any pixel associated to a cluster
    has the cluster index as its value"""
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
    df: pandas.DataFrame,
    nPix: np.ndarray,
    weightName: str | None = None,
    pixelMatchScale: int = 1,
) -> np.ndarray:
    """Fill a source counts map from a reduced dataframe for one input
    catalog"""
    if weightName is None:
        weights = None
    else:
        weights = df[weightName].values
    return fillCountsMapFromArrays(
        df["xCell"] / pixelMatchScale,
        df["yCell"] / pixelMatchScale,
        nPix=np.ceil(nPix / pixelMatchScale).astype(int),
        weights=weights,
    )


def filterFootprints(
    fpSet: afwDetect.FootprintSet,
    buf: int,
    pixelMatchScale: int = 1,
) -> afwDetect.FootprintSet:
    """Remove footprints within `buf` pixels of the celll edge"""
    region = fpSet.getRegion()
    width, height = region.getWidth(), region.getHeight()
    outList = []
    maxX = width - buf
    maxY = height - buf
    for fp in fpSet.getFootprints():
        cent = fp.getCentroid()
        xC = cent.getX() * pixelMatchScale
        yC = cent.getY() * pixelMatchScale
        if xC < buf or xC > maxX or yC < buf or yC > maxY:
            continue
        outList.append(fp)
    fpSetOut = afwDetect.FootprintSet(fpSet.getRegion())
    fpSetOut.setFootprints(outList)
    return fpSetOut


def getFootprints(
    countsMap: np.ndarray,
    buf: int,
    pixelMatchScale: int = 1,
) -> dict:
    """Take a source counts map and do clustering using Footprint detection"""
    image = afwImage.ImageF(countsMap.astype(np.float32))
    footprintsOrig = afwDetect.FootprintSet(image, afwDetect.Threshold(0.5))
    if buf == 0:
        footprints = footprintsOrig
    else:
        footprints = filterFootprints(footprintsOrig, buf, pixelMatchScale)
    footprintKey = afwImage.ImageI(np.full(countsMap.shape, -1, dtype=np.int32))
    for i, footprint in enumerate(footprints.getFootprints()):
        footprint.spans.setImage(footprintKey, i, doClip=True)
    return dict(image=image, footprints=footprints, footprintKey=footprintKey)


def associateSourcesToFootprints(
    data: list[pandas.DataFrame],
    clusterKey: np.ndarray,
    pixelMatchScale: int = 1,
) -> list[np.ndarray]:
    """Loop through data and associate sources to clusters"""
    return [findClusterIds(df, clusterKey, pixelMatchScale) for df in data]
