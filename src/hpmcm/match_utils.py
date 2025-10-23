from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import utils
from .cluster import ClusterData
from .object import ObjectData

if TYPE_CHECKING:
    from .cell import CellData


def heirarchicalProcessObject(
    objData: ObjectData, cellData: CellData, pixelR2Cut: float, recurse: int = 0
) -> None:
    """Recursively process an object and make sub-objects

    Parameters
    ----------
    objData:
        Object being processed

    cellData:
        Associated cell

    pixelR2Cut:
        Distance cut for associated, in pixels**2

    recurse:
        Current recursion level


    Notes
    -----
    This function will test if the input objData is good,
    i.e., that all the sources lie within the pixelR2Cut,
    and will split objData into multiple objects if
    some of the sources lie outside the cut

    The new objects will be added to the parent cluster of
    the original object
    """
    if recurse > cellData.matcher.maxSubDivision:
        return
    objData.recurse = recurse

    if objData.nSrc == 0:
        print("Empty object", objData.nSrc, objData.nUnique, recurse)
        return

    assert objData.data is not None
    if objData.mask.sum() == 1:
        objData.xCent = objData.data.xPix.values[0]
        objData.yCent = objData.data.yPix.values[0]
        objData.dist2 = np.zeros((1), float)
        objData.rmsDist = 0.0
        objData.snrMean = objData.data.SNR.values[0]
        objData.snrRms = 0.0
        return

    sumSnr = np.sum(objData.data.SNR)
    objData.xCent = np.sum(objData.data.xPix * objData.data.SNR) / sumSnr
    objData.yCent = np.sum(objData.data.yPix * objData.data.SNR) / sumSnr
    objData.dist2 = np.array(
        (objData.xCent - objData.data.xPix) ** 2
        + (objData.yCent - objData.data.yPix) ** 2
    )
    objData.rmsDist = np.sqrt(np.mean(objData.dist2))
    objData.snrMean = np.mean(objData.data.SNR.values)
    objData.snrRms = np.std(objData.data.SNR.values)

    subMask = objData.dist2 < pixelR2Cut
    if subMask.all():
        return

    if recurse >= cellData.matcher.maxSubDivision:
        return

    heirarchicalSplitObject(objData, cellData, pixelR2Cut, recurse=recurse + 1)
    return


def heirarchicalSplitObject(
    objData: ObjectData, cellData: CellData, pixelR2Cut: float, recurse: int = 0
) -> None:
    """Split up a cluster keeping only one source per input
    catalog

    Parameters
    ----------
    objData:
        Object being processed

    cellData:
        Associated cell

    pixelR2Cut:
        Distance cut for associated, in pixels**2

    recurse:
        Current recursion level

    Notes
    -----
    This function actually does the splitting of objects
    """

    if recurse > cellData.matcher.maxSubDivision:
        return
    objData.recurse = recurse
    objData.extract()

    assert objData.data is not None

    bbox = objData.parentCluster.footprint.getBBox()
    zoomFactors = [1, 2, 4, 8, 16, 32]
    zoomFactor = zoomFactors[recurse]

    nPix = zoomFactor * np.array([bbox.getHeight(), bbox.getWidth()])
    zoomX = zoomFactor * objData.data.xCluster / objData.parentCluster.pixelMatchScale
    zoomY = zoomFactor * objData.data.yCluster / objData.parentCluster.pixelMatchScale

    countsMap = utils.fillCountsMapFromArrays(zoomX, zoomY, nPix)

    fpDict = utils.getFootprints(countsMap, buf=0)
    footprints = fpDict["footprints"]
    nFootprints = len(footprints.getFootprints())
    if nFootprints == 1:
        if recurse >= cellData.matcher.maxSubDivision:
            return
        heirarchicalSplitObject(objData, cellData, pixelR2Cut, recurse=recurse + 1)
        return

    footprintKey = fpDict["footprintKey"]
    footprintIds = utils.findClusterIdsFromArrays(zoomX, zoomY, footprintKey)

    biggest = np.argmax(np.bincount(footprintIds))
    biggestMask = np.zeros(objData.mask.shape, dtype=bool)

    for iFp in range(nFootprints):
        subMask = footprintIds == iFp
        count = 0
        newMask = np.zeros(objData.mask.shape, dtype=bool)
        for i, val in enumerate(objData.mask):
            if val:
                newMask[i] = subMask[count]
                count += 1
        assert subMask.sum() == newMask.sum()

        if iFp == biggest:
            biggestMask = newMask
            continue

        newObject = objData.parentCluster.addObject(cellData, newMask)
        heirarchicalProcessObject(newObject, cellData, pixelR2Cut, recurse=recurse)

    objData.mask = biggestMask
    objData.extract()
    heirarchicalProcessObject(objData, cellData, pixelR2Cut, recurse=recurse)
    return


def heirarchicalProcessCluster(
    cluster: ClusterData, cellData: CellData, pixelR2Cut: float
) -> list[ObjectData]:
    """Function that is called recursively to
    split clusters until they consist only of sources within
    the match radius of the cluster centroid.

    Recursively process a cluster and make associated objects

    Parameters
    ----------
    cluster:
        Cluster being processed

    cellData:
        Associated cell

    pixelR2Cut:
        Distance cut for associated, in pixels**2

    Returns
    -------
    Objects produced in cluster processing
    """
    cluster.extract(cellData)
    assert cluster.data is not None

    cluster.nSrc = len(cluster.data.iCat)
    cluster.nUnique = len(np.unique(cluster.data.iCat.values))
    if cluster.nSrc == 0:
        print("Empty cluster", cluster.nSrc, cluster.nUnique)
        return cluster.objects

    if cluster.nSrc == 1:
        cluster.xCent = cluster.data.xPix.values[0]
        cluster.yCent = cluster.data.yPix.values[0]
        cluster.dist2 = np.zeros((1))
        cluster.rmsDist = 0.0
        cluster.snrMean = cluster.data.SNR.values[0]
        cluster.snrRms = 0.0
        initialObject = cluster.addObject(cellData)
        heirarchicalProcessObject(initialObject, cellData, pixelR2Cut)
        return cluster.objects

    sumSnr = np.sum(cluster.data.SNR)
    cluster.xCent = np.sum(cluster.data.xPix * cluster.data.SNR) / sumSnr
    cluster.yCent = np.sum(cluster.data.yPix * cluster.data.SNR) / sumSnr
    cluster.dist2 = (cluster.xCent - cluster.data.xPix) ** 2 + (
        cluster.yCent - cluster.data.yPix
    ) ** 2
    cluster.rmsDist = np.sqrt(np.mean(cluster.dist2))
    cluster.snrMean = np.mean(cluster.data.SNR.values)
    cluster.snrRms = np.std(cluster.data.SNR.values)

    initialObject = cluster.addObject(cellData)
    heirarchicalProcessObject(initialObject, cellData, pixelR2Cut)
    return cluster.objects
