from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure

from .cell import CellData
from .cluster import ClusterData
from .match import Match


def showShearObjs(matcher: Match, iK: tuple[int, int]) -> Figure | SubFigure:
    """Draw the objects in a cluster

    Parameters
    ----------
    matcher:
        Match object

    iK:
        Indices of the particular cluster

    Returns
    -------
    Figure showing the cluster in question
    """
    cellData = matcher.cellDict[iK[0]]
    cluster = cellData.clusterDict[iK[1]]
    extent = cluster.footprint.extent()
    cluster.extract(cellData)
    xOffset = cellData.minPix[0] + 25
    yOffset = cellData.minPix[1] + 25
    assert cluster.data is not None
    xOff = cluster.data.xPix - xOffset
    yOff = cluster.data.yPix - yOffset
    catIndices = cluster.catIndices
    image = cluster.footprint.cutout
    img = plt.imshow(image, origin="lower", extent=extent)
    colors = ["red", "blue", "green", "cyan", "orange"]
    markers = [".", "<", ">", "v", "^"]
    for iObj, obj in enumerate(cluster.objects):
        for x_, y_, i_ in zip(xOff[obj.mask], yOff[obj.mask], catIndices[obj.mask]):
            img.axes.scatter(x_, y_, c=colors[iObj % 5], marker=markers[i_ % 5])
    _cb = plt.colorbar(label="Objects per pixel")
    assert img.axes.figure is not None
    return img.axes.figure


def showShearObj(matcher: Match, iK: tuple[int, int]) -> Figure | SubFigure:
    """Draw a single object

    Parameters
    ----------
    matcher:
        Match object

    iK:
        Indices of the particular object

    Returns
    -------
    Figure showing the object in question
    """

    cellData = matcher.cellDict[iK[0]]
    theObj = cellData.objectDict[iK[1]]
    cluster = theObj.parentCluster
    extent = cluster.footprint.extent()
    cluster.extract(cellData)
    xOffset = cellData.minPix[0] + 25
    yOffset = cellData.minPix[1] + 25
    assert cluster.data is not None
    xOff = cluster.data.xPix - xOffset
    yOff = cluster.data.yPix - yOffset
    catIndices = cluster.catIndices
    img = plt.imshow(cluster.footprint.cutout, origin="lower", extent=extent)
    markers = [".", "<", ">", "v", "^"]
    for _iObj, obj in enumerate(cluster.objects):
        if obj.objectId == theObj.objectId:
            color = "red"
        else:  # pragma: no cover
            color = "blue"
        for x_, y_, i_ in zip(xOff[obj.mask], yOff[obj.mask], catIndices[obj.mask]):
            img.axes.scatter(x_, y_, c=color, marker=markers[i_ % 5])
    _cb = plt.colorbar(label="Objects per pixel")
    assert img.axes.figure is not None
    return img.axes.figure


def showCluster(
    image: np.ndarray,
    cluster: ClusterData,
    cellData: CellData,
    mask: np.ndarray | None = None,
) -> Figure | SubFigure:
    """Draw a cluster

    Parameters
    ----------
    image:
        Counts map used to make clusters

    cluster:
        Cluster being draw

    cellData:
        Parent Cell for the cluster

    mask:
        Mask showing which sources are in the cluster

    Returns
    -------
    Figure showing the cluster in question
    """
    extent = cluster.footprint.extent()
    cluster.extract(cellData)
    xOffset = cellData.minPix[0] + 25
    yOffset = cellData.minPix[1] + 25
    xOff = cluster.xCluster
    yOff = cluster.yCluster
    if mask is not None:  # pragma: no cover
        xOff = xOff[mask]
        yOff = yOff[mask]
    xC = cluster.xCent - xOffset
    yC = cluster.yCent - yOffset

    img = plt.imshow(
        image[cluster.footprint.sliceX][cluster.footprint.sliceY],
        origin="lower",
        extent=extent,
        cmap="grey",
    )
    _cb = plt.colorbar(label="Objects per pixel")
    try:
        assert cluster.data is not None
        xOff_u = xOff - cluster.data.dxShear
        yOff_u = yOff - cluster.data.dyShear
        img.axes.scatter(xOff_u, yOff_u, marker="x")
    except Exception:  # pragma: no cover
        pass
    img.axes.scatter(xOff, yOff)
    img.axes.scatter(xC, yC, marker="+", c="green")
    img.axes.set_xlabel("x [pixels]")
    img.axes.set_ylabel("y [pixels]")
    assert img.axes.figure is not None
    return img.axes.figure


def showObjects(
    image: np.ndarray,
    cluster: ClusterData,
    cellData: CellData,
) -> Figure | SubFigure:
    """Draw a cluster, showing the objects

    Parameters
    ----------
    image:
        Counts map used to make clusters

    cluster:
        Cluster being draw

    cellData:
        Parent Cell for the cluster

    mask:
        Mask showing which sources are in the cluster

    Returns
    -------
    Figure showing the cluster in question
    """
    extent = cluster.footprint.extent()
    cluster.extract(cellData)
    xOffset = cellData.minPix[0]
    yOffset = cellData.minPix[1]
    xOff = cluster.xCluster
    yOff = cluster.yCluster
    img = plt.imshow(
        image[cluster.footprint.sliceX][cluster.footprint.sliceY],
        origin="lower",
        extent=extent,
    )
    _cb = plt.colorbar(label="Objects per pixel")
    colors = ["red", "blue", "green", "cyan", "orange"]
    for iObj, obj in enumerate(cluster.objects):
        xC = obj.xCent - xOffset
        yC = obj.yCent - yOffset
        img.axes.scatter(
            xOff[obj.mask],
            yOff[obj.mask],
            c=colors[iObj % 5],
            s=1 + np.ceil(iObj / 5),
        )
        print(1 + np.ceil(iObj / 5))
        img.axes.scatter(xC, yC, marker="+", c=colors[iObj % 6])
    assert img.axes.figure is not None
    return img.axes.figure


def showObjectsV2(
    image: np.ndarray,
    cluster: ClusterData,
    cellData: CellData,
) -> Figure | SubFigure:
    """Draw a cluster, showing the objects

    Parameters
    ----------
    image:
        Counts map used to make clusters

    cluster:
        Cluster being draw

    cellData:
        Parent Cell for the cluster

    Returns
    -------
    Figure showing the cluster in question
    """
    extent = cluster.footprint.extent()
    cluster.extract(cellData)
    xOffset = cellData.minPix[0]
    yOffset = cellData.minPix[1]
    xOff = cluster.xPix - xOffset
    yOff = cluster.yPix - yOffset
    img = plt.imshow(
        image[cluster.footprint.sliceX][cluster.footprint.sliceY],
        origin="lower",
        extent=extent,
    )
    _cb = plt.colorbar()
    colors = ["red", "blue", "green", "cyan", "orange"]
    for xOff_, yOff_, iCat_ in zip(xOff, yOff, cluster.catIndices):
        if iCat_ % 5 == 0 and iCat_ != 20:
            continue
        img.axes.scatter(
            xOff_, yOff_, c=colors[iCat_ % 5], s=20 - 3 * np.ceil(iCat_ / 5)
        )
    assert img.axes.figure is not None
    return img.axes.figure
