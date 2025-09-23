import lsst.afw.image as afwImage
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure

from .cell import CellData
from .cluster import ClusterData


def showCluster(
    image: afwImage.ImageF,
    cluster: ClusterData,
    cellData: CellData,
    mask: np.ndarray | None = None,
) -> Figure | SubFigure:
    """Draw a cluster"""
    extent = (
        0,
        cluster.footprint.getBBox().getWidth(),
        0,
        cluster.footprint.getBBox().getHeight(),
    )
    cluster.extract(cellData)
    xOffset = cellData.minPix[0] + cluster.footprint.getBBox().getBeginY()
    yOffset = cellData.minPix[1] + cluster.footprint.getBBox().getBeginX()
    xOff = cluster.xCluster
    yOff = cluster.yCluster
    if mask is not None:
        xOff = xOff[mask]
        yOff = yOff[mask]
    xC = cluster.xCent - xOffset
    yC = cluster.yCent - yOffset
    img = plt.imshow(
        image[cluster.footprint.getBBox()].array, origin="lower", extent=extent
    )
    _cb = plt.colorbar()
    img.axes.scatter(yOff, xOff)
    img.axes.scatter(yC, xC, marker="+", c="green")
    assert img.axes.figure is not None
    return img.axes.figure


def showObjects(
    image: afwImage.ImageF,
    cluster: ClusterData,
    cellData: CellData,
) -> Figure | SubFigure:
    """Draw a cluster, showing the objects"""
    extent = (
        0,
        cluster.footprint.getBBox().getWidth(),
        0,
        cluster.footprint.getBBox().getHeight(),
    )
    cluster.extract(cellData)
    xOffset = cellData.minPix[0] + cluster.footprint.getBBox().getBeginY()
    yOffset = cellData.minPix[1] + cluster.footprint.getBBox().getBeginX()
    xOff = cluster.xCluster
    yOff = cluster.yCluster
    img = plt.imshow(
        image[cluster.footprint.getBBox()].array, origin="lower", extent=extent
    )
    _cb = plt.colorbar()
    colors = ["red", "blue", "green", "cyan", "orange"]
    for iObj, obj in enumerate(cluster.objects):
        xC = obj.xCent - xOffset
        yC = obj.yCent - yOffset
        img.axes.scatter(
            yOff[obj.mask],
            xOff[obj.mask],
            c=colors[iObj % 5],
            s=1 + np.ceil(iObj / 5),
        )
        print(1 + np.ceil(iObj / 5))
        img.axes.scatter(yC, xC, marker="+", c=colors[iObj % 6])
    assert img.axes.figure is not None
    return img.axes.figure


def showObjectsV2(
    image: afwImage.ImageF,
    cluster: ClusterData,
    cellData: CellData,
) -> Figure | SubFigure:
    """Draw a cluster, showing the objects"""
    extent = (
        0,
        cluster.footprint.getBBox().getWidth(),
        0,
        cluster.footprint.getBBox().getHeight(),
    )
    cluster.extract(cellData)
    xOffset = cellData.minPix[0] + cluster.footprint.getBBox().getBeginY()
    yOffset = cellData.minPix[1] + cluster.footprint.getBBox().getBeginX()
    xOff = cluster.xPix - xOffset
    yOff = cluster.yPix - yOffset
    img = plt.imshow(
        image[cluster.footprint.getBBox()].array, origin="lower", extent=extent
    )
    _cb = plt.colorbar()
    colors = ["red", "blue", "green", "cyan", "orange"]
    for xOff_, yOff_, iCat_ in zip(xOff, yOff, cluster.catIndices):
        if iCat_ % 5 == 0 and iCat_ != 20:
            continue
        img.axes.scatter(
            yOff_, xOff_, c=colors[iCat_ % 5], s=20 - 3 * np.ceil(iCat_ / 5)
        )
    assert img.axes.figure is not None
    return img.axes.figure
