from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure

from .cell import CellData
from .cluster import ClusterData
from .match import Match


def showShearObjs(matcher: Match, i_k: tuple[int, int]) -> Figure | SubFigure:
    """Draw the objects in a cluster

    Parameters
    ----------
    matcher:
        Match object

    i_k:
        Indices of the particular cluster

    Returns
    -------
    Figure showing the cluster in question
    """
    cell_data = matcher.cell_dict[i_k[0]]
    cluster = cell_data.cluster_dict[i_k[1]]
    extent = cluster.footprint.extent()
    cluster.extract(cell_data)
    x_offset = cell_data.min_pix[0] + 25
    y_offset = cell_data.min_pix[1] + 25
    assert cluster.data is not None
    x_off = cluster.data.x_pix - x_offset
    y_off = cluster.data.y_pix - y_offset
    catalog_ids = cluster.catalog_id
    image = cluster.footprint.cutout
    img = plt.imshow(image, origin="lower", extent=extent)
    colors = ["red", "blue", "green", "cyan", "orange"]
    markers = [".", "<", ">", "v", "^"]
    for i_obj, obj in enumerate(cluster.objects):
        for x_, y_, i_ in zip(x_off[obj.mask], y_off[obj.mask], catalog_ids[obj.mask]):
            img.axes.scatter(x_, y_, c=colors[i_obj % 5], marker=markers[i_ % 5])
    _cb = plt.colorbar(label="Objects per pixel")
    assert img.axes.figure is not None
    return img.axes.figure


def showShearObj(matcher: Match, i_k: tuple[int, int]) -> Figure | SubFigure:
    """Draw a single object

    Parameters
    ----------
    matcher:
        Match object

    i_k:
        Indices of the particular object

    Returns
    -------
    Figure showing the object in question
    """

    cell_data = matcher.cell_dict[i_k[0]]
    the_obj = cell_data.object_dict[i_k[1]]
    cluster = the_obj.parent_cluster
    extent = cluster.footprint.extent()
    cluster.extract(cell_data)
    x_offset = cell_data.min_pix[0] + 25
    y_offset = cell_data.min_pix[1] + 25
    assert cluster.data is not None
    x_off = cluster.data.x_pix - x_offset
    y_off = cluster.data.y_pix - y_offset
    catalog_ids = cluster.catalog_id
    img = plt.imshow(cluster.footprint.cutout, origin="lower", extent=extent)
    markers = [".", "<", ">", "v", "^"]
    for _i_obj, obj in enumerate(cluster.objects):
        if obj.object_id == the_obj.object_id:
            color = "red"
        else:  # pragma: no cover
            color = "blue"
        for x_, y_, i_ in zip(x_off[obj.mask], y_off[obj.mask], catalog_ids[obj.mask]):
            img.axes.scatter(x_, y_, c=color, marker=markers[i_ % 5])
    _cb = plt.colorbar(label="Objects per pixel")
    assert img.axes.figure is not None
    return img.axes.figure


def showCluster(
    image: np.ndarray,
    cluster: ClusterData,
    cell_data: CellData,
    mask: np.ndarray | None = None,
) -> Figure | SubFigure:
    """Draw a cluster

    Parameters
    ----------
    image:
        Counts map used to make clusters

    cluster:
        Cluster being draw

    cell_data:
        Parent Cell for the cluster

    mask:
        Mask showing which sources are in the cluster

    Returns
    -------
    Figure showing the cluster in question
    """
    extent = cluster.footprint.extent()
    cluster.extract(cell_data)
    x_offset = cell_data.min_pix[0] + 25
    y_offset = cell_data.min_pix[1] + 25
    x_off = cluster.x_cluster
    y_off = cluster.y_cluster
    if mask is not None:  # pragma: no cover
        x_off = x_off[mask]
        y_off = y_off[mask]
    x_c = cluster.x_cent - x_offset
    y_c = cluster.y_cent - y_offset

    img = plt.imshow(
        image[cluster.footprint.slice_x][cluster.footprint.slice_y],
        origin="lower",
        extent=extent,
        cmap="grey",
    )
    _cb = plt.colorbar(label="Objects per pixel")
    try:
        assert cluster.data is not None
        x_off_u = x_off - cluster.data.dxShear
        y_off_u = y_off - cluster.data.dyShear
        img.axes.scatter(x_off_u, y_off_u, marker="x")
    except Exception:  # pragma: no cover
        pass
    img.axes.scatter(x_off, y_off)
    img.axes.scatter(x_c, y_c, marker="+", c="green")
    img.axes.set_xlabel("x [pixels]")
    img.axes.set_ylabel("y [pixels]")
    assert img.axes.figure is not None
    return img.axes.figure


def showObjects(
    image: np.ndarray,
    cluster: ClusterData,
    cell_data: CellData,
) -> Figure | SubFigure:
    """Draw a cluster, showing the objects

    Parameters
    ----------
    image:
        Counts map used to make clusters

    cluster:
        Cluster being draw

    cell_data:
        Parent Cell for the cluster

    mask:
        Mask showing which sources are in the cluster

    Returns
    -------
    Figure showing the cluster in question
    """
    extent = cluster.footprint.extent()
    cluster.extract(cell_data)
    x_offset = cell_data.min_pix[0]
    y_offset = cell_data.min_pix[1]
    x_off = cluster.x_cluster
    y_off = cluster.y_cluster
    img = plt.imshow(
        image[cluster.footprint.slice_x][cluster.footprint.slice_y],
        origin="lower",
        extent=extent,
    )
    _cb = plt.colorbar(label="Objects per pixel")
    colors = ["red", "blue", "green", "cyan", "orange"]
    for i_obj, obj in enumerate(cluster.objects):
        x_c = obj.x_cent - x_offset
        y_c = obj.y_cent - y_offset
        img.axes.scatter(
            x_off[obj.mask],
            y_off[obj.mask],
            c=colors[i_obj % 5],
            s=1 + np.ceil(i_obj / 5),
        )
        print(1 + np.ceil(i_obj / 5))
        img.axes.scatter(x_c, y_c, marker="+", c=colors[i_obj % 6])
    assert img.axes.figure is not None
    return img.axes.figure


def showObjectsV2(
    image: np.ndarray,
    cluster: ClusterData,
    cell_data: CellData,
) -> Figure | SubFigure:
    """Draw a cluster, showing the objects

    Parameters
    ----------
    image:
        Counts map used to make clusters

    cluster:
        Cluster being draw

    cell_data:
        Parent Cell for the cluster

    Returns
    -------
    Figure showing the cluster in question
    """
    extent = cluster.footprint.extent()
    cluster.extract(cell_data)
    x_offset = cell_data.min_pix[0]
    y_offset = cell_data.min_pix[1]
    x_off = cluster.x_pix - x_offset
    y_off = cluster.y_pix - y_offset
    img = plt.imshow(
        image[cluster.footprint.slice_x][cluster.footprint.slice_y],
        origin="lower",
        extent=extent,
    )
    _cb = plt.colorbar()
    colors = ["red", "blue", "green", "cyan", "orange"]
    for x_off_, y_off_, i_cat_ in zip(x_off, y_off, cluster.catalog_id):
        if i_cat_ % 5 == 0 and i_cat_ != 20:
            continue
        img.axes.scatter(
            x_off_, y_off_, c=colors[i_cat_ % 5], s=20 - 3 * np.ceil(i_cat_ / 5)
        )
    assert img.axes.figure is not None
    return img.axes.figure
