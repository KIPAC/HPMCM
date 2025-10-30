from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import utils
from .cluster import ClusterData
from .object import ObjectData

if TYPE_CHECKING:
    from .cell import CellData


def heirarchicalProcessObject(
    obj_data: ObjectData, cell_data: CellData, pixel_r2_cut: float, recurse: int = 0
) -> None:
    """Recursively process an object and make sub-objects

    Parameters
    ----------
    obj_data:
        Object being processed

    cell_data:
        Associated cell

    pixel_r2_cut:
        Distance cut for associated, in pixels**2

    recurse:
        Current recursion level


    Notes
    -----
    This function will test if the input obj_data is good,
    i.e., that all the sources lie within the pixel_r2_cut,
    and will split obj_data into multiple objects if
    some of the sources lie outside the cut

    The new objects will be added to the parent cluster of
    the original object
    """
    if recurse > cell_data.matcher.max_sub_division:
        return
    obj_data.recurse = recurse

    if obj_data.n_src == 0:
        print("Empty object", obj_data.n_src, obj_data.n_unique, recurse)
        return

    assert obj_data.data is not None
    if obj_data.mask.sum() == 1:
        obj_data.x_cent = obj_data.data.x_pix.values[0]
        obj_data.y_cent = obj_data.data.y_pix.values[0]
        obj_data.dist_2 = np.zeros((1), float)
        obj_data.rms_dist = 0.0
        obj_data.snr_mean = obj_data.data.snr.values[0]
        obj_data.snr_rms = 0.0
        return

    sum_snr = np.sum(obj_data.data.snr)
    obj_data.x_cent = np.sum(obj_data.data.x_pix * obj_data.data.snr) / sum_snr
    obj_data.y_cent = np.sum(obj_data.data.y_pix * obj_data.data.snr) / sum_snr
    obj_data.dist_2 = np.array(
        (obj_data.x_cent - obj_data.data.x_pix) ** 2
        + (obj_data.y_cent - obj_data.data.y_pix) ** 2
    )
    obj_data.rms_dist = np.sqrt(np.mean(obj_data.dist_2))
    obj_data.snr_mean = np.mean(obj_data.data.snr.values)
    obj_data.snr_rms = np.std(obj_data.data.snr.values)

    sub_mask = obj_data.dist_2 < pixel_r2_cut
    if sub_mask.all():
        return

    if recurse >= cell_data.matcher.max_sub_division:
        return

    heirarchicalSplitObject(obj_data, cell_data, pixel_r2_cut, recurse=recurse + 1)
    return


def heirarchicalSplitObject(
    obj_data: ObjectData, cell_data: CellData, pixel_r2_cut: float, recurse: int = 0
) -> None:
    """Split up a cluster keeping only one source per input
    catalog

    Parameters
    ----------
    obj_data:
        Object being processed

    cell_data:
        Associated cell

    pixel_r2_cut:
        Distance cut for associated, in pixels**2

    recurse:
        Current recursion level

    Notes
    -----
    This function actually does the splitting of objects
    """

    if recurse > cell_data.matcher.max_sub_division:
        return
    obj_data.recurse = recurse
    obj_data.extract()

    assert obj_data.data is not None

    slice_x = obj_data.parent_cluster.footprint.slice_x
    slice_y = obj_data.parent_cluster.footprint.slice_y
    zoom_factors = [1, 2, 4, 8, 16, 32]
    zoom_factor = zoom_factors[recurse]

    n_pix = zoom_factor * np.array(
        [slice_x.stop - slice_x.start, slice_y.stop - slice_y.start]
    )
    zoom_x = zoom_factor * obj_data.data.x_cluster / obj_data.parent_cluster.pixel_match_scale
    zoom_y = zoom_factor * obj_data.data.y_cluster / obj_data.parent_cluster.pixel_match_scale

    counts_map = utils.fillCountsMapFromArrays(zoom_x, zoom_y, n_pix)

    fp_dict = utils.getFootprints(counts_map, buf=0)
    footprints = fp_dict["footprints"]
    n_footprints = len(footprints.footprints)
    if n_footprints == 1:
        if recurse >= cell_data.matcher.max_sub_division:
            return
        heirarchicalSplitObject(obj_data, cell_data, pixel_r2_cut, recurse=recurse + 1)
        return

    footprint_key = fp_dict["footprint_key"]
    footprint_ids = utils.findClusterIdsFromArrays(
        zoom_x.values.astype(int), zoom_y.values.astype(int), footprint_key
    )

    biggest = np.argmax(np.bincount(footprint_ids))
    biggest_mask = np.zeros(obj_data.mask.shape, dtype=bool)

    for i_fp in range(n_footprints):
        sub_mask = footprint_ids == i_fp
        count = 0
        new_mask = np.zeros(obj_data.mask.shape, dtype=bool)
        for i, val in enumerate(obj_data.mask):
            if val:
                new_mask[i] = sub_mask[count]
                count += 1
        assert sub_mask.sum() == new_mask.sum()

        if i_fp == biggest:
            biggest_mask = new_mask
            continue

        new_object = obj_data.parent_cluster.addObject(cell_data, new_mask)
        heirarchicalProcessObject(new_object, cell_data, pixel_r2_cut, recurse=recurse)

    obj_data.mask = biggest_mask
    obj_data.extract()
    heirarchicalProcessObject(obj_data, cell_data, pixel_r2_cut, recurse=recurse)
    return


def heirarchicalProcessCluster(
    cluster: ClusterData, cell_data: CellData, pixel_r2_cut: float
) -> list[ObjectData]:
    """Function that is called recursively to
    split clusters until they consist only of sources within
    the match radius of the cluster centroid.

    Recursively process a cluster and make associated objects

    Parameters
    ----------
    cluster:
        Cluster being processed

    cell_data:
        Associated cell

    pixel_r2_cut:
        Distance cut for associated, in pixels**2

    Returns
    -------
    Objects produced in cluster processing
    """
    cluster.extract(cell_data)
    assert cluster.data is not None

    cluster.n_src = len(cluster.data.i_cat)
    cluster.n_unique = len(np.unique(cluster.data.i_cat.values))
    if cluster.n_src == 0:
        print("Empty cluster", cluster.n_src, cluster.n_unique)
        return cluster.objects

    if cluster.n_src == 1:
        cluster.x_cent = cluster.data.x_pix.values[0]
        cluster.y_cent = cluster.data.y_pix.values[0]
        cluster.dist_2 = np.zeros((1))
        cluster.rms_dist = 0.0
        cluster.snr_mean = cluster.data.snr.values[0]
        cluster.snr_rms = 0.0
        initial_object = cluster.addObject(cell_data)
        heirarchicalProcessObject(initial_object, cell_data, pixel_r2_cut)
        return cluster.objects

    sum_snr = np.sum(cluster.data.snr)
    cluster.x_cent = np.sum(cluster.data.x_pix * cluster.data.snr) / sum_snr
    cluster.y_cent = np.sum(cluster.data.y_pix * cluster.data.snr) / sum_snr
    cluster.dist_2 = (cluster.x_cent - cluster.data.x_pix) ** 2 + (
        cluster.y_cent - cluster.data.y_pix
    ) ** 2
    cluster.rms_dist = np.sqrt(np.mean(cluster.dist_2))
    cluster.snr_mean = np.mean(cluster.data.snr.values)
    cluster.snr_rms = np.std(cluster.data.snr.values)

    initial_object = cluster.addObject(cell_data)
    heirarchicalProcessObject(initial_object, cell_data, pixel_r2_cut)
    return cluster.objects
