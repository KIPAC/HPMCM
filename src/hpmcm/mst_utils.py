from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from .cell import CellData
    from .cluster import ClusterData
    from .object import ObjectData


def mst_group_sources(x: np.ndarray, y: np.ndarray, cut_pix: float) -> np.ndarray:
    """Group sources by proximity using MST-equivalent single-linkage clustering.

    Two sources end up in the same group if and only if there is a chain of
    sources connecting them where each consecutive pair is within ``cut_pix``
    pixels of each other.  This is identical to building the full minimum
    spanning tree and cutting every edge longer than ``cut_pix``, but is
    computed more efficiently by finding connected components of the sparse
    near-neighbour graph.

    Parameters
    ----------
    x, y:
        Cell-frame pixel coordinates, one entry per source (all catalogs
        concatenated).
    cut_pix:
        Maximum pair distance in cell pixels to keep connected.

    Returns
    -------
    Integer component labels with shape ``(N,)``.  Sources with the same
    label belong to the same group.  Labels are 0-indexed and contiguous.
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.zeros(1, dtype=np.int32)

    xy = np.column_stack([x, y])
    tree = cKDTree(xy)
    pairs = list(tree.query_pairs(r=cut_pix))

    if not pairs:
        return np.arange(n, dtype=np.int32)

    pairs_arr = np.array(pairs)
    rows = pairs_arr[:, 0]
    cols = pairs_arr[:, 1]
    ones = np.ones(len(rows), dtype=np.float32)

    graph = csr_matrix(
        (
            np.concatenate([ones, ones]),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(n, n),
    )
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int32)


def mstProcessCluster(
    cluster: ClusterData,
    cell_data: CellData,
    pixel_r2_cut: float,
    mst_cut_pix: float,
) -> list[ObjectData]:
    """Process a cluster into objects using MST-based splitting.

    Replaces :py:func:`~hpmcm.match_utils.heirarchicalProcessCluster` for
    MST matchers.  Computes cluster centroid statistics then delegates
    per-object refinement to :py:func:`mstProcessObject`.

    Parameters
    ----------
    cluster:
        Cluster to process.
    cell_data:
        Parent cell.
    pixel_r2_cut:
        Maximum squared distance from object centroid (cell pixels) for a
        source to be considered well-matched.
    mst_cut_pix:
        Initial MST grouping radius in cell pixels.  Halved at each
        recursion level by :py:func:`mstSplitObject`.
    """
    cluster.extract(cell_data)
    assert cluster.data is not None

    cluster.n_src = len(cluster.data.i_cat)
    cluster.n_unique = len(np.unique(cluster.data.i_cat.values))

    if cluster.n_src == 0:
        print("Empty cluster", cluster.n_src, cluster.n_unique)
        return cluster.objects

    if cluster.n_src == 1:
        cluster.x_cent = cluster.data.x_cell.values[0]
        cluster.y_cent = cluster.data.y_cell.values[0]
        cluster.dist_2 = np.zeros((1))
        cluster.rms_dist = 0.0
        cluster.snr_mean = cluster.data.snr.values[0]
        cluster.snr_rms = 0.0
        initial_object = cluster.addObject(cell_data)
        mstProcessObject(initial_object, cell_data, pixel_r2_cut, mst_cut_pix)
        return cluster.objects

    sum_snr = np.sum(cluster.data.snr)
    cluster.x_cent = np.sum(cluster.data.x_cell * cluster.data.snr) / sum_snr
    cluster.y_cent = np.sum(cluster.data.y_cell * cluster.data.snr) / sum_snr
    cluster.dist_2 = (
        (cluster.x_cent - cluster.data.x_cell) ** 2
        + (cluster.y_cent - cluster.data.y_cell) ** 2
    )
    cluster.rms_dist = np.sqrt(np.mean(cluster.dist_2))
    cluster.snr_mean = np.mean(cluster.data.snr.values)
    cluster.snr_rms = np.std(cluster.data.snr.values)

    initial_object = cluster.addObject(cell_data)
    mstProcessObject(initial_object, cell_data, pixel_r2_cut, mst_cut_pix)
    return cluster.objects


def mstProcessObject(
    obj_data: ObjectData,
    cell_data: CellData,
    pixel_r2_cut: float,
    mst_cut_pix: float,
    recurse: int = 0,
) -> None:
    """Refine one object, splitting it if sources are confused or out of range.

    Replaces :py:func:`~hpmcm.match_utils.heirarchicalProcessObject` for
    MST matchers.  When splitting is needed, delegates to
    :py:func:`mstSplitObject` with ``mst_cut_pix / 2``.

    Parameters
    ----------
    obj_data:
        Object to refine.
    cell_data:
        Parent cell.
    pixel_r2_cut:
        Maximum squared distance from centroid (cell pixels) for a source
        to be well-matched.
    mst_cut_pix:
        MST cut radius in cell pixels for the *current* recursion level.
        :py:func:`mstSplitObject` will be called with ``mst_cut_pix / 2``.
    recurse:
        Current recursion depth.
    """
    if recurse > cell_data.matcher.max_sub_division:
        return
    obj_data.recurse = recurse

    if obj_data.n_src == 0:
        print("Empty object", obj_data.n_src, obj_data.n_unique, recurse)
        return

    assert obj_data.data is not None
    if obj_data.mask.sum() == 1:
        obj_data.x_cent = obj_data.data.x_cell.values[0]
        obj_data.y_cent = obj_data.data.y_cell.values[0]
        obj_data.dist_2 = np.zeros((1), float)
        obj_data.rms_dist = 0.0
        obj_data.snr_mean = obj_data.data.snr.values[0]
        obj_data.snr_rms = 0.0
        return

    sum_snr = np.sum(obj_data.data.snr)
    obj_data.x_cent = np.sum(obj_data.data.x_cell * obj_data.data.snr) / sum_snr
    obj_data.y_cent = np.sum(obj_data.data.y_cell * obj_data.data.snr) / sum_snr
    obj_data.dist_2 = np.array(
        (obj_data.x_cent - obj_data.data.x_cell) ** 2
        + (obj_data.y_cent - obj_data.data.y_cell) ** 2
    )
    obj_data.rms_dist = np.sqrt(np.mean(obj_data.dist_2))
    obj_data.snr_mean = np.mean(obj_data.data.snr.values)
    obj_data.snr_rms = np.std(obj_data.data.snr.values)

    sub_mask = obj_data.dist_2 < pixel_r2_cut
    if (obj_data.n_unique == obj_data.n_src) and sub_mask.all():
        return

    if recurse >= cell_data.matcher.max_sub_division:
        return

    mstSplitObject(obj_data, cell_data, pixel_r2_cut, mst_cut_pix / 2, recurse=recurse + 1)


def mstSplitObject(
    obj_data: ObjectData,
    cell_data: CellData,
    pixel_r2_cut: float,
    mst_cut_pix: float,
    recurse: int = 0,
) -> None:
    """Split a confused object by re-grouping its sources at a finer MST cut.

    Replaces :py:func:`~hpmcm.match_utils.heirarchicalSplitObject` for
    MST matchers.  Runs :py:func:`mst_group_sources` with ``mst_cut_pix``
    on the sources in the object.  If the sources fall into more than one
    group the object is divided into sub-objects, each of which is then
    refined by :py:func:`mstProcessObject` at the same ``mst_cut_pix``.
    If only one group is found, the cut is halved and the split is retried.

    Parameters
    ----------
    obj_data:
        Object to split.
    cell_data:
        Parent cell.
    pixel_r2_cut:
        Maximum squared distance from centroid (cell pixels).
    mst_cut_pix:
        MST grouping radius in cell pixels for this split attempt.
        Halved on each retry.
    recurse:
        Current recursion depth.
    """
    if recurse > cell_data.matcher.max_sub_division:
        return
    obj_data.recurse = recurse
    obj_data.extract()
    assert obj_data.data is not None

    x = obj_data.data.x_cell.values
    y = obj_data.data.y_cell.values
    labels = mst_group_sources(x, y, mst_cut_pix)
    n_groups = len(np.unique(labels))

    if n_groups == 1:
        if recurse >= cell_data.matcher.max_sub_division:
            return
        mstSplitObject(obj_data, cell_data, pixel_r2_cut, mst_cut_pix / 2, recurse=recurse + 1)
        return

    biggest_label = int(np.argmax(np.bincount(labels)))
    biggest_mask = np.zeros(obj_data.mask.shape, dtype=bool)

    for label in np.unique(labels):
        sub_mask_local = labels == label

        # Map from indices-within-object back to indices-within-cluster
        new_mask = np.zeros(obj_data.mask.shape, dtype=bool)
        count = 0
        for i, val in enumerate(obj_data.mask):
            if val:
                new_mask[i] = sub_mask_local[count]
                count += 1
        assert sub_mask_local.sum() == new_mask.sum()

        if label == biggest_label:
            biggest_mask = new_mask
            continue

        new_object = obj_data.parent_cluster.addObject(cell_data, new_mask)
        mstProcessObject(new_object, cell_data, pixel_r2_cut, mst_cut_pix, recurse=recurse)

    obj_data.mask = biggest_mask
    obj_data.extract()
    mstProcessObject(obj_data, cell_data, pixel_r2_cut, mst_cut_pix, recurse=recurse)
