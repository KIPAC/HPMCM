from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


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
