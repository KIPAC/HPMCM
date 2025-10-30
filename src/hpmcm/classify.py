from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .cluster import ClusterData
    from .match import Match


def clusterStats(cluster_dict: OrderedDict[int, ClusterData]) -> np.ndarray:
    """Helper function to get stats about the clusters

    Parameters
    ----------
    cluster_dict:
        Dict from clusterId to ClusterData object

    Returns
    -------
    Cluster Statistics (nClusters, n_orphan, n_mixed, n_confused)

    Notes
    -----
    Return array contains

    nClusters: Total number of clusters

    n_orphan: Number of single source clusters (i.e., single detections)

    n_mixed: Number of clusters with more than one source from each input catalog

    n_confused: Number of souces with more than four cases of duplication
    """
    n_orphan = 0
    n_mixed = 0
    n_confused = 0
    for val in cluster_dict.values():
        if val.n_src == 1:
            n_orphan += 1
        if val.n_src != val.n_unique:
            n_mixed += 1
            if val.n_src > val.n_unique + 3:  # pragma: no cover
                n_confused += 1
    return np.array([len(cluster_dict), n_orphan, n_mixed, n_confused])


def printSummaryStats(matcher: Match) -> np.ndarray:
    """Helper function to print info about clusters"""
    stats = np.zeros((4), int)
    for key, cell_data in matcher.cell_dict.items():
        cell_stats = clusterStats(cell_data.cluster_dict)
        print(
            f"{key:5}: "
            f"{cell_stats[0]:5} "
            f"{cell_stats[1]:5} "
            f"{cell_stats[2]:5} "
            f"{cell_stats[3]:5}"
        )
        stats += cell_stats
    return stats


def classifyClusters(matcher: Match, **kwargs: Any) -> dict[str, list]:
    """Sort clusters by their properties

    This will return a dict of lists of clusters of various types
    """
    nsrcs = []

    cut1 = []
    cut2 = []

    used = []
    ideal_faint = []
    ideal = []

    faint = []
    edge_mixed = []
    mixed = []
    edge_missing = []
    edge_extra = []

    missing = []
    two_missing = []
    many_missing = []
    extra = []
    caught = []

    cell_edge = kwargs.get("cell_edge", 75)
    edge_cut = kwargs.get("edge_cut", 2)
    snr_cut = kwargs.get("snr_cut", 7.5)

    n_cat = len(matcher.red_data)

    for i_c, cell_data in matcher.cell_dict.items():
        cd = cell_data.cluster_dict

        for key, c in cd.items():
            k = (i_c, key)

            assert c.data is not None

            nsrcs.append(c.n_src)

            if (np.fabs(c.data.x_cell_coadd) > cell_edge).all() or (
                np.fabs(c.data.y_cell_coadd) > cell_edge
            ).all():
                cut1.append(k)
                continue
            if (
                np.fabs(c.data.x_cell_coadd.mean()) > cell_edge
                or np.fabs(c.data.y_cell_coadd.mean()) > cell_edge
            ):
                cut2.append(k)
                continue

            used.append(k)

            edge_case = False
            is_faint = False
            if (np.fabs(c.data.x_cell_coadd) > cell_edge - edge_cut).any() or (
                np.fabs(c.data.y_cell_coadd) > cell_edge - edge_cut
            ).any():
                edge_case = True
            if (c.data.snr < snr_cut).any():
                is_faint = True

            if c.n_src == c.n_unique and c.n_src == n_cat and is_faint:
                ideal_faint.append(k)
            elif c.n_src == c.n_unique and c.n_src == n_cat:
                ideal.append(k)
            elif c.n_src < n_cat and is_faint:
                faint.append(k)
            elif (
                c.n_src == n_cat and c.n_unique != n_cat and edge_case
            ):  # pragma: no cover
                edge_mixed.append(k)
            elif c.n_src == n_cat and c.n_unique != n_cat:  # pragma: no cover
                mixed.append(k)
            elif c.n_src < n_cat and edge_case:  # pragma: no cover
                edge_missing.append(k)
            elif c.n_src > n_cat and edge_case:  # pragma: no cover
                edge_extra.append(k)
            elif c.n_src == n_cat - 1:  # pragma: no cover
                missing.append(k)
            elif c.n_src == n_cat - 2:  # pragma: no cover
                two_missing.append(k)
            elif c.n_src < n_cat - 2:
                many_missing.append(k)
            elif c.n_src > n_cat:
                extra.append(k)
            else:  # pragma: no cover
                caught.append(k)

    return dict(
        nsrcs=nsrcs,
        cut1=cut1,
        cut2=cut2,
        used=used,
        ideal_faint=ideal_faint,
        ideal=ideal,
        faint=faint,
        edge_mixed=edge_mixed,
        mixed=mixed,
        edge_missing=edge_missing,
        edge_extra=edge_extra,
        missing=missing,
        two_missing=two_missing,
        many_missing=many_missing,
        extra=extra,
        caught=caught,
    )


def matchObjectsAgainstRef(matcher: Match, **kwargs: Any) -> dict[str, list]:
    """Match objects against the reference catalog"""
    nsrcs = []
    used = []

    ideal_faint = []
    ideal = []

    faint = []
    not_in_ref = []
    not_in_ref_faint = []
    in_ref = []

    extra = []
    missing = []
    two_missing = []
    many_missing = []
    caught = []

    snr_cut = kwargs.get("snr_cut", 7.5)

    n_cat = len(matcher.red_data)

    for i_c, cell_data in matcher.cell_dict.items():
        od = cell_data.object_dict
        for key, c in od.items():
            k = (i_c, key)

            assert c.data is not None
            nsrcs.append(c.n_src)
            used.append(k)

            is_faint = False
            if (c.data.snr < snr_cut).any():
                is_faint = True

            if (c.catalog_id == 0).any():
                in_ref.append(k)
            else:
                if is_faint:
                    not_in_ref_faint.append(k)
                else:
                    not_in_ref.append(k)
                continue

            if c.n_src == c.n_unique and c.n_src == n_cat and is_faint:
                ideal_faint.append(k)
            elif c.n_src == c.n_unique and c.n_src == n_cat:
                ideal.append(k)
            elif is_faint:
                faint.append(k)
            elif c.n_src == n_cat - 1:
                missing.append(k)
            elif c.n_src == n_cat - 2:  # pragma: no cover
                two_missing.append(k)
            elif c.n_src < n_cat - 2:  # pragma: no cover
                many_missing.append(k)
            elif c.n_src > n_cat:
                extra.append(k)
            else:  # pragma: no cover
                caught.append(k)

    return dict(
        nsrcs=nsrcs,
        used=used,
        ideal_faint=ideal_faint,
        ideal=ideal,
        faint=faint,
        missing=missing,
        in_ref=in_ref,
        not_in_ref=not_in_ref,
        not_in_ref_faint=not_in_ref_faint,
        extra=extra,
        two_missing=two_missing,
        many_missing=many_missing,
        caught=caught,
    )


def printObjectMatchTypes(o_dict: dict) -> None:
    """Print numbers of different types of object matches"""
    print("All            ", len(o_dict["nsrcs"]))
    print("Used           ", len(o_dict["used"]))
    print("  New            ", len(o_dict["not_in_ref"]))
    print("  New (faint)    ", len(o_dict["not_in_ref_faint"]))
    print("In Ref         ", len(o_dict["in_ref"]))
    print("Faint          ", len(o_dict["faint"]))
    print("Good           ", len(o_dict["ideal"]))
    print("  Good (faint)   ", len(o_dict["ideal_faint"]))
    print("Missing        ", len(o_dict["missing"]))
    print("Two Missing    ", len(o_dict["two_missing"]))
    print("All Missing    ", len(o_dict["many_missing"]))
    print("Extra          ", len(o_dict["extra"]))
    print("Caught         ", len(o_dict["caught"]))


def classifyObjects(matcher: Match, **kwargs: Any) -> dict[str, list]:
    """Sort objects by their properties

    This will return a dict of lists of objects
    """

    nsrcs = []

    cut1 = []
    cut2 = []

    used = []
    ideal_faint = []
    ideal = []

    faint = []
    edge_mixed = []
    mixed = []
    edge_missing = []
    edge_extra = []

    orphan = []
    missing = []
    two_missing = []
    many_missing = []
    extra = []
    caught = []

    cell_edge = kwargs.get("cell_edge", 75)
    edge_cut = kwargs.get("edge_cut", 2)
    snr_cut = kwargs.get("snr_cut", 7.5)

    n_cat = len(matcher.red_data)

    for i_c, cell_data in matcher.cell_dict.items():
        od = cell_data.object_dict

        for key, c in od.items():
            k = (i_c, key)

            assert c.data is not None

            nsrcs.append(c.n_src)

            try:
                if (np.fabs(c.data.x_cell_coadd) > cell_edge).all() or (
                    np.fabs(c.data.y_cell_coadd) > cell_edge
                ).all():
                    cut1.append(k)
                    continue
            except Exception:
                pass
            try:
                if (
                    np.fabs(c.data.x_cell_coadd.mean()) > cell_edge
                    or np.fabs(c.data.y_cell_coadd.mean()) > cell_edge
                ):
                    cut2.append(k)
                    continue
            except Exception:
                pass

            used.append(k)

            edge_case = False
            is_faint = False
            try:
                if (np.fabs(c.data.x_cell_coadd) > cell_edge - edge_cut).any() or (
                    np.fabs(c.data.y_cell_coadd) > cell_edge - edge_cut
                ).any():
                    edge_case = True
            except Exception:
                edge_case = False
            if (c.data.snr < snr_cut).any():
                is_faint = True
            if c.n_src == c.n_unique and c.n_src == n_cat and is_faint:
                ideal_faint.append(k)
            elif c.n_src == c.n_unique and c.n_src == n_cat:
                ideal.append(k)
            elif c.n_src < n_cat and is_faint:
                faint.append(k)
            elif (
                c.n_src == n_cat and c.n_unique != n_cat and edge_case
            ):  # pragma: no cover
                edge_mixed.append(k)
            elif c.n_src == n_cat and c.n_unique != n_cat:
                mixed.append(k)
            elif c.n_src < n_cat and edge_case:  # pragma: no cover
                edge_missing.append(k)
            elif c.n_src < n_cat and c.parent_cluster.n_src >= n_cat:  # pragma: no cover
                orphan.append(k)
            elif c.n_src == n_cat - 1:
                missing.append(k)
            elif c.n_src == n_cat - 2:  # pragma: no cover
                two_missing.append(k)
            elif c.n_src < n_cat - 2:
                many_missing.append(k)
            elif c.n_src > n_cat and edge_case:  # pragma: no cover
                edge_extra.append(k)
            elif c.n_src > n_cat:
                extra.append(k)
            else:  # pragma: no cover
                caught.append(k)

    return dict(
        nsrcs=nsrcs,
        cut1=cut1,
        cut2=cut2,
        used=used,
        ideal_faint=ideal_faint,
        ideal=ideal,
        faint=faint,
        edge_mixed=edge_mixed,
        mixed=mixed,
        edge_missing=edge_missing,
        edge_extra=edge_extra,
        orphan=orphan,
        missing=missing,
        two_missing=two_missing,
        many_missing=many_missing,
        extra=extra,
        caught=caught,
    )


def printClusterTypes(cluster_types: dict[str, list]) -> None:
    """Print numbers of different types of clusters"""
    print(
        "All Clusters:                                  ",
        len(cluster_types["nsrcs"]),
    )
    print("cut 1                                          ", len(cluster_types["cut1"]))
    print("cut 2                                          ", len(cluster_types["cut2"]))
    print("Used:                                          ", len(cluster_types["used"]))
    print(
        "good (n source from n catalogs):               ",
        len(cluster_types["ideal"]),
    )
    print(
        "good faint                                     ",
        len(cluster_types["ideal_faint"]),
    )
    print(
        "faint (< n sources, snr < cut):                ",
        len(cluster_types["faint"]),
    )
    print(
        "mixed (n source from < n catalogs):            ",
        len(cluster_types["mixed"]),
    )
    print(
        "edge_mixed (mixed near edge of cell):          ",
        len(cluster_types["edge_mixed"]),
    )
    print(
        "edge_missing (< n sources, near edge of cell): ",
        len(cluster_types["edge_missing"]),
    )
    print(
        "edge_extra (> n sources, near edge of cell):   ",
        len(cluster_types["edge_extra"]),
    )
    print(
        "faint (< n sources, snr < cut):                ",
        len(cluster_types["faint"]),
    )
    print(
        "one missing (n-1 sources, not near edge):      ",
        len(cluster_types["missing"]),
    )
    print(
        "two missing (n-2 sources, not near edge):      ",
        len(cluster_types["two_missing"]),
    )
    print(
        "many missing (< n-2 sources, not near edge):   ",
        len(cluster_types["many_missing"]),
    )
    print(
        "extra (> n sources, not near edge):            ",
        len(cluster_types["extra"]),
    )


def printObjectTypes(object_types: dict[str, list]) -> None:
    """Print numbers of different types of objects"""
    print("All Objects:                                   ", len(object_types["nsrcs"]))
    print("cut 1                                          ", len(object_types["cut1"]))
    print("cut 2                                          ", len(object_types["cut2"]))
    print("Used:                                          ", len(object_types["used"]))
    print("good (n source from n catalogs):               ", len(object_types["ideal"]))
    print(
        "good faint                                     ",
        len(object_types["ideal_faint"]),
    )
    print("faint (< n sources, snr < cut):                ", len(object_types["faint"]))
    print("mixed (n source from < n catalogs):            ", len(object_types["mixed"]))
    print(
        "edge_mixed (mixed near edge of cell):          ",
        len(object_types["edge_mixed"]),
    )
    print(
        "edge_missing (< n sources, near edge of cell): ",
        len(object_types["edge_missing"]),
    )
    print(
        "edge_extra (> n sources, near edge of cell):   ",
        len(object_types["edge_extra"]),
    )
    print("faint (< n sources, snr < cut):                ", len(object_types["faint"]))
    print(
        "orphan (split off from larger cluster          ",
        len(object_types["orphan"]),
    )
    print(
        "one missing (n-1 sources, not near edge):      ",
        len(object_types["missing"]),
    )
    print(
        "two missing (n-2 sources, not near edge):      ",
        len(object_types["two_missing"]),
    )
    print(
        "many missing (< n-2 sources, not near edge):   ",
        len(object_types["many_missing"]),
    )
    print("extra (> n sources, not near edge):            ", len(object_types["extra"]))
