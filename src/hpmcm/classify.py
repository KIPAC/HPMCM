from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .cluster import ClusterData
    from .match import Match


def clusterStats(clusterDict: OrderedDict[int, ClusterData]) -> np.ndarray:
    """Helper function to get stats about the clusters

    Parameters
    ----------
    clusterDict:
        Dict from clusterId to ClusterData object

    Returns
    -------
    Cluster Statistics (nClusters, nOrphan, nMixed, nConfused)

    Notes
    -----
    Return array contains

    nClusters: Total number of clusters

    nOrphan: Number of single source clusters (i.e., single detections)

    nMixed: Number of clusters with more than one source from each input catalog

    nConfused: Number of souces with more than four cases of duplication
    """
    nOrphan = 0
    nMixed = 0
    nConfused = 0
    for val in clusterDict.values():
        if val.nSrc == 1:
            nOrphan += 1
        if val.nSrc != val.nUnique:
            nMixed += 1
            if val.nSrc > val.nUnique + 3:  # pragma: no cover
                nConfused += 1
    return np.array([len(clusterDict), nOrphan, nMixed, nConfused])


def printSummaryStats(matcher: Match) -> np.ndarray:
    """Helper function to print info about clusters"""
    stats = np.zeros((4), int)
    for key, cellData in matcher.cellDict.items():
        cellStats = clusterStats(cellData.clusterDict)
        print(
            f"{key:5}: "
            f"{cellStats[0]:5} "
            f"{cellStats[1]:5} "
            f"{cellStats[2]:5} "
            f"{cellStats[3]:5}"
        )
        stats += cellStats
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

    cell_edge = kwargs.get("cellEdge", 75)
    edge_cut = kwargs.get("edgeCut", 2)
    snr_cut = kwargs.get("SNRCut", 7.5)

    n_cat = len(matcher.redData)

    for iC, cellData in matcher.cellDict.items():
        cd = cellData.clusterDict

        for key, c in cd.items():
            k = (iC, key)

            assert c.data is not None

            nsrcs.append(c.nSrc)

            if (np.fabs(c.data.xCellCoadd) > cell_edge).all() or (
                np.fabs(c.data.yCellCoadd) > cell_edge
            ).all():
                cut1.append(k)
                continue
            if (
                np.fabs(c.data.xCellCoadd.mean()) > cell_edge
                or np.fabs(c.data.yCellCoadd.mean()) > cell_edge
            ):
                cut2.append(k)
                continue

            used.append(k)

            edge_case = False
            is_faint = False
            if (np.fabs(c.data.xCellCoadd) > cell_edge - edge_cut).any() or (
                np.fabs(c.data.yCellCoadd) > cell_edge - edge_cut
            ).any():
                edge_case = True
            if (c.data.SNR < snr_cut).any():
                is_faint = True

            if c.nSrc == c.nUnique and c.nSrc == n_cat and is_faint:
                ideal_faint.append(k)
            elif c.nSrc == c.nUnique and c.nSrc == n_cat:
                ideal.append(k)
            elif c.nSrc < n_cat and is_faint:
                faint.append(k)
            elif c.nSrc == n_cat and c.nUnique != n_cat and edge_case:  # pragma: no cover
                edge_mixed.append(k)
            elif c.nSrc == n_cat and c.nUnique != n_cat:  # pragma: no cover
                mixed.append(k)
            elif c.nSrc < n_cat and edge_case:  # pragma: no cover
                edge_missing.append(k)
            elif c.nSrc > n_cat and edge_case:  # pragma: no cover
                edge_extra.append(k)
            elif c.nSrc == n_cat - 1:  # pragma: no cover
                missing.append(k)
            elif c.nSrc == n_cat - 2:  # pragma: no cover
                two_missing.append(k)
            elif c.nSrc < n_cat - 2:
                many_missing.append(k)
            elif c.nSrc > n_cat:
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

    snr_cut = kwargs.get("SNRCut", 7.5)

    n_cat = len(matcher.redData)

    for iC, cellData in matcher.cellDict.items():
        od = cellData.objectDict
        for key, c in od.items():
            k = (iC, key)

            assert c.data is not None
            nsrcs.append(c.nSrc)
            used.append(k)

            is_faint = False
            if (c.data.SNR < snr_cut).any():
                is_faint = True

            if (c.catIndices == 0).any():
                in_ref.append(k)
            else:
                if is_faint:
                    not_in_ref_faint.append(k)
                else:
                    not_in_ref.append(k)
                continue

            if c.nSrc == c.nUnique and c.nSrc == n_cat and is_faint:
                ideal_faint.append(k)
            elif c.nSrc == c.nUnique and c.nSrc == n_cat:
                ideal.append(k)
            elif is_faint:
                faint.append(k)
            elif c.nSrc == n_cat - 1:
                missing.append(k)
            elif c.nSrc == n_cat - 2:  # pragma: no cover
                two_missing.append(k)
            elif c.nSrc < n_cat - 2:  # pragma: no cover
                many_missing.append(k)
            elif c.nSrc > n_cat:
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


def printObjectMatchTypes(oDict: dict) -> None:
    """Print numbers of different types of object matches"""
    print("All            ", len(oDict["nsrcs"]))
    print("Used           ", len(oDict["used"]))
    print("  New            ", len(oDict["not_in_ref"]))
    print("  New (faint)    ", len(oDict["not_in_ref_faint"]))
    print("In Ref         ", len(oDict["in_ref"]))
    print("Faint          ", len(oDict["faint"]))
    print("Good           ", len(oDict["ideal"]))
    print("  Good (faint)   ", len(oDict["ideal_faint"]))
    print("Missing        ", len(oDict["missing"]))
    print("Two Missing    ", len(oDict["two_missing"]))
    print("All Missing    ", len(oDict["many_missing"]))
    print("Extra          ", len(oDict["extra"]))
    print("Caught         ", len(oDict["caught"]))


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

    cell_edge = kwargs.get("cellEdge", 75)
    edge_cut = kwargs.get("edgeCut", 2)
    snr_cut = kwargs.get("SNRCut", 7.5)

    n_cat = len(matcher.redData)

    for iC, cellData in matcher.cellDict.items():
        od = cellData.objectDict

        for key, c in od.items():
            k = (iC, key)

            assert c.data is not None

            nsrcs.append(c.nSrc)

            try:
                if (np.fabs(c.data.xCellCoadd) > cell_edge).all() or (
                    np.fabs(c.data.yCellCoadd) > cell_edge
                ).all():
                    cut1.append(k)
                    continue
            except Exception:
                pass
            try:
                if (
                    np.fabs(c.data.xCellCoadd.mean()) > cell_edge
                    or np.fabs(c.data.yCellCoadd.mean()) > cell_edge
                ):
                    cut2.append(k)
                    continue
            except Exception:
                pass

            used.append(k)

            edge_case = False
            is_faint = False
            try:
                if (np.fabs(c.data.xCellCoadd) > cell_edge - edge_cut).any() or (
                    np.fabs(c.data.yCellCoadd) > cell_edge - edge_cut
                ).any():
                    edge_case = True
            except Exception:
                edge_case = False
            if (c.data.SNR < snr_cut).any():
                is_faint = True
            if c.nSrc == c.nUnique and c.nSrc == n_cat and is_faint:
                ideal_faint.append(k)
            elif c.nSrc == c.nUnique and c.nSrc == n_cat:
                ideal.append(k)
            elif c.nSrc < n_cat and is_faint:
                faint.append(k)
            elif c.nSrc == n_cat and c.nUnique != n_cat and edge_case:  # pragma: no cover
                edge_mixed.append(k)
            elif c.nSrc == n_cat and c.nUnique != n_cat:
                mixed.append(k)
            elif c.nSrc < n_cat and edge_case:  # pragma: no cover
                edge_missing.append(k)
            elif c.nSrc < n_cat and c.parentCluster.nSrc >= n_cat:  # pragma: no cover
                orphan.append(k)
            elif c.nSrc == n_cat - 1:
                missing.append(k)
            elif c.nSrc == n_cat - 2:  # pragma: no cover
                two_missing.append(k)
            elif c.nSrc < n_cat - 2:
                many_missing.append(k)
            elif c.nSrc > n_cat and edge_case:  # pragma: no cover
                edge_extra.append(k)
            elif c.nSrc > n_cat:
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


def printClusterTypes(clusterTypes: dict[str, list]) -> None:
    """Print numbers of different types of clusters"""
    print(
        "All Clusters:                                  ",
        len(clusterTypes["nsrcs"]),
    )
    print("cut 1                                          ", len(clusterTypes["cut1"]))
    print("cut 2                                          ", len(clusterTypes["cut2"]))
    print("Used:                                          ", len(clusterTypes["used"]))
    print(
        "good (n source from n catalogs):               ",
        len(clusterTypes["ideal"]),
    )
    print(
        "good faint                                     ",
        len(clusterTypes["ideal_faint"]),
    )
    print(
        "faint (< n sources, SNR < cut):                ",
        len(clusterTypes["faint"]),
    )
    print(
        "mixed (n source from < n catalogs):            ",
        len(clusterTypes["mixed"]),
    )
    print(
        "edge_mixed (mixed near edge of cell):          ",
        len(clusterTypes["edge_mixed"]),
    )
    print(
        "edge_missing (< n sources, near edge of cell): ",
        len(clusterTypes["edge_missing"]),
    )
    print(
        "edge_extra (> n sources, near edge of cell):   ",
        len(clusterTypes["edge_extra"]),
    )
    print(
        "faint (< n sources, SNR < cut):                ",
        len(clusterTypes["faint"]),
    )
    print(
        "one missing (n-1 sources, not near edge):      ",
        len(clusterTypes["missing"]),
    )
    print(
        "two missing (n-2 sources, not near edge):      ",
        len(clusterTypes["two_missing"]),
    )
    print(
        "many missing (< n-2 sources, not near edge):   ",
        len(clusterTypes["many_missing"]),
    )
    print(
        "extra (> n sources, not near edge):            ",
        len(clusterTypes["extra"]),
    )


def printObjectTypes(objectTypes: dict[str, list]) -> None:
    """Print numbers of different types of objects"""
    print("All Objects:                                   ", len(objectTypes["nsrcs"]))
    print("cut 1                                          ", len(objectTypes["cut1"]))
    print("cut 2                                          ", len(objectTypes["cut2"]))
    print("Used:                                          ", len(objectTypes["used"]))
    print("good (n source from n catalogs):               ", len(objectTypes["ideal"]))
    print(
        "good faint                                     ",
        len(objectTypes["ideal_faint"]),
    )
    print("faint (< n sources, SNR < cut):                ", len(objectTypes["faint"]))
    print("mixed (n source from < n catalogs):            ", len(objectTypes["mixed"]))
    print(
        "edge_mixed (mixed near edge of cell):          ",
        len(objectTypes["edge_mixed"]),
    )
    print(
        "edge_missing (< n sources, near edge of cell): ",
        len(objectTypes["edge_missing"]),
    )
    print(
        "edge_extra (> n sources, near edge of cell):   ",
        len(objectTypes["edge_extra"]),
    )
    print("faint (< n sources, SNR < cut):                ", len(objectTypes["faint"]))
    print(
        "orphan (split off from larger cluster          ",
        len(objectTypes["orphan"]),
    )
    print(
        "one missing (n-1 sources, not near edge):      ",
        len(objectTypes["missing"]),
    )
    print(
        "two missing (n-2 sources, not near edge):      ",
        len(objectTypes["two_missing"]),
    )
    print(
        "many missing (< n-2 sources, not near edge):   ",
        len(objectTypes["many_missing"]),
    )
    print("extra (> n sources, not near edge):            ", len(objectTypes["extra"]))
