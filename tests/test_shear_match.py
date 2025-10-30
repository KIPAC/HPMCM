import glob
import os

import numpy as np

import hpmcm

DATADIR = "examples/test_data"  # Input data directory
SHEAR_ST = "0p01"  # Applied shear as a string
SHEAR = 0.01  # Decimal version of applied shear
CATALOG_TYPE = "wmom"  # which object characterization to use
TRACT = 10463  # which tract to study

BASEFILE = os.path.join(DATADIR, f"shear_{CATALOG_TYPE}_{SHEAR_ST}_match_{TRACT}")

PIXEL_R2CUT = 4.0  # Cut at distance**2 = 4 pixels
PIXEL_MATCH_SCALE = 1  # Use pixel scale to do matching
SNR_CUT = 10.0  # Cut on signal-to-noise


def testShearMatch(setup_data: int) -> None:
    """Run ShearMatch on 400 cells"""

    assert setup_data == 0

    # get the data
    source_tablesfiles = sorted(
        glob.glob(
            os.path.join(DATADIR, f"shear_{CATALOG_TYPE}_{SHEAR_ST}_uncleaned_{TRACT}_*.pq")
        )
    )
    source_tablesfiles.reverse()
    catalog_ids: list[int] = list(np.arange(len(source_tablesfiles)))

    # Create matcher
    matcher = hpmcm.ShearMatch.createShearMatch(
        pixel_R2_cut=PIXEL_R2CUT, pixel_match_scale=PIXEL_MATCH_SCALE, deshear=-1 * SHEAR
    )

    # Reduce the input data
    matcher.reduceData(source_tablesfiles, catalog_ids)

    # Make sure it got the right number of cells
    assert matcher.n_cell[0] == 200
    assert matcher.n_cell[1] == 200

    # Define the range of cells to run over
    x_range = range(50, 70)
    y_range = range(170, 190)

    # Run the analysis
    matcher.analysisLoop(x_range, y_range)

    # Extract some data
    stats = matcher.extractStats()
    shear_stats = matcher.extractShearStats()

    assert stats is not None
    assert shear_stats is not None
    obj_shear = shear_stats[1]
    assert obj_shear is not None

    # Test the classification codes
    hpmcm.classify.printSummaryStats(matcher)

    obj_lists = hpmcm.classify.classifyObjects(matcher, SNRCut=10.0)
    hpmcm.classify.printObjectTypes(obj_lists)

    cluster_lists = hpmcm.classify.classifyClusters(matcher, SNRCut=10.0)
    hpmcm.classify.printClusterTypes(cluster_lists)

    # Make sure the match efficiency is high
    n_good = len(obj_lists["ideal"])
    bad_list = [
        "edge_mixed",
        "edge_missing",
        "edge_extra",
        "orphan",
        "missing",
        "two_missing",
        "many_missing",
        "extra",
        "caught",
    ]
    n_bad = np.sum([len(obj_lists[x]) for x in bad_list])
    effic = n_good / (n_good + n_bad)
    assert effic > 0.95

    # Get a particular cell and rerun the analysis to test visualization
    # and classification functions
    cell_idx = matcher.getCellIdx(50, 170)
    cell = matcher.cell_dict[cell_idx]
    od = cell.analyze(None, 4)
    assert od is not None

    cluster = matcher.getCluster(cluster_lists["ideal"][0])
    obj = matcher.getObject(obj_lists["ideal"][0])

    assert len(cluster.x_cluster)
    assert len(cluster.y_cluster)
    assert len(cluster.x_pix)
    assert len(cluster.y_pix)

    assert len(obj.x_cluster)
    assert len(obj.y_cluster)
    assert len(obj.x_pix)
    assert len(obj.y_pix)

    hpmcm.viz_utils.showShearObjs(matcher, cluster_lists["ideal"][0])
    hpmcm.viz_utils.showShearObj(matcher, obj_lists["ideal"][0])
    hpmcm.viz_utils.showCluster(od["image"], cluster, cell)
    hpmcm.viz_utils.showObjects(od["image"], cluster, cell)
    hpmcm.viz_utils.showObjectsV2(od["image"], cluster, cell)


def testShearReport(setup_data: int) -> None:
    """Run the shearReport function"""
    assert setup_data == 0

    output_file = os.path.join(DATADIR, "dummy")
    hpmcm.shear_utils.shearReport(
        BASEFILE,
        output_file,
        SHEAR,
        cat_type=CATALOG_TYPE,
        tract=TRACT,
        snr_cut=SNR_CUT,
    )
