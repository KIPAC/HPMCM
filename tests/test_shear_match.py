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
    catalog_ids = np.arange(len(source_tablesfiles))

    
    # Create matcher
    matcher = hpmcm.ShearMatch.createShearMatch(
        pixelR2Cut=PIXEL_R2CUT, pixelMatchScale=PIXEL_MATCH_SCALE, deshear=-1 * SHEAR
    )

    # Reduce the input data
    matcher.reduceData(source_tablesfiles, catalog_ids)

    # Make sure it got the right number of cells
    assert matcher.nCell[0] == 200
    assert matcher.nCell[1] == 200

    # Define the range of cells to run over
    xRange = range(50, 70)
    yRange = range(170, 190)

    # Run the analysis
    matcher.analysisLoop(xRange, yRange)

    # Extract some data
    stats = matcher.extractStats()
    shear_stats = matcher.extractShearStats()

    assert stats is not None
    assert shear_stats is not None
    obj_shear = shear_stats[1]
    assert obj_shear is not None

    # Test the classification codes
    hpmcm.classify.printSummaryStats(matcher)

    objLists = hpmcm.classify.classifyObjects(matcher, SNRCut=10.0)
    hpmcm.classify.printObjectTypes(objLists)

    clusterLists = hpmcm.classify.classifyClusters(matcher, SNRCut=10.0)
    hpmcm.classify.printClusterTypes(clusterLists)

    # Make sure the match efficiency is high
    n_good = len(objLists["ideal"])
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
    n_bad = np.sum([len(objLists[x]) for x in bad_list])
    effic = n_good / (n_good + n_bad)
    assert effic > 0.95

    # Get a particular cell and rerun the analysis to test visualization
    # and classification functions
    cellIdx = matcher.getCellIdx(50, 170)
    cell = matcher.cellDict[cellIdx]
    od = cell.analyze(None, 4)
    cluster = matcher.getCluster(clusterLists["ideal"][0])
    obj = matcher.getObject(objLists["ideal"][0])

    assert len(cluster.xCluster)
    assert len(cluster.yCluster)
    assert len(cluster.xPix)
    assert len(cluster.yPix)

    assert len(obj.xCluster)
    assert len(obj.yCluster)
    assert len(obj.xPix)
    assert len(obj.yPix)

    hpmcm.viz_utils.showShearObjs(matcher, clusterLists["ideal"][0])
    hpmcm.viz_utils.showShearObj(matcher, objLists["ideal"][0])
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
        catType=CATALOG_TYPE,
        tract=TRACT,
        snrCut=SNR_CUT,
    )
