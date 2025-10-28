import glob
import os

import numpy as np
import hpmcm

DATADIR = "examples/test_data"  # Input data directory
SHEAR_ST = "0p01"  # Applied shear as a string
TRACT = 10463  # which tract to study

REF_DIR = (37.9, 7.0)  # RA, DEC in deg of center of match region
REGION_SIZE = (0.375, 0.375)  # Size of match region in degrees
PIXEL_SIZE = 0.5 / 3600.0  # Size of pixels used in matching
PIXEL_R2CUT = 4.0  # Cut at distance**2 = 4 pixels


def testWCSMatch(setup_data: int) -> None:
    """Run WcsMatch on a large part of a tract"""

    assert setup_data == 0

    # get the data
    
    source_tablesfiles = sorted(
        glob.glob(os.path.join(DATADIR, f"shear_*_{SHEAR_ST}_cleaned_{TRACT}_ns.pq"))
    )
    source_tablesfiles.append(os.path.join(DATADIR, f"object_{TRACT}.pq"))
    source_tablesfiles.reverse()
    source_tablesfiles = [source_tablesfiles[0], source_tablesfiles[1]]
    catalog_ids = np.arange(len(source_tablesfiles))

    # Create matcher
    matcher = hpmcm.WcsMatch.create(
        REF_DIR, REGION_SIZE, pixelSize=PIXEL_SIZE, pixelR2Cut=PIXEL_R2CUT
    )

    # Reduce the input data
    matcher.reduceData(source_tablesfiles, catalog_ids)

    # Make sure it got the right number of cells
    assert matcher.nCell[0] == 3
    assert matcher.nCell[1] == 3

    # Define the range of cells to run over
    xRange = range(1, 2)
    yRange = range(1, 2)

    # Run the analysis
    matcher.analysisLoop(xRange, yRange)

    # Extract some data
    stats = matcher.extractStats()
    assert stats is not None

    # Test the classification codes
    objLists = hpmcm.classify.classifyObjects(matcher, SNRCut=10.0)
    hpmcm.classify.printObjectTypes(objLists)

    # Test the classification codes
    odict = hpmcm.classify.matchObjectsAgainstRef(matcher, snrCut=10.0)
    hpmcm.classify.printObjectMatchTypes(odict)

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

    assert effic > 0.85

    # Get a particular cell and rerun the analysis to test visualization
    # and classification functions
    cell = matcher.cellDict[matcher.getCellIdx(1, 1)]
    _od = cell.analyze(None, 4)
    _cluster = list(cell.clusterDict.values())[0]
