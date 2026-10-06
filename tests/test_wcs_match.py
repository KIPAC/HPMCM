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
    catalog_ids: list[int] = list(np.arange(len(source_tablesfiles)))

    # Create matcher
    matcher = hpmcm.WcsMatch.create(
        REF_DIR, REGION_SIZE, pixel_size=PIXEL_SIZE, pixel_R2_cut=PIXEL_R2CUT
    )

    # Reduce the input data
    matcher.reduceData(source_tablesfiles, catalog_ids)

    # Make sure it got the right number of cells
    assert matcher.n_cell[0] == 4
    assert matcher.n_cell[1] == 4

    # Define the range of cells to run over
    x_range = range(1, 2)
    y_range = range(1, 2)

    # Run the analysis
    matcher.analysisLoop(x_range, y_range)

    # Extract some data
    stats = matcher.extractStats()
    assert stats is not None

    # Test the classification codes
    obj_lists = hpmcm.classify.classifyObjects(matcher, SNRCut=10.0)
    hpmcm.classify.printObjectTypes(obj_lists)

    # Test the classification codes
    odict = hpmcm.classify.matchObjectsAgainstRef(matcher, snrCut=10.0)
    hpmcm.classify.printObjectMatchTypes(odict)

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

    assert effic > 0.85

    # Get a particular cell and rerun the analysis to test visualization
    # and classification functions
    cell = matcher.cell_dict[matcher.getCellIdx(1, 1)]
    _od = cell.analyze(None, 4)
    _cluster = list(cell.cluster_dict.values())[0]


def testWCSMatchPixelMatchScale(setup_data: int) -> None:
    """Verify pixel_match_scale > 1 works the same as pms=1 for WcsMatch.

    Runs the same single-cell analysis with pixel_match_scale=2 and checks
    that:
    - Objects are detected and clustered without error
    - footprint.extent() is scaled correctly (slice bounds × pms)
    - Match efficiency is still reasonable
    """
    assert setup_data == 0

    source_tablesfiles = sorted(
        glob.glob(os.path.join(DATADIR, f"shear_*_{SHEAR_ST}_cleaned_{TRACT}_ns.pq"))
    )
    source_tablesfiles.append(os.path.join(DATADIR, f"object_{TRACT}.pq"))
    source_tablesfiles.reverse()
    source_tablesfiles = [source_tablesfiles[0], source_tablesfiles[1]]
    catalog_ids: list[int] = list(np.arange(len(source_tablesfiles)))

    matcher = hpmcm.WcsMatch.create(
        REF_DIR,
        REGION_SIZE,
        pixel_size=PIXEL_SIZE,
        pixel_R2_cut=PIXEL_R2CUT,
        pixel_match_scale=2,
    )
    assert matcher.pixel_match_scale == 2

    matcher.reduceData(source_tablesfiles, catalog_ids)
    matcher.analysisLoop(range(1, 2), range(1, 2))

    stats = matcher.extractStats()
    assert stats is not None

    cell = matcher.cell_dict[matcher.getCellIdx(1, 1)]
    assert cell.pixel_match_scale == 2

    # Verify the counts map is half-resolution (pixel_match_scale = 2)
    # so the footprint slice bounds are in counts-map coords, and
    # extent() should scale them back to cell-pixel coords.
    for cluster in cell.cluster_dict.values():
        assert cluster.pixel_match_scale == 2
        x0, x1, y0, y1 = cluster.footprint.extent(cluster.pixel_match_scale)
        # extent with pms=2 should differ from extent(1) by factor 2
        x0_raw, x1_raw, y0_raw, y1_raw = cluster.footprint.extent(1)
        assert x0 == x0_raw * 2
        assert x1 == x1_raw * 2
        assert y0 == y0_raw * 2
        assert y1 == y1_raw * 2

    obj_lists = hpmcm.classify.classifyObjects(matcher, SNRCut=10.0)
    n_good = len(obj_lists["ideal"])
    bad_list = [
        "edge_mixed", "edge_missing", "edge_extra", "orphan",
        "missing", "two_missing", "many_missing", "extra", "caught",
    ]
    n_bad = np.sum([len(obj_lists[x]) for x in bad_list])
    effic = n_good / (n_good + n_bad)
    assert effic > 0.75
