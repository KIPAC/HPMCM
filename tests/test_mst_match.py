import glob
import os

import numpy as np
import pandas

import hpmcm

DATADIR = "examples/test_data"
SHEAR_ST = "0p01"
SHEAR = 0.01
CATALOG_TYPE = "wmom"
TRACT = 10463

PIXEL_SIZE = 0.5 / 3600.0
PIXEL_R2CUT = 4.0

REF_DIR = (37.9, 7.0)
REGION_SIZE = (0.375, 0.375)
OUR_GEOM = hpmcm.shear_utils.ShearCellGeometry(ref_dir=REF_DIR)


def testMSTWcsMatchBasic(setup_data: int) -> None:
    """MSTWcsMatch on a single cell gives reasonable match efficiency."""
    assert setup_data == 0

    source_tablesfiles = sorted(
        glob.glob(os.path.join(DATADIR, f"shear_*_{SHEAR_ST}_cleaned_{TRACT}_ns.pq"))
    )
    source_tablesfiles.append(os.path.join(DATADIR, f"object_{TRACT}.pq"))
    source_tablesfiles.reverse()
    source_tablesfiles = [source_tablesfiles[0], source_tablesfiles[1]]
    catalog_ids: list[int] = list(np.arange(len(source_tablesfiles)))

    matcher = hpmcm.MSTWcsMatch.create(
        REF_DIR, REGION_SIZE, PIXEL_SIZE, mst_cut=1.0, pixel_r2_cut=PIXEL_R2CUT
    )
    matcher.reduceData(source_tablesfiles, catalog_ids)

    assert matcher.n_cell[0] == 4
    assert matcher.n_cell[1] == 4

    matcher.analysisLoop(range(1, 2), range(1, 2))

    stats = matcher.extractStats()
    assert stats is not None

    obj_lists = hpmcm.classify.classifyObjects(matcher, SNRCut=10.0)
    n_good = len(obj_lists["ideal"])
    bad_list = [
        "edge_mixed", "edge_missing", "edge_extra", "orphan",
        "missing", "two_missing", "many_missing", "extra", "caught",
    ]
    n_bad = int(np.sum([len(obj_lists[x]) for x in bad_list]))
    assert n_good > 0
    effic = n_good / (n_good + n_bad)
    assert effic > 0.75


def testMSTWcsMatchSynthetic() -> None:
    """MST groups close sources together and keeps distant sources separate."""
    # 0.5 arcsec/pixel grid; mst_cut=1.5 arcsec -> 3.0 pixels
    region_size = (0.1, 0.1)
    mst_cut = 1.5

    matcher = hpmcm.MSTWcsMatch.create(
        (0.0, 0.0), region_size, PIXEL_SIZE, mst_cut, pixel_r2_cut=PIXEL_R2CUT
    )

    n_pix = (np.array(region_size) / PIXEL_SIZE).astype(int)
    cx, cy = float(n_pix[0]) / 2.0, float(n_pix[1]) / 2.0

    # Catalogs 0 and 1: sources 0.71 pixels apart — should merge into one cluster
    # Catalog 2: source 160+ pixels away — should be its own cluster
    df0 = pandas.DataFrame(
        {"id": [0], "ra": [0.0], "dec": [0.0], "x_pix": [cx], "y_pix": [cy], "snr": [10.0]}
    )
    df1 = pandas.DataFrame(
        {
            "id": [1],
            "ra": [0.0],
            "dec": [0.0],
            "x_pix": [cx + 0.5],
            "y_pix": [cy + 0.5],
            "snr": [10.0],
        }
    )
    df2 = pandas.DataFrame(
        {
            "id": [2],
            "ra": [0.0],
            "dec": [0.0],
            "x_pix": [cx - 160.0],
            "y_pix": [cy - 160.0],
            "snr": [10.0],
        }
    )

    matcher.setReducedData([df0, df1, df2], [0, 1, 2])
    matcher.analysisLoop()

    stats = matcher.extractStats(central_only=False)
    assert stats is not None

    total_objects = sum(len(c.object_dict) for c in matcher.cell_dict.values())
    assert total_objects == 2

    total_clusters = sum(len(c.cluster_dict) for c in matcher.cell_dict.values())
    assert total_clusters == 2

    # The object containing catalogs 0 and 1 should have n_src=2
    objects = [
        obj
        for cell in matcher.cell_dict.values()
        for obj in cell.object_dict.values()
    ]
    n_src_values = sorted(obj.n_src for obj in objects)
    assert n_src_values == [1, 2]


def testMSTWcsMatchGeometryDict() -> None:
    """mst_cut is persisted in _geometryDict."""
    matcher = hpmcm.MSTWcsMatch.create(
        (0.0, 0.0), (0.1, 0.1), PIXEL_SIZE, mst_cut=1.25
    )
    geo = matcher._geometryDict()
    assert "mst_cut" in geo
    assert geo["mst_cut"] == 1.25
    assert geo["class"] == "MSTWcsMatch"


def testMSTShearMatchBasic(setup_data: int) -> None:
    """MSTShearMatch on a subset of cells gives reasonable match efficiency."""
    assert setup_data == 0

    source_tablesfiles = sorted(
        glob.glob(
            os.path.join(DATADIR, f"shear_{CATALOG_TYPE}_{SHEAR_ST}_uncleaned_{TRACT}_*.pq")
        )
    )
    source_tablesfiles.reverse()
    catalog_ids: list[int] = list(np.arange(len(source_tablesfiles)))

    matcher = hpmcm.MSTShearMatch.createShearMatch(
        geometry=OUR_GEOM,
        mst_cut=0.5,
        pixel_r2_cut=PIXEL_R2CUT,
        deshear=-1 * SHEAR,
    )
    matcher.reduceData(source_tablesfiles, catalog_ids)

    assert matcher.n_cell[0] == 201
    assert matcher.n_cell[1] == 201

    matcher.analysisLoop(range(50, 55), range(170, 175))

    stats = matcher.extractStats()
    shear_stats = matcher.extractShearStats()

    assert stats is not None
    assert shear_stats is not None
    assert shear_stats["object_shear"] is not None

    obj_lists = hpmcm.classify.classifyObjects(matcher, SNRCut=10.0)
    n_good = len(obj_lists["ideal"])
    bad_list = [
        "edge_mixed", "edge_missing", "edge_extra", "orphan",
        "missing", "two_missing", "many_missing", "extra", "caught",
    ]
    n_bad = int(np.sum([len(obj_lists[x]) for x in bad_list]))
    assert n_good > 0
    effic = n_good / (n_good + n_bad)
    assert effic > 0.80
