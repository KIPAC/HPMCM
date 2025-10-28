import hpmcm
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def test_shear_match(setup_data: int) -> None:
    assert setup_data==0

    DATADIR = "examples/test_data" # Input data directory
    shear_st = "0p01"       # Applied shear as a string
    shear = 0.01            # Decimal version of applied shear
    shear_type = "wmom"     # which object characterization to use 
    tract = 10463           # which tract to study

    SOURCE_TABLEFILES = sorted(glob.glob(os.path.join(DATADIR, f"shear_{shear_type}_{shear_st}_uncleaned_{tract}_*.pq")))
    SOURCE_TABLEFILES.reverse()
    VISIT_IDS = np.arange(len(SOURCE_TABLEFILES))

    PIXEL_R2CUT = 4.         # Cut at distance**2 = 4 pixels
    PIXEL_MATCH_SCALE = 1    # Use pixel scale to do matching
    

    matcher = hpmcm.ShearMatch.createShearMatch(pixelR2Cut=PIXEL_R2CUT, pixelMatchScale=PIXEL_MATCH_SCALE, deshear=-1*shear)

    matcher.reduceData(SOURCE_TABLEFILES, VISIT_IDS)

    assert matcher.nCell[0] == 200
    assert matcher.nCell[1] == 200
        
    xRange = range(50, 70)
    yRange = range(170, 190)

    matcher.analysisLoop(xRange, yRange)

    cell = matcher.cellDict[matcher.getCellIdx(50, 170)]
    od = cell.analyze(None, 4)

    cluster = list(cell.clusterDict.values())[0]
    
    stats = matcher.extractStats()
    shear_stats = matcher.extractShearStats()
    obj_shear = shear_stats[1]


    objLists = hpmcm.classify.classifyObjects(matcher, SNRCut=10.)
    hpmcm.classify.printObjectTypes(objLists)

    clusterLists = hpmcm.classify.classifyClusters(matcher, SNRCut=10.)
    hpmcm.classify.printClusterTypes(clusterLists)


    n_good = len(objLists['ideal'])
    bad_list = ['edge_mixed', 'edge_missing', 'edge_extra', 'orphan', 'missing', 'two_missing', 'many_missing', 'extra', 'caught']
    n_bad = np.sum([len(objLists[x]) for x in bad_list])
    effic = n_good/(n_good+n_bad)
    effic_err = np.sqrt(effic*(1-effic)/(n_good+n_bad))

    assert effic > 0.95


def test_shear_report(setup_data: int) -> None:
    DATADIR = "examples/test_data" # Input data directory
    shear_st = "0p01"       # Applied shear as a string
    shear = 0.01            # Decimal version of applied shear
    shear_type = "wmom"     # which object characterization to use 
    snr_cut=10.

    tract = 10463
    catType = "wmom"
    basefile = os.path.join(DATADIR, f"shear_{shear_type}_{shear_st}_match_{tract}")
    output_file = os.path.join(DATADIR, "dummy")
    hpmcm.shear_utils.shearReport(
        basefile, output_file, shear, catType=catType, tract=tract, snrCut=snr_cut
    )

