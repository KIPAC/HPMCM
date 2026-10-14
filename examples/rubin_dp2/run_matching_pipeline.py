#!/usr/bin/env python
"""
HPMCM full matching pipeline for a single Rubin tract.

Combines the actions of StandarizeInputs.ipynb and RunMatching.ipynb:
  1. Standardize inputs  — split raw catalogs by shear type and reduce columns
  2. Rubin shear matching
  3. DESC shear matching
  4. WCS meta-matching across all object catalogs

Edit the CONFIGURATION block below for your tract, then run:

    python run_matching_pipeline.py
"""
from __future__ import annotations

import tables_io

import hpmcm
from hpmcm import shear_utils, utils

# ============================================================
# CONFIGURATION — edit this block before running
# ============================================================

TRACT: int = 8006

# Tract centre (RA, Dec in degrees); used to build the shear-match WCS.
REF_DIR: tuple[float, float] = (340.5, -9.6694214876)

# ── Raw input files ───────────────────────────────────────────────────────────
# Full path to the raw joined Rubin metadetect catalog (all shear types in one
# file, produced before split-by-type).
RUBIN_MD_RAW_FILE: str = f"{TRACT}/shear_object_{TRACT}.parq"

# Full path to the raw joined DESC metadetect catalog.
DESC_MD_RAW_FILE: str = f"{TRACT}/mdet-run-{TRACT}.parq"

# Full path to the Rubin object catalog.
OBJECT_RAW_FILE: str = f"{TRACT}/object_{TRACT}.parq"

# Full path to the anacal catalog.
ANACAL_RAW_FILE: str = f"{TRACT}/anacal_merged_{TRACT}.parq"

# ── Shear parameters ──────────────────────────────────────────────────────────
# Magnitude of the applied shear (|g|).  Used to set the matcher's deshear
# parameter (-RUBIN_SHEAR / -DESC_SHEAR) so matching is done on de-sheared
# coordinates.
RUBIN_SHEAR: float = 0.01
DESC_SHEAR:  float = 0.01

# Shear catalog types to include in each matching run.
# Rubin MD produces five catalogs; DESC MD produces three.
RUBIN_SHEAR_NAMES: list[str] = ["ns", "1m", "1p", "2m", "2p"]
DESC_SHEAR_NAMES:  list[str] = ["ns", "1m", "1p"]

# ── Shear matching parameters ─────────────────────────────────────────────────
SHEAR_PIXEL_MATCH_SCALE: int   = 2
SHEAR_PIXEL_R2_CUT:      float = 16.0

# ── WCS meta-matching parameters ──────────────────────────────────────────────
# Pixel scale in degrees per pixel (0.2 arcsec/pix = 0.2/3600 deg/pix).
WCS_PIXEL_SCALE:       float = 0.2 / 3600.0
# Side length of the matching region in pixels.
WCS_N_PIX_REGION:      int   = 30_000
WCS_CELL_SIZE:         int   = 150
WCS_CELL_BUFFER:       int   = 5
WCS_CELL_MAX_OBJECT:   int   = 10_000
WCS_PIXEL_MATCH_SCALE: int   = 4
WCS_PIXEL_R2_CUT:      float = 16.0

# ── Step control ──────────────────────────────────────────────────────────────
# Set any flag to False to skip that step (e.g. if the output files already
# exist from a previous run).
REDO_RUBIN_MD:      bool = True  # split + reduce raw Rubin MD catalog
REDO_DESC_MD:       bool = True  # split + reduce raw DESC MD catalog
REDO_OBJECT:        bool = True  # reduce Rubin object catalog
REDO_ANACAL:        bool = True  # reduce anacal catalog

REMATCH_RUBIN:      bool = True  # run Rubin shear matching and save
REMATCH_DESC:       bool = True  # run DESC shear matching and save
REMATCH_OBJECTS:    bool = True  # run WCS meta-matching and save

BUILD_RUBIN_JOINED: bool = True  # build joined Rubin object table
BUILD_DESC_JOINED:  bool = True  # build joined DESC object table

# Set True to run the analysis on a single cell only (quick smoke-test).
TEST_ONLY: bool = False

# ============================================================
# END CONFIGURATION — no edits needed below this line
# ============================================================

# ── Derived intermediate file paths ───────────────────────────────────────────
# These follow the naming conventions of splitRubinMDByTypeAndClean /
# splitDESCMDByTypeAndClean and the reduce* helpers.

def _rubin_split_file(st: str) -> str:
    """Per-type Rubin file written by splitRubinMDByTypeAndClean."""
    return RUBIN_MD_RAW_FILE.replace(".parq", f"_uncleaned_{TRACT}_{st}.parq")

def _desc_split_file(st: str) -> str:
    """Per-type DESC file written by splitDESCMDByTypeAndClean."""
    return DESC_MD_RAW_FILE.replace(".parq", f"_uncleaned_{TRACT}_{st}.parq")

def _rubin_reduced_file(st: str) -> str:
    """Column-reduced Rubin file written by reduceRubinMDTable."""
    return f"{TRACT}/shear_object_reduced_uncleaned_{TRACT}_{st}.parq"

def _desc_reduced_file(st: str) -> str:
    """Column-reduced DESC file written by reduceDESCMDTable."""
    return f"{TRACT}/mdet-run-reduced_uncleaned_{TRACT}_{st}.parq"

RUBIN_SHEAR_FILES: list[str] = [_rubin_reduced_file(st) for st in RUBIN_SHEAR_NAMES]
RUBIN_SHEAR_IDS:   list[int]  = list(range(len(RUBIN_SHEAR_FILES)))

DESC_SHEAR_FILES:  list[str] = [_desc_reduced_file(st) for st in DESC_SHEAR_NAMES]
DESC_SHEAR_IDS:    list[int]  = list(range(len(DESC_SHEAR_FILES)))

# Meta-matching input files: object catalog, anacal, and the two shear-match
# joined object tables (produced by this script in steps 2 and 3).
ALL_FILES: list[str] = [
    f"{TRACT}/object_reduced_{TRACT}.parq",
    f"{TRACT}/anacal_reduced_{TRACT}.parq",
    f"{TRACT}/desc_md_match_object_joined.pq",
    f"{TRACT}/rubin_md_match_object_joined.pq",
]
ALL_IDS: list[int] = list(range(len(ALL_FILES)))

OUR_GEOM = shear_utils.ShearCellGeometry(ref_dir=REF_DIR)
_WCS_REGION_DEG = WCS_N_PIX_REGION * WCS_PIXEL_SCALE

_TEST_RANGE = range(100, 101)  # single cell used when TEST_ONLY=True

# ── Step 1a: Standardize Rubin MD catalog ─────────────────────────────────────

if REDO_RUBIN_MD:
    print("Standardizing Rubin MD catalog...")
    shear_utils.splitRubinMDByTypeAndClean(
        RUBIN_MD_RAW_FILE, TRACT, RUBIN_SHEAR, deshear=False
    )
    for st in RUBIN_SHEAR_NAMES:
        utils.reduceRubinMDTable(_rubin_split_file(st), _rubin_reduced_file(st))
    print("  Done.")

# ── Step 1b: Standardize DESC MD catalog ──────────────────────────────────────

if REDO_DESC_MD:
    print("Standardizing DESC MD catalog...")
    shear_utils.splitDESCMDByTypeAndClean(
        DESC_MD_RAW_FILE, TRACT, DESC_SHEAR, deshear=False
    )
    for st in DESC_SHEAR_NAMES:
        utils.reduceDESCMDTable(_desc_split_file(st), _desc_reduced_file(st))
    print("  Done.")

# ── Step 1c: Reduce object catalog ────────────────────────────────────────────

if REDO_OBJECT:
    print("Standardizing object catalog...")
    utils.reduceObjectTable(OBJECT_RAW_FILE, f"{TRACT}/object_reduced_{TRACT}.parq")
    print("  Done.")

# ── Step 1d: Reduce anacal catalog ────────────────────────────────────────────

if REDO_ANACAL:
    print("Standardizing anacal catalog...")
    utils.reduceAnacalTable(ANACAL_RAW_FILE, f"{TRACT}/anacal_reduced_{TRACT}.parq")
    print("  Done.")

# ── Step 2: Rubin shear matching ──────────────────────────────────────────────

if REMATCH_RUBIN:
    print("Running Rubin shear matching...")
    rubin_shear_match = hpmcm.ShearMatch.createShearMatch(
        geometry=OUR_GEOM,
        pixel_match_scale=SHEAR_PIXEL_MATCH_SCALE,
        pixel_r2_cut=SHEAR_PIXEL_R2_CUT,
        deshear=-RUBIN_SHEAR,
        shear_names=RUBIN_SHEAR_NAMES,
    )
    rubin_shear_match.reduceData(RUBIN_SHEAR_FILES, RUBIN_SHEAR_IDS)
    if TEST_ONLY:
        rubin_shear_match.analysisLoop(_TEST_RANGE, _TEST_RANGE)
    else:
        rubin_shear_match.analysisLoop()
    rubin_stats = rubin_shear_match.extractStats()
    rubin_stats.update(**rubin_shear_match.extractShearStats())
    tables_io.write(rubin_stats, f"{TRACT}/rubin_md_match_")
    rubin_shear_match.save(f"{TRACT}/rubin_md_match")
    print("  Done.")

# ── Step 2 (cont.): Build joined Rubin object table ───────────────────────────

if BUILD_RUBIN_JOINED:
    print("Building joined Rubin object table...")
    rubin_obj_stats = tables_io.read(f"{TRACT}/rubin_md_match_object_stats.pq")
    rubin_obj_assoc = tables_io.read(f"{TRACT}/rubin_md_match_object_assoc.pq")
    joined_rubin = hpmcm.output_tables.buildJoinedObjectTable(
        rubin_obj_stats,
        rubin_obj_assoc,
        RUBIN_SHEAR_FILES,
        RUBIN_SHEAR_IDS,
        object_shear=f"{TRACT}/rubin_md_match_object_shear.pq",
    )
    stats_cols  = [f"{b}_gaussFlux"    for b in "griz"]
    stats_cols += [f"{b}_gaussFluxErr" for b in "griz"]
    drop_cols = []
    for col_prefix in ["tract", "ra", "dec", "gauss_snr", "patch_x", "patch_y",
                        "cell_idx_x", "cell_idx_y", "shear", "meta_step"]:
        drop_cols += [f"{col_prefix}_{i}" for i in RUBIN_SHEAR_IDS]
    rd_joined_rubin = hpmcm.reduceJoinedTable(
        joined_rubin,
        catalog_ids=RUBIN_SHEAR_IDS,
        drop_cols=drop_cols,
        stats_cols=stats_cols,
    )
    tables_io.write(rd_joined_rubin, f"{TRACT}/rubin_md_match_object_joined.pq")
    print("  Done.")

# ── Step 3: DESC shear matching ───────────────────────────────────────────────

if REMATCH_DESC:
    print("Running DESC shear matching...")
    desc_shear_match = hpmcm.ShearMatch.createShearMatch(
        geometry=OUR_GEOM,
        pixel_match_scale=SHEAR_PIXEL_MATCH_SCALE,
        pixel_r2_cut=SHEAR_PIXEL_R2_CUT,
        deshear=-DESC_SHEAR,
        shear_names=DESC_SHEAR_NAMES,
    )
    desc_shear_match.reduceData(DESC_SHEAR_FILES, DESC_SHEAR_IDS)
    if TEST_ONLY:
        desc_shear_match.analysisLoop(_TEST_RANGE, _TEST_RANGE)
    else:
        desc_shear_match.analysisLoop()
    desc_stats = desc_shear_match.extractStats()
    desc_stats.update(**desc_shear_match.extractShearStats())
    tables_io.write(desc_stats, f"{TRACT}/desc_md_match_")
    desc_shear_match.save(f"{TRACT}/desc_md_match")
    print("  Done.")

# ── Step 3 (cont.): Build joined DESC object table ────────────────────────────

if BUILD_DESC_JOINED:
    print("Building joined DESC object table...")
    desc_obj_stats = tables_io.read(f"{TRACT}/desc_md_match_object_stats.pq")
    desc_obj_assoc = tables_io.read(f"{TRACT}/desc_md_match_object_assoc.pq")
    joined_desc = hpmcm.output_tables.buildJoinedObjectTable(
        desc_obj_stats,
        desc_obj_assoc,
        DESC_SHEAR_FILES,
        DESC_SHEAR_IDS,
        object_shear=f"{TRACT}/desc_md_match_object_shear.pq",
    )
    stats_cols  = [f"flux_{b}"     for b in "riz"]
    stats_cols += [f"flux_err_{b}" for b in "riz"]
    drop_cols = []
    for col_prefix in ["ra", "dec", "gauss_snr", "patch_x", "patch_y",
                        "cell_idx_x", "cell_idx_y", "shear", "meta_step"]:
        drop_cols += [f"{col_prefix}_{i}" for i in DESC_SHEAR_IDS]
    rd_joined_desc = hpmcm.reduceJoinedTable(
        joined_desc,
        catalog_ids=DESC_SHEAR_IDS,
        drop_cols=drop_cols,
        stats_cols=stats_cols,
    )
    tables_io.write(rd_joined_desc, f"{TRACT}/desc_md_match_object_joined.pq")
    print("  Done.")

# ── Step 4: WCS meta-matching ─────────────────────────────────────────────────

if REMATCH_OBJECTS:
    print("Running WCS meta-matching...")
    wcs_match = hpmcm.WcsMatch.create(
        ref_dir=REF_DIR,
        region_size=(_WCS_REGION_DEG, _WCS_REGION_DEG),
        pixel_size=WCS_PIXEL_SCALE,
        cell_size=WCS_CELL_SIZE,
        cell_buffer=WCS_CELL_BUFFER,
        cell_max_object=WCS_CELL_MAX_OBJECT,
        pixel_match_scale=WCS_PIXEL_MATCH_SCALE,
        pixel_r2_cut=WCS_PIXEL_R2_CUT,
    )
    wcs_match.reduceData(ALL_FILES, ALL_IDS)
    if TEST_ONLY:
        wcs_match.analysisLoop(_TEST_RANGE, _TEST_RANGE)
    else:
        wcs_match.analysisLoop()
    stats = wcs_match.extractStats()
    tables_io.write(stats, f"{TRACT}/all_match_")
    wcs_match.save(f"{TRACT}/all_match")
    print("  Done.")

print("Pipeline complete.")
