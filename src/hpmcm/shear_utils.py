from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas
import tables_io

from .shear_data import ShearData

if TYPE_CHECKING:
    from .cell import CellData
    from .shear_match import ShearMatch


# These are the names of the various shear catalogs
# They are in reverse "alphabetic" order
SHEAR_NAMES = ["ns", "2p", "2m", "1p", "1m"]

# These are the coeffs for the various shear catalogs
DESHEAR_COEFFS = np.array(
    [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, -1, -1, 0],
        [1, 0, 0, -1],
        [-1, 0, 0, 1],
    ]
)

# These parameters will have to change if the cells change
CELL_INNER_SIZE = 150
CELL_BUFFER = 25
CELL_OFFSET = 0.5
N_PATCH = 20

# These are calculated from the above
CELL_OUTER_SIZE = CELL_INNER_SIZE + (2 * CELL_BUFFER)
CLEAN_CELL_CUT = int(CELL_INNER_SIZE) / 2
UNCLEAN_CELL_CUT = CLEAN_CELL_CUT + 5
PATCH_OFFSET = (N_PATCH + 1 / 2)


def shearStats(df: pandas.DataFrame) -> dict:
    """Return the shear statistics

    {st} is the shear catalog name, one of "ns", "2p", "2m", "1p", "1m"

    {i}, {j} index the shear parameters 1, 2

    Parameters
    ----------
    df:
        Input DataFrame, must have :py:class:`hpmcm.ShearTable` schema

    Returns
    -------
    Shear stats in a dict.

    Notes
    -----
    Shear stats include:

    +-----------------+-----------------------------------------------------+
    | Key             | Description                                         |
    +=================+=====================================================+
    | n_{st}          | Number of sources from that catalog                 |
    +-----------------+-----------------------------------------------------+
    | g_{i}_{st}      | g_{i} shear parameter for that catalog              |
    +-----------------+-----------------------------------------------------+
    | delta_g_{i}_{j} | g_{i,j} shear measurement: g_{i}_{j}p - g_{i}_{j}m  |
    +-----------------+-----------------------------------------------------+
    | good            | True if every catalog has one source in this object |
    +-----------------+-----------------------------------------------------+

    If the matching is not good, then delta_g_1 = delta_g_2 = np.nan
    """
    # Extract arrays once to avoid repeated DataFrame indexing
    i_cat_arr = df["i_cat"].values
    g_1_arr = df["g_1"].values
    g_2_arr = df["g_2"].values

    out_dict: dict[str, float | int] = {}
    all_good = True
    for i, name_ in enumerate(SHEAR_NAMES):
        mask = i_cat_arr == i
        n_cat = int(mask.sum())
        if n_cat != 1:
            all_good = False
        out_dict[f"n_{name_}"] = n_cat
        if n_cat:
            out_dict[f"g_1_{name_}"] = float(g_1_arr[mask].mean())
            out_dict[f"g_2_{name_}"] = float(g_2_arr[mask].mean())
        else:
            out_dict[f"g_1_{name_}"] = np.nan
            out_dict[f"g_2_{name_}"] = np.nan
    if all_good:
        out_dict["delta_g_1_1"] = out_dict["g_1_1p"] - out_dict["g_1_1m"]
        out_dict["delta_g_2_2"] = out_dict["g_2_2p"] - out_dict["g_2_2m"]
        out_dict["delta_g_1_2"] = out_dict["g_1_2p"] - out_dict["g_1_2m"]
        out_dict["delta_g_2_1"] = out_dict["g_2_1p"] - out_dict["g_2_1m"]
    else:
        out_dict["delta_g_1_1"] = np.nan
        out_dict["delta_g_2_2"] = np.nan
        out_dict["delta_g_1_2"] = np.nan
        out_dict["delta_g_2_1"] = np.nan
    out_dict["good"] = all_good
    return out_dict


def shearReport(
    basefile: str,
    output_file_base: str | None,
    shear: float,
    cat_type: str,
    tract: int,
    snr_cut: float = 7.5,
) -> ShearData:
    """Report on the shear calibration

    Parameters
    ----------
    basefile
        Input base file name (see notes)

    output_file_base:
        Output file name (see notes)

    shear:
        Applied shear

    cat_type:
        Catalog type (one of ["pgauss", "gauss", "wmom"]

    tract:
        Tract, written to output data

    snr_cut:
        Signal-to-noise cut.

    Returns
    -------
    The computed ShearData object

    Notes
    -----
    This will read the object shear data from "{basefile}_cluster_shear.pq"
    This will read the object statistics from "{basefile}_cluster_stats.pq"

    If output_file_base is not None:
    This will write the shear stats to "{output_file_base}.pkl"
    This will write the figures to "{output_file_base}_{figure}.png"
    """
    t = tables_io.read(f"{basefile}_cluster_shear.pq")
    t2 = tables_io.read(f"{basefile}_cluster_stats.pq")

    shear_data = ShearData(t, t2, shear, cat_type, tract, snr_cut=snr_cut)

    if output_file_base is not None:
        shear_data.save(f"{output_file_base}.pkl")
        shear_data.savefigs(output_file_base)

    return shear_data


def mergeShearReports(
    inputs: list[str],
    output_file: str,
) -> None:
    """Merge reports on the shear calibration

    Parameters
    ----------
    inputs:
        List of input ShearData pickle files

    output_file:
        Where to write the merged file
    """
    out_dict: dict[str, list] = defaultdict(list)
    for input_ in inputs:
        for key, val in ShearData.load(input_).toDict().items():
            out_dict[key].append(val)

    out_df = pandas.DataFrame(out_dict)
    out_df.to_parquet(output_file)


def splitByTypeAndClean(
    basefile: str,
    tract: int,
    shear: float,
    cat_type: str,
    *,
    clean: bool = False,
) -> None:  # pragma: no cover
    """Split a parquet file by shear catalog type and tract

    Parameters
    ----------
    basefile:
        Original file name

    tract:
        Tract to select

    shear:
        Applied shear, saved to output

    cat_type:
        Catalog type to select

    clean:
        Remove duplicates

    Notes
    -----
    This will create 5 files with the pattern:
    "{basefile}_uncleaned_{tract}_{type}.pq"

    +--------------+-------------------------------------+
    | Column       | Description                         |
    +==============+=====================================+
    | id           | Index of object inside catalog      |
    +--------------+-------------------------------------+
    | orig_id      | Original object id                  |
    +--------------+-------------------------------------+
    | cell_idx_x   | X-index of Cell                     |
    +--------------+-------------------------------------+
    | cell_idx_y   | Y-index of Cell                     |
    +--------------+-------------------------------------+
    | x_cell_coadd | X-coordinate in cell frame          |
    +--------------+-------------------------------------+
    | y_cell_coadd | Y-coordinate in cell frame          |
    +--------------+-------------------------------------+
    | x_pix        | X-coordinate in global WCS frame    |
    +--------------+-------------------------------------+
    | y_pix        | Y-coordinate in global WCS frame    |
    +--------------+-------------------------------------+
    | g_1          | Shear g_1 component estimate        |
    +--------------+-------------------------------------+
    | g_2          | Shear g_2 component estimate        |
    +--------------+-------------------------------------+
    | snr          | Signal-to-noise ratio               |
    +--------------+-------------------------------------+

    """
    p = tables_io.read(basefile)
    if clean:
        clean_st = "cleaned"
        cell_cut = CLEAN_CELL_CUT
    else:
        clean_st = "uncleaned"
        cell_cut = UNCLEAN_CELL_CUT
    for type_ in SHEAR_NAMES:
        mask = p["shear_type"] == type_
        sub = p[mask]

        # Filter on tract and patch centrality before computing derived columns
        right_tract = sub["tract"] == tract
        central_to_patch = (
            (np.fabs(sub["cell_x"].values - PATCH_OFFSET) < (N_PATCH / 2))
            & (np.fabs(sub["cell_y"].values - PATCH_OFFSET) < (N_PATCH / 2))
        )
        sub = sub[right_tract & central_to_patch].copy(deep=True)

        cell_idx_x = (N_PATCH * sub["patch_x"].values + sub["cell_x"].values).astype(
            int
        )
        cell_idx_y = (N_PATCH * sub["patch_y"].values + sub["cell_y"].values).astype(
            int
        )
        cent_x = CELL_INNER_SIZE * (cell_idx_x - CELL_OFFSET)
        cent_y = CELL_INNER_SIZE * (cell_idx_y - CELL_OFFSET)
        x_cell_coadd = sub["col"].values - cent_x
        y_cell_coadd = sub["row"].values - cent_y

        central_to_cell = (np.fabs(x_cell_coadd) < cell_cut) & (
            np.fabs(y_cell_coadd) < cell_cut
        )
        cleaned = sub[central_to_cell].copy(deep=True)

        cleaned["x_cell_coadd"] = x_cell_coadd[central_to_cell]
        cleaned["y_cell_coadd"] = y_cell_coadd[central_to_cell]
        cleaned["x_pix"] = cleaned["col"] + CELL_BUFFER
        cleaned["y_pix"] = cleaned["row"] + CELL_BUFFER
        cleaned["snr"] = (
            cleaned[f"{cat_type}_band_flux_r"] / cleaned[f"{cat_type}_band_flux_err_r"]
        )
        cleaned["g_1"] = cleaned[f"{cat_type}_g_1"]
        cleaned["g_2"] = cleaned[f"{cat_type}_g_2"]
        cleaned["cell_idx_x"] = cell_idx_x[central_to_cell]
        cleaned["cell_idx_y"] = cell_idx_y[central_to_cell]
        cleaned["orig_id"] = cleaned["id"]
        cleaned["id"] = np.arange(len(cleaned))
        cleaned["shear"] = shear
        cleaned.to_parquet(basefile.replace(".parq", f"_{clean_st}_{tract}_{type_}.pq"))


def reduceShearDataForCell(
    cell: CellData, i_cat: int, dataframe: pandas.DataFrame
) -> pandas.DataFrame:
    """Filters dataframe to keep only sources in the cell

    Parameters
    ----------
    cell:
        The cell being analyzed

    i_cat:
        Catalog index

    dataframe:
        Input dataframe


    Returns
    -------
    Filtered datasets


    Notes
    -----
    This will optionally deshear the source positions if `matcher.deshear`
    is not None.

    This will add these columns to the output dataframes

    +-----------+-------------------------------------+
    | Column    | Description                         |
    +===========+=====================================+
    | x_cell    | X-coordinate in cell frame          |
    +-----------+-------------------------------------+
    | y_cell    | Y-coordinate in cell frame          |
    +-----------+-------------------------------------+
    | x_pix     | X-coordinate in global WCS frame    |
    +-----------+-------------------------------------+
    | y_pix     | Y-coordinate in global WCS frame    |
    +-----------+-------------------------------------+
    | dx_shear  | Change in X position when desheared |
    +-----------+-------------------------------------+
    | dy_shear  | Change in Y position when desheared |
    +-----------+-------------------------------------+

    """

    matcher = cell.matcher

    if TYPE_CHECKING:
        assert isinstance(matcher, ShearMatch)

    filtered_idx = matcher.getCellIndices(dataframe) == cell.idx
    reduced = dataframe[filtered_idx]

    # Work on numpy arrays for vectorized arithmetic
    x_cell_orig = reduced["x_cell_coadd"].values
    y_cell_orig = reduced["y_cell_coadd"].values
    x_pix_orig = reduced["x_pix"].values
    y_pix_orig = reduced["y_pix"].values

    coeffs = DESHEAR_COEFFS[i_cat]
    if matcher.deshear is not None:
        dx_shear = matcher.deshear * (
            x_cell_orig * coeffs[0] + y_cell_orig * coeffs[2]
        )
        dy_shear = matcher.deshear * (
            x_cell_orig * coeffs[1] + y_cell_orig * coeffs[3]
        )
        x_cell = x_cell_orig + dx_shear
        y_cell = y_cell_orig + dy_shear
        x_pix = x_pix_orig + dx_shear
        y_pix = y_pix_orig + dy_shear
    else:  # pragma: no cover
        dx_shear = np.zeros(len(reduced))
        dy_shear = np.zeros(len(reduced))
        x_cell = x_cell_orig
        y_cell = y_cell_orig
        x_pix = x_pix_orig
        y_pix = y_pix_orig

    x_cell = (x_cell + (CELL_OUTER_SIZE / 2)) / matcher.pixel_match_scale
    y_cell = (y_cell + (CELL_OUTER_SIZE / 2)) / matcher.pixel_match_scale
    filtered_bounds = (
        (x_cell >= 0) & (x_cell < cell.n_pix[0])
        & (y_cell >= 0) & (y_cell < cell.n_pix[1])
    )

    # Single copy at the end
    red = reduced[filtered_bounds].copy(deep=True)
    red["x_cell"] = x_cell[filtered_bounds]
    red["y_cell"] = y_cell[filtered_bounds]
    red["x_pix"] = x_pix[filtered_bounds]
    red["y_pix"] = y_pix[filtered_bounds]
    if matcher.deshear is not None:
        red["dx_shear"] = dx_shear[filtered_bounds]
        red["dy_shear"] = dy_shear[filtered_bounds]
    return red


def makeMatchedShearSourceCatalogs(
    source_base_name: str,
    match_base_name: str,
) -> dict[str, pandas.DataFrame]:
    """Use the associations to join the source tables to their match objects

    Parameters
    ----------
    source_base_name:
        Base file name for source catalogs

    match_base_name:
        Base file name for match tables

    Returns
    -------
    Dict of tables, keyed by shear type, which have the
    source catalogs joined to the associated objects
    """
    keys = ['object_stats', 'object_assoc', 'object_shear']
    shear_types = {v: k for k, v in enumerate(SHEAR_NAMES)}
    td = tables_io.read(match_base_name, keys=keys)
    itd = tables_io.read(source_base_name, keys=list(shear_types.keys()))
    # Stats and shear tables are row-aligned; concat is cheaper than merge on synthetic index
    merged_object = pandas.concat(
        [td['object_stats'].reset_index(drop=True),
         td['object_shear'].reset_index(drop=True)],
        axis=1,
    )
    # Remove duplicate column names (keep first occurrence)
    merged_object = merged_object.loc[:, ~merged_object.columns.duplicated()]
    merged_object_assoc = td['object_assoc'].merge(
        merged_object, on="object_id", how="inner", suffixes=["_assoc", "_object"]
    )

    out_dict: dict[str, pandas.DataFrame] = {}

    # Process 'ns' (i_cat==0) first so it's available for left-joins below
    ns_sources = itd['ns'].copy()
    ns_sources['source_id'] = ns_sources['id']
    ns_mask = merged_object_assoc.catalog_id == 0
    ns_matched = merged_object_assoc[ns_mask].merge(
        ns_sources, on="source_id", how="inner", suffixes=["_object", "_source"]
    )
    out_dict['ns'] = ns_matched

    for cat_type_, i_cat_ in shear_types.items():
        if i_cat_ == 0:
            continue
        merged_object_assoc_mask = merged_object_assoc.catalog_id == i_cat_
        merged_object_assoc_masked = merged_object_assoc[merged_object_assoc_mask]
        sources = itd[cat_type_].copy()
        sources['source_id'] = sources['id']
        matched_source = merged_object_assoc_masked.merge(
            sources, on="source_id", how="inner", suffixes=["_object", "_source"]
        )
        fully_merged = matched_source.merge(
            out_dict['ns'], on='object_id', how='left', suffixes=['', '_ns']
        )
        out_dict[cat_type_] = fully_merged

    return out_dict
