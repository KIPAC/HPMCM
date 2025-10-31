from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas
import tables_io

from .shear_data import ShearData

if TYPE_CHECKING:
    from .cell import CellData
    from .shear_match import ShearMatch


# These are the names of the various shear catalogs
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


def shearStats(df: pandas.DataFrame) -> dict:
    """Return the shear statistics

    {st} is the shear type, one of "gauss", "pgauss", "wmom"

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
    | delta_g_{i}_{j} | g_{i,j} shear measurment: g_{i}_{j}p - g_{i}_{j}m   |
    +-----------------+-----------------------------------------------------+
    | good            | True if every catalog has one source in this object |
    +-----------------+-----------------------------------------------------+

    If the matching is not good, then delta_g_1 = delta_g_2 = np.nan
    """
    out_dict: dict[str, float | int] = {}
    all_good = True
    for i, name_ in enumerate(SHEAR_NAMES):
        mask = df.i_cat == i
        n_cat = mask.sum()
        if n_cat != 1:
            all_good = False
        out_dict[f"n_{name_}"] = int(n_cat)
        if n_cat:
            out_dict[f"g_1_{name_}"] = df[mask].g_1.values.mean()
            out_dict[f"g_2_{name_}"] = df[mask].g_2.values.mean()
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
) -> None:
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
        Tract, written to outout data

    snr_cut:
        Signal-to-noise cut.

    Notes
    -----
    This will read the object shear data from "{basefile}_cluster_shear.pq"
    This will read the object statisticis from "{basefile}_cluster_stats.pq"

    This will write the shear stats to "{output_file_base}.pkl"
    This will write the figures to "{output_file_base}_{figure}.png"
    """
    t = tables_io.read(f"{basefile}_cluster_shear.pq")
    t2 = tables_io.read(f"{basefile}_cluster_stats.pq")

    shear_data = ShearData(t, t2, shear, cat_type, tract, snr_cut=snr_cut)

    if output_file_base is not None:
        shear_data.save(f"{output_file_base}.pkl")
        shear_data.savefigs(output_file_base)


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
    out_dict: dict[str, Any] = {}
    for input_ in inputs:
        shear_data = ShearData.load(input_)
        input_dict = shear_data.toDict()
        for key, val in input_dict.items():
            if key in out_dict:
                out_dict[key].append(val)
            else:
                out_dict[key] = [val]

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
        cell_cut = 75
    else:
        clean_st = "uncleaned"
        cell_cut = 80
    for type_ in SHEAR_NAMES:
        mask = p["shear_type"] == type_
        sub = p[mask].copy(deep=True)
        cell_idx_x = (20 * sub["patch_x"].values + sub["cell_x"].values).astype(int)
        cell_idx_y = (20 * sub["patch_y"].values + sub["cell_y"].values).astype(int)
        cent_x = 150 * cell_idx_x - 75
        cent_y = 150 * cell_idx_y - 75
        x_cell_coadd = sub["col"] - cent_x
        y_cell_coadd = sub["row"] - cent_y
        sub["x_pix"] = sub["col"] + 25
        sub["y_pix"] = sub["row"] + 25
        sub["x_cell_coadd"] = x_cell_coadd
        sub["y_cell_coadd"] = y_cell_coadd
        sub["snr"] = sub[f"{cat_type}_band_flux_r"] / sub[f"{cat_type}_band_flux_err_r"]
        sub["g_1"] = sub[f"{cat_type}_g_1"]
        sub["g_2"] = sub[f"{cat_type}_g_2"]
        sub["cell_idx_x"] = cell_idx_x
        sub["cell_idx_y"] = cell_idx_y
        sub["orig_id"] = sub.id
        sub["id"] = np.arange(len(sub))
        central_to_cell = np.bitwise_and(
            np.fabs(x_cell_coadd) < cell_cut, np.fabs(y_cell_coadd) < cell_cut
        )
        central_to_patch = np.bitwise_and(
            np.fabs(sub["cell_x"].values - 10.5) < 10,
            np.fabs(sub["cell_y"].values - 10.5) < 10,
        )
        right_tract = sub["tract"] == tract
        central = np.bitwise_and(central_to_cell, central_to_patch)
        selected = np.bitwise_and(right_tract, central)
        cleaned = sub[selected].copy(deep=True)
        cleaned["shear"] = np.repeat(shear, len(cleaned))
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
    | dy_shear  | Change in X position when desheared |
    +-----------+-------------------------------------+

    """

    matcher = cell.matcher

    if TYPE_CHECKING:
        assert isinstance(matcher, ShearMatch)

    filtered_idx = matcher.getCellIndices(dataframe) == cell.idx
    reduced = dataframe[filtered_idx].copy(deep=True)

    x_cell_orig = reduced["x_cell_coadd"]
    y_cell_orig = reduced["y_cell_coadd"]
    x_pix_orig = reduced["x_pix"]
    y_pix_orig = reduced["y_pix"]

    if matcher.deshear is not None:
        # De-shear in the cell frame to do matching
        dx_shear = matcher.deshear * (
            x_cell_orig * DESHEAR_COEFFS[i_cat][0]
            + y_cell_orig * DESHEAR_COEFFS[i_cat][2]
        )
        dy_shear = matcher.deshear * (
            x_cell_orig * DESHEAR_COEFFS[i_cat][1]
            + y_cell_orig * DESHEAR_COEFFS[i_cat][3]
        )
        x_cell = x_cell_orig + dx_shear
        y_cell = y_cell_orig + dy_shear
        x_pix = x_pix_orig + dx_shear
        y_pix = y_pix_orig + dy_shear
    else:  # pragma: no cover
        dx_shear = np.zeros(len(dataframe))
        dy_shear = np.zeros(len(dataframe))
        x_cell = x_cell_orig
        y_cell = y_cell_orig
        x_pix = x_pix_orig
        y_pix = y_pix_orig

    x_cell = (x_cell + 100) / matcher.pixel_match_scale
    y_cell = (y_cell + 100) / matcher.pixel_match_scale
    filtered_x = np.bitwise_and(x_cell >= 0, x_cell < cell.n_pix[0])
    filtered_y = np.bitwise_and(y_cell >= 0, y_cell < cell.n_pix[1])
    filtered_bounds = np.bitwise_and(filtered_x, filtered_y)
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
    """Use the associations to join the source tables to their match obects

    Parameters
    ----------
    source_base_name:
        _base file name for souces catalogs

    match_base_name:
        _base file naem for match tables

    Returns
    -------
    Dict of tables, keyed by shear type, which have the
    souces catalogs joined to the associated objects
    """
    keys = ["object_stats", "object_assoc", "object_shear"]
    shear_types = {v: k for k, v in enumerate(SHEAR_NAMES)}
    td = tables_io.read(match_base_name, keys=keys)
    itd = tables_io.read(source_base_name, keys=list(shear_types.keys()))
    td["object_stats"]["idx"] = np.arange(len(td["object_stats"]))
    td["object_shear"]["idx"] = np.arange(len(td["object_shear"]))
    merged_object = td["object_stats"].merge(
        td["object_shear"], on="idx", how="inner", suffixes=["_l", "_r"]
    )
    merged_object_assoc = td["object_assoc"].merge(
        merged_object, on="objectId", how="inner", suffixes=["_l", "_r"]
    )
    out_dict = {}
    for cat_type_, i_cat_ in shear_types.items():
        merged_object_assoc_mask = merged_object_assoc.catalogId == i_cat_
        merged_object_assoc_masked = merged_object_assoc[merged_object_assoc_mask]
        sources = itd[cat_type_]
        sources["sourceId"] = sources["id"]
        matched_source = merged_object_assoc_masked.merge(
            sources, on="sourceId", how="inner", suffixes=["_l", "_r"]
        )
        out_dict[cat_type_] = matched_source
    return out_dict
