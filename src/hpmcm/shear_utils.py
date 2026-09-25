from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
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


@dataclass
class ShearCellGeometry:
    """Geometry and configuration for the shear coadd cell grid and WCS matching.

    Attributes
    ----------
    cell_inner_size:
        Inner cell size in coadd pixels.
    cell_buffer:
        Buffer around the inner region in coadd pixels. x_cell = 0 is the
        outer edge; the inner region starts at x_cell = cell_buffer.
    cell_inner_buffer:
        Extra pixels beyond the inner region retained in uncleaned catalogs.
    pixel_offset:
        Half-pixel correction applied when computing x_cell_coadd.
    n_cell_in_patch:
        Number of inner cells per patch side.
    n_cell_patch_buffer:
        Extra cell indices in the patch buffer region.
    pixel_size:
        WCS pixel size in degrees.
    match_buffer:
        Overlap in pixels between adjacent WCS cells used during matching.
    tract_size:
        Tract size in WCS pixels [x, y].
    ref_dir:
        Reference direction (RA, DEC) in degrees for the tract centre.
        When set, ShearMatch builds a WCS from this and pixel_size so that
        pixToWorld returns valid RA/Dec for objects and clusters.
        If None (default), pixToWorld returns NaN.
    wcs_ctype:
        WCS projection type used when building the pixel-to-sky transform.
        Default "TAN" (gnomonic) matches the Rubin sky map projection.
    """

    cell_inner_size: int = 150
    cell_buffer: int = 50
    cell_inner_buffer: int = 5
    pixel_offset: float = 0.5
    n_cell_in_patch: int = 20
    n_cell_patch_buffer: int = 1
    pixel_size: float = 0.2 / 3600.0
    match_buffer: int = 25
    tract_size: np.ndarray = field(default_factory=lambda: np.array([30000, 30000]))
    ref_dir: tuple[float, float] | None = None
    wcs_ctype: str = "TAN"

    @property
    def cell_outer_size(self) -> int:
        """Total cell size including both buffers."""
        return self.cell_inner_size + 2 * self.cell_buffer

    @property
    def patch_offset(self) -> float:
        """Cell-index offset to the centre of a patch."""
        return (self.n_cell_in_patch + self.n_cell_patch_buffer) / 2


DEFAULT_GEOMETRY = ShearCellGeometry()


def innerCellMask(
    df: pandas.DataFrame,
    geometry: ShearCellGeometry = DEFAULT_GEOMETRY,
) -> np.ndarray:
    """Return a boolean mask selecting sources within the inner cell region.

    The inner region spans [cell_buffer, cell_buffer + cell_inner_size) in both
    x_cell and y_cell, where x_cell = 0 is the outer edge of the cell.

    Parameters
    ----------
    df:
        DataFrame with x_cell and y_cell columns in the cell frame.
    geometry:
        Cell geometry configuration.

    Returns
    -------
    Boolean array, True for sources within the inner cell region.
    """
    lo = geometry.cell_buffer
    hi = geometry.cell_buffer + geometry.cell_inner_size
    return (
        (df["x_cell"].values >= lo)
        & (df["x_cell"].values < hi)
        & (df["y_cell"].values >= lo)
        & (df["y_cell"].values < hi)
    )


def shearStats(
    df: pandas.DataFrame,
    shear_names: list[str] = SHEAR_NAMES,
) -> dict:
    """Return the shear statistics

    {st} is a shear catalog name drawn from shear_names.

    {i}, {j} index the shear parameters 1, 2.

    Parameters
    ----------
    df:
        Input DataFrame, must have :py:class:`hpmcm.ShearTable` schema.

    shear_names:
        Ordered list of active catalog names. Defaults to all five
        (SHEAR_NAMES). Pass ["ns", "1p", "1m"] for 3-catalog mode.
        i_cat values in df correspond to positions in this list.

    Returns
    -------
    Shear stats in a dict.

    Notes
    -----
    All keys from the full SHEAR_NAMES schema are always present in the
    output. Inactive catalogs (not in shear_names) have n=0 and g=nan.
    delta_g_{i}_1 is computed when "1p" and "1m" are both active;
    delta_g_{i}_2 when "2p" and "2m" are both active; nan otherwise.

    +-----------------+-----------------------------------------------------+
    | Key             | Description                                         |
    +=================+=====================================================+
    | n_{st}          | Number of sources from that catalog                 |
    +-----------------+-----------------------------------------------------+
    | g_{i}_{st}      | g_{i} shear parameter for that catalog              |
    +-----------------+-----------------------------------------------------+
    | delta_g_{i}_{j} | g_{i,j} shear measurement: g_{i}_{j}p - g_{i}_{j}m  |
    +-----------------+-----------------------------------------------------+
    | good            | True if every active catalog has one source         |
    +-----------------+-----------------------------------------------------+
    """
    i_cat_arr = df["i_cat"].values
    g_1_arr = df["g_1"].values
    g_2_arr = df["g_2"].values

    # Pre-fill all schema keys so output is always schema-compatible
    out_dict: dict[str, float | int] = {}
    for name_ in SHEAR_NAMES:
        out_dict[f"n_{name_}"] = 0
        out_dict[f"g_1_{name_}"] = np.nan
        out_dict[f"g_2_{name_}"] = np.nan

    all_good = True
    for i, name_ in enumerate(shear_names):
        mask = i_cat_arr == i
        n_cat = int(mask.sum())
        if n_cat != 1:
            all_good = False
        out_dict[f"n_{name_}"] = n_cat
        if n_cat:
            out_dict[f"g_1_{name_}"] = float(g_1_arr[mask].mean())
            out_dict[f"g_2_{name_}"] = float(g_2_arr[mask].mean())

    has_g1 = "1p" in shear_names and "1m" in shear_names
    has_g2 = "2p" in shear_names and "2m" in shear_names
    if all_good:
        out_dict["delta_g_1_1"] = out_dict["g_1_1p"] - out_dict["g_1_1m"] if has_g1 else np.nan
        out_dict["delta_g_2_1"] = out_dict["g_2_1p"] - out_dict["g_2_1m"] if has_g1 else np.nan
        out_dict["delta_g_1_2"] = out_dict["g_1_2p"] - out_dict["g_1_2m"] if has_g2 else np.nan
        out_dict["delta_g_2_2"] = out_dict["g_2_2p"] - out_dict["g_2_2m"] if has_g2 else np.nan
    else:
        out_dict["delta_g_1_1"] = np.nan
        out_dict["delta_g_2_1"] = np.nan
        out_dict["delta_g_1_2"] = np.nan
        out_dict["delta_g_2_2"] = np.nan
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


def splitRubinMDByTypeAndClean(
    basefile: str,
    tract: int,
    shear: float,
    *,
    clean: bool = False,
    geometry: ShearCellGeometry = DEFAULT_GEOMETRY,
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

    """
    p = tables_io.read(basefile)
    clean_st = "cleaned" if clean else "uncleaned"

    for type_ in SHEAR_NAMES:
        mask = p["metaStep"] == type_
        sub = p[mask]

        # Filter on tract and patch centrality before computing derived columns
        right_tract = sub["tract"] == tract
        central_to_patch = (
            np.fabs(sub["cell_x"].values - geometry.patch_offset)
            < (geometry.n_cell_in_patch / 2)
        ) & (
            np.fabs(sub["cell_y"].values - geometry.patch_offset)
            < (geometry.n_cell_in_patch / 2)
        )

        print(f"Centeral to patch {central_to_patch.sum()} {len(central_to_patch)}")
        sub = sub[right_tract & central_to_patch].copy(deep=True)

        if "patch_x" not in sub.columns:
            sub["patch_x"] = sub["patch"] % 10
            sub["patch_y"] = sub["patch"] // 10

        cell_idx_x = (
            geometry.n_cell_in_patch * sub["patch_x"].values + sub["cell_x"].values
        ).astype(int)
        cell_idx_y = (
            geometry.n_cell_in_patch * sub["patch_y"].values + sub["cell_y"].values
        ).astype(int)

        # x_cell_coadd = 0 at the left edge of the inner region (cell_idx * cell_inner_size)
        x_cell_coadd = sub["x"].values - (cell_idx_x - geometry.n_cell_patch_buffer) * geometry.cell_inner_size
        y_cell_coadd = sub["y"].values - (cell_idx_y - geometry.n_cell_patch_buffer) * geometry.cell_inner_size

        buf = 0 if clean else geometry.cell_inner_buffer
        central_to_cell = (
            (x_cell_coadd >= -buf)
            & (x_cell_coadd < geometry.cell_inner_size + buf)
            & (y_cell_coadd >= -buf)
            & (y_cell_coadd < geometry.cell_inner_size + buf)
        )
        print(f"Centeral to cell {central_to_cell.sum()} {len(central_to_cell)}")
        
        cleaned = sub[central_to_cell].copy(deep=True)

        cleaned["x_cell_coadd"] = x_cell_coadd[central_to_cell]
        cleaned["y_cell_coadd"] = y_cell_coadd[central_to_cell]
        cleaned["x_pix"] = cleaned["x"]
        cleaned["y_pix"] = cleaned["y"]
        cleaned["cell_idx_x"] = cell_idx_x[central_to_cell]
        cleaned["cell_idx_y"] = cell_idx_y[central_to_cell]
        cleaned["id"] = cleaned["shearObjectId"]
        cleaned["shear"] = shear
        cleaned["meta_step"] = np.full(len(cleaned), type_)
        cleaned.to_parquet(
            basefile.replace(".parq", f"_{clean_st}_{tract}_{type_}.parq")
        )


def splitDESCMDByTypeAndClean(
    basefile: str,
    tract: int,
    shear: float,
    *,
    clean: bool = False,
    geometry: ShearCellGeometry = DEFAULT_GEOMETRY,
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

    """
    p = tables_io.read(basefile)
    clean_st = "cleaned" if clean else "uncleaned"
    for type_ in SHEAR_NAMES:
        mask = p["mcal_step"] == type_
        sub = p[mask]

        # Filter on tract and patch centrality before computing derived columns
        right_tract = np.ones(len(sub)).astype(bool)
        central_to_patch = (
            np.fabs(sub["cell_i"].values - geometry.patch_offset)
            < (geometry.n_cell_in_patch / 2)
        ) & (
            np.fabs(sub["cell_j"].values - geometry.patch_offset)
            < (geometry.n_cell_in_patch / 2)
        )
        print(f"Centeral to patch {central_to_patch.sum()} {len(central_to_patch)}")
        sub = sub[right_tract & central_to_patch].copy(deep=True)

        cell_idx_x = (
            geometry.n_cell_in_patch * sub["patch_x"].values + sub["cell_j"].values
        ).astype(int)
        cell_idx_y = (
            geometry.n_cell_in_patch * sub["patch_y"].values + sub["cell_i"].values
        ).astype(int)

        # xcell is outer-edge referenced (0 to cell_outer_size); subtract cell_buffer
        # so that x_cell_coadd = 0 at the left edge of the inner region
        x_cell_coadd = sub["xcell"].values - geometry.cell_buffer
        y_cell_coadd = sub["ycell"].values - geometry.cell_buffer

        buf = 0 if clean else geometry.cell_inner_buffer
        central_to_cell = (
            (x_cell_coadd >= -buf)
            & (x_cell_coadd < geometry.cell_inner_size + buf)
            & (y_cell_coadd >= -buf)
            & (y_cell_coadd < geometry.cell_inner_size + buf)
        )
        print(f"Centeral to cell {central_to_cell.sum()} {len(central_to_cell)}")
            
        cleaned = sub[central_to_cell].copy(deep=True)
        
        cleaned["x_cell_coadd"] = x_cell_coadd[central_to_cell]
        cleaned["y_cell_coadd"] = y_cell_coadd[central_to_cell]
        cleaned["x_pix"] = cleaned["x"]
        cleaned["y_pix"] = cleaned["y"]
        cleaned["cell_idx_x"] = cell_idx_x[central_to_cell]
        cleaned["cell_idx_y"] = cell_idx_y[central_to_cell]
        cleaned["id"] = np.arange(len(cleaned))
        cleaned["shear"] = shear
        cleaned["meta_step"] = np.full(len(cleaned), type_)
        cleaned.to_parquet(
            basefile.replace(".parq", f"_{clean_st}_{tract}_{type_}.parq")
        )
        
        

def deshearSourcesForCell(
    dataframe: pandas.DataFrame,
    cell_idx_x: int,
    cell_idx_y: int,
    shear_name: str,
    deshear: float | None,
    cell_buffer: int = DEFAULT_GEOMETRY.cell_buffer,
    cell_inner_size: int = DEFAULT_GEOMETRY.cell_inner_size,
) -> pandas.DataFrame:
    """Filter and deshear sources belonging to a specific cell.

    Parameters
    ----------
    dataframe:
        Input dataframe with cell_idx_x, cell_idx_y, x_cell_coadd,
        y_cell_coadd, x_pix, and y_pix columns.

    cell_idx_x:
        X cell index to select (matched against dataframe["cell_idx_x"]).

    cell_idx_y:
        Y cell index to select (matched against dataframe["cell_idx_y"]).

    shear_name:
        Shear catalog name; one of SHEAR_NAMES ("ns", "2p", "2m", "1p", "1m").
        Selects the deshear coefficients from DESHEAR_COEFFS.

    deshear:
        Deshear factor (-1 * applied shear). If None no deshearing is applied
        and dx_shear / dy_shear columns are not added to the output.

    cell_buffer:
        Buffer in pixels around the inner cell region.

    cell_inner_size:
        Size of the inner cell region in coadd pixels. Used to locate the
        cell centre, about which the shear transformation is applied.

    Returns
    -------
    DataFrame containing only the sources in the requested cell with
    x_cell, y_cell, x_pix, y_pix columns set to the desheared positions.
    If deshear is not None, dx_shear and dy_shear columns are also added.

    Notes
    -----
    The deshear correction is applied relative to the centre of the inner
    cell region (x_cell_coadd = cell_inner_size / 2), so sources at the
    centre receive zero correction and the maximum correction magnitude is
    |deshear| * cell_inner_size / 2 at the inner edges. The final x_cell
    values therefore remain close to the undesheared positions.
    x_cell and y_cell are in regular cell pixel space (same frame as
    CellData.x_cell: x_cell = 0 at the outer edge of the cell,
    x_cell = cell_buffer at the inner left edge). pixel_match_scale is
    NOT applied here; fillCountsMapFromDf handles that conversion, keeping
    this consistent with the non-shear CellData.reduceDataframe path.
    No bounds filtering is applied; call reduceShearDataForCell for that.
    """
    if shear_name not in SHEAR_NAMES:
        raise ValueError(f"shear_name must be one of {SHEAR_NAMES}, got {shear_name!r}")

    mask = (dataframe["cell_idx_x"] == cell_idx_x) & (
        dataframe["cell_idx_y"] == cell_idx_y
    )
    reduced = dataframe[mask]

    x_cell_orig = reduced["x_cell_coadd"].values
    y_cell_orig = reduced["y_cell_coadd"].values
    x_pix_orig = reduced["x_pix"].values
    y_pix_orig = reduced["y_pix"].values

    # Shear is applied relative to the centre of the inner cell region
    x_centre = x_cell_orig - cell_inner_size / 2
    y_centre = y_cell_orig - cell_inner_size / 2

    coeffs = DESHEAR_COEFFS[SHEAR_NAMES.index(shear_name)]
    if deshear is not None:
        dx_shear: np.ndarray = deshear * (
            x_centre * coeffs[0] + y_centre * coeffs[2]
        )
        dy_shear: np.ndarray = deshear * (
            x_centre * coeffs[1] + y_centre * coeffs[3]
        )
        x_cell = x_cell_orig + dx_shear
        y_cell = y_cell_orig + dy_shear
        x_pix = x_pix_orig + dx_shear
        y_pix = y_pix_orig + dy_shear
    else:
        x_cell = x_cell_orig
        y_cell = y_cell_orig
        x_pix = x_pix_orig
        y_pix = y_pix_orig

    x_cell = x_cell + cell_buffer
    y_cell = y_cell + cell_buffer

    red = reduced.copy(deep=True)
    red["x_cell"] = x_cell
    red["y_cell"] = y_cell
    red["x_pix"] = x_pix
    red["y_pix"] = y_pix
    if deshear is not None:
        red["dx_shear"] = dx_shear
        red["dy_shear"] = dy_shear
    return red


def reduceShearDataForCell(
    cell: CellData,
    shear_name: str,
    dataframe: pandas.DataFrame,
    geometry: ShearCellGeometry = DEFAULT_GEOMETRY,
) -> pandas.DataFrame:
    """Filter and deshear sources for a cell, then clip to the cell footprint.

    Parameters
    ----------
    cell:
        The cell being analyzed.

    shear_name:
        Shear catalog name; one of SHEAR_NAMES ("ns", "2p", "2m", "1p", "1m").

    dataframe:
        Input dataframe.

    Returns
    -------
    Filtered dataframe with x_cell, y_cell, x_pix, y_pix columns added.
    If matcher.deshear is not None, dx_shear and dy_shear are also added.

    Notes
    -----
    Delegates deshearing to deshearSourcesForCell, then filters to
    sources within the cell footprint (0 <= x_cell < cell.n_pix[0]).
    """
    matcher = cell.matcher
    
    if TYPE_CHECKING:
        assert isinstance(matcher, ShearMatch)

    n_cell_y = int(matcher.n_cell[1])
    cell_idx_x = int(cell.idx // n_cell_y)
    cell_idx_y = int(cell.idx % n_cell_y)

    red = deshearSourcesForCell(
        dataframe,
        cell_idx_x=cell_idx_x,
        cell_idx_y=cell_idx_y,
        shear_name=shear_name,
        deshear=matcher.deshear,
        cell_buffer=geometry.cell_buffer,
        cell_inner_size=geometry.cell_inner_size,
    )

    in_bounds = (
        (red["x_cell"] >= 0)
        & (red["x_cell"] < cell.n_pix[0])
        & (red["y_cell"] >= 0)
        & (red["y_cell"] < cell.n_pix[1])
    )
    return red[in_bounds]


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
    keys = ["object_stats", "object_assoc", "object_shear"]
    shear_types = {v: k for k, v in enumerate(SHEAR_NAMES)}
    td = tables_io.read(match_base_name, keys=keys)
    itd = tables_io.read(source_base_name, keys=list(shear_types.keys()))
    # Stats and shear tables are row-aligned; concat is cheaper than merge on synthetic index
    merged_object = pandas.concat(
        [
            td["object_stats"].reset_index(drop=True),
            td["object_shear"].reset_index(drop=True),
        ],
        axis=1,
    )
    # Remove duplicate column names (keep first occurrence)
    merged_object = merged_object.loc[:, ~merged_object.columns.duplicated()]
    merged_object_assoc = td["object_assoc"].merge(
        merged_object, on="object_id", how="inner", suffixes=["_assoc", "_object"]
    )

    out_dict: dict[str, pandas.DataFrame] = {}

    # Process 'ns' (i_cat==0) first so it's available for left-joins below
    ns_sources = itd["ns"].copy()
    ns_sources["source_id"] = ns_sources["id"]
    ns_mask = merged_object_assoc.catalog_id == 0
    ns_matched = merged_object_assoc[ns_mask].merge(
        ns_sources, on="source_id", how="inner", suffixes=["_object", "_source"]
    )
    out_dict["ns"] = ns_matched

    for cat_type_, i_cat_ in shear_types.items():
        if i_cat_ == 0:
            continue
        merged_object_assoc_mask = merged_object_assoc.catalog_id == i_cat_
        merged_object_assoc_masked = merged_object_assoc[merged_object_assoc_mask]
        sources = itd[cat_type_].copy()
        sources["source_id"] = sources["id"]
        matched_source = merged_object_assoc_masked.merge(
            sources, on="source_id", how="inner", suffixes=["_object", "_source"]
        )
        fully_merged = matched_source.merge(
            out_dict["ns"], on="object_id", how="left", suffixes=["", "_ns"]
        )
        out_dict[cat_type_] = fully_merged

    return out_dict
