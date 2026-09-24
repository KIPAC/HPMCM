from __future__ import annotations

import numpy as np
import pandas
import tables_io

from .footprint import FootprintSet


def findClusterIdsFromArrays(
    x_locals: np.ndarray,
    y_locals: np.ndarray,
    cluster_key: np.ndarray,
) -> np.ndarray:
    """Associate sources to clusters using `clusterkey`
    which is a map where any pixel associated to a cluster
    has the cluster index as its value

    Parameters
    ----------
    x_locals:
        Local pixel x-positions

    y_locals:
        Local pixel y-positions

    cluster_key:
        2D-map of cluster Ids by pixel position

    Returns
    -------
    Ids of associated clusters
    """
    return np.array(
        [
            cluster_key[x_local_, y_local_]
            for x_local_, y_local_ in zip(x_locals, y_locals)
        ]
    ).astype(np.int32)


def findClusterIds(
    df: pandas.DataFrame,
    cluster_key: np.ndarray,
    pixel_match_scale: int = 1,
) -> np.ndarray:
    """Associate sources to clusters using `clusterkey`
    which is a map where any pixel associated to a cluster
    has the cluster index as its value

    Parameters
    ----------
    df:
        DataFrame with local pixel positions (x_cell, y_cell)

    cluster_key:
        2D-map of cluster Ids by pixel position

    pixel_match_scale:
        Scale-factor to use in making cluster map

    Returns
    -------
    Ids of associated clusters
    """
    return findClusterIdsFromArrays(
        np.floor(df["x_cell"] / pixel_match_scale).astype(int),
        np.floor(df["y_cell"] / pixel_match_scale).astype(int),
        cluster_key,
    )


def fillCountsMapFromArrays(
    x_locals: np.ndarray,
    y_locals: np.ndarray,
    n_pix: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fill a source counts map

    Parameters
    ----------
    x_locals:
        Local pixel x-positions

    y_locals:
        Local pixel y-positions

    n_pix:
        Number of pixels in x,y for counts map

    weights:
        If provided, weights to apply for each entry in counts map

    Returns
    -------
    Counts map of source in cell, projected into n_pix,n_pix grid
    """
    hist = np.histogram2d(
        x_locals,
        y_locals,
        bins=(n_pix[0], n_pix[1]),
        range=((0, n_pix[0]), (0, n_pix[1])),
        weights=weights,
    )
    return hist[0]


def fillCountsMapFromDf(
    df: pandas.DataFrame,
    n_pix: np.ndarray,
    weight_name: str | None = None,
    pixel_match_scale: int = 1,
) -> np.ndarray:
    """Fill a source counts map from a reduced dataframe for one input
    catalog

    Parameters
    ----------
    df:
        DataFrame with local pixel positions (x_cell, y_cell)

    n_pix:
        Number of pixels in x,y for counts map

    weight_name:
        If provided column to use for weights

    pixel_match_scale:
        Scale-factor to use in making cluster map

    Returns
    -------
    Counts map of source in cell, projected into n_pix,n_pix grid
    """
    if weight_name is None:
        weights = None
    else:  # pragma: no cover
        weights = df[weight_name].values
    return fillCountsMapFromArrays(
        df["x_cell"] / pixel_match_scale,
        df["y_cell"] / pixel_match_scale,
        n_pix=np.ceil(n_pix / pixel_match_scale).astype(int),
        weights=weights,
    )


def getFootprints(
    counts_map: np.ndarray,
    buf: int,
    pixel_match_scale: int = 1,
) -> dict:
    """Take a source counts map and do clustering using Footprint detection

    Parameters
    ----------
    counts_map:
        Map of source counts

    buf:
        Number of pixels in cell-edge buffer

    pixel_match_scale:
        Scale-factor used in making cluster map

    Returns
    -------
    Footprint data

    +---------------+------------------+---------------------------------+
    | Key           | Type             | Description                     |
    +===============+==================+=================================+
    | image         | np.ndarray       | Counts map of sources           |
    +---------------+------------------+---------------------------------+
    | footprints    | FootprintSet     | Clustering footprints           |
    +---------------+------------------+---------------------------------+
    | footprint_key | np.ndarray       | Array with cluster associations |
    +---------------+------------------+---------------------------------+

    """
    footprints_orig = FootprintSet.detect(counts_map)
    footprints = footprints_orig.filter(buf, pixel_match_scale)
    footprint_key = footprints.fp_key
    return dict(image=counts_map, footprints=footprints, footprint_key=footprint_key)


def associateSourcesToFootprints(
    data: list[pandas.DataFrame],
    cluster_key: np.ndarray,
    pixel_match_scale: int = 1,
) -> list[np.ndarray]:
    """Loop through data and associate sources to clusters

    Parameters
    ----------
    data:
        Input DataFrames

    cluster_key:
        2D-map of cluster Ids by pixel position

    pixel_match_scale:
        Scale-factor used in making cluster map

    Returns
    -------
    Lists of clusters associated to each source
    output[i][j] will give the id of the cluster associated
    to source j in input catalog i.
    """
    return [findClusterIds(df, cluster_key, pixel_match_scale) for df in data]


def reduceObjectTable(
    basefile: str,
    outfile: str,
    extra_cols: list[str] | None = None,
) -> None:  # pragma: no cover
    """Reduce an object table to just the colums needed for matching

    Parameters
    ----------
    basefile:
        Original file name

    outfile:
        Output file name

    extra_cols:
        Extra columns to copy

    Notes
    -----
    This will produce a DataFrame with at least these columns:

    +-----------------------+---------------------------------------------------------------+
    | Column name           | Description                                                   |
    +=======================+===============================================================+
    | id                    | source ID                                                     |
    +-----------------------+---------------------------------------------------------------+
    | tract                 | Tract source was found in                                     |
    +-----------------------+---------------------------------------------------------------+
    | patch                 | Patch source was found in                                     |
    +-----------------------+---------------------------------------------------------------+
    | ra                    | RA in degrees                                                 |
    +-----------------------+---------------------------------------------------------------+
    | dec                   | DEC in degress                                                |
    +-----------------------+---------------------------------------------------------------+
    | snr                   | Signal-to-Noise of source, used for filtering and centroiding |
    +-----------------------+---------------------------------------------------------------+
    | {band}_gaapPsfFlux    | Flux, for band in u,g,r,i,z,y                                 |
    +-----------------------+---------------------------------------------------------------+
    | {band}_gaapPsfFluxErr | Flux error, for band in u,g,r,i,z,y                           |
    +-----------------------+---------------------------------------------------------------+

    """
    t = tables_io.read(basefile)
    cols = ["tract", "patch", "coord_ra", "coord_dec", "objectId"]
    cols += [f"{band}_gaapPsfFlux" for band in "ugrizy"]
    cols += [f"{band}_gaapPsfFluxErr" for band in "ugrizy"]

    if extra_cols is not None:
        cols += extra_cols

    tout = t[cols].copy(deep=True)

    tout["ra"] = tout["coord_ra"]
    tout["dec"] = tout["coord_dec"]

    snr = np.zeros(len(tout["i_gaapPsfFlux"]))
    for band in "riz":
        snr += np.where(
            np.isfinite(tout[f"{band}_gaapPsfFlux"]),
            tout[f"{band}_gaapPsfFlux"] / tout[f"{band}_gaapPsfFluxErr"],
            0,
        )
    tout["snr"] = snr
    tout["id"] = tout["objectId"]

    tout.to_parquet(outfile)


def reduceAnacalTable(
    basefile: str,
    outfile: str,
    extra_cols: list[str] | None = None,
) -> None:  # pragma: no cover
    """Reduce an object table to just the colums needed for matching

    Parameters
    ----------
    basefile:
        Original file name

    outfile:
        Output file name

    extra_cols:
        Extra columns to copy

    Notes
    -----
    This will produce a DataFrame with at least these columns:

    +-----------------------------+---------------------------------------------------------------+
    | Column name                 | Description                                                   |
    +=============================+===============================================================+
    | id                          | source ID                                                     |
    +-----------------------------+---------------------------------------------------------------+
    | tract                       | Tract source was found in                                     |
    +-----------------------------+---------------------------------------------------------------+
    | patch                       | Patch source was found in                                     |
    +-----------------------------+---------------------------------------------------------------+
    | ra                          | RA in degrees                                                 |
    +-----------------------------+---------------------------------------------------------------+
    | dec                         | DEC in degress                                                |
    +-----------------------------+---------------------------------------------------------------+
    | snr                         | Signal-to-Noise of source, used for filtering and centroiding |
    +-----------------------------+---------------------------------------------------------------+
    | lsst_{band}_flux_gauss2     | Flux, for band in u,g,r,i,z,y                                 |
    +-----------------------------+---------------------------------------------------------------+
    | lsst_{band}_flux_gauss2_err | Flux error, for band in u,g,r,i,z,y                           |
    +-----------------------------+---------------------------------------------------------------+

    """
    t = tables_io.read(basefile)
    cols = ["tract_id", "patch_x", "patch_y", "ra", "dec", "object_id"]
    cols += [f"lsst_{band}_flux_gauss2" for band in "riz"]
    cols += [f"lsst_{band}_flux_gauss2_err" for band in "riz"]
    if extra_cols is not None:
        cols += extra_cols

    tout = t[cols].copy(deep=True)

    tout["tract"] = tout["tract_id"]

    snr = np.zeros(len(tout["lsst_i_flux_gauss2_err"]))
    for band in "riz":
        snr += np.where(
            np.isfinite(tout[f"lsst_{band}_flux_gauss2"]),
            tout[f"lsst_{band}_flux_gauss2"] / tout[f"lsst_{band}_flux_gauss2_err"],
            0,
        )
    tout["snr"] = snr
    tout["id"] = tout["object_id"]

    tout.to_parquet(outfile)


def reduceRubinMDTable(
    basefile: str,
    outfile: str,
    extra_cols: list[str] | None = None,
) -> None:  # pragma: no cover
    """Reduce an object table to just the colums needed for matching

    Parameters
    ----------
    basefile:
        Original file name

    outfile:
        Output file name

    extra_cols:
        Extra columns to copy

    Notes
    -----
    This will produce a DataFrame with at least these columns:

    +-----------------------------+---------------------------------------------------------------+
    | Column name                 | Description                                                   |
    +=============================+===============================================================+
    | id                          | source ID                                                     |
    +-----------------------------+---------------------------------------------------------------+
    | tract                       | Tract source was found in                                     |
    +-----------------------------+---------------------------------------------------------------+
    | patch                       | Patch source was found in                                     |
    +-----------------------------+---------------------------------------------------------------+
    | ra                          | RA in degrees                                                 |
    +-----------------------------+---------------------------------------------------------------+
    | dec                         | DEC in degress                                                |
    +-----------------------------+---------------------------------------------------------------+
    | snr                         | Signal-to-Noise of source, used for filtering and centroiding |
    +-----------------------------+---------------------------------------------------------------+
    | {band}_gaussFlux            | Flux, for band in u,g,r,i,z,y                                 |
    +-----------------------------+---------------------------------------------------------------+
    | {band}_gaussFluxErr         | Flux error, for band in u,g,r,i,z,y                           |
    +-----------------------------+---------------------------------------------------------------+

    """
    t = tables_io.read(basefile)
    cols = [
        "tract", "ra", "dec", "shearObjectId", "gauss_snr",
        "patch_x", "patch_y", "x_cell_coadd", "y_cell_coadd",
        "x_pix", "y_pix", "cell_idx_x", "cell_idx_y",
        "id", "shear", "meta_step",
    ]
    cols += [f"{band}_gaussFlux" for band in "griz"]
    cols += [f"{band}_gaussFluxErr" for band in "griz"]
    if extra_cols is not None:
        cols += extra_cols

    tout = t[cols].copy(deep=True)

    tout["snr"] = tout["gauss_snr"]

    tout.to_parquet(outfile)


def reduceDESCMDTable(
    basefile: str,
    outfile: str,
    extra_cols: list[str] | None = None,
) -> None:  # pragma: no cover
    """Reduce an object table to just the colums needed for matching

    Parameters
    ----------
    basefile:
        Original file name

    outfile:
        Output file name

    extra_cols:
        Extra columns to copy

    Notes
    -----
    This will produce a DataFrame with at least these columns:

    +-----------------------------+---------------------------------------------------------------+
    | Column name                 | Description                                                   |
    +=============================+===============================================================+
    | id                          | source ID                                                     |
    +-----------------------------+---------------------------------------------------------------+
    | tract                       | Tract source was found in                                     |
    +-----------------------------+---------------------------------------------------------------+
    | patch                       | Patch source was found in                                     |
    +-----------------------------+---------------------------------------------------------------+
    | ra                          | RA in degrees                                                 |
    +-----------------------------+---------------------------------------------------------------+
    | dec                         | DEC in degress                                                |
    +-----------------------------+---------------------------------------------------------------+
    | snr                         | Signal-to-Noise of source, used for filtering and centroiding |
    +-----------------------------+---------------------------------------------------------------+
    | {band}_gaussFlux            | Flux, for band in u,g,r,i,z,y                                 |
    +-----------------------------+---------------------------------------------------------------+
    | {band}_gaussFluxErr         | Flux error, for band in u,g,r,i,z,y                           |
    +-----------------------------+---------------------------------------------------------------+

    """
    t = tables_io.read(basefile)
    cols = [
        "patch_x", "patch_y", "ra", "dec", "id", "s2n",
        "x_cell_coadd", "y_cell_coadd", "x_pix", "y_pix",
        "cell_idx_x", "cell_idx_y", "shear", "meta_step",
    ]
    cols += [f"flux_{band}" for band in "riz"]
    cols += [f"flux_err_{band}" for band in "riz"]
    if extra_cols is not None:
        cols += extra_cols

    tout = t[cols].copy(deep=True)

    tout["snr"] = tout["s2n"]

    tout.to_parquet(outfile)
