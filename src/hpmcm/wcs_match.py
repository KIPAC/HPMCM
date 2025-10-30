from __future__ import annotations

from typing import Any

import numpy as np
import pandas
from astropy import wcs

from .match import Match


def createGlobalWcs(
    ref_dir: tuple[float, float],
    pix_size: float,
    n_pix: np.ndarray,
) -> wcs.WCS:
    """Helper function to create the WCS used to project the
    sources in a skymap


    Parameters
    ----------
    ref_dir:
        Reference Direction (RA, DEC) in degrees

    pix_size:
        Pixel size in degrees

    n_pix:
        Number of pixels in x, y

    Returns
    -------
    WCS to create the pixel grid
    """
    w = wcs.WCS(naxis=2)
    w.wcs.cdelt = [-pix_size, pix_size]
    w.wcs.crpix = [n_pix[0] / 2, n_pix[1] / 2]
    w.wcs.crval = [ref_dir[0], ref_dir[1]]
    return w


class WcsMatch(Match):
    """Class to do N-way matching using a provided WCS.

    This uses the provided WCS to define a Skymap that covers the full region
    begin matched.

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames
    that contain the following columns.

    +-------------+---------------------------------------------------------------+
    | Column name | Description                                                   |
    +=============+===============================================================+
    | id          | source ID                                                     |
    +-------------+---------------------------------------------------------------+
    | ra          | RA in degrees                                                 |
    +-------------+---------------------------------------------------------------+
    | dec         | DEC in degress                                                |
    +-------------+---------------------------------------------------------------+
    | snr         | Signal-to-Noise of source, used for filtering and centroiding |
    +-------------+---------------------------------------------------------------+
    """

    def __init__(
        self,
        match_wcs: wcs.WCS,
        **kwargs: Any,
    ):
        self.wcs: wcs.WCS = match_wcs
        pixel_size: float = self.wcs.wcs.cdelt[1]
        n_pix_side: np.ndarray = np.ceil(2 * np.array(self.wcs.wcs.crpix)).astype(int)
        Match.__init__(self, pixel_size=pixel_size, n_pix_side=n_pix_side, **kwargs)

    @classmethod
    def create(
        cls,
        ref_dir: tuple[float, float],
        region_size: tuple[float, float],
        pixel_size: float,
        **kwargs: Any,
    ) -> WcsMatch:
        """Helper function to create a Match object.

        This will use the provided arguments to create a WCS, and then
        create and return a `WcsMatch` matcher.

        Parameters
        ----------
        ref_dir:
            Reference Direction (RA, DEC) in degrees

        region_size:
            Size of match region, in degrees

        pix_size:
            Pixel size in degrees

        Returns
        -------
        Object to create matches for the requested region
        """
        n_pix = (np.array(region_size) / pixel_size).astype(int)
        match_wcs = createGlobalWcs(ref_dir, pixel_size, n_pix)
        return cls(match_wcs, n_pixels=n_pix, **kwargs)

    def _getPixValues(self, df: pandas.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_pix, y_pix = self.wcs.wcs_world2pix(df["ra"].values, df["dec"].values, 0)
        return x_pix, y_pix

    def pixToWorld(
        self,
        x_pix: np.ndarray,
        y_pix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert local coords in pixels to world coordinates (RA, DEC)"""
        assert self.wcs is not None
        return self.wcs.wcs_pix2world(x_pix, y_pix, 0)

    def reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame

        Notes
        -----
        This applies a trivial cut on signal-to-noise (snr>1).
        and adds the pixel coordinates to the DataFrame.

        This will add these columns to the output dataframes


        +--------------+-------------------------------------+
        | Column       | Description                         |
        +==============+=====================================+
        | id           | Index of object inside catalog      |
        +--------------+-------------------------------------+
        | ra           | Source RA                           |
        +--------------+-------------------------------------+
        | dec          | Source DEC                          |
        +--------------+-------------------------------------+
        | x_pix        | X-coordinate in global WCS frame    |
        +--------------+-------------------------------------+
        | y_pix        | Y-coordinate in global WCS frame    |
        +--------------+-------------------------------------+
        | snr          | Signal-to-noise ratio               |
        +--------------+-------------------------------------+

        """
        df_clean = df[(df.snr > 1)]
        x_pix, y_pix = self.wcs.wcs_world2pix(
            df_clean["ra"].values, df_clean["dec"].values, 0
        )
        df_clean["x_pix"] = x_pix
        df_clean["y_pix"] = y_pix
        filtered = (
            (df_clean.x_pix >= 0)
            & (df_clean.x_pix < self.n_pix_side[0])
            & (df_clean.y_pix >= 0)
            & (df_clean.y_pix < self.n_pix_side[1])
        )

        df_red = df_clean[filtered].copy(deep=True)

        return df_red[
            [
                "id",
                "ra",
                "dec",
                "x_pix",
                "y_pix",
                "snr",
            ]
        ]
