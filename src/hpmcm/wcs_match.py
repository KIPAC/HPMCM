from __future__ import annotations

from typing import Any

import numpy as np
import pandas
from astropy import wcs

from .match import Match


def createGlobalWcs(
    refDir: tuple[float, float],
    pixSize: float,
    nPix: np.ndarray,
) -> wcs.WCS:
    """Helper function to create the WCS used to project the
    sources in a skymap


    Parameters
    ----------
    refDir:
        Reference Direction (RA, DEC) in degrees

    pixSize:
        Pixel size in degrees

    nPix:
        Number of pixels in x, y


    Returns
    -------
    wcs.WCS
        WCS to create the pixel grid
    """
    w = wcs.WCS(naxis=2)
    w.wcs.cdelt = [-pixSize, pixSize]
    w.wcs.crpix = [nPix[0] / 2, nPix[1] / 2]
    w.wcs.crval = [refDir[0], refDir[1]]
    return w


class WcsMatch(Match):
    """Class to do N-way matching

    Uses a provided WCS to define a Skymap that covers the full region
    begin matched.

    Uses that WCS to assign pixel locations to all sources in the input catalogs

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames
    that contain the following columns.

    "id" : source ID
    "ra" : RA in degrees
    "dec": DEC in degress
    "SNR": Signal-to-Noise of source, used for filtering and centroiding
    """

    def __init__(
        self,
        matchWcs: wcs.WCS,
        **kwargs: Any,
    ):
        self._wcs: wcs.WCS = matchWcs
        pixSize: float = self._wcs.wcs.cdelt[1]
        nPixSide: np.ndarray = np.ceil(2 * np.array(self._wcs.wcs.crpix)).astype(int)
        Match.__init__(self, pixSize=pixSize, nPixSide=nPixSide, **kwargs)

    @classmethod
    def create(
        cls,
        refDir: tuple[float, float],
        regionSize: tuple[float, float],
        pixSize: float,
        **kwargs: Any,
    ) -> Match:
        """Helper function to create a Match object

        Parameters
        ----------
        refDir:
            Reference Direction (RA, DEC) in degrees

        pixSize:
            Pixel size in degrees

        Returns
        -------
        Match:
            Object to create matches for the requested region
        """
        nPix = (np.array(regionSize) / pixSize).astype(int)
        matchWcs = createGlobalWcs(refDir, pixSize, nPix)
        return cls(matchWcs, nPixels=nPix, **kwargs)

    @property
    def wcs(self) -> wcs.WCS:
        """Return the WCS used to pixelize the region"""
        return self._wcs

    def _getPixValues(self, df: pandas.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xPix, yPix = self._wcs.wcs_world2pix(df["ra"].values, df["dec"].values, 0)
        return xPix, yPix

    def pixToWorld(
        self,
        xPix: np.ndarray,
        yPix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert locals in pixels to world coordinates (RA, DEC)"""
        assert self._wcs is not None
        return self._wcs.wcs_pix2world(xPix, yPix, 0)

    def _reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame"""
        df_clean = df[(df.SNR > 1)]
        xPix, yPix = self._wcs.wcs_world2pix(
            df_clean["ra"].values, df_clean["dec"].values, 0
        )
        df_red = df_clean.copy(deep=True)

        df_red["xPix"] = xPix
        df_red["yPix"] = yPix

        return df_red[
            [
                "id",
                "ra",
                "dec",
                "xPix",
                "yPix",
                "SNR",
            ]
        ]
