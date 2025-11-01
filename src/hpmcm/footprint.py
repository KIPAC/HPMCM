from __future__ import annotations

import numpy as np
from photutils.segmentation import detect_sources
from scipy import ndimage


class Footprint:
    """Wraps the slices returns by `ndimage.find_objects`

    Attributes
    ----------
    image: np.ndaray
        Original image

    slice_x: slice
        Slice of footprint in X

    slice_y: slice
        Slice of footprint in Y

    cutout: np.ndaray
        Image cutout
    """

    def __init__(self, image: np.ndarray, slices: tuple[slice, slice]):
        """Build a Footprint

        Parameters
        ----------
        image:
            Original image (counts map of source positions)

        slice:
            Slices in X and Y
        """
        self.image = image
        self.slice_x = slices[0]
        self.slice_y = slices[1]
        self.cutout = self.image[self.slice_x, self.slice_y]

    def extent(self) -> tuple[int, int, int, int]:
        """Return the extent of the Footprint, for use by `matplotlib`"""
        return (
            self.slice_x.start,
            self.slice_x.stop,
            self.slice_y.start,
            self.slice_y.stop,
        )


class FootprintSet:
    """Wraps Footprints detected by `photutils.segmentation.detect_sources`

    Attributes
    ----------
    image: np.ndarray
        Original image

    footprints: list[Footprint]
        Footprints found in the image

    fp_key: np.ndarray
        Map of footprint associations: -1 -> no association
    """

    def __init__(
        self, image: np.ndarray, fp_key: np.ndarray, footprints: list[Footprint]
    ):
        """Create a FootprintSet

        Parameters
        ----------
        image:
            Original image (counts map of source positions)

        fp_key:
            Map of footprint associations: -1 -> no association

        footprints:
            List of Footprints in this set
        """
        self.image = image
        self.footprints = footprints
        self.fp_key = fp_key

    @classmethod
    def detect(cls, image: np.ndarray) -> FootprintSet:
        """Create a FootprintSet from a countsMap

        Parameters
        ----------
        image:
            Original image (counts map of source positions)

        Returns
        -------
        Newly created FootprintSet
        """
        fp_key = detect_sources(image, 0.5, 1).data
        slices = ndimage.find_objects(fp_key)
        fp_key = fp_key - 1
        footprints: list[Footprint] = []
        for slice_pair_ in slices:
            footprints.append(Footprint(image, slice_pair_))
        return cls(image, fp_key, footprints)

    def filter(self, buf: int, pixel_match_scale: int = 1) -> FootprintSet:
        """Remove footprints outside the central region of cell"""
        if buf == 0:
            return self

        # The easiest way to do this is to mask out the outer
        # region in the footprint key
        n_exclude = np.floor(buf / pixel_match_scale).astype(int)
        n_x, n_y = self.fp_key.shape
        self.fp_key[0:n_exclude] = -1
        self.fp_key[n_x - n_exclude : n_x] = -1
        self.fp_key[:, 0:n_exclude] = -1
        self.fp_key[:, n_y - n_exclude : n_y] = -1
        return self
