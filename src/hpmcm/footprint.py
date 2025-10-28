from __future__ import annotations

import numpy as np
from scipy import ndimage
from photutils.segmentation import detect_sources


class Footprint:
    """Wraps the slices returns by `ndimage.find_objects`

    Attributes
    ----------
    image: np.ndaray
        Original image

    sliceX: slice
        Slice of footprint in X

    sliceY: slice
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
        self.sliceX = slices[0]
        self.sliceY = slices[1]
        self.cutout = self.image[self.sliceX, self.sliceY]

    def extent(self) -> tuple[int, int, int, int]:
        """Return the extent of the Footprint, for use by `matplotlib`"""
        return (
            self.sliceX.start,
            self.sliceX.stop,
            self.sliceY.start,
            self.sliceY.stop,
        )


class FootprintSet:
    """Wraps Footprints detected by `photutils.segmentation.detect_sources`

    Attributes
    ----------
    image: np.ndarray
        Original image

    footprints: list[Footprint]
        Footprints found in the image

    fpKey: np.ndarray
        Map of footprint associations: -1 -> no association
    """

    def __init__(
        self, image: np.ndarray, fpKey: np.ndarray, footprints: list[Footprint]
    ):
        """Create a FootprintSet

        Parameters
        ----------
        image:
            Original image (counts map of source positions)

        fpKey:
            Map of footprint associations: -1 -> no association

        footprints:
            List of Footprints in this set
        """
        self.image = image
        self.footprints = footprints
        self.fpKey = fpKey

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
        fpKey = detect_sources(image, 0.5, 1).data
        slices = ndimage.find_objects(fpKey)
        fpKey = fpKey - 1
        footprints: list[Footprint] = []
        for slicePair in slices:
            footprints.append(Footprint(image, slicePair))
        return cls(image, fpKey, footprints)

    def filter(self, buf: int, pixelMatchScale: int = 1) -> FootprintSet:
        """Remove footprints outside the central region of cell"""
        if buf == 0:
            return self

        # The easiest way to do this is to mask out the outer
        # region in the footprint key
        nExclude = np.floor(buf/pixelMatchScale).astype(int)
        nX, nY = self.fpKey.shape
        self.fpKey[0:nExclude] = -1
        self.fpKey[nX-nExclude:nX] = -1
        self.fpKey[:,0:nExclude] = -1
        self.fpKey[:,nY-nExclude:nY] = -1
        return self
