from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas
import pyarrow.parquet as pq
from astropy import wcs
from astropy.io import fits
from astropy.table import vstack

from .cell import CellData
from .cluster import ClusterData

COLUMNS = ["ra", "dec", "id", "patch_x", "patch_y", "cell_x", "cell_y", "row", "col"]


def createGlobalWcs(
    refDir: tuple[float, float],
    pixSize: float,
    nPix: np.ndarray,
) -> wcs.WCS:
    """Helper function to create the WCS used to project the
    sources in a skymap"""
    w = wcs.WCS(naxis=2)
    w.wcs.cdelt = [-pixSize, pixSize]
    w.wcs.crpix = [nPix[0] / 2, nPix[1] / 2]
    w.wcs.crval = [refDir[0], refDir[1]]
    return w


def clusterStats(clusterDict: OrderedDict[int, ClusterData]) -> np.ndarray:
    """Helper function to get stats about the clusters

    'Orphan'   means single source clusters (i.e., single detections)
    'Mixed`    means there is more that one source from at least one
               input catalog
    'Confused' means there are more than four cases of duplication
    """
    nOrphan = 0
    nMixed = 0
    nConfused = 0
    for val in clusterDict.values():
        if val.nSrc == 1:
            nOrphan += 1
        if val.nSrc != val.nUnique:
            nMixed += 1
            if val.nSrc > val.nUnique + 3:
                nConfused += 1
    return np.array([len(clusterDict), nOrphan, nMixed, nConfused])


class Match:
    """Class to do N-way matching

    Uses a provided WCS to define a Skymap that covers the full region
    begin matched.

    Uses that WCS to assign pixel locations to all sources in the input catalogs

    Iterates over cells and does source clustering in each cell
    using Footprint detection on a Skymap of source counts per pixel.

    Assigns each input source to a cluster.

    At that stage the clusters are not the final product as they can include
    more than one soruce from a given catalog.

    Loops over clusters and processes each cluster to resolve confusion.

    If there is not a unqiue source per-catalog redo the clustering with
    half-size pixels to try to split the cluster (down to minimum pixel scale)

    Parameters
    ----------
    _redData : `list`, [`Dataframe`]
        Reduced dataframes with only the columns needed for matching

    _clusters : `OrderedDict`, [`tuple`, `CellData`]
        Dictionary providing access to cell data
    """

    def __init__(
        self,
        matchWcs: wcs.WCS,
        **kwargs: Any,
    ):
        self._wcs: wcs.WCS = matchWcs
        self._pixSize: float = self._wcs.wcs.cdelt[1]
        self._nPixSide: np.ndarray = np.ceil(2 * np.array(self._wcs.wcs.crpix)).astype(
            int
        )
        self._cellSize: int = kwargs.get("cellSize", 3000)
        self._cellBuffer: int = kwargs.get("cellBuffer", 10)
        self._cellMaxObject: int = kwargs.get("cellMaxObject", 100000)
        self._pixelR2Cut: float = kwargs.get("pixelR2Cut", 1.0)
        self._maxSubDivision: int = kwargs.get("maxSubDivision", 3)
        self._catType: str = kwargs.get("catalogType", "wmom")
        self._nCell: np.ndarray = np.ceil(self._nPixSide / self._cellSize)
        self._redData: OrderedDict[int, pandas.DataFrame] = OrderedDict()
        self._clusters: OrderedDict[tuple[int, int], CellData] = OrderedDict()

    @classmethod
    def create(
        cls,
        refDir: tuple[float, float],
        regionSize: tuple[float, float],
        pixSize: float,
        **kwargs: Any,
    ) -> Match:
        """Make an `NWayMatch` object from inputs"""
        nPix = (np.array(regionSize) / pixSize).astype(int)
        matchWcs = createGlobalWcs(refDir, pixSize, nPix)
        return cls(matchWcs, **kwargs)

    @property
    def redData(self) -> OrderedDict[int, pandas.DataFrame]:
        """Return the dictionary of reduced data, i.e., just the columns
        need for matching"""
        return self._redData

    @property
    def nCell(self) -> np.ndarray:
        """Return the number of cells in X,Y"""
        return self._nCell

    def pixToArcsec(self) -> float:
        """Convert pixel size (in degrees) to arcseconds"""
        return 3600.0 * self._pixSize

    def pixToWorld(
        self,
        xPix: np.ndarray,
        yPix: np.ndarray,
    ) -> np.ndarray:
        """Convert locals in pixels to world coordinates (RA, DEC)"""
        return self._wcs.wcs_pix2world(xPix, yPix, 0)

    def getIdOffset(
        self,
        ix: int,
        iy: int,
    ) -> int:
        """Get the ID offset to use for a given cell"""
        cellIdx = self._nCell[1] * ix + iy
        return int(self._cellMaxObject * cellIdx)

    def reduceData(
        self,
        inputFiles: list[str],
        visitIds: list[int],
    ) -> None:
        """Read input files and filter out only the columns we need"""
        for fName, vid in zip(inputFiles, visitIds):
            self._redData[vid] = self._reduceDataFrame(fName)

    def analyzeCell(
        self,
        ix: int,
        iy: int,
        fullData: bool = False,
    ) -> dict | None:
        """Analyze a single cell

        Returns an OrderedDict

        'srd' : `CellData`
            The analysis data for the Cell

        if fullData is True the return dict will include

        'image' : `afwImage.ImageI`
            Image of cell source counts map
        'countsMap' : `np.array`
            Numpy array with same
        'clusters' : `afwDetect.FootprintSet`
            Clusters as dectected by finding FootprintSet on source counts map
        'clusterKey' : `afwImage.ImageI`
            Map of cell with pixels filled with index of
            associated Footprints
        """
        iCell = np.array([ix, iy])
        cellStep = np.array([self._cellSize, self._cellSize])
        corner = iCell * cellStep
        idOffset = self.getIdOffset(ix, iy)
        cellData = CellData(self, idOffset, corner, cellStep, self._cellBuffer)
        cellData.reduceData(list(self._redData.values()))
        oDict = cellData.analyze(pixelR2Cut=self._pixelR2Cut)
        if oDict is None:
            return None
        if fullData:
            oDict["cellData"] = cellData
            return oDict
        if cellData.nObjects >= self._cellMaxObject:
            print("Too many object in a cell", cellData.nObjects, self._cellMaxObject)
        return dict(cellData=cellData)

    def analysisLoop(self) -> None:
        """Does clustering for all cells"""
        self._clusters.clear()

        for ix in range(int(self._nCell[0])):
            sys.stdout.write(f"{ix:%2i}")
            sys.stdout.flush()
            for iy in range(int(self._nCell[1])):
                sys.stdout.write(".")
                sys.stdout.flush()
                iCell = (ix, iy)
                odict = self.analyzeCell(ix, iy)
                if odict is None:
                    continue
                cellData = odict["cellData"]
                self._clusters[iCell] = cellData
            sys.stdout.write("!\n")

    def extractStats(self) -> Any:
        """Extracts cluster statisistics"""
        clusterAssocTables = []
        objectAssocTables = []
        clusterStatsTables = []
        objectStatsTables = []

        for ix in range(int(self._nCell[0])):
            for iy in range(int(self._nCell[1])):
                iCell = (ix, iy)
                cellData = self._clusters[iCell]
                clusterAssocTables.append(cellData.getClusterAssociations())
                objectAssocTables.append(cellData.getObjectAssociations())
                clusterStatsTables.append(cellData.getClusterStats())
                objectStatsTables.append(cellData.getObjectStats())
        hduList = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.table_to_hdu(vstack(clusterAssocTables)),
                fits.table_to_hdu(vstack(objectAssocTables)),
                fits.table_to_hdu(vstack(clusterStatsTables)),
                fits.table_to_hdu(vstack(objectStatsTables)),
            ]
        )
        return hduList

    def printSummaryStats(self) -> np.ndarray:
        """Helper function to print info about clusters"""
        stats = np.zeros((4), int)
        for key, cellData in self._clusters.items():
            cellStats = clusterStats(cellData.clusterDict)
            print(
                f"{key[0]:%3} "
                f"{key[1]:%3}: "
                f"{cellStats[0]:%8i} "
                f"{cellStats[1]:%8i} "
                f"{cellStats[2]:%8i} "
                f"{cellStats[3]:%8i}"
            )
            stats += cellStats
        return stats

    def _reduceDataFrame(
        self,
        fName: str,
    ) -> pandas.DataFrame:
        """Read and reduce a single input file"""
        columns = COLUMNS.copy()
        if self._catType == "wmom":
            columns += [
                "wmom_band_flux_r",
                "wmom_band_flux_err_r",
                "wmom_g_1",
                "wmom_g_2",
            ]
        elif self._catType == "gauss":
            columns += ["gauss_band_mag_r", "gauss_mag_r_err", "gauss_g_1", "gauss_g_2"]
        # parq = pq.read_pandas(fName, columns=columns)
        parq = pq.read_pandas(fName)
        df = parq.to_pandas()
        if self._catType == "wmom":
            df["SNR"] = df["wmom_band_flux_r"] / df["wmom_band_flux_err_r"]
        elif self._catType == "gauss":
            # df['SNR'] = df['PsFlux']/df['PsFluxErr']
            df["SNR"] = 1.0 / df["gauss_mag_r_err"]
        # select sources that have SNR > 5.
        # You may start with 10 or even 50 if you want to start with just the brightest objects
        # AND
        # Centroid_flag is True if there was a problem fitting the position (centroid)
        # AND
        # sky_source is True if it is a measurement of blank sky.
        # sky_sources should have SNR < 5 or the Centroid_flag set,
        # but explicitly filter just to make sure.
        # AND
        # detect_isPrimary = True to remove duplicate rows from deblending:
        # If a source has been deblended, the parent is marked detect_isPrimary=False and its children True.
        # df_clean = df[(df.SNR > 5) & ~df.Centroid_flag & ~df.sky_source & df.detect_isPrimary]
        df_clean = df[(df.SNR > 1)]
        xcell, ycell = self._wcs.wcs_world2pix(
            df_clean["ra"].values, df_clean["dec"].values, 0
        )
        # df_red = df_clean[
        #    [
        #        "ra",
        #        "dec",
        #        "SNR",
        #        "id",
        #        "patch_x",
        #        "patch_y",
        #        "cell_x",
        #        "cell_y",
        #        "row",
        #        "col",
        #        f"{self._catType}_g_1",
        #        f"{self._catType}_g_2",
        #    ]
        # ].copy(deep=True)
        df_red = df_clean.copy(deep=True)

        idx_x = 20 * df_red["patch_x"].values + df_red["cell_x"].values
        idx_y = 20 * df_red["patch_y"].values + df_red["cell_y"].values
        cent_x = 150 * idx_x - 75
        cent_y = 150 * idx_y - 75
        local_x = df_red["col"] - cent_x
        local_y = df_red["row"] - cent_y

        df_red["xcell"] = xcell
        df_red["ycell"] = ycell
        df_red["local_x"] = local_x
        df_red["local_y"] = local_y
        return df_red
        # return df_red[
        #    [
        #        "ra",
        #        "dec",
        #        "SNR",
        #        "id",
        #        "xcell",
        #        "ycell",
        #        "local_x",
        #        "local_y",
        #        f"{self._catType}_g_1",
        #        f"{self._catType}_g_2",
        #    ]
        # ]
