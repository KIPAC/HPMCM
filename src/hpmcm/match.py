from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Iterable
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


def clusterStats(clusterDict: OrderedDict[int, ClusterData]) -> np.ndarray:
    """Helper function to get stats about the clusters

    Parameters
    ----------
    clusterDict:
        Dict from clusterId to ClusterData object

    Returns
    -------
    nClusters, nOrphans, nMixed, nConfused

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
    _fullData: list[DataFrame]
        Full input DataFrames

    _redData : list[DataFrame]
        Reduced DataFrames with only the columns needed for matching

    _cellDict : OrderedDict[tuple[int, int], CellData]
        Dictionary providing access to cell data

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames
    that contain the following columns.

    "id" : source ID
    "ra" : RA in degrees
    "dec": DEC in degress
    "SNR": Signal-to-Noise of source, used for filtering and centroiding

    Optional, used in post-processing:
    "xCell_coadd": X-postion in cell-based coadd used for metadetect
    "yCell_coadd": Y-postion in cell-based coadd used for metadetect
    "g_1": shape measurement
    "g_2": shape measurement
    """

    def __init__(
        self,
        matchWcs: wcs.WCS | None,
        **kwargs: Any,
    ):
        self._wcs: wcs.WCS | None = matchWcs
        if self._wcs is not None:
            self._pixSize: float = self._wcs.wcs.cdelt[1]
            self._nPixSide: np.ndarray = np.ceil(
                2 * np.array(self._wcs.wcs.crpix)
            ).astype(int)
        else:
            self._pixSize = kwargs.get("pixelSize", 1.0)
            self._nPixSide = kwargs["nPixels"]
        self._cellSize: int = kwargs.get("cellSize", 3000)
        self._cellBuffer: int = kwargs.get("cellBuffer", 10)
        self._cellMaxObject: int = kwargs.get("cellMaxObject", 100000)
        self._pixelR2Cut: float = kwargs.get("pixelR2Cut", 1.0)
        self._maxSubDivision: int = kwargs.get("maxSubDivision", 3)
        self._pixelMatchScale: int = kwargs.get("pixelMatchScale", 1)
        self._catType: str = kwargs.get("catalogType", "wmom")
        self._nCell: np.ndarray = np.ceil(self._nPixSide / self._cellSize)

        self._fullData: OrderedDict[int, pandas.DataFrame] = OrderedDict()
        self._redData: OrderedDict[int, pandas.DataFrame] = OrderedDict()
        self._cellDict: OrderedDict[tuple[int, int], CellData] = OrderedDict()

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
        use_wcs = kwargs.pop("useWCS", True)
        if use_wcs:
            matchWcs = createGlobalWcs(refDir, pixSize, nPix)
        else:
            matchWcs = None
            kwargs["nPixels"] = nPix
        return cls(matchWcs, **kwargs)

    @classmethod
    def createCoaddCellsForTract(
        cls,
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
        nPix = np.array([30000, 30000])
        matchWcs = None
        kw = dict(
            pixelSize=0.2*3600.,
            nPixels=nPix,
            cellSize=150,
            cellBuffer=25,
            cellMaxObject=1000,
        )
        return cls(matchWcs, **kw, **kwargs)

    @property
    def fullData(self) -> OrderedDict[int, pandas.DataFrame]:
        """Return the dictionary of full data as passed to the Match object"""
        return self._fullData

    @property
    def redData(self) -> OrderedDict[int, pandas.DataFrame]:
        """Return the dictionary of reduced data, i.e., just the columns
        need for matching"""
        return self._redData

    @property
    def wcs(self) -> wcs.WCS:
        """Return the WCS used to pixelize the region"""
        return self._wcs

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
        assert self._wcs is not None
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
        """Read input files and filter out only the columns we need

        Each input file should have an associated visitId.
        This is used to test if we have more than one-source
        per input catalog.

        If the inputs files have a pre-defined ID associated with them
        that can be used.   Otherwise it is fine just to give a range from
        0 to nInputs.
        """
        for fName, vid in zip(inputFiles, visitIds):
            self._fullData[vid] = self._readDataFrame(fName)
            self._redData[vid] = self._reduceDataFrame(self._fullData[vid])
            self._fullData[vid].set_index('id', inplace=True)

    def analyzeCell(
        self,
        ix: int,
        iy: int,
        fullData: bool = False,
    ) -> dict | None:
        """Analyze a single cell

        Parameters
        ----------
        ix:
            Cell index in x-coord

        iy:
            Cell index in y-coord

        Returns
        -------
        dict:
            Dict with the information listed below

        'cellData' : `CellData`
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
        cellKey = (ix, iy)
        iCell = np.array(cellKey).astype(int)
        cellStep = np.array([self._cellSize, self._cellSize])
        corner = iCell * cellStep
        idOffset = self.getIdOffset(ix, iy)
        cellData = CellData(self, idOffset, corner, cellStep, iCell, self._cellBuffer)
        cellData.reduceData(list(self._redData.values()))
        oDict = cellData.analyze(pixelR2Cut=self._pixelR2Cut)
        if cellData.nObjects >= self._cellMaxObject:
            print("Too many object in a cell", cellData.nObjects, self._cellMaxObject)

        self._cellDict[cellKey] = cellData
        if oDict is None:
            return None
        if fullData:
            oDict["cellData"] = cellData
            return oDict

        return dict(cellData=cellData)

    def analysisLoop(
        self, xRange: Iterable | None = None, yRange: Iterable | None = None
    ) -> None:
        """Does matching for all cells"""
        self._cellDict.clear()

        if xRange is None:
            xRange = range(int(self._nCell[0]))
        if yRange is None:
            yRange = range(int(self._nCell[1]))

        for ix in xRange:
            for iy in yRange:
                odict = self.analyzeCell(ix, iy)
                if odict is None:
                    continue
            if ix == 0:
                pass
            elif ix % 10 == 0:
                sys.stdout.write(f" {ix}!\n")
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()

        sys.stdout.write(" Done!\n")
        sys.stdout.flush()


    def extractShearStats(self) -> list[pandas.DataFrame]:
        clusterShearStatsTables = []
        objectShearStatsTables = []

        for ix in range(int(self._nCell[0])):
            for iy in range(int(self._nCell[1])):
                iCell = (ix, iy)
                if iCell not in self._cellDict:
                    continue
                cellData = self._cellDict[iCell]
                clusterShearStatsTables.append(cellData.getClusterShearStats())
                objectShearStatsTables.append(cellData.getObjectShearStats())

        return [
            pandas.concat(clusterShearStatsTables),
            pandas.concat(objectShearStatsTables),
        ]

        
    def extractStats(self) -> list[pandas.DataFrame]:
        """Extracts cluster statisistics"""
        clusterAssocTables = []
        objectAssocTables = []
        clusterStatsTables = []
        objectStatsTables = []

        for ix in range(int(self._nCell[0])):
            for iy in range(int(self._nCell[1])):
                iCell = (ix, iy)
                if iCell not in self._cellDict:
                    continue
                cellData = self._cellDict[iCell]
                clusterAssocTables.append(cellData.getClusterAssociations())
                objectAssocTables.append(cellData.getObjectAssociations())
                clusterStatsTables.append(cellData.getClusterStats())
                objectStatsTables.append(cellData.getObjectStats())

        return [
            pandas.concat(clusterAssocTables),
            pandas.concat(objectAssocTables),
            pandas.concat(clusterStatsTables),
            pandas.concat(objectStatsTables),
        ]

    def printSummaryStats(self) -> np.ndarray:
        """Helper function to print info about clusters"""
        stats = np.zeros((4), int)
        for key, cellData in self._cellDict.items():
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

    def _readDataFrame(
        self,
        fName: str,
    ) -> pandas.DataFrame:
        """Read a single input file"""
        parq = pq.read_pandas(fName)
        df = parq.to_pandas()
        return df

    def _reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame"""
        df_clean = df[(df.SNR > 1)]
        if self._wcs is not None:
            xPix, yPix = self._wcs.wcs_world2pix(
                df_clean["ra"].values, df_clean["dec"].values, 0
            )
        else:
            xPix, yPix = (
                df_clean["col"].values + 25,
                df_clean["row"].values + 25,
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
                "xCell_coadd",
                "yCell_coadd",
                "SNR",
                "g_1",
                "g_2",
                "idx_x",
                "idx_y",
            ]
        ]


    def classifyClusters(self, **kwargs: Any) -> dict[str, list]:

        nsrcs = []

        cut1 = []
        cut2 = []

        used = []
        ideal_faint = []        
        ideal = []

        faint = []
        edge_mixed = []
        mixed = []
        edge_missing = []
        edge_extra = []

        missing = []
        two_missing = []
        many_missing = []
        extra = []
        caught = []

        cell_edge = kwargs.get('cellEdge', 75)        
        edge_cut = kwargs.get('edgeCut', 2)
        snr_cut = kwargs.get('SNRCut', 7.5)

        n_cat = len(self._redData)
        
        for iC, cellData in self._cellDict.items():
            cd = cellData.clusterDict
            
            for key, c in cd.items():
                k = (iC, key)

                nsrcs.append(c._nSrc)    

                if (np.fabs(c.data.xCell_coadd) > cell_edge).all() or (np.fabs(c.data.yCell_coadd) > cell_edge).all():
                    cut1.append(k)
                    continue
                if np.fabs(c.data.xCell_coadd.mean()) > cell_edge or np.fabs(c.data.yCell_coadd.mean()) > cell_edge:
                    cut2.append(k)
                    continue

                used.append(k)
        
                edge_case = False
                is_faint = False
                if (np.fabs(c._data.xCell_coadd) > cell_edge-edge_cut).any() or (np.fabs(c._data.yCell_coadd) > cell_edge-edge_cut).any():
                    edge_case = True
                if (c._data.SNR < snr_cut).any():
                    is_faint = True
                                        
                if c._nSrc == c._nUnique and c._nSrc == n_cat and is_faint:
                    ideal_faint.append(k)
                elif c._nSrc == c._nUnique and c._nSrc == n_cat:
                    ideal.append(k)
                elif c._nSrc < n_cat and is_faint:
                    faint.append(k)                    
                elif c._nSrc == n_cat and c._nUnique != n_cat and edge_case:
                    edge_mixed.append(k)
                elif c._nSrc == n_cat and c._nUnique != n_cat:
                    mixed.append(k)
                elif c._nSrc < n_cat and edge_case:
                    edge_missing.append(k)
                elif c._nSrc > n_cat and edge_case:
                    edge_extra.append(k)    
                elif c._nSrc == n_cat - 1:
                    missing.append(k)
                elif c._nSrc == n_cat - 2:
                    two_missing.append(k)
                elif c._nSrc < n_cat - 2:
                    many_missing.append(k)
                elif c._nSrc > n_cat:
                    extra.append(k)
                else:
                    caught.append(k)

        return dict(
            nsrcs=nsrcs,
            cut1=cut1,
            cut2=cut2,
            used=used,
            ideal_faint=ideal_faint,
            ideal=ideal,
            faint=faint,
            edge_mixed=edge_mixed,
            mixed=mixed,
            edge_missing=edge_missing,
            edge_extra=edge_extra,
            missing=missing,
            two_missing=two_missing,
            many_missing=many_missing,
            extra=extra,
            caught=caught,
        )
    
        
    
    def classifyObjects(self, **kwargs: Any) -> dict[str, list]:

        nsrcs = []

        cut1 = []
        cut2 = []

        used = []
        ideal_faint = []        
        ideal = []

        faint = []
        edge_mixed = []
        mixed = []
        edge_missing = []
        edge_extra = []

        orphan = []
        missing = []
        two_missing = []
        many_missing = []
        extra = []
        caught = []

        cell_edge = kwargs.get('cellEdge', 75)        
        edge_cut = kwargs.get('edgeCut', 2)
        snr_cut = kwargs.get('SNRCut', 7.5)

        n_cat = len(self._redData)
        
        for iC, cellData in self._cellDict.items():
            od = cellData.objectDict
            
            for key, c in od.items():
                k = (iC, key)

                nsrcs.append(c._nSrc)    

                if (np.fabs(c.data.xCell_coadd) > cell_edge).all() or (np.fabs(c.data.yCell_coadd) > cell_edge).all():
                    cut1.append(k)
                    continue
                if np.fabs(c.data.xCell_coadd.mean()) > cell_edge or np.fabs(c.data.yCell_coadd.mean()) > cell_edge:
                    cut2.append(k)
                    continue

                used.append(k)
        
                edge_case = False
                is_faint = False
                if (np.fabs(c._data.xCell_coadd) > cell_edge-edge_cut).any() or (np.fabs(c._data.yCell_coadd) > cell_edge-edge_cut).any():
                    edge_case = True
                if (c._data.SNR < snr_cut).any():
                    is_faint = True
                                        
                if c._nSrc == c._nUnique and c._nSrc == n_cat and is_faint:
                    ideal_faint.append(k)
                elif c._nSrc == c._nUnique and c._nSrc == n_cat:
                    ideal.append(k)
                elif c._nSrc < n_cat and is_faint:
                    faint.append(k)                    
                elif c._nSrc == n_cat and c._nUnique != n_cat and edge_case:
                    edge_mixed.append(k)
                elif c._nSrc == n_cat and c._nUnique != n_cat:
                    mixed.append(k)
                elif c._nSrc < n_cat and edge_case:
                    edge_missing.append(k)
                elif c._nSrc > n_cat and edge_case:
                    edge_extra.append(k)    
                elif c._nSrc < n_cat and c._parentCluster.nSrc >= n_cat:
                    orphan.append(k)
                elif c._nSrc == n_cat - 1:
                    missing.append(k)
                elif c._nSrc == n_cat - 2:
                    two_missing.append(k)
                elif c._nSrc < n_cat - 2:
                    many_missing.append(k)
                elif c._nSrc > n_cat:
                    extra.append(k)
                else:
                    caught.append(k)

        return dict(
            nsrcs=nsrcs,
            cut1=cut1,
            cut2=cut2,
            used=used,
            ideal_faint=ideal_faint,
            ideal=ideal,
            faint=faint,
            edge_mixed=edge_mixed,
            mixed=mixed,
            edge_missing=edge_missing,
            edge_extra=edge_extra,
            orphan=orphan,
            missing=missing,
            two_missing=two_missing,
            many_missing=many_missing,
            extra=extra,
            caught=caught,
        )


    @staticmethod
    def printClusterTypes(clusterTypes: dict[str, list]):
        print("All Clusters:                                  ", len(clusterTypes["nsrcs"]))
        print("cut 1                                          ", len(clusterTypes["cut1"]))
        print("cut 2                                          ", len(clusterTypes["cut2"]))        
        print("Used:                                          ", len(clusterTypes["used"]))
        print("good (n source from n catalogs):               ", len(clusterTypes["ideal"]))
        print("good faint                                     ", len(clusterTypes["ideal_faint"]))
        print("faint (< n sources, SNR < cut):                ", len(clusterTypes["faint"]))        
        print("mixed (n source from < n catalogs):            ", len(clusterTypes["mixed"]))
        print("edge_mixed (mixed near edge of cell):          ", len(clusterTypes["edge_mixed"]))
        print("edge_missing (< n sources, near edge of cell): ", len(clusterTypes["edge_missing"]))
        print("edge_extra (> n sources, near edge of cell):   ", len(clusterTypes["edge_extra"]))
        print("faint (< n sources, SNR < cut):                ", len(clusterTypes["faint"]))
        print("one missing (n-1 sources, not near edge):      ", len(clusterTypes["missing"]))
        print("two missing (n-2 sources, not near edge):      ", len(clusterTypes["two_missing"]))
        print("many missing (< n-2 sources, not near edge):   ", len(clusterTypes["many_missing"]))
        print("extra (> n sources, not near edge):            ", len(clusterTypes["extra"]))
    
    
    @staticmethod
    def printObjectTypes(objectTypes: dict[str, list]):
        print("All Objects:                                   ", len(objectTypes["nsrcs"]))
        print("cut 1                                          ", len(objectTypes["cut1"]))
        print("cut 2                                          ", len(objectTypes["cut2"]))        
        print("Used:                                          ", len(objectTypes["used"]))
        print("good (n source from n catalogs):               ", len(objectTypes["ideal"]))
        print("good faint                                     ", len(objectTypes["ideal_faint"]))
        print("faint (< n sources, SNR < cut):                ", len(objectTypes["faint"]))        
        print("mixed (n source from < n catalogs):            ", len(objectTypes["mixed"]))
        print("edge_mixed (mixed near edge of cell):          ", len(objectTypes["edge_mixed"]))
        print("edge_missing (< n sources, near edge of cell): ", len(objectTypes["edge_missing"]))
        print("edge_extra (> n sources, near edge of cell):   ", len(objectTypes["edge_extra"]))
        print("faint (< n sources, SNR < cut):                ", len(objectTypes["faint"]))
        print("orphan (split off from larger cluster          ", len(objectTypes["orphan"]))
        print("one missing (n-1 sources, not near edge):      ", len(objectTypes["missing"]))
        print("two missing (n-2 sources, not near edge):      ", len(objectTypes["two_missing"]))
        print("many missing (< n-2 sources, not near edge):   ", len(objectTypes["many_missing"]))
        print("extra (> n sources, not near edge):            ", len(objectTypes["extra"]))


    def getCluster(self, iK: tuple[tuple[int, int], int]) -> ClusterData:
        cellData = self._cellDict[iK[0]]
        cluster = cellData.clusterDict[iK[1]]
        return cluster


    def getObject(self, iK: tuple[tuple[int, int], int]) -> ObjectData:
        cellData = self._cellDict[iK[0]]
        theObj = cellData.objectDict[iK[1]]
        return theObj        
