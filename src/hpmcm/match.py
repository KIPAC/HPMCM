from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas

from . import input_tables, output_tables
from .cell import CellData
from .cluster import ClusterData
from .object import ObjectData


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

    Attributes
    ----------
    pixSize : float
        Pixel size in arcseconds

    nPixSide: int
        Number of pixels in the match region

    cellSize: int
        Number of pixels in a Cell

    cellBuffer: int
        Number of overlapping pixel in a Cell

    cellMaxObject: int
        Max number of objects in a cell, used to make unique IDs

    maxSubDivision: int
        Maximum number of cell sub-divisions

    pixelR2Cut: float
        Distance cut for Object membership, in pixels**2

    nCell: np.ndarray
        Number of cells in match region

    fullData: list[DataFrame]
        Full input DataFrames

    redData : list[DataFrame]
        Reduced DataFrames with only the columns needed for matching

    cellDict : OrderedDict[int, CellData]
        Dictionary providing access to cell data

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames.
    The expected columns depend on which sub-class of `Match` is being used.

    Four output tables are produced:

    +----------------+---------------------------------------------------+
    | Key            | Class                                             |
    +================+===================================================+
    | _cluster_assoc | :py:class:`hpmcm.output_tables.ClusterAssocTable` |
    +----------------+---------------------------------------------------+
    | _cluster_stats | :py:class:`hpmcm.output_tables.ClusterStatsTable` |
    +----------------+---------------------------------------------------+
    | _object_assoc  | :py:class:`hpmcm.output_tables.ObjectAssocTable`  |
    +----------------+---------------------------------------------------+
    | _object_stats  | :py:class:`hpmcm.output_tables.ObjectStatsTable`  |
    +----------------+---------------------------------------------------+

    """

    inputTableClass: type = input_tables.SourceTable
    extraCols: list[str] = []

    def __init__(
        self,
        **kwargs: Any,
    ):
        self.pixSize = kwargs["pixelSize"]
        self.nPixSide = kwargs["nPixels"]
        self.cellSize: int = kwargs.get("cellSize", 1000)
        self.cellBuffer: int = kwargs.get("cellBuffer", 10)
        self.cellMaxObject: int = kwargs.get("cellMaxObject", 100000)
        self.maxSubDivision: int = kwargs.get("maxSubDivision", 3)
        self.pixelR2Cut: float = kwargs.get("pixelR2Cut", 1.0)
        self.nCell: np.ndarray = np.ceil(self.nPixSide / self.cellSize)

        self.fullData: OrderedDict[int, pandas.DataFrame] = OrderedDict()
        self.redData: OrderedDict[int, pandas.DataFrame] = OrderedDict()
        self.cellDict: OrderedDict[int, CellData] = OrderedDict()

    def pixToArcsec(self) -> float:
        """Convert pixel size (in degrees) to arcseconds"""
        return 3600.0 * self.pixSize

    def pixToWorld(
        self,
        xPix: np.ndarray,
        yPix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert local coords in pixels to world coordinates (RA, DEC)"""
        return np.repeat(np.nan, len(xPix)), np.repeat(np.nan, len(yPix))

    def getCellIdx(
        self,
        ix: int,
        iy: int,
    ) -> int:
        """Get the Index to use for a given cell"""
        return int(self.nCell[1] * ix + iy)

    def getIdOffset(
        self,
        ix: int,
        iy: int,
    ) -> int:
        """Get the ID offset to use for a given cell"""
        cellIdx = self.getCellIdx(ix, iy)
        return int(self.cellMaxObject * cellIdx)

    def reduceData(
        self,
        inputFiles: list[str],
        catalogId: list[int],
    ) -> None:
        """Read input files and filter out only the columns we need

        Each input file should have an associated catalogId.
        This is used to test if we have more than one-source
        per input catalog.

        If the inputs files have a pre-defined ID associated with them
        that can be used.   Otherwise it is fine just to give a range from
        0 to nInputs.
        """
        for fName, cid in zip(inputFiles, catalogId):
            self.fullData[cid] = self._readDataFrame(fName)
            self.redData[cid] = self.reduceDataFrame(self.fullData[cid])
            self.fullData[cid].set_index("id", inplace=True)

    def _buildCellData(
        self,
        idOffset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
    ) -> CellData:
        return CellData(self, idOffset, corner, size, idx, self.cellBuffer)

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
        Output of cell analysis


        Notes
        -----
        cellData : CellData : The analysis data for the Cell

        image : afwImage.ImageI : Image of cell source counts map

        countsMap : np.array : Numpy array with cell source counts

        clusters : afwDetect.FootprintSet : Clusters as dectected by finding FootprintSet on source counts map

        clusterKey : afwImage.ImageI : Map of cell with pixels filled with index of associated Footprints

        Notes
        -----
        If fullData is False, only cellData will be returned
        """
        iCell = self.getCellIdx(ix, iy)
        cellStep = np.array([self.cellSize, self.cellSize])
        corner = np.array([ix - 1, iy - 1]) * cellStep
        idOffset = self.getIdOffset(ix, iy)
        cellData = self._buildCellData(idOffset, corner, cellStep, iCell)
        cellData.reduceData(list(self.redData.values()))
        oDict = cellData.analyze(pixelR2Cut=self.pixelR2Cut)
        if cellData.nObjects >= self.cellMaxObject:
            print("Too many object in a cell", cellData.nObjects, self.cellMaxObject)

        self.cellDict[iCell] = cellData
        if oDict is None:
            return None
        if fullData:
            oDict["cellData"] = cellData
            return oDict

        return dict(cellData=cellData)

    def analysisLoop(
        self, xRange: Iterable | None = None, yRange: Iterable | None = None
    ) -> None:
        """Does matching for all cells.

        This stores the results, but does not write or return them.

        Parameters
        ----------
        xRange:
            Range of cells to analysze in X.  None -> Entire range.

        yRange:
            Range of cells to analysis in Y.  None -> Entire range.
        """
        self.cellDict.clear()

        if xRange is None:
            xRange = range(int(self.nCell[0]))
        if yRange is None:
            yRange = range(int(self.nCell[1]))

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

    def extractStats(self) -> list[pandas.DataFrame]:
        """Extracts cluster statisistics

        Returns
        -------
        DataFrames with matching info,
        :py:class:`hpmcm.output_tables.ClusterAssocTable`,
        :py:class:`hpmcm.output_tables.ObjectAssocTable`.
        :py:class:`hpmcm.output_tables.ClusterStatsTable`.
        :py:class:`hpmcm.output_tables.ObjectStatsTable`.

        """
        clusterAssocTables = []
        objectAssocTables = []
        clusterStatsTables = []
        objectStatsTables = []

        for ix in range(int(self.nCell[0])):
            for iy in range(int(self.nCell[1])):
                iCell = self.getCellIdx(ix, iy)
                if iCell not in self.cellDict:
                    continue
                cellData = self.cellDict[iCell]
                clusterAssocTables.append(
                    output_tables.ClusterAssocTable.buildFromCellData(cellData).data,
                )
                objectAssocTables.append(
                    output_tables.ObjectAssocTable.buildFromCellData(cellData).data,
                )
                clusterStatsTables.append(
                    output_tables.ClusterStatsTable.buildFromCellData(cellData).data,
                )
                objectStatsTables.append(
                    output_tables.ObjectStatsTable.buildFromCellData(cellData).data,
                )
        return [
            pandas.concat(clusterAssocTables),
            pandas.concat(objectAssocTables),
            pandas.concat(clusterStatsTables),
            pandas.concat(objectStatsTables),
        ]

    def _readDataFrame(
        self,
        fName: str,
    ) -> pandas.DataFrame:
        """Read a single input file"""
        # FIXME, we want to use this function
        # return self.inputTableClass.read(fName, self.extraCols)
        import pyarrow.parquet as pq

        parq = pq.read_pandas(fName)
        df = parq.to_pandas()
        return df

    def _getPixValues(self, df: pandas.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame

        Parameters
        ----------
        df:
            Input data frame

        Returns
        -------
        Reduced DataFrame
        """
        raise NotImplementedError()

    def getCluster(self, iK: tuple[int, int]) -> ClusterData:
        """Get a particular cluster

        Parameters
        ----------
        iK:
            CellId, ClusterId

        Returns
        -------
        Requested cluster
        """
        cellData = self.cellDict[iK[0]]
        cluster = cellData.clusterDict[iK[1]]
        return cluster

    def getObject(self, iK: tuple[int, int]) -> ObjectData:
        """Get a particular object

        Parameters
        ----------
        iK:
            CellId, ObjectId

        Returns
        -------
        Requested object
        """
        cellData = self.cellDict[iK[0]]
        theObj = cellData.objectDict[iK[1]]
        return theObj
