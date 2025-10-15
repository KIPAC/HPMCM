from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas

from . import shear_utils, utils

if TYPE_CHECKING:
    from .cell import CellData
    from .cluster import ClusterData


RECURSE_MAX = 4


def makeObjectAssocTable(cellData: CellData) -> pandas.DataFrame:
    """Small function to create object association table

    Parameters
    ----------
    cellData:
        Cell we are making table for

    Returns
    -------
    pandas.DataFrame:
        Object Association table    

    Notes
    -----
    The data frame will have the following data

    object: int
        Object Id

    parent: int
        Parent Cluster Id

    id: int
        Source Id

    idx: int
        Source Index in respective catalog

    cat: int
        Index of catalog

    distance: float
        Distance between source and object centroid (in arcsec)

    cellIdx: int
        Index of parent cell
    """
    clusterIds = []
    objectIds = []
    sourceIds = []
    sourceIdxs = []
    catIdxs = []
    distancesList: list[np.ndarray] = []

    for obj in cellData.objectDict.values():
        clusterIds.append(
            np.full((obj.nSrc), obj.parentCluster.iCluster, dtype=int)
        )
        objectIds.append(np.full((obj.nSrc), obj.objectId, dtype=int))
        sourceIds.append(obj.sourceIds())
        sourceIdxs.append(obj.sourceIdxs())
        catIdxs.append(obj.catIndices)
        assert obj.dist2.size
        distancesList.append(obj.dist2)
    if not distancesList:
        return pandas.DataFrame(
            dict(
                object=np.array([], int),
                parent=np.array([], int),
                id=np.array([], int),
                idx=np.array([], int),
                cat=np.array([], int),
                distance=[],
            )
        )
    distances = np.hstack(distancesList)
    distances = cellData.matcher.pixToArcsec() * np.sqrt(distances)
    data = dict(
        object=np.hstack(objectIds),
        parent=np.hstack(clusterIds),
        id=np.hstack(sourceIds),
        idx=np.hstack(sourceIdxs),
        cat=np.hstack(catIdxs),
        distance=distances,
        cellIdx=np.repeat(cellData.idx, len(distances)).astype(int),
    )
    return pandas.DataFrame(data)


def makeObjectStatsTable(cellData: CellData) -> pandas.DataFrame:
    """Small function to create object association table

    Parameters
    ----------
    cellData:
        Cell we are making table for

    Returns
    -------
    pandas.DataFrame:
        Object Association table    
        

    """
    nObj = self.nObjects
    clusterIds = np.zeros((nObj), dtype=int)
    objectIds = np.zeros((nObj), dtype=int)
    nSrcs = np.zeros((nObj), dtype=int)
    nUniques = np.zeros((nObj), dtype=int)
    distRms = np.zeros((nObj), dtype=float)
    xCents = np.zeros((nObj), dtype=float)
    yCents = np.zeros((nObj), dtype=float)
    SNRs = np.zeros((nObj), dtype=float)
    SNRRms = np.zeros((nObj), dtype=float)
    
    for idx, obj in enumerate(cellData.objectDict.values()):
        clusterIds[idx] = obj.parentCluster.iCluster
        objectIds[idx] = obj.objectId
        nSrcs[idx] = obj.nSrc
        nUniques[idx] = obj.nUnique
        distRms[idx] = obj.rmsDist
        xCents[idx] = obj.xCent
        yCents[idx] = obj.yCent
        assert obj.data is not None
        sumSNR = obj.data.SNR.sum()
        xCents[idx] = np.sum(obj.data.SNR * obj.data.xCell) / sumSNR
        yCents[idx] = np.sum(obj.data.SNR * obj.data.yCell) / sumSNR
        SNRs[idx] = obj.snrMean
        SNRRms[idx] = obj.snrRms

    ra, dec = cellData._getRaDec(xCents, yCents)
    distRms *= cellData.matcher.pixToArcsec()

    data = dict(
        clusterIds=clusterIds,
        objectIds=objectIds,
        nUniques=nUniques,
        nSrcs=nSrcs,
        distRms=distRms,
        ra=ra,
        dec=dec,
        xCents=xCents,
        yCents=yCents,
        SNRs=SNRs,
        SNRRms=SNRRms,
        cellIdx=np.repeat(cellData.idx, len(distRms)).astype(int),
    )

    return pandas.DataFrame(data)



class ObjectData:
    """Small class to define 'Objects', i.e., sets of associated sources

    Attributes
    ----------
    parentCluster : ClusterData
        Parent cluster that this object is build from

    objectId : int
        Unique object identifier

    mask : np.ndarray[bool]
        Mask of which source in parent cluster are in this object

    catIndices: np.ndarray[int]
        Indicies showing which catalog each source belongs to

    nSrc: int
        Number of sources in this object

    nUnique: int
        Number of catalogs contributing to this object

    data: pandas.DataFrame
        Data for the sources in this object

    recurse: int
        Recursion level needed to make this object

    xCent: float
        X Centeroid of this object in global pixel coordinates

    yCent: float
        Y Centeroid of this object in global pixel coordinates

    rmsDist: float
        RMS distance of sources to the centroid

    dist2: np.ndarray
        Distances of sources to the centroid
    """

    def __init__(
        self,
        cluster: ClusterData,
        objectId: int,
        mask: np.ndarray | None,
        recurse: int = 0,
    ):
        """Build from `ClusterData`, an objectId and mask specifying with sources
        in the cluster are part of the object"""
        self.parentCluster: ClusterData = cluster
        self.objectId: int = objectId
        if mask is None:
            self.mask = np.ones((self.parentCluster.nSrc), dtype=bool)
        else:
            self.mask = mask
        self.catIndices: np.ndarray = self.parentCluster.catIndices[self.mask]
        self.nSrc: int = self.catIndices.size
        self.nUnique: int = np.unique(self.catIndices).size
        self.data: pandas.DataFrame | None = None

        self.recurse: int = recurse
        self.xCent: float = np.nan
        self.yCent: float = np.nan
        self.rmsDist: float = np.nan
        self.dist2: np.ndarray = np.array([])
        self.extract()

    @property
    def xCluster(self) -> np.ndarray:
        """Return the x-position of the sources within the footprint"""
        assert self.data is not None
        return self.data.xCluster

    @property
    def yCluster(self) -> np.ndarray:
        """Return the y-position of the sources within the footprint"""
        assert self.data is not None
        return self.data.yCluster

    @property
    def xPix(self) -> np.ndarray:
        """Return the x-position of the sources within the cell"""
        assert self.data is not None
        return self.data.xPix

    @property
    def yPix(self) -> np.ndarray:
        """Return the y-position of the sources within the cell"""
        assert self.data is not None
        return self.data.yPix

    def sourceIds(self) -> np.ndarray:
        """Return the source ids for the sources in the object"""
        return self.parentCluster.srcIds[self.mask]

    def sourceIdxs(self) -> np.ndarray:
        """Return the source indices for the sources in the object"""
        return self.parentCluster.srcIdxs[self.mask]

    def _updateCatIndices(self) -> None:
        self.catIndices = self.parentCluster.catIndices[self.mask]
        self.nSrc = self.catIndices.size
        self.nUnique = np.unique(self.catIndices).size

    def extract(self) -> None:
        """Extract data from parent cluster"""
        assert self.parentCluster.data is not None
        self.data = self.parentCluster.data.iloc[self.mask]
        self._updateCatIndices()


class ShearObjectData(ObjectData):
    """Subclass of ObjectData that can compute shear statisitics"""

    def shearStats(self) -> dict:
        """Return the shear statistics"""
        assert self.data is not None
        return shear_utils.shearStats(self.data)
