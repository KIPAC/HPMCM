

class ClusterData:
    """ Class to store data about clusters

    Parameters
    ----------
    iCluster : `int`
        Cluster ID
    origCluster : `int`
        Id of the original cluster this cluster was made from
    nSrc : `int`
        Number of sources in this cluster
    nUnique : `int`
        Number of catalogs contributing sources to this cluster
    catIndices : `np.array`, [`int`]
        Indices of the catalogs of sources associated to this cluster
    sourcdIds : `np.array`, [`int`]
        Sources IDs of the sources associated to this cluster
    sourcdIdxs : `np.array`, [`int`]
        Indices of the sources with their respective catalogs
    xCent : `float`
        X-pixel value of cluster centroid (in WCS used to do matching)
    yCent : `float`
        Y-pixel value of cluster centroid (in WCS used to do matching)
    """
    def __init__(self, iCluster, footprint, sources, origCluster=None):
        self._iCluster = iCluster
        self._footprint = footprint
        if origCluster is None:
            self._origCluster = self._iCluster
        else:
            self._origCluster = origCluster
        self._catIndices = sources[0]
        self._sourceIds = sources[1]
        self._sourceIdxs = sources[2]
        self._nSrc =  self._catIndices.size
        self._nUnique = len(np.unique(self._catIndices))
        self._objects = []
        self._data = None
        self._xCent = None
        self._yCent = None
        self._dist2 = None
        self._rmsDist = None
        self._xCell = None
        self._yCell = None
        self._snr = None
        
    def extract(self, subRegionData):
        """ Extract the xCell, yCell and snr data from
        the sources in this cluster
        """

        series_list = []
        iCat_list = []
        src_idx_list = []
        
        for i, (iCat, srcIdx) in enumerate(zip(self._catIndices, self._sourceIdxs)):

            series_list.append(subRegionData.data[iCat].iloc[srcIdx])
            icat_list.append(iCat)
            src_idx_list.append(srcIdx)

        self._data = pandas.DataFrame(series_list)
        self._data['iCat'] = iCat_list
        self._data['idx'] = src_idx_list

        self._xCell = self._data['xcell'].values()
        self._yCell = self._data['ycell'].values()
        self._snr = self._data['snr'].values()
        
    def clearTempData(self):
        """ Remove temporary data only used when making objects """
        self._data = None
        self._xCell = None
        self._yCell = None
        self._snr = None

    @property
    def iCluster(self):
        """ Return the cluster ID """
        return self._iCluster

    @property
    def nSrc(self):
        """ Return the number of sources associated to the cluster """
        return self._nSrc

    @property
    def nUnique(self):
        """ Return the number of catalogs contributing sources to the cluster """
        return self._nUnique

    @property
    def sourceIds(self):
        """ Return the source IDs associated to this cluster """
        return self._sourceIds

    @property
    def dist2(self):
        """ Return an array with the distance squared (in cells)
        between each source and the cluster centroid """
        return self._dist2

    @property
    def objects(self):
        """ Return the objects associated with this cluster """
        return self._objects

    def processCluster(self, cellData, pixelR2Cut):
        """ Function that is called recursively to
        split clusters until they:

        1.  Consist only of sources with the match radius of the cluster
        centroid.

        2.  Have at most one source per input catalog
        """
        self._nSrc =  self._catIndices.size
        self._nUnique = len(np.unique(self._catIndices))
        if self._nSrc == 0:
            print("Empty cluster", self._nSrc, self._nUnique)
            return self._objects
        self.extract(subRegionData)
        if self._nSrc == 1:
            self._xCent = self._xCell[0]
            self._yCent = self._yCell[0]
            self._dist2 = np.zeros((1))
            self._rmsDist = 0.
            initialObject = self.addObject(cellData)
            initialObject.processObject(cellData, pixelR2Cut)
            self.clearTempData()
            return self._objects

        sumSnr = np.sum(self._snr)
        self._xCent = np.sum(self._xCell*self._snr) / sumSnr
        self._yCent = np.sum(self._yCell*self._snr) / sumSnr
        self._dist2 = (self._xCent - self._xCell)**2 + (self._yCent - self._yCell)**2
        self._rmsDist = np.sqrt(np.mean(self._dist2))
        
        initialObject = self.addObject(subRegionData)
        initialObject.processObject(subRegionData, pixelR2Cut)
        self.clearTempData()
        return self._objects

    def addObject(self, subRegionData, mask=None):
        """ Add a new object to this cluster """
        newObject = subRegionData.addObject(self, mask)
        self._objects.append(newObject)
        return newObject
