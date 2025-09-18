
class Match:
    """ Class to do N-way matching

    Uses a provided WCS to define a Skymap that covers the full region
    begin matched.

    Uses that WCS to assign cell locations to all sources in the input catalogs

    Iterates over sub-regions and does source clustering in each sub-region
    using Footprint detection on a Skymap of source counts per cell.

    Assigns each input source to a cluster.

    At that stage the clusters are not the final product as they can include
    more than one soruce from a given catalog.

    Loops over clusters and processes each cluster to:

       1. Remove outliers outside the match radius w.r.t. the cluster centroid.
       2. Resolve cases of confusion, where multiple sources from a single
       catalog contribute to a cluster.

    Parameters
    ----------
    _redData : `list`, [`Dataframe`]
        Reduced dataframes with only the columns needed for matching

    _clusters : `OrderedDict`, [`tuple`, `SubregionData`]
        Dictionary providing access to subregion data
    """

    def __init__(self, matchWcs, **kwargs):
        self._wcs = matchWcs
        self._cellSize = self._wcs.wcs.cdelt[1]
        self._nCellSide = np.ceil(2*np.array(self._wcs.wcs.crpix)).astype(int)
        self._subRegionSize = kwargs.get('subRegionSize', 3000)
        self._subRegionBuffer = kwargs.get('subRegionBuffer', 10)
        self._subregionMaxObject = kwargs.get('subregionMaxObject', 100000)
        self._pixelR2Cut = kwargs.get('pixelR2Cut', 1.0)
        self._nSubRegion = np.ceil(self._nCellSide/self._subRegionSize)
        self._redData = OrderedDict()
        self._clusters = None

    def cellToArcsec(self):
        return 3600. * self._cellSize

    def cellToWorld(self, xCell, yCell):
        return self._wcs.wcs_pix2world(xCell, yCell, 0)
    
    @classmethod
    def create(cls, refDir, regionSize, cellSize, **kwargs):
        """ Make an `NWayMatch` object from inputs """
        nCell = (np.array(regionSize)/cellSize).astype(int)
        matchWcs = createGlobalWcs(refDir, cellSize, nCell)
        return cls(matchWcs, **kwargs)

    @property
    def redData(self):
        """ Return the dictionary of reduced data, i.e., just the columns
        need for matching """
        return self._redData

    @property
    def nSubRegion(self):
        """ Return the number of sub-regions in X,Y """
        return self._nSubRegion

    def reduceData(self, inputFiles, visitIds):
        """ Read input files and filter out only the columns we need """
        for fName, vid in zip(inputFiles, visitIds):
            self._redData[vid] = self.reduceDataFrame(fName)

    def reduceDataFrame(self, fName):
        """ Read and reduce a single input file """
        parq = pq.read_pandas(fName, columns=COLUMNS)
        df = parq.to_pandas()
        #df['SNR'] = df['PsFlux']/df['PsFluxErr']
        df['SNR'] = 1./df['gauss_mag_r_err']
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
        #df_clean = df[(df.SNR > 5) & ~df.Centroid_flag & ~df.sky_source & df.detect_isPrimary]
        df_clean = df[(df.SNR > 1)]
        xcell, ycell = self._wcs.wcs_world2pix(df_clean['ra'].values, df_clean['dec'].values, 0)
        df_red = df_clean[["ra", "dec", "SNR", "id", "patch_x", "patch_y", "cell_x", "cell_y", "row", "col", "gauss_g_1", "gauss_g_2"]].copy(deep=True)
                              
        idx_x = 20*df_red['patch_x'].values + df_red['cell_x'].values
        idx_y = 20*df_red['patch_y'].values + df_red['cell_y'].values
        cent_x = 150*idx_x -75
        cent_y = 150*idx_y -75
        local_x = df_red['col'] - cent_x
        local_y = df_red['row'] - cent_y

        df_red['xcell'] = xcell
        df_red['ycell'] = ycell
        df_red['local_x'] = local_x
        df_red['local_y'] = local_y
        return df_red[["ra", "dec", "SNR", "id", "xcell", "ycell", "local_x", "local_y", "gauss_g_1", "gauss_g_2"]]

    def reduceCatalog(self, catalog):
        """ Reduce a catalog """
        raise NotImplementedError()

    def add(self, catalog, vid):
        """ Add a catalog to the data set being matched """
        self._redData[vid] = self.reduceCatalog(catalog)

    def getIdOffset(self, ix, iy):
        """ Get the ID offset to use for a given sub-region """
        subRegionIdx = self._nSubRegion[1]*ix + iy
        return int(self._subregionMaxObject * subRegionIdx)

    def analyzeSubregion(self, ix, iy, fullData=False):
        """ Analyze a single subregion

        Returns an OrderedDict

        'srd' : `SubregionData`
            The analysis data for the sub-region

        if fullData is True the return dict will include

        'image' : `afwImage.ImageI`
            Image of subregion source counts map
        'countsMap' : `np.array`
            Numpy array with same
        'clusters' : `afwDetect.FootprintSet`
            Clusters as dectected by finding FootprintSet on source counts map
        'clusterKey' : `afwImage.ImageI`
            Map of subregion with pixels filled with index of
            associated Footprints
        """
        iSubRegion = np.array([ix, iy])
        corner = iSubRegion * self._subRegionSize
        idOffset = self.getIdOffset(ix, iy)
        srd = SubregionData(self, idOffset, corner, self._subRegionSize, self._subRegionBuffer)
        srd.reduceData(self._redData.values())
        oDict = srd.analyze(pixelR2Cut=self._pixelR2Cut)
        if oDict is None:
            return None
        if fullData:
            oDict['srd'] = srd
            return oDict
        if srd.nObjects >= self._subregionMaxObject:
            print("Too many object in a subregion", srd.nObjects, elf._subregionMaxObject)
        return dict(srd=srd)

    def finish(self):
        """ Does clusering for all subregions

        Does not store source counts maps for the counts regions
        """
        self._clusters = OrderedDict()
        nAssoc = 0
        clusterAssocTables = []
        objectAssocTables = []
        clusterStatsTables = []
        objectStatsTables = []

        for ix in range(int(self._nSubRegion[0])):
            sys.stdout.write("%2i " % ix)
            sys.stdout.flush()
            for iy in range(int(self._nSubRegion[1])):
                sys.stdout.write('.')
                sys.stdout.flush()
                iSubRegion = (ix, iy)
                odict = self.analyzeSubregion(ix, iy)
                if odict is None:
                    continue
                subregionData = odict['srd']
                self._clusters[iSubRegion] = subregionData
                clusterAssocTables.append(subregionData.getClusterAssociations())
                objectAssocTables.append(subregionData.getObjectAssociations())
                clusterStatsTables.append(subregionData.getClusterStats())
                objectStatsTables.append(subregionData.getObjectStats())
                
            sys.stdout.write('!\n')

        sys.stdout.write("Making association vectors\n")
        hduList = fits.HDUList([fits.PrimaryHDU(),
                                fits.table_to_hdu(vstack(clusterAssocTables)),
                                fits.table_to_hdu(vstack(objectAssocTables)),
                                fits.table_to_hdu(vstack(clusterStatsTables)),
                                fits.table_to_hdu(vstack(objectStatsTables))])
        return hduList

    def allStats(self):
        """ Helper function to print info about clusters """
        stats = np.zeros((4), int)
        for key, srd in self._clusters.items():
            subRegionStats = clusterStats(srd._clusterDict)
            print("%3i, %3i: %8i %8i %8i %8i" % (key[0], key[1], subRegionStats[0], subRegionStats[1], subRegionStats[2], subRegionStats[3]))
            stats += subRegionStats
        return stats
