from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.figure import Figure

from .table import TableColumnInfo, TableInterface

SHEAR_NAMES = ["ns", "2p", "2m", "1p", "1m"]


class ShearTable(TableInterface):
    """Interface of table with shear information"""

    schema = TableInterface.schema.copy()
    schema["good"] = TableColumnInfo(bool, "Has unique match")
    for name_ in SHEAR_NAMES:
        schema[f"n_{name_}"] = TableColumnInfo(
            float, f"number of sourcrs from catalog {name_}"
        )
        for i in [1, 2]:
            schema[f"g_{i}_{name_}"] = TableColumnInfo(
                float, f"g {i} for catalog {name_}"
            )
    for i in [1, 2]:
        for j in [1, 2]:
            schema[f"delta_g_{i}_{j}"] = TableColumnInfo(
                float, f"delta g {i} for {j}p - {j}m"
            )


class ShearHistogramStats:
    """Simple class to store stats about a histogram

    Attributes
    ----------
    w: float
        Sum of weights

    mean: float
        Histogram mean

    std: float
        Histogram standard deviation

    error: float
        Error on histogram mean

    invVar: float
        Inverse Variance
    """

    def __init__(
        self,
        weights: np.ndarray,
        binCenters: np.ndarray,
    ):
        """Compute the stats from a histogram"""
        self.w = np.sum(weights)
        self.mean = np.sum(weights * binCenters) / self.w
        deltas = binCenters - self.mean
        var = np.sum(weights * deltas * deltas) / self.w
        self.std = np.sqrt(var)
        self.error = self.std / np.sqrt(self.w)
        self.invVar = 1.0 / (self.error * self.error)


class ShearHistograms:
    """Simple class to store histogram relating to shear calibration

    {type} is the matching type, one of "good", "bad", "all"

    {i}, {j} are the components of the shear: 1, 2

    Attributes
    ----------
    binEdges: np.ndarray
        Bin edges for all histograms

    binCenters: np.ndarray
        Bin centers for all histograms

    central: slice
        Slice to select central region of histogram

    centralEdges: slice
        Slice to select edges for central region of histogram

    good_delta_g_{i}_{j}: np.ndarray
        Histogram of g_{i}_{j}p - g_{i}_{j}m for all well-matched objects

    {type}_g_{i}_{cat}: np.ndarray
       Histogram of all g_{i} value of all objects of {type} in {cat}
    """

    def __init__(
        self,
        good: pandas.DataFrame,
        bad: pandas.DataFrame,
        catType: str,
    ):

        if catType != "pgauss":
            self.binEdges = np.linspace(-1, 1, 2001)
            self.central = slice(800, 1200)
            self.centralEdges = slice(800, 1201)
        else:
            self.binEdges = np.linspace(-10, 10, 20001)
            self.central = slice(9800, 10200)
            self.centralEdges = slice(9800, 10201)

        self.binCenters = (self.binEdges[1:] + self.binEdges[0:-1]) / 2.0

        self.good_delta_g_1_1 = np.histogram(good.delta_g_1_1, bins=self.binEdges)[0]
        self.good_delta_g_2_2 = np.histogram(good.delta_g_2_2, bins=self.binEdges)[0]
        self.good_delta_g_1_2 = np.histogram(good.delta_g_1_2, bins=self.binEdges)[0]
        self.good_delta_g_2_1 = np.histogram(good.delta_g_2_1, bins=self.binEdges)[0]

        self.good_g_2_2p = np.histogram(
            good.g_2_2p, weights=good.n_2p, bins=self.binEdges
        )[0]
        self.good_g_2_2m = np.histogram(
            -1 * good.g_2_2m, weights=good.n_2m, bins=self.binEdges
        )[0]
        self.good_g_1_1p = np.histogram(
            good.g_1_1p, weights=good.n_1p, bins=self.binEdges
        )[0]
        self.good_g_1_1m = np.histogram(
            -1 * good.g_1_1m, weights=good.n_1m, bins=self.binEdges
        )[0]

        self.good_g_2_1p = np.histogram(
            good.g_2_1p, weights=good.n_1p, bins=self.binEdges
        )[0]
        self.good_g_2_1m = np.histogram(
            -1 * good.g_2_1m, weights=good.n_1m, bins=self.binEdges
        )[0]
        self.good_g_1_2p = np.histogram(
            good.g_1_2p, weights=good.n_2p, bins=self.binEdges
        )[0]
        self.good_g_1_2m = np.histogram(
            -1 * good.g_1_2m, weights=good.n_2m, bins=self.binEdges
        )[0]

        self.bad_g_2_2p = np.histogram(
            bad.g_2_2p, weights=bad.n_2p, bins=self.binEdges
        )[0]
        self.bad_g_2_2m = np.histogram(
            -1 * bad.g_2_2m, weights=bad.n_2m, bins=self.binEdges
        )[0]
        self.bad_g_1_1p = np.histogram(
            bad.g_1_1p, weights=bad.n_1p, bins=self.binEdges
        )[0]
        self.bad_g_1_1m = np.histogram(
            -1 * bad.g_1_1m, weights=bad.n_1m, bins=self.binEdges
        )[0]

        self.bad_g_2_1p = np.histogram(
            bad.g_2_1p, weights=bad.n_1p, bins=self.binEdges
        )[0]
        self.bad_g_2_1m = np.histogram(
            -1 * bad.g_2_1m, weights=bad.n_1m, bins=self.binEdges
        )[0]
        self.bad_g_1_2p = np.histogram(
            bad.g_1_2p, weights=bad.n_2p, bins=self.binEdges
        )[0]
        self.bad_g_1_2m = np.histogram(
            -1 * bad.g_1_2m, weights=bad.n_2m, bins=self.binEdges
        )[0]

        self.all_g_2_2p = self.bad_g_2_2p + self.good_g_2_2p
        self.all_g_2_2m = self.bad_g_2_2m + self.good_g_2_2m
        self.all_g_1_1p = self.bad_g_1_1p + self.good_g_1_1p
        self.all_g_1_1m = self.bad_g_1_1m + self.good_g_1_1m

        self.all_g_2_1p = self.bad_g_2_1p + self.good_g_2_1p
        self.all_g_2_1m = self.bad_g_2_1m + self.good_g_2_1m
        self.all_g_1_2p = self.bad_g_1_2p + self.good_g_1_2p
        self.all_g_1_2m = self.bad_g_1_2m + self.good_g_1_2m

    def plotMetacalib(
        self,
        stats_g_1: ShearHistogramStats | None = None,
        stats_g_2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot delta_g_1_1 and delta_g_2_2 for fully matched objects"""

        if stats_g_1 is None:
            stats_g_1 = ShearHistogramStats(self.good_delta_g_1_1, self.binCenters)
        if stats_g_2 is None:
            stats_g_2 = ShearHistogramStats(self.good_delta_g_2_1, self.binCenters)
        assert stats_g_1 is not None
        assert stats_g_2 is not None

        fig, axes = plt.subplots()
        axes.stairs(
            self.good_delta_g_1_1[self.central],
            self.binEdges[self.centralEdges],
            label=f"g_1,1: {stats_g_1.mean:.6f} +- {stats_g_1.error:.6f}",
        )
        axes.stairs(
            self.good_delta_g_2_2[self.central],
            self.binEdges[self.centralEdges],
            label=f"g_2,2: {stats_g_2.mean:.6f} +- {stats_g_2.error:.6f}",
        )
        axes.axvline(x=0, color="red", linestyle="--", linewidth=2)
        axes.legend()
        fig.tight_layout()
        return fig

    def plotMetaDetect(
        self,
        hist1p: np.ndarray,
        hist1m: np.ndarray,
        hist2p: np.ndarray,
        hist2m: np.ndarray,
        stats_g_1: ShearHistogramStats | None = None,
        stats_g_2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for all objects of a particular type"""

        if stats_g_1 is None:
            stats_g_1 = ShearHistogramStats(hist1p + hist1m, self.binCenters)
        if stats_g_2 is None:
            stats_g_2 = ShearHistogramStats(hist2p + hist2m, self.binCenters)
        assert stats_g_1 is not None
        assert stats_g_2 is not None

        fig, axes = plt.subplots()
        axes.stairs(
            (hist1p + hist1m)[self.central],
            self.binEdges[self.centralEdges],
            label=f"g1: {200*stats_g_1.mean:.4f} +- {200*stats_g_1.error:.4f}",
        )
        axes.stairs(
            (hist2p + hist2m)[self.central],
            self.binEdges[self.centralEdges],
            label=f"g2: {200*stats_g_2.mean:.4f} +- {200*stats_g_2.error:.4f}",
        )
        axes.legend()
        fig.tight_layout()
        return fig

    def plotMetaDetectGood(
        self,
        stats_g_1: ShearHistogramStats | None = None,
        stats_g_2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for all fully matched objects"""

        return self.plotMetaDetect(
            self.good_g_1_1p,
            self.good_g_1_1m,
            self.good_g_2_2p,
            self.good_g_2_2m,
            stats_g_1,
            stats_g_2,
        )

    def plotMetaDetectAll(
        self,
        stats_g_1: ShearHistogramStats | None = None,
        stats_g_2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for all objects"""

        return self.plotMetaDetect(
            self.all_g_1_1p,
            self.all_g_1_1m,
            self.all_g_2_2p,
            self.all_g_2_2m,
            stats_g_1,
            stats_g_2,
        )

    def plotMetaDetectBad(
        self,
        stats_g_1: ShearHistogramStats | None = None,
        stats_g_2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for non-fully matched objects"""
        return self.plotMetaDetect(
            self.bad_g_1_1p,
            self.bad_g_1_1m,
            self.bad_g_2_2p,
            self.bad_g_2_2m,
            stats_g_1,
            stats_g_2,
        )


class ShearStats:
    """Simple class to store shear statisitics

    {type} is the matching type, one of "good", "bad", "all"

    {i}, {j} are the components of the shear: 1, 2

    Attributes
    ----------
    delta_g_{i}_{i}: ShearHistogramStats
        Stats for g_{i}_{j}p - g_{i}_{j}m for fully matched objects

    {type}_g_{i}_{j}: ShearHistogramStats
        Stats for g_{i}_{j} for objects of {type}
    """

    def __init__(
        self,
        hists: ShearHistograms,
    ):

        self.delta_g_1_1 = ShearHistogramStats(hists.good_delta_g_1_1, hists.binCenters)
        self.delta_g_2_2 = ShearHistogramStats(hists.good_delta_g_2_2, hists.binCenters)

        self.delta_g_1_2 = ShearHistogramStats(hists.good_delta_g_1_2, hists.binCenters)
        self.delta_g_2_1 = ShearHistogramStats(hists.good_delta_g_2_1, hists.binCenters)

        self.good_g_1_1 = ShearHistogramStats(
            hists.good_g_1_1p + hists.good_g_1_1m, hists.binCenters
        )
        self.good_g_2_2 = ShearHistogramStats(
            hists.good_g_2_2p + hists.good_g_2_2m, hists.binCenters
        )
        self.good_g_1_2 = ShearHistogramStats(
            hists.good_g_1_2p + hists.good_g_1_2m, hists.binCenters
        )
        self.good_g_2_1 = ShearHistogramStats(
            hists.good_g_2_1p + hists.good_g_2_1m, hists.binCenters
        )

        self.bad_g_1_1 = ShearHistogramStats(
            hists.bad_g_1_1p + hists.bad_g_1_1m, hists.binCenters
        )
        self.bad_g_2_2 = ShearHistogramStats(
            hists.bad_g_2_2p + hists.bad_g_2_2m, hists.binCenters
        )
        self.bad_g_1_2 = ShearHistogramStats(
            hists.bad_g_1_2p + hists.bad_g_1_2m, hists.binCenters
        )
        self.bad_g_2_1 = ShearHistogramStats(
            hists.bad_g_2_1p + hists.bad_g_2_1m, hists.binCenters
        )

        self.all_g_1_1 = ShearHistogramStats(
            hists.all_g_1_1p + hists.all_g_1_1m, hists.binCenters
        )
        self.all_g_2_2 = ShearHistogramStats(
            hists.all_g_2_2p + hists.all_g_2_2m, hists.binCenters
        )
        self.all_g_1_2 = ShearHistogramStats(
            hists.all_g_1_2p + hists.all_g_1_2m, hists.binCenters
        )
        self.all_g_2_1 = ShearHistogramStats(
            hists.all_g_2_1p + hists.all_g_2_1m, hists.binCenters
        )


class ShearData:
    """Collection of shear related data for a single catalog

    Attritubes
    ----------
    shear: float
        Applied shear

    catType: str
        Catalog type

    tract: int
        Tract

    nObjects: int
        Number of objects in catalog

    nInCell: int
        Nubmer of objects in the central region of cell

    nUsed: int
        Number of objects passing SNR cut and in central region of cell

    nGood: int
        Number of fully matched objects

    nBad: int
        Number on non-fully matched objects

    nAll: int
        Number of fully and non-fully matched objects

    effic: float
        Efficiency to fully match objects

    efficErr: float
        Error on efficiency to fully match objects

    hists: ShearHistograms
        Histograms of shear data

    stats: ShearStats
        Summary statistics
    """

    def __init__(
        self,
        shearTable: pandas.DataFrame,
        statsTable: pandas.DataFrame,
        shear: float,
        catType: str,
        tract: int,
        snrCut: float = 7.5,
    ):
        shearTable["idx"] = np.arange(len(shearTable))
        statsTable["idx"] = np.arange(len(statsTable))
        merged = shearTable.merge(statsTable, on="idx")

        self.shear = shear
        self.catType = catType
        self.tract = tract

        in_cell_mask = np.bitwise_and(
            np.fabs(merged.xCent - 100) < 75,
            np.fabs(merged.yCent - 100) < 75,
        )
        in_cell = merged[in_cell_mask]
        bright_mask = in_cell.SNR > snrCut

        used = in_cell[bright_mask]
        good_mask = used.good
        good = used[good_mask]
        bad = used[~good_mask]

        self.nObjects = len(merged)
        self.nInCell = len(in_cell)
        self.nUsed = len(used)
        self.nGood = len(good)
        self.nBad = len(bad)
        self.nAll = self.nGood + self.nBad
        self.effic = self.nGood / self.nAll
        self.efficErr = float(
            np.sqrt(self.nGood * self.nBad * self.nAll) / (self.nAll * self.nAll)
        )

        print("")
        print(f"Report: {catType} {shear:.4f} {tract}")
        print("All Objects:                               ", self.nObjects)
        print("InCell                                     ", self.nInCell)
        print("Used                                       ", self.nUsed)
        print("Good                                       ", self.nGood)
        print("Bad                                        ", self.nBad)
        print(
            "Efficiency                                 ",
            f"{self.effic:.4f} +- {self.efficErr:.4f}",
        )
        self.hists = ShearHistograms(good, bad, self.catType)
        self.stats = ShearStats(self.hists)

    def makePlots(self) -> dict[str, Figure]:
        """Make the standard plots"""
        plotDict = dict(
            metacalib=self.hists.plotMetacalib(
                self.stats.delta_g_1_1, self.stats.delta_g_2_2
            ),
            metadetect_good=self.hists.plotMetaDetectGood(
                self.stats.good_g_1_1, self.stats.good_g_2_2
            ),
            metadetect_bad=self.hists.plotMetaDetectBad(
                self.stats.bad_g_1_1, self.stats.bad_g_2_2
            ),
            metadetect_all=self.hists.plotMetaDetectAll(
                self.stats.all_g_1_1, self.stats.all_g_2_2
            ),
        )
        return plotDict

    def toDict(self) -> dict:
        """Convert self to a dict"""
        outDict = {}
        outDict["shear"] = self.shear
        outDict["n_objects"] = self.nObjects
        outDict["n_in_cell"] = self.nInCell
        outDict["n_used"] = self.nUsed
        outDict["n_good"] = self.nGood
        outDict["n_bad"] = self.nBad
        outDict["efficiency"] = self.effic
        outDict["efficiency_err"] = self.efficErr

        prefixes = [
            "mc_delta_g_1_1",
            "mc_delta_g_2_2",
            "mc_delta_g_1_2",
            "mc_delta_g_2_1",
            "md_all_g_1_1",
            "md_all_g_2_2",
            "md_all_g_1_2",
            "md_all_g_2_1",
            "md_good_g_1_1",
            "md_good_g_2_2",
            "md_good_g_1_2",
            "md_good_g_2_1",
            "md_bad_g_1_1",
            "md_bad_g_2_2",
            "md_bad_g_1_2",
            "md_bad_g_2_1",
        ]
        all_stats = [
            self.stats.delta_g_1_1,
            self.stats.delta_g_2_2,
            self.stats.delta_g_1_2,
            self.stats.delta_g_2_1,
            self.stats.all_g_1_1,
            self.stats.all_g_2_2,
            self.stats.all_g_1_2,
            self.stats.all_g_2_1,
            self.stats.good_g_1_1,
            self.stats.good_g_2_2,
            self.stats.good_g_1_2,
            self.stats.good_g_2_1,
            self.stats.bad_g_1_1,
            self.stats.bad_g_2_2,
            self.stats.bad_g_1_2,
            self.stats.bad_g_2_1,
        ]

        for prefix_, stats_ in zip(prefixes, all_stats):
            outDict[f"{prefix_}"] = stats_.mean
            outDict[f"{prefix_}_std"] = stats_.std
            outDict[f"{prefix_}_err"] = stats_.error
            outDict[f"{prefix_}_invVar"] = stats_.invVar
        return outDict

    def save(self, filepath: str) -> None:
        """Save to a pickle file"""
        with open(filepath, "wb") as fOut:
            pickle.dump(self, fOut)

    @classmethod
    def load(cls, filepath: str) -> ShearData:
        """Load from a pickle file"""
        with open(filepath, "rb") as fIn:
            loaded = pickle.load(fIn)
        return loaded

    def savefigs(self, outputFileBase: str) -> None:
        """Save all the figures"""
        plots = self.makePlots()
        for key, plot_ in plots.items():
            plot_.savefig(f"{outputFileBase}_{key}.png")
