from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.figure import Figure


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

    inv_var: float
        Inverse Variance
    """

    def __init__(
        self,
        weights: np.ndarray,
        bin_centers: np.ndarray,
    ):
        """Compute the stats from a histogram"""
        self.w = np.sum(weights)
        self.mean = np.sum(weights * bin_centers) / self.w
        deltas = bin_centers - self.mean
        var = np.sum(weights * deltas * deltas) / self.w
        self.std = np.sqrt(var)
        self.error = self.std / np.sqrt(self.w)
        self.inv_var = 1.0 / (self.error * self.error)


class ShearHistograms:
    """Simple class to store histogram relating to shear calibration


    Attributes
    ----------
    bin_edges: np.ndarray
        Bin edges for all histograms

    bin_centers: np.ndarray
        Bin centers for all histograms

    central: slice
        Slice to select central region of histogram

    centralEdges: slice
        Slice to select edges for central region of histogram

    {type}_g{i}_{cat}: np.ndarray
       Histogram of all g{i} value of all objects of {type} in {cat}
    """

    def __init__(
        self,
        good: pandas.DataFrame,
        bad: pandas.DataFrame,
        catType: str,
    ):

        if catType != "pgauss":
            self.bin_edges = np.linspace(-1, 1, 2001)
            self.central = slice(800, 1200)
            self.centralEdges = slice(800, 1201)
        else:
            self.bin_edges = np.linspace(-10, 10, 20001)
            self.central = slice(9800, 10200)
            self.centralEdges = slice(9800, 10201)

        self.bin_centers = (self.bin_edges[1:] + self.bin_edges[0:-1]) / 2.0

        self.good_delta_g1 = np.histogram(good.delta_g_1, bins=self.bin_edges)[0]
        self.good_delta_g2 = np.histogram(good.delta_g_2, bins=self.bin_edges)[0]

        self.good_g2_2p = np.histogram(
            good.g2_2p, weights=good.n_2p, bins=self.bin_edges
        )[0]
        self.good_g2_2m = np.histogram(
            -1 * good.g2_2m, weights=good.n_2m, bins=self.bin_edges
        )[0]
        self.good_g1_1p = np.histogram(
            good.g1_1p, weights=good.n_1p, bins=self.bin_edges
        )[0]
        self.good_g1_1m = np.histogram(
            -1 * good.g1_1m, weights=good.n_1m, bins=self.bin_edges
        )[0]

        self.bad_g2_2p = np.histogram(bad.g2_2p, weights=bad.n_2p, bins=self.bin_edges)[
            0
        ]
        self.bad_g2_2m = np.histogram(
            -1 * bad.g2_2m, weights=bad.n_2m, bins=self.bin_edges
        )[0]
        self.bad_g1_1p = np.histogram(bad.g1_1p, weights=bad.n_1p, bins=self.bin_edges)[
            0
        ]
        self.bad_g1_1m = np.histogram(
            -1 * bad.g1_1m, weights=bad.n_1m, bins=self.bin_edges
        )[0]

        self.all_g2_2p = self.bad_g2_2p + self.good_g2_2p
        self.all_g2_2m = self.bad_g2_2m + self.good_g2_2m
        self.all_g1_1p = self.bad_g1_1p + self.good_g1_1p
        self.all_g1_1m = self.bad_g1_1m + self.good_g1_1m

    def plotMetacalib(
        self,
        statsG1: ShearHistogramStats | None = None,
        statsG2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot delta_g1 and delta_g2 for fully matched objects"""

        if statsG1 is None:
            statsG1 = ShearHistogramStats(self.good_delta_g1, self.bin_centers)
        if statsG2 is None:
            statsG2 = ShearHistogramStats(self.good_delta_g2, self.bin_centers)
        assert statsG1 is not None
        assert statsG2 is not None

        fig, axes = plt.subplots()
        axes.stairs(
            self.good_delta_g1[self.central],
            self.bin_edges[self.centralEdges],
            label=f"g1: {statsG1.mean:.6f} +- {statsG1.error:.6f}",
        )
        axes.stairs(
            self.good_delta_g2[self.central],
            self.bin_edges[self.centralEdges],
            label=f"g2: {statsG2.mean:.6f} +- {statsG2.error:.6f}",
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
        statsG1: ShearHistogramStats | None = None,
        statsG2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for all objects of a particular type"""

        if statsG1 is None:
            statsG1 = ShearHistogramStats(hist1p + hist1m, self.bin_centers)
        if statsG2 is None:
            statsG2 = ShearHistogramStats(hist2p + hist2m, self.bin_centers)
        assert statsG1 is not None
        assert statsG2 is not None

        fig, axes = plt.subplots()
        axes.stairs(
            (hist1p + hist1m)[self.central],
            self.bin_edges[self.centralEdges],
            label=f"g1: {200*statsG1.mean:.4f} +- {200*statsG1.error:.4f}",
        )
        axes.stairs(
            (hist2p + hist2m)[self.central],
            self.bin_edges[self.centralEdges],
            label=f"g2: {200*statsG2.mean:.4f} +- {200*statsG2.error:.4f}",
        )
        axes.legend()
        fig.tight_layout()
        return fig

    def plotMetaDetectGood(
        self,
        statsG1: ShearHistogramStats | None = None,
        statsG2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for all fully matched objects"""

        return self.plotMetaDetect(
            self.good_g1_1p,
            self.good_g1_1m,
            self.good_g2_2p,
            self.good_g2_2m,
            statsG1,
            statsG2,
        )

    def plotMetaDetectAll(
        self,
        statsG1: ShearHistogramStats | None = None,
        statsG2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for all objects"""

        return self.plotMetaDetect(
            self.all_g1_1p,
            self.all_g1_1m,
            self.all_g2_2p,
            self.all_g2_2m,
            statsG1,
            statsG2,
        )

    def plotMetaDetectBad(
        self,
        statsG1: ShearHistogramStats | None = None,
        statsG2: ShearHistogramStats | None = None,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for non-fully matched objects"""
        return self.plotMetaDetect(
            self.bad_g1_1p,
            self.bad_g1_1m,
            self.bad_g2_2p,
            self.bad_g2_2m,
            statsG1,
            statsG2,
        )


class ShearStats:
    """Simple class to store shear statisitics

    Attributes
    ----------
    delta_g1: ShearHistogramStats
        Stats for g1 deltas for fully matched objects

    delta_g2: ShearHistogramStats
        Stats for g2 deltas for fully matched objects

    {type}_g1: ShearHistogramStats
        Stats for g1 for objects of {type}

    {type}_g2: ShearHistogramStats
        Stats for g2 for objects of {type}
    """

    def __init__(
        self,
        hists: ShearHistograms,
    ):

        self.delta_g1 = ShearHistogramStats(hists.good_delta_g1, hists.bin_centers)
        self.delta_g2 = ShearHistogramStats(hists.good_delta_g2, hists.bin_centers)

        self.good_g1 = ShearHistogramStats(
            hists.good_g1_1p + hists.good_g1_1m, hists.bin_centers
        )
        self.good_g2 = ShearHistogramStats(
            hists.good_g2_2p + hists.good_g2_2m, hists.bin_centers
        )

        self.bad_g1 = ShearHistogramStats(
            hists.bad_g1_1p + hists.bad_g1_1m, hists.bin_centers
        )
        self.bad_g2 = ShearHistogramStats(
            hists.bad_g2_2p + hists.bad_g2_2m, hists.bin_centers
        )

        self.all_g1 = ShearHistogramStats(
            hists.all_g1_1p + hists.all_g1_1m, hists.bin_centers
        )
        self.all_g2 = ShearHistogramStats(
            hists.all_g2_2p + hists.all_g2_2m, hists.bin_centers
        )


class ShearData:
    """Collection of shear related data for a single catalog

    Attritubes
    ----------
    self.shear: float
        Applied shear

    self.catType: str
        Catalog type

    self.tract: int
        Tract

    self.nObjects: int
        Number of objects in catalog

    self.nInCell: int
        Nubmer of objects in the central region of cell

    self.nUsed: int
        Number of objects passing SNR cut and in central region of cell

    self.nGood: int
        Number of fully matched objects

    self.nBad: int
        Number on non-fully matched objects

    self.nAll: int
        Number of fully and non-fully matched objects

    self.effic: float
        Efficiency to fully match objects

    self.efficErr: float
        Error on efficiency to fully match objects

    self.hists = ShearHistograms
        Histograms of shear data

    self.stats = ShearStats
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
            np.fabs(merged.xCents - 100) < 75,
            np.fabs(merged.yCents - 100) < 75,
        )
        in_cell = merged[in_cell_mask]
        bright_mask = in_cell.SNRs > snrCut

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
                self.stats.delta_g1, self.stats.delta_g2
            ),
            metadetect_good=self.hists.plotMetaDetectGood(
                self.stats.good_g1, self.stats.good_g2
            ),
            metadetect_bad=self.hists.plotMetaDetectBad(
                self.stats.bad_g1, self.stats.bad_g2
            ),
            metadetect_all=self.hists.plotMetaDetectAll(
                self.stats.all_g1, self.stats.all_g2
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
            "mc_delta_g1",
            "mc_delta_g2",
            "md_all_g1",
            "md_all_g2",
            "md_good_g1",
            "md_good_g2",
            "md_bad_g1",
            "md_bad_g2",
        ]
        all_stats = [
            self.stats.delta_g1,
            self.stats.delta_g2,
            self.stats.all_g1,
            self.stats.all_g2,
            self.stats.good_g1,
            self.stats.good_g2,
            self.stats.bad_g1,
            self.stats.bad_g2,
        ]

        for prefix_, stats_ in zip(prefixes, all_stats):
            outDict[f"{prefix_}"] = stats_.mean
            outDict[f"{prefix_}_std"] = stats_.std
            outDict[f"{prefix_}_err"] = stats_.error
            outDict[f"{prefix_}_inv_var"] = stats_.inv_var
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
