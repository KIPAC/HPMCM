from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.figure import Figure

SHEAR_NAMES = ["ns", "2p", "2m", "1p", "1m"]


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


class ShearProfileHistogramStats:
    """Simple class to store stats about a 2d histogram

    Attributes
    ----------
    w: np.array
        Sum of weights

    mean: np.array
        Histogram mean

    std: np.array
        Histogram standard deviation

    error: np.array
        Error on histogram mean

    inv_var: np.array
        Inverse Variance
    """

    def __init__(
        self,
        hist_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
    ):
        """Compute the stats from a histogram"""
        weights = hist_2d[0]
        y_bins = hist_2d[2]
        y_bin_centers = 0.5 * (y_bins[0:-1] + y_bins[1:])

        self.w = np.sum(weights, axis=1)
        self.mean = np.sum(weights * y_bin_centers, axis=1) / self.w
        deltas = y_bin_centers - np.expand_dims(self.mean, -1)
        var = np.sum(weights * deltas * deltas, axis=1) / self.w
        self.std = np.sqrt(var)
        self.error = self.std / np.sqrt(self.w)
        self.inv_var = 1.0 / (self.error * self.error)


class ShearHistograms:
    """Simple class to store histogram relating to shear calibration

    {type} is the matching type, one of "good", "bad", "all"

    {i}, {j} are the components of the shear: 1, 2

    Attributes
    ----------
    bin_edges: np.ndarray
        Bin edges for all histograms

    bin_centers: np.ndarray
        Bin centers for all histograms

    central: slice
        Slice to select central region of histogram

    central_edges: slice
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
        cat_type: str,
    ):

        if cat_type != "pgauss":
            self.bin_edges = np.linspace(-1, 1, 2001)
            self.central = slice(800, 1200)
            self.central_edges = slice(800, 1201)
        else:
            self.bin_edges = np.linspace(-10, 10, 20001)
            self.central = slice(9800, 10200)
            self.central_edges = slice(9800, 10201)

        self.bin_centers = (self.bin_edges[1:] + self.bin_edges[0:-1]) / 2.0

        self.good_delta_g_1_1 = np.histogram(good.delta_g_1_1, bins=self.bin_edges)[0]
        self.good_delta_g_2_2 = np.histogram(good.delta_g_2_2, bins=self.bin_edges)[0]
        self.good_delta_g_1_2 = np.histogram(good.delta_g_1_2, bins=self.bin_edges)[0]
        self.good_delta_g_2_1 = np.histogram(good.delta_g_2_1, bins=self.bin_edges)[0]

        self.good_g_2_2p = np.histogram(
            good.g_2_2p, weights=good.n_2p, bins=self.bin_edges
        )[0]
        self.good_g_2_2m = np.histogram(
            -1 * good.g_2_2m, weights=good.n_2m, bins=self.bin_edges
        )[0]
        self.good_g_1_1p = np.histogram(
            good.g_1_1p, weights=good.n_1p, bins=self.bin_edges
        )[0]
        self.good_g_1_1m = np.histogram(
            -1 * good.g_1_1m, weights=good.n_1m, bins=self.bin_edges
        )[0]

        self.good_g_2_1p = np.histogram(
            good.g_2_1p, weights=good.n_1p, bins=self.bin_edges
        )[0]
        self.good_g_2_1m = np.histogram(
            -1 * good.g_2_1m, weights=good.n_1m, bins=self.bin_edges
        )[0]
        self.good_g_1_2p = np.histogram(
            good.g_1_2p, weights=good.n_2p, bins=self.bin_edges
        )[0]
        self.good_g_1_2m = np.histogram(
            -1 * good.g_1_2m, weights=good.n_2m, bins=self.bin_edges
        )[0]

        self.bad_g_2_2p = np.histogram(
            bad.g_2_2p, weights=bad.n_2p, bins=self.bin_edges
        )[0]
        self.bad_g_2_2m = np.histogram(
            -1 * bad.g_2_2m, weights=bad.n_2m, bins=self.bin_edges
        )[0]
        self.bad_g_1_1p = np.histogram(
            bad.g_1_1p, weights=bad.n_1p, bins=self.bin_edges
        )[0]
        self.bad_g_1_1m = np.histogram(
            -1 * bad.g_1_1m, weights=bad.n_1m, bins=self.bin_edges
        )[0]

        self.bad_g_2_1p = np.histogram(
            bad.g_2_1p, weights=bad.n_1p, bins=self.bin_edges
        )[0]
        self.bad_g_2_1m = np.histogram(
            -1 * bad.g_2_1m, weights=bad.n_1m, bins=self.bin_edges
        )[0]
        self.bad_g_1_2p = np.histogram(
            bad.g_1_2p, weights=bad.n_2p, bins=self.bin_edges
        )[0]
        self.bad_g_1_2m = np.histogram(
            -1 * bad.g_1_2m, weights=bad.n_2m, bins=self.bin_edges
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
        shear: float = 0.01,
        *,
        use_central: bool = True,
    ) -> Figure:
        """Plot delta_g_1_1 and delta_g_2_2 for fully matched objects"""

        if stats_g_1 is None:
            stats_g_1 = ShearHistogramStats(self.good_delta_g_1_1, self.bin_centers)
        if stats_g_2 is None:
            stats_g_2 = ShearHistogramStats(self.good_delta_g_2_1, self.bin_centers)
        assert stats_g_1 is not None
        assert stats_g_2 is not None

        fig, axes = plt.subplots()
        if use_central:
            val_slice = self.central
            edge_slice = self.central_edges
        else:
            val_slice = slice(0, len(self.good_delta_g_1_1))
            edge_slice = slice(0, len(self.bin_edges))

        axes.stairs(
            self.good_delta_g_1_1[val_slice],
            self.bin_edges[edge_slice],
            label=f"R_11: {stats_g_1.mean/shear:.4f} +- {stats_g_1.error/shear:.4f}",
        )
        axes.stairs(
            self.good_delta_g_2_2[val_slice],
            self.bin_edges[edge_slice],
            label=f"R_22: {stats_g_2.mean/shear:.4f} +- {stats_g_2.error/shear:.4f}",
        )
        axes.axvline(x=stats_g_1.mean, color="blue", linestyle="-", linewidth=2)
        axes.axvline(x=stats_g_2.mean, color="orange", linestyle="-", linewidth=2)
        axes.axvline(
            x=stats_g_1.mean + stats_g_1.error,
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        axes.axvline(
            x=stats_g_2.mean + stats_g_2.error,
            color="orange",
            linestyle="--",
            linewidth=2,
        )
        axes.axvline(
            x=stats_g_1.mean - stats_g_1.error,
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        axes.axvline(
            x=stats_g_2.mean - stats_g_2.error,
            color="orange",
            linestyle="--",
            linewidth=2,
        )
        axes.legend(loc="upper right")
        axes.set_xlabel(r"$\delta g$")
        axes.set_ylabel("Counts")
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
        shear: float = 0.01,
        title: str = "",
        *,
        use_central: bool = True,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for all objects of a particular type"""

        if stats_g_1 is None:
            stats_g_1 = ShearHistogramStats(hist1p + hist1m, self.bin_centers)
        if stats_g_2 is None:
            stats_g_2 = ShearHistogramStats(hist2p + hist2m, self.bin_centers)
        assert stats_g_1 is not None
        assert stats_g_2 is not None

        fig, axes = plt.subplots()
        if use_central:
            val_slice = self.central
            edge_slice = self.central_edges
        else:
            val_slice = slice(0, len(hist1p))
            edge_slice = slice(0, len(hist1p) + 1)

        axes.stairs(
            (hist1p + hist1m)[val_slice],
            self.bin_edges[edge_slice],
            label=f"R_11: {2*stats_g_1.mean/shear:.4f} +- {2*stats_g_1.error/shear:.4f}",
        )
        axes.stairs(
            (hist2p + hist2m)[val_slice],
            self.bin_edges[edge_slice],
            label=f"R_22: {2*stats_g_2.mean/shear:.4f} +- {2*stats_g_2.error/shear:.4f}",
        )
        axes.set_xlabel("g")
        axes.set_ylabel("Counts")
        axes.axvline(x=stats_g_1.mean, color="blue", linestyle="-", linewidth=2)
        axes.axvline(x=stats_g_2.mean, color="orange", linestyle="-", linewidth=2)
        axes.axvline(
            x=stats_g_1.mean + stats_g_1.error,
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        axes.axvline(
            x=stats_g_2.mean + stats_g_2.error,
            color="orange",
            linestyle="--",
            linewidth=2,
        )
        axes.axvline(
            x=stats_g_1.mean - stats_g_1.error,
            color="blue",
            linestyle="--",
            linewidth=2,
        )
        axes.axvline(
            x=stats_g_2.mean - stats_g_2.error,
            color="orange",
            linestyle="--",
            linewidth=2,
        )

        axes.legend(loc="upper right")
        axes.set_title(title)
        fig.tight_layout()
        return fig

    def plotMetaDetectGood(
        self,
        stats_g_1: ShearHistogramStats | None = None,
        stats_g_2: ShearHistogramStats | None = None,
        shear: float = 0.01,
        *,
        use_central: bool = True,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for all fully matched objects"""

        return self.plotMetaDetect(
            self.good_g_1_1p,
            self.good_g_1_1m,
            self.good_g_2_2p,
            self.good_g_2_2m,
            stats_g_1,
            stats_g_2,
            shear=shear,
            title="Good Matches",
            use_central=use_central,
        )

    def plotMetaDetectAll(
        self,
        stats_g_1: ShearHistogramStats | None = None,
        stats_g_2: ShearHistogramStats | None = None,
        shear: float = 0.01,
        *,
        use_central: bool = True,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for all objects"""

        return self.plotMetaDetect(
            self.all_g_1_1p,
            self.all_g_1_1m,
            self.all_g_2_2p,
            self.all_g_2_2m,
            stats_g_1,
            stats_g_2,
            shear=shear,
            title="All Clusters",
            use_central=use_central,
        )

    def plotMetaDetectBad(
        self,
        stats_g_1: ShearHistogramStats | None = None,
        stats_g_2: ShearHistogramStats | None = None,
        shear: float = 0.01,
        *,
        use_central: bool = True,
    ) -> Figure:
        """Plot hist1p - hist1m and hist2p - hist2m for non-fully matched objects"""
        return self.plotMetaDetect(
            self.bad_g_1_1p,
            self.bad_g_1_1m,
            self.bad_g_2_2p,
            self.bad_g_2_2m,
            stats_g_1,
            stats_g_2,
            shear=shear,
            title="Bad Matches",
            use_central=use_central,
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

        self.delta_g_1_1 = ShearHistogramStats(hists.good_delta_g_1_1, hists.bin_centers)
        self.delta_g_2_2 = ShearHistogramStats(hists.good_delta_g_2_2, hists.bin_centers)

        self.delta_g_1_2 = ShearHistogramStats(hists.good_delta_g_1_2, hists.bin_centers)
        self.delta_g_2_1 = ShearHistogramStats(hists.good_delta_g_2_1, hists.bin_centers)

        self.good_g_1_1 = ShearHistogramStats(
            hists.good_g_1_1p + hists.good_g_1_1m, hists.bin_centers
        )
        self.good_g_2_2 = ShearHistogramStats(
            hists.good_g_2_2p + hists.good_g_2_2m, hists.bin_centers
        )
        self.good_g_1_2 = ShearHistogramStats(
            hists.good_g_1_2p + hists.good_g_1_2m, hists.bin_centers
        )
        self.good_g_2_1 = ShearHistogramStats(
            hists.good_g_2_1p + hists.good_g_2_1m, hists.bin_centers
        )

        self.bad_g_1_1 = ShearHistogramStats(
            hists.bad_g_1_1p + hists.bad_g_1_1m, hists.bin_centers
        )
        self.bad_g_2_2 = ShearHistogramStats(
            hists.bad_g_2_2p + hists.bad_g_2_2m, hists.bin_centers
        )
        self.bad_g_1_2 = ShearHistogramStats(
            hists.bad_g_1_2p + hists.bad_g_1_2m, hists.bin_centers
        )
        self.bad_g_2_1 = ShearHistogramStats(
            hists.bad_g_2_1p + hists.bad_g_2_1m, hists.bin_centers
        )

        self.all_g_1_1 = ShearHistogramStats(
            hists.all_g_1_1p + hists.all_g_1_1m, hists.bin_centers
        )
        self.all_g_2_2 = ShearHistogramStats(
            hists.all_g_2_2p + hists.all_g_2_2m, hists.bin_centers
        )
        self.all_g_1_2 = ShearHistogramStats(
            hists.all_g_1_2p + hists.all_g_1_2m, hists.bin_centers
        )
        self.all_g_2_1 = ShearHistogramStats(
            hists.all_g_2_1p + hists.all_g_2_1m, hists.bin_centers
        )


class ShearData:
    """Collection of shear related data for a single catalog

    Attritubes
    ----------
    shear: float
        Applied shear

    cat_type: str
        Catalog type

    tract: int
        Tract

    n_objects: int
        Number of objects in catalog

    n_in_cell: int
        Nubmer of objects in the central region of cell

    n_used: int
        Number of objects passing snr cut and in central region of cell

    n_good: int
        Number of fully matched objects

    n_bad: int
        Number on non-fully matched objects

    n_all: int
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
        shear_table: pandas.DataFrame,
        stats_table: pandas.DataFrame,
        shear: float,
        cat_type: str,
        tract: int,
        snr_cut: float = 7.5,
    ):
        shear_table["idx"] = np.arange(len(shear_table))
        stats_table["idx"] = np.arange(len(stats_table))
        merged = shear_table.merge(stats_table, on="idx")

        self.shear = shear
        self.cat_type = cat_type
        self.tract = tract

        in_cell_mask = np.bitwise_and(
            np.fabs(merged.x_cent - 100) < 75,
            np.fabs(merged.y_cent - 100) < 75,
        )
        in_cell = merged[in_cell_mask]
        bright_mask = in_cell.snr > snr_cut

        used = in_cell[bright_mask]
        good_mask = used.good
        good = used[good_mask]
        bad = used[~good_mask]

        self.n_objects = len(merged)
        self.n_in_cell = len(in_cell)
        self.n_used = len(used)
        self.n_good = len(good)
        self.n_bad = len(bad)
        self.n_all = self.n_good + self.n_bad
        self.effic = self.n_good / self.n_all
        self.effic_err = float(np.sqrt(self.effic * (1.0 - self.effic) / self.n_all))

        print("")
        print(f"Report: {cat_type} {shear:.4f} {tract}")
        print("All Objects:                               ", self.n_objects)
        print("InCell                                     ", self.n_in_cell)
        print("Used                                       ", self.n_used)
        print("Good                                       ", self.n_good)
        print("Bad                                        ", self.n_bad)
        print(
            "Efficiency                                 ",
            f"{self.effic:.4f} +- {self.effic_err:.4f}",
        )
        self.hists = ShearHistograms(good, bad, self.cat_type)
        self.stats = ShearStats(self.hists)

    def makePlots(self, *, use_central: bool = True) -> dict[str, Figure]:
        """Make the standard plots"""
        plot_dict = dict(
            metacalib=self.hists.plotMetacalib(
                self.stats.delta_g_1_1,
                self.stats.delta_g_2_2,
                shear=self.shear,
                use_central=use_central,
            ),
            metadetect_good=self.hists.plotMetaDetectGood(
                self.stats.good_g_1_1,
                self.stats.good_g_2_2,
                shear=self.shear,
                use_central=use_central,
            ),
            metadetect_bad=self.hists.plotMetaDetectBad(
                self.stats.bad_g_1_1,
                self.stats.bad_g_2_2,
                shear=self.shear,
                use_central=use_central,
            ),
            metadetect_all=self.hists.plotMetaDetectAll(
                self.stats.all_g_1_1,
                self.stats.all_g_2_2,
                shear=self.shear,
                use_central=use_central,
            ),
        )
        return plot_dict

    def toDict(self) -> dict:
        """Convert self to a dict"""
        out_dict = {}
        out_dict["shear"] = self.shear
        out_dict["n_objects"] = self.n_objects
        out_dict["n_in_cell"] = self.n_in_cell
        out_dict["n_used"] = self.n_used
        out_dict["n_good"] = self.n_good
        out_dict["n_bad"] = self.n_bad
        out_dict["efficiency"] = self.effic
        out_dict["efficiency_err"] = self.effic_err

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
            out_dict[f"{prefix_}"] = stats_.mean
            out_dict[f"{prefix_}_std"] = stats_.std
            out_dict[f"{prefix_}_err"] = stats_.error
            out_dict[f"{prefix_}_inv_var"] = stats_.inv_var
        return out_dict

    def save(self, filepath: str) -> None:
        """Save to a pickle file"""
        with open(filepath, "wb") as f_out:
            pickle.dump(self, f_out)

    @classmethod
    def load(cls, filepath: str) -> ShearData:
        """Load from a pickle file"""
        with open(filepath, "rb") as f_in:
            loaded = pickle.load(f_in)
        return loaded

    def savefigs(self, output_file_base: str) -> None:
        """Save all the figures"""
        plots = self.makePlots()
        for key, plot_ in plots.items():
            plot_.savefig(f"{output_file_base}_{key}.png")
