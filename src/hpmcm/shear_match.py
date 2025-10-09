from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import pandas
import tables_io
import yaml

from .cell import CellData, ShearCellData
from .match import Match

COLUMNS = ["ra", "dec", "id", "patch_x", "patch_y", "cell_x", "cell_y", "row", "col"]


class ShearMatch(Match):
    """Class to do N-way matching for shear calibration.

    Uses cell-based coadds toassign pixel locations to all sources in
    the input catalogs

    Expects 5 input catalogs.

    Attributes
    ----------
    _maxSubDivision: int
        Maximum number of cell sub-diviions
    
    _pixelMatchScale: int
        Number of cells to merge in original counts map
    
    _catType: str
        Shear catalog type
    
    _deshear: float
        Deshearing parameter

    Notes
    -----
    This expectes a list of parquet files with pandas DataFrames
    that contain the following columns.

    "id" : source ID
    "xCell_coadd": X-postion in cell-based coadd used for metadetect
    "yCell_coadd": Y-postion in cell-based coadd used for metadetect
    "g_1": shape measurement
    "g_2": shape measurement
    "SNR": Signal-to-Noise of source, used for filtering and centroiding

    """

    def __init__(
        self,
        **kwargs: Any,
    ):
        self._maxSubDivision: int = kwargs.get("maxSubDivision", 3)
        self._pixelMatchScale: int = kwargs.get("pixelMatchScale", 1)
        self._catType: str = kwargs.get("catalogType", "wmom")
        self._deshear: float | None = kwargs.get("deshear", None)
        Match.__init__(self, **kwargs)

    @property
    def deshear(self) -> float | None:
        """Return the deshearing parameter"""
        return self._deshear

    @classmethod
    def createShearMatch(
        cls,
        **kwargs: Any,
    ) -> ShearMatch:
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
        kw = dict(
            pixelSize=0.2 * 3600.0,
            nPixels=nPix,
            cellSize=150,
            cellBuffer=25,
            cellMaxObject=1000,
        )
        return cls(**kw, **kwargs)

    @classmethod
    def splitByTypeAndClean(
        cls,
        basefile: str,
        tract: int,
        shear: float,
    ) -> None:
        """Split a parquet file by shear catalog type"""
        TYPES = ["ns", "1m", "2m", "1p", "2p"]
        p = tables_io.read(basefile)
        for type_ in TYPES:
            mask = p["shear_type"] == type_
            sub = p[mask].copy(deep=True)
            idx_x = (20 * sub["patch_x"].values + sub["cell_x"].values).astype(int)
            idx_y = (20 * sub["patch_y"].values + sub["cell_y"].values).astype(int)
            cent_x = 150 * idx_x - 75
            cent_y = 150 * idx_y - 75
            xCell_coadd = sub["col"] - cent_x
            yCell_coadd = sub["row"] - cent_y
            sub["xCell_coadd"] = xCell_coadd
            sub["yCell_coadd"] = yCell_coadd
            sub["SNR"] = sub["wmom_band_flux_r"] / sub["wmom_band_flux_err_r"]
            sub["g_1"] = sub["wmom_g_1"]
            sub["g_2"] = sub["wmom_g_2"]
            sub["idx_x"] = idx_x
            sub["idx_y"] = idx_y
            sub["orig_id"] = sub.id
            sub["id"] = np.arange(len(sub))
            central_to_cell = np.bitwise_and(
                np.fabs(xCell_coadd) < 80, np.fabs(yCell_coadd) < 80
            )
            central_to_patch = np.bitwise_and(
                np.fabs(sub["cell_x"].values - 10.5) < 10,
                np.fabs(sub["cell_y"].values - 10.5) < 10,
            )
            right_tract = sub["tract"] == tract
            central = np.bitwise_and(central_to_cell, central_to_patch)
            selected = np.bitwise_and(right_tract, central)
            cleaned = sub[selected].copy(deep=True)
            cleaned["shear"] = np.repeat(shear, len(cleaned))
            cleaned.to_parquet(
                basefile.replace(".parq", f"_uncleaned_{tract}_{type_}.pq")
            )

    def _buildCellData(
        self,
        idOffset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: np.ndarray,
    ) -> CellData:
        return ShearCellData(self, idOffset, corner, size, idx, self._cellBuffer)

    def extractShearStats(self) -> list[pandas.DataFrame]:
        """Extract shear stats"""
        clusterShearStatsTables = []
        objectShearStatsTables = []

        for ix in range(int(self._nCell[0])):
            for iy in range(int(self._nCell[1])):
                iCell = (ix, iy)
                if iCell not in self._cellDict:
                    continue
                cellData = self._cellDict[iCell]
                assert isinstance(cellData, ShearCellData)
                clusterShearStatsTables.append(cellData.getClusterShearStats())
                objectShearStatsTables.append(cellData.getObjectShearStats())

        return [
            pandas.concat(clusterShearStatsTables),
            pandas.concat(objectShearStatsTables),
        ]

    def _getPixValues(self, df: pandas.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xPix, yPix = (
            df["col"].values + 25,
            df["row"].values + 25,
        )
        return xPix, yPix

    def _reduceDataFrame(
        self,
        df: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """Reduce a single input DataFrame"""
        df_clean = df[(df.SNR > 1)]
        xPix, yPix = self._getPixValues(df_clean)
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

    @staticmethod
    def stats(
        weights: np.ndarray,
        bin_centers: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Compute the stats from a histogram"""
        w = np.sum(weights)
        mean = np.sum(weights * bin_centers) / w
        deltas = bin_centers - mean
        var = np.sum(weights * deltas * deltas) / w
        std = np.sqrt(var)
        error = std / np.sqrt(w)
        inv_var = 1.0 / (error * error)
        return float(mean), float(std), float(error), float(inv_var)

    @classmethod
    def shear_report(
        cls,
        basefile: str,
        outputFileBase: str,
        shear: float,
    ) -> None:
        """Report on the shear calibration"""
        t = tables_io.read(f"{basefile}_object_shear.pq")
        t2 = tables_io.read(f"{basefile}_object_stats.pq")
        t["idx"] = np.arange(len(t))
        t2["idx"] = np.arange(len(t2))
        merged = t.merge(t2, on="idx")

        in_cell_mask = np.bitwise_and(
            np.fabs(merged.xCents - 100) < 75,
            np.fabs(merged.yCents - 100) < 75,
        )
        in_cell = merged[in_cell_mask]
        bright_mask = in_cell.SNRs > 7.5

        used = in_cell[bright_mask]

        good_mask = used.good
        good = used[good_mask]
        bad = used[~good_mask]

        nGood = len(good)
        nBad = len(bad)
        nAll = nGood + nBad
        effic =  nGood / nAll
        efficErr = float(np.sqrt(nGood*nBad*nAll)/(nAll*nAll))
        
        outDict = {}
        outDict["shear"] = shear
        outDict["n_objects"] = len(merged)
        outDict["n_in_cell"] = len(in_cell)
        outDict["n_used"] = len(used)
        outDict["n_good"] = nGood
        outDict["n_bad"] = nBad
        outDict["efficiency"] = effic
        outDict["efficiency_err"] = efficErr
        
        print("All Objects:                               ", len(merged))
        print("Usable                                     ", len(in_cell))
        print("Bright                                     ", len(used))
        print("Good                                       ", len(good))
        print("Bad                                        ", len(bad))
        print(
            "Efficiency                                 ",
            f"{effic:.4f} +- {efficErr:.4f}"
        )

        bin_edges = np.linspace(-1, 1, 2001)
        bin_centers = (bin_edges[1:] + bin_edges[0:-1]) / 2.0

        good_delta_g1 = np.histogram(good.delta_g_1, bins=bin_edges)[0]
        good_delta_g2 = np.histogram(good.delta_g_2, bins=bin_edges)[0]

        good_g2_2p = np.histogram(good.g2_2p, weights=good.n_2p, bins=bin_edges)[0]
        good_g2_2m = np.histogram(-1 * good.g2_2m, weights=good.n_2m, bins=bin_edges)[0]
        good_g1_1p = np.histogram(good.g1_1p, weights=good.n_1p, bins=bin_edges)[0]
        good_g1_1m = np.histogram(-1 * good.g1_1m, weights=good.n_1m, bins=bin_edges)[0]

        bad_g2_2p = np.histogram(bad.g2_2p, weights=bad.n_2p, bins=bin_edges)[0]
        bad_g2_2m = np.histogram(-1 * bad.g2_2m, weights=bad.n_2m, bins=bin_edges)[0]
        bad_g1_1p = np.histogram(bad.g1_1p, weights=bad.n_1p, bins=bin_edges)[0]
        bad_g1_1m = np.histogram(-1 * bad.g1_1m, weights=bad.n_1m, bins=bin_edges)[0]

        all_g2_2p = bad_g2_2p + good_g2_2p
        all_g2_2m = bad_g2_2m + good_g2_2m
        all_g1_1p = bad_g1_1p + good_g1_1p
        all_g1_1m = bad_g1_1m + good_g1_1m

        stats_delta_g1 = cls.stats(good_delta_g1, bin_centers)
        stats_delta_g2 = cls.stats(good_delta_g2, bin_centers)

        stats_good_g1 = cls.stats(good_g1_1p + good_g1_1m, bin_centers)
        stats_good_g2 = cls.stats(good_g2_2p + good_g2_2m, bin_centers)

        stats_bad_g1 = cls.stats(bad_g1_1p + bad_g1_1m, bin_centers)
        stats_bad_g2 = cls.stats(bad_g2_2p + bad_g2_2m, bin_centers)

        stats_all_g1 = cls.stats(all_g1_1p + all_g1_1m, bin_centers)
        stats_all_g2 = cls.stats(all_g2_2p + all_g2_2m, bin_centers)
        
        outDict["mc_delta_g1"] = stats_delta_g1[0]
        outDict["mc_delta_g1_std"] = stats_delta_g1[1]        
        outDict["mc_delta_g1_err"] = stats_delta_g1[2]
        outDict["mc_delta_g1_inv_var"] = stats_delta_g1[3]

        outDict["mc_delta_g2"] = stats_delta_g2[0]
        outDict["mc_delta_g2_std"] = stats_delta_g2[1]        
        outDict["mc_delta_g2_err"] = stats_delta_g2[2]
        outDict["mc_delta_g2_inv_var"] = stats_delta_g2[3]

        outDict["md_all_g1"] = stats_all_g1[0]
        outDict["md_all_g1_std"] = stats_all_g1[1]        
        outDict["md_all_g1_err"] = stats_all_g1[2]
        outDict["md_all_g1_inv_var"] = stats_all_g1[3]

        outDict["md_all_g2"] = stats_all_g2[0]
        outDict["md_all_g2_std"] = stats_all_g2[1]        
        outDict["md_all_g2_err"] = stats_all_g2[2]
        outDict["md_all_g2_inv_var"] = stats_all_g2[3]

        outDict["md_good_g1"] = stats_good_g1[0]
        outDict["md_good_g1_std"] = stats_good_g1[1]        
        outDict["md_good_g1_err"] = stats_good_g1[2]
        outDict["md_good_g1_inv_var"] = stats_good_g1[3]

        outDict["md_good_g2"] = stats_good_g2[0]
        outDict["md_good_g2_std"] = stats_good_g2[1]        
        outDict["md_good_g2_err"] = stats_good_g2[2]
        outDict["md_good_g2_inv_var"] = stats_good_g2[3]

        outDict["md_bad_g1"] = stats_bad_g1[0]
        outDict["md_bad_g1_std"] = stats_bad_g1[1]        
        outDict["md_bad_g1_err"] = stats_bad_g1[2]
        outDict["md_bad_g1_inv_var"] = stats_bad_g1[3]

        outDict["md_bad_g2"] = stats_bad_g2[0]
        outDict["md_bad_g2_std"] = stats_bad_g2[1]        
        outDict["md_bad_g2_err"] = stats_bad_g2[2]
        outDict["md_bad_g2_inv_var"] = stats_bad_g2[3]

        with open(f"{outputFileBase}.yaml", "w") as fout:
            yaml.safe_dump(outDict, fout)
                           
        fig_mc, axes_mc = plt.subplots()
        axes_mc.stairs(
            good_delta_g1[800:1200],
            bin_edges[800:1201],
            label=f"g1: {stats_delta_g1[0]:.6f} +- {stats_delta_g1[2]:.6f}",
        )
        axes_mc.stairs(
            good_delta_g2[800:1200],
            bin_edges[800:1201],
            label=f"g2: {stats_delta_g2[0]:.6f} +- {stats_delta_g2[2]:.6f}",
        )
        axes_mc.axvline(x=0, color="red", linestyle="--", linewidth=2)
        axes_mc.legend()
        fig_mc.tight_layout()
        fig_mc.savefig(f"{outputFileBase}_metacalib.png")

        fig_md_good, axes_md_good = plt.subplots()
        axes_md_good.stairs(
            (good_g1_1p + good_g1_1m)[800:1200],
            bin_edges[800:1201],
            label=f"g1: {200*stats_good_g1[0]:.4f} +- {200*stats_good_g1[2]:.4f}",
        )
        axes_md_good.stairs(
            (good_g2_2p + good_g2_2m)[800:1200],
            bin_edges[800:1201],
            label=f"g2: {200*stats_good_g2[0]:.4f} +- {200*stats_good_g2[2]:.4f}",
        )
        axes_md_good.legend()
        fig_md_good.tight_layout()
        fig_md_good.savefig(f"{outputFileBase}_md_good.png")

        fig_md_bad, axes_md_bad = plt.subplots()
        axes_md_bad.stairs(
            (bad_g1_1p + bad_g1_1m)[800:1200],
            bin_edges[800:1201],
            label=f"g1: {200*stats_bad_g1[0]:.4f} +- {200*stats_bad_g1[2]:.4f}",
        )
        axes_md_bad.stairs(
            (bad_g2_2p + bad_g2_2m)[800:1200],
            bin_edges[800:1201],
            label=f"g2: {200*stats_bad_g2[0]:.4f} +- {200*stats_bad_g2[2]:.4f}",
        )
        axes_md_bad.legend()
        fig_md_bad.tight_layout()
        fig_md_bad.savefig(f"{outputFileBase}_md_bad.png")

        print("Shear                                      ", f"{shear}")

        print("MetaCalib:")
        print(
            "g1:                                         ",
            f"{100*stats_delta_g1[0]:.4f} +- {100*stats_delta_g1[2]:.4f}"
        )
        print(
            "g2:                                         ",
            f"{100*stats_delta_g2[0]:.4f} +- {100*stats_delta_g2[2]:.4f}"
        )

        print("MetaDetect Good:")
        print(
            "g1:                                         ",
            f"{200*stats_good_g1[0]:.4f} +- {200*stats_good_g1[2]:.4f}"
        )
        print(
            "g2:                                         ",
            f"{200*stats_good_g2[0]:.4f} +- {200*stats_good_g2[2]:.4f}"
        )

        print("MetaDetect bad:")
        print(
            "g1:                                         ",
            f"{200*stats_bad_g1[0]:.4f} +- {200*stats_bad_g1[2]:.4f}"
        )
        print(
            "g2:                                         ",
            f"{200*stats_bad_g2[0]:.4f} +- {200*stats_bad_g2[2]:.4f}"
        )


    @classmethod
    def merge_shear_reports(
        cls,
        inputs: list[str],
        outputFile: str,
    ) -> None:
        """Merge report on the shear calibration"""

        outDict = {}
        for input_ in inputs:
            with open(input_, 'r') as fin:
                inputDict = yaml.safe_load(fin)

            for key, val in inputDict.items():
                if key in outDict:
                    outDict[key].append(val)
                else:
                    outDict[key] = [val]

        outDF = pandas.DataFrame(outDict)
        outDF.to_parquet(outputFile)
