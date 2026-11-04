from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas

from .cell import CellData
from .mst_cell import MSTCellData, MSTShearCellData
from .shear_match import ShearMatch
from .shear_utils import DEFAULT_GEOMETRY, ShearCellGeometry
from .wcs_match import WcsMatch, createGlobalWcs


class MSTWcsMatch(WcsMatch):
    """WcsMatch that uses MST-based initial grouping instead of footprint detection.

    Sources within ``mst_cut`` arcseconds of each other (via a chain of
    neighbours) are placed in the same initial cluster.  The subsequent
    hierarchical confusion-resolution is identical to the standard
    :py:class:`~hpmcm.wcs_match.WcsMatch` pipeline.

    Parameters
    ----------
    match_wcs:
        Astropy WCS defining the pixel grid.
    mst_cut:
        Maximum source-to-source separation in **arcseconds** for two sources
        to be connected in the initial MST grouping.
    **kwargs:
        Forwarded to :py:class:`~hpmcm.wcs_match.WcsMatch`.
    """

    def __init__(
        self,
        match_wcs: Any,
        mst_cut: float,
        **kwargs: Any,
    ) -> None:
        self.mst_cut: float = mst_cut
        WcsMatch.__init__(self, match_wcs, **kwargs)

    def _buildCellData(
        self,
        id_offset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
    ) -> CellData:
        return MSTCellData(self, id_offset, corner, size, idx, self.cell_buffer)

    def _geometryDict(self) -> dict[str, Any]:
        d = super()._geometryDict()
        d["mst_cut"] = float(self.mst_cut)
        return d

    @classmethod
    def create(
        cls,
        ref_dir: tuple[float, float],
        region_size: tuple[float, float],
        pixel_size: float,
        mst_cut: float,
        **kwargs: Any,
    ) -> MSTWcsMatch:
        """Create an MSTWcsMatch for a given sky region.

        Parameters
        ----------
        ref_dir:
            Reference direction (RA, DEC) in degrees.
        region_size:
            Size of the match region in degrees.
        pixel_size:
            WCS pixel size in degrees.
        mst_cut:
            MST grouping radius in arcseconds.
        **kwargs:
            Forwarded to the constructor.
        """
        n_pix = (np.array(region_size) / pixel_size).astype(int)
        match_wcs = createGlobalWcs(ref_dir, pixel_size, n_pix)
        kwargs.setdefault("n_cell_buffer", 1)
        return cls(match_wcs, mst_cut, n_pixels=n_pix, **kwargs)

    @classmethod
    def load(
        cls,
        save_dir: str | Path,
        x_range: tuple[int, int] | None = None,
        y_range: tuple[int, int] | None = None,
    ) -> MSTWcsMatch:
        """Restore an MSTWcsMatch from a directory written by save().

        Parameters
        ----------
        save_dir:
            Directory written by ``save()``.
        x_range:
            Optional ``(x_min, x_max)`` inclusive bounds on the cell x-index.
        y_range:
            Optional ``(y_min, y_max)`` inclusive bounds on the cell y-index.
        """
        save_dir = Path(save_dir)
        with open(save_dir / "geometry.json") as fh:
            geo = json.load(fh)

        ref_dir = tuple(geo["ref_dir"])
        n_pix = np.array(geo["n_pixels"])
        match_wcs = createGlobalWcs(
            ref_dir, geo["pixel_size"], n_pix, ctype=geo.get("wcs_ctype", "TAN")
        )
        matcher = cls(
            match_wcs,
            geo["mst_cut"],
            n_pixels=n_pix,
            cell_size=geo["cell_size"],
            cell_buffer=geo["cell_buffer"],
            cell_max_object=geo["cell_max_object"],
            max_sub_division=geo["max_sub_division"],
            pixel_r2_cut=geo["pixel_r2_cut"],
            n_cell_buffer=geo["n_cell_buffer"],
            pixel_match_scale=geo.get("pixel_match_scale", 1),
        )
        matcher.catalog_id_map = {int(k): int(v) for k, v in geo["catalog_id_map"].items()}

        cluster_assoc = pandas.read_parquet(save_dir / "cluster_assoc.parquet")
        object_assoc = pandas.read_parquet(save_dir / "object_assoc.parquet")
        cluster_stats = pandas.read_parquet(save_dir / "cluster_stats.parquet")
        object_stats = pandas.read_parquet(save_dir / "object_stats.parquet")

        cluster_assoc, object_assoc = matcher._filterAssocByRange(
            cluster_assoc, object_assoc, x_range, y_range
        )

        matcher._loadReducedData(save_dir)
        per_cell_data = matcher._buildPerCellData(cluster_assoc)
        matcher._reconstructCells(
            cluster_assoc, object_assoc, cluster_stats, object_stats, per_cell_data
        )
        return matcher


class MSTShearMatch(ShearMatch):
    """ShearMatch that uses MST-based initial grouping instead of footprint detection.

    Sources within ``mst_cut`` arcseconds of each other (via a chain of
    neighbours) are placed in the same initial cluster.  The subsequent
    hierarchical confusion-resolution is identical to the standard
    :py:class:`~hpmcm.shear_match.ShearMatch` pipeline.

    Parameters
    ----------
    mst_cut:
        Maximum source-to-source separation in **arcseconds** for two sources
        to be connected in the initial MST grouping.
    **kwargs:
        Forwarded to :py:class:`~hpmcm.shear_match.ShearMatch`.
    """

    def __init__(
        self,
        mst_cut: float,
        **kwargs: Any,
    ) -> None:
        self.mst_cut: float = mst_cut
        ShearMatch.__init__(self, **kwargs)

    def _buildCellData(
        self,
        id_offset: int,
        corner: np.ndarray,
        size: np.ndarray,
        idx: int,
    ) -> CellData:
        return MSTShearCellData(self, id_offset, corner, size, idx, self.cell_buffer)

    def _geometryDict(self) -> dict[str, Any]:
        d = super()._geometryDict()
        d["mst_cut"] = float(self.mst_cut)
        return d

    @classmethod
    def createShearMatch(  # type: ignore[override]
        cls,
        geometry: ShearCellGeometry = DEFAULT_GEOMETRY,
        mst_cut: float = 1.0,
        **kwargs: Any,
    ) -> MSTShearMatch:
        """Create an MSTShearMatch for a given shear cell geometry.

        Parameters
        ----------
        geometry:
            Cell geometry and WCS matching configuration.
        mst_cut:
            MST grouping radius in arcseconds.
        **kwargs:
            Forwarded to the constructor.
        """
        kw: dict[str, Any] = dict(
            pixel_size=geometry.pixel_size,
            n_pixels=geometry.tract_size,
            cell_size=geometry.cell_inner_size,
            cell_buffer=geometry.match_buffer,
            cell_max_object=1000,
            n_cell_buffer=1,
            geometry=geometry,
        )
        kw.update(kwargs)
        return cls(mst_cut, **kw)

    @classmethod
    def load(  # type: ignore[override]
        cls,
        save_dir: str | Path,
        x_range: tuple[int, int] | None = None,
        y_range: tuple[int, int] | None = None,
    ) -> MSTShearMatch:
        """Restore an MSTShearMatch from a directory written by save().

        Parameters
        ----------
        save_dir:
            Directory written by ``save()``.
        x_range:
            Optional ``(x_min, x_max)`` inclusive bounds on the cell x-index.
        y_range:
            Optional ``(y_min, y_max)`` inclusive bounds on the cell y-index.
        """
        save_dir = Path(save_dir)
        with open(save_dir / "geometry.json") as fh:
            geo = json.load(fh)

        sg = geo["shear_geometry"]
        geometry = ShearCellGeometry(**sg)

        matcher = cls(
            geo["mst_cut"],
            pixel_size=geo["pixel_size"],
            n_pixels=np.array(geo["n_pixels"]),
            cell_size=geo["cell_size"],
            cell_buffer=geo["cell_buffer"],
            cell_max_object=geo["cell_max_object"],
            max_sub_division=geo["max_sub_division"],
            pixel_r2_cut=geo["pixel_r2_cut"],
            n_cell_buffer=geo["n_cell_buffer"],
            shear_names=geo["shear_names"],
            deshear=geo["deshear"],
            catalogType=geo["cat_type"],
            pixel_match_scale=geo["pixel_match_scale"],
            geometry=geometry,
        )
        matcher.catalog_id_map = {int(k): int(v) for k, v in geo["catalog_id_map"].items()}

        cluster_assoc = pandas.read_parquet(save_dir / "cluster_assoc.parquet")
        object_assoc = pandas.read_parquet(save_dir / "object_assoc.parquet")
        cluster_stats = pandas.read_parquet(save_dir / "cluster_stats.parquet")
        object_stats = pandas.read_parquet(save_dir / "object_stats.parquet")

        cluster_assoc, object_assoc = matcher._filterAssocByRange(
            cluster_assoc, object_assoc, x_range, y_range
        )

        matcher._loadReducedData(save_dir)
        per_cell_data = matcher._buildPerCellData(cluster_assoc)
        matcher._reconstructCells(
            cluster_assoc, object_assoc, cluster_stats, object_stats, per_cell_data
        )
        return matcher
