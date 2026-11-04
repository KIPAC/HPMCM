from __future__ import annotations

import numpy as np

from . import match_utils, mst_utils
from .cell import CellData, ShearCellData
from .footprint import Footprint


class MSTCellMixin:
    """Mixin that replaces footprint-based ``analyze()`` with MST grouping.

    Mix this before ``CellData`` or ``ShearCellData`` in the MRO to get
    MST-based initial clustering.  The mixin reads attributes that are
    guaranteed to exist on all ``CellData`` subclasses (``n_src``, ``data``,
    ``matcher``, ``pixel_match_scale``, ``id_offset``, ``cluster_dict``) and
    delegates cluster building to ``_buildClusterData``, which is overridden
    by shear subclasses.
    """

    def analyze(  # type: ignore[override]
        self,
        weight_name: str | None = None,
        pixel_r2_cut: float = 2.0,
    ) -> dict | None:
        """Analyze this cell using MST-based source grouping."""
        if self.n_src == 0:  # type: ignore[attr-defined]
            return None
        self._buildMSTClusterData(pixel_r2_cut)
        return {}

    def _buildMSTClusterData(self, pixel_r2_cut: float) -> None:
        """Group sources via MST and build ClusterData for each component."""
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        i_cat_parts: list[np.ndarray] = []
        src_id_parts: list[np.ndarray] = []
        src_idx_parts: list[np.ndarray] = []

        for i_cat, df in enumerate(self.data):  # type: ignore[attr-defined]
            x_parts.append(df["x_cell"].values)
            y_parts.append(df["y_cell"].values)
            i_cat_parts.append(np.full(len(df), i_cat, dtype=np.int32))
            src_id_parts.append(df["id"].values)
            src_idx_parts.append(np.arange(len(df), dtype=np.int32))

        x_all = np.concatenate(x_parts)
        y_all = np.concatenate(y_parts)
        i_cat_all = np.concatenate(i_cat_parts)
        src_id_all = np.concatenate(src_id_parts)
        src_idx_all = np.concatenate(src_idx_parts)

        matcher = self.matcher  # type: ignore[attr-defined]
        cut_pix: float = matcher.mst_cut / matcher.pixToArcsec()
        labels = mst_utils.mst_group_sources(x_all, y_all, cut_pix)

        pms: int = self.pixel_match_scale  # type: ignore[attr-defined]
        id_offset: int = self.id_offset  # type: ignore[attr-defined]

        for comp_id in np.unique(labels):
            mask = labels == comp_id
            sources = np.array(
                [i_cat_all[mask], src_id_all[mask], src_idx_all[mask]]
            )
            x_comp = x_all[mask]
            y_comp = y_all[mask]
            x_min_ds = int(np.floor(x_comp.min() / pms))
            x_max_ds = int(np.floor(x_comp.max() / pms)) + 1
            y_min_ds = int(np.floor(y_comp.min() / pms))
            y_max_ds = int(np.floor(y_comp.max() / pms)) + 1
            footprint = Footprint.from_bounds(x_min_ds, x_max_ds, y_min_ds, y_max_ds)
            i_cluster = int(comp_id) + id_offset
            cluster = self._buildClusterData(i_cluster, footprint, sources)  # type: ignore[attr-defined]
            self.cluster_dict[i_cluster] = cluster  # type: ignore[attr-defined]
            match_utils.heirarchicalProcessCluster(cluster, self, pixel_r2_cut)  # type: ignore[arg-type]


class MSTCellData(MSTCellMixin, CellData):
    """CellData that uses MST-based source grouping instead of footprint detection."""


class MSTShearCellData(MSTCellMixin, ShearCellData):
    """ShearCellData that uses MST-based source grouping instead of footprint detection."""
