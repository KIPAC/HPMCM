"""
The :py:mod:`hpmcm` package collects a set of tools to do catalog
cross matching using the following strategy.

1. The match region is mapped onto a single WCS and each sources
   is projected into pixel coordinates in that WCS.  This is defined
   using a sub-class of :py:class:`hpmcm.match.Match`

2. The match region is sub-divided into
   :py:class:`cells <hpmcm.cell.CellData>`, which include some
   overlap region between them.

3. For each cell, a "counts map" of sources in that cell is produced,
   counting the number of sources in each pixel of the projected WCS.

4. A clustering algorithm is run to extract the "footprints" of sets
   of adjacent pixels that contain sources.   Those are then used to
   create a set of :py:class:`clusters <hpmcm.cluster.ClusterData>`

5. Each cluster in then refined into
   :py:class:`objects <hpmcm.cluster.ObjectData>`, by repeating the
   counts map and footprint detection using progressive smaller pixels
   until each objects consists only of sources within the requested
   match radius.  At each iteration the pixel scale is halved.

6. Information about the source-to-object and source-to-clusters
   associations, as well information about the resulting objects
   and clusters are collected into :py:mod:`hpmcm.output_tables`


The two sub-classes of `Match` are:

1. :py:class:`WcsMatch <hpmcm.wcs_match.WcsMatch>` allows the user to
   define the WCS used for the projection and takes any number of input
   catalogs.

2. :py:class:`ShearMatch <hpmcm.shear_match.ShearMatch>` uses a predefined
   WCS that was used to create the meta-detection ShearObject catalogs, and
   expects exactly 5 input catalogs: a reference catalog and 4 counterfactual
   shear catalogs.
"""

from . import classify, match_utils, package_utils, shear_utils, utils, viz_utils
from ._version import __version__
from .cell import CellData, ShearCellData
from .cluster import ClusterData, ShearClusterData
from .footprint import Footprint, FootprintSet
from .match import Match
from .object import ObjectData, ShearObjectData
from .output_tables import (
    ClusterAssocTable,
    ClusterStatsTable,
    ObjectAssocTable,
    ObjectStatsTable,
    ShearTable,
)
from .shear_data import ShearData
from .shear_match import ShearMatch
from .table import TableColumnInfo, TableInterface
from .wcs_match import WcsMatch

__all__ = [
    "CellData",
    "ClusterAssocTable",
    "ClusterData",
    "ClusterStatsTable",
    "Footprint",
    "FootprintSet",
    "Match",
    "ObjectAssocTable",
    "ObjectData",
    "ObjectStatsTable",
    "ShearCellData",
    "ShearClusterData",
    "ShearData",
    "ShearTable",
    "ShearObjectData",
    "ShearMatch",
    "TableColumnInfo",
    "TableInterface",
    "WcsMatch",
    "classify",
    "match_utils",
    "package_utils",
    "shear_utils",
    "utils",
    "viz_utils",
    "__version__",
]
