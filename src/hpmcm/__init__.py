"""
The :py:mod:`hpmcm` package collects a set of tools to do catalog
cross matching.
"""

from . import classify, match_utils, shear_utils, utils, viz_utils
from ._version import __version__
from .cell import CellData, ShearCellData
from .cluster import ClusterAssocTable, ClusterData, ClusterStatsTable, ShearClusterData
from .match import Match
from .object import ObjectAssocTable, ObjectData, ObjectStatsTable, ShearObjectData
from .shear_data import ShearData, ShearTable
from .shear_match import ShearMatch
from .table import TableColumnInfo, TableInterface
from .wcs_match import WcsMatch

__all__ = [
    "CellData",
    "ClusterAssocTable",
    "ClusterData",
    "ClusterStatsTable",
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
    "match_utils",
    "shear_utils",
    "utils",
    "viz_utils",
    "__version__",
]
