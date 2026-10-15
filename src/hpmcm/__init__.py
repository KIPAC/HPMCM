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
   :py:class:`objects <hpmcm.object.ObjectData>`, by repeating the
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
   WCS that was used to create the meta-detection ShearObject catalogs.
   By default it expects 5 input catalogs (a reference catalog ``ns`` and
   4 counterfactual shear catalogs ``1p``, ``1m``, ``2p``, ``2m``), but the
   active set is configurable via the ``shear_names`` parameter, allowing
   e.g. 3-catalog mode (``["ns", "1p", "1m"]``).


Additional features
-------------------

* **Serialization** — both matchers support :py:meth:`save()
  <hpmcm.match.Match.save>` and :py:meth:`load()
  <hpmcm.wcs_match.WcsMatch.load>` to persist a completed match to disk and
  reload it, optionally filtering by cell range (``x_range`` / ``y_range``)
  for parallelised workflows.

* **Joined tables** — :py:func:`buildJoinedObjectTable
  <hpmcm.output_tables.buildJoinedObjectTable>` and
  :py:func:`buildJoinedClusterTable
  <hpmcm.output_tables.buildJoinedClusterTable>` combine the association and
  statistics tables into a single flat table for downstream analysis.

* **In-memory catalog loading** — :py:meth:`Match.setReducedData()
  <hpmcm.match.Match.setReducedData>` lets you pass already-reduced
  DataFrames directly to any matcher, bypassing file I/O.
  :py:meth:`ShearMatch.splitAndReduce()
  <hpmcm.shear_match.ShearMatch.splitAndReduce>` accepts a single joined
  DataFrame (all shear types concatenated with a type column), splits it by
  type, and loads the result — no intermediate per-type files needed.

* **Raw catalog preparation** — :py:func:`shear_utils.prepareRubinMDForMatch
  <hpmcm.shear_utils.prepareRubinMDForMatch>` and
  :py:func:`shear_utils.prepareDESCMDForMatch
  <hpmcm.shear_utils.prepareDESCMDForMatch>` apply the same spatial
  filtering and coordinate computation as the ``split-rubin`` / ``split-desc``
  CLI commands but return an in-memory joined DataFrame instead of writing
  files.  Pass the result directly to ``splitAndReduce``.
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
    ClusterShearTable,
    ClusterStatsTable,
    ObjectAssocTable,
    ObjectShearTable,
    ObjectStatsTable,
    ShearTable,
    buildJoinedClusterTable,
    buildJoinedObjectTable,
    computeColumnStats,
    reduceJoinedTable,
)
from .shear_data import ShearData
from .shear_match import ShearMatch
from .table import TableColumnInfo, TableInterface
from .wcs_match import WcsMatch

__all__ = [
    "CellData",
    "ClusterAssocTable",
    "ClusterData",
    "ClusterShearTable",
    "ClusterStatsTable",
    "buildJoinedClusterTable",
    "buildJoinedObjectTable",
    "computeColumnStats",
    "reduceJoinedTable",
    "Footprint",
    "FootprintSet",
    "Match",
    "ObjectAssocTable",
    "ObjectData",
    "ObjectShearTable",
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
