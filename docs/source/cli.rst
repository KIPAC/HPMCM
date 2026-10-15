*********
hpmcm CLI
*********

=========================
Matching using global WCS
=========================

.. click:: hpmcm.cli.commands:wcsMatchCommand
   :prog: hpmcm wcs match
   :nested: full


=====================================
Matching using cell-based coadd frame
=====================================

.. click:: hpmcm.cli.commands:shearMatchCommand
   :prog: hpmcm shear match
   :nested: full

Input catalog modes
-------------------

The ``shear match`` command accepts catalogs in two ways:

**Multiple positional inputs** — one pre-split parquet file per shear type,
passed as positional arguments.  Files should be in reverse alphabetical order
(``ns`` last) to match the expected catalog-ID ordering::

    hpmcm shear match ns.pq 1p.pq 1m.pq 2p.pq 2m.pq --output-file out.pq

This is the classic workflow: split the raw catalog with ``split-rubin`` or
``split-desc`` first, then pass the resulting per-type files here.

**Single joined catalog** — pass ``--joined-input`` with one file that contains
all shear types.  Use ``--catalog-format`` to tell the command what format the
file is in:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - ``--catalog-format``
     - Input expected
     - Extra options
   * - ``pre-processed`` *(default)*
     - Coordinates already computed (``x_cell_coadd``, ``cell_idx_x/y``
       present); shear type identified by ``--shear-type-col`` (default
       ``meta_step``).
     - ``--shear-type-col`` if the column is named differently
   * - ``rubin-md``
     - Raw joined Rubin MD catalog (``metaStep``, ``cell_x/y``, ``patch``,
       ``x/y``, ``shearObjectId``, …).  Tract/patch filtering and coordinate
       computation are applied automatically.
     - ``--tract`` *(required)*
   * - ``desc-md``
     - Raw joined DESC MD catalog (``mcal_step``, ``cell_i/j``,
       ``xcell/ycell``, ``x/y``, …).  Patch filtering and coordinate
       computation are applied automatically.
     - —

Example — raw Rubin MD catalog::

    hpmcm shear match \
        --joined-input raw_rubin.parq \
        --catalog-format rubin-md \
        --tract 10463 \
        --deshear --shear 0.01 \
        --output-file out.pq

Example — raw DESC MD catalog::

    hpmcm shear match \
        --joined-input raw_desc.parq \
        --catalog-format desc-md \
        --deshear --shear 0.01 \
        --output-file out.pq

Example — pre-processed joined catalog (e.g.\ files concatenated after
running ``split-rubin``, where the type column is ``meta_step``)::

    hpmcm shear match \
        --joined-input joined.parq \
        --output-file out.pq

When ``--joined-input`` is used, the command internally calls
:func:`~hpmcm.shear_utils.prepareRubinMDForMatch` or
:func:`~hpmcm.shear_utils.prepareDESCMDForMatch` (for raw formats) and then
:meth:`~hpmcm.shear_match.ShearMatch.splitAndReduce` to populate the matcher
in memory without writing intermediate files.

================================================
Splitting Rubin input catalogs for shear matching
================================================

.. click:: hpmcm.cli.commands:shearSplitRubinCommand
   :prog: hpmcm shear split-rubin
   :nested: full


================================================
Splitting DESC input catalogs for shear matching
================================================

.. click:: hpmcm.cli.commands:shearSplitDESCCommand
   :prog: hpmcm shear split-desc
   :nested: full


================================
Making shear calibration reports
================================


.. click:: hpmcm.cli.commands:shearReportCommand
   :prog: hpmcm shear report
   :nested: full


=================================
Merging shear calibration reports
=================================


.. click:: hpmcm.cli.commands:shearMergeReportsCommand
   :prog: hpmcm shear merge-reports
   :nested: full
