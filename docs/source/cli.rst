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


=============================================
Splitting Rubin input catalogs for shear matching
=============================================

.. click:: hpmcm.cli.commands:shearSplitRubinCommand
   :prog: hpmcm shear split-rubin
   :nested: full


=============================================
Splitting DESC input catalogs for shear matching
=============================================

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
