*********
hpmcm CLI
*********
	    
=========================
Matching using global WCS
=========================

.. click:: hpmcm.cli.commands:wcs_match_command
   :prog: hpmcm wcs match
   :nested: full

	    
=====================================	    
Matching using cell-based coadd frame
=====================================

.. click:: hpmcm.cli.commands:shear_match_command
   :prog: hpmcm shear match
   :nested: full

	    
===========================================
Splitting input catalogs for shear matching
===========================================

.. click:: hpmcm.cli.commands:shear_split_command
   :prog: hpmcm shear split
   :nested: full


================================
Making shear calibration reports
================================

	    
.. click:: hpmcm.cli.commands:shear_report_command
   :prog: hpmcm shear report
   :nested: full

	    
=================================
Merging shear calibration reports
=================================

	    
.. click:: hpmcm.cli.commands:shear_merge_reports_command
   :prog: hpmcm shear merge-reports
   :nested: full
