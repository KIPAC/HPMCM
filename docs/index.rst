#########################################################################
rail_projects: a toolkit for managing `RAIL`-based data analysis projects
#########################################################################

***********
Description
***********

`rail_projects` is a tool-kit to manage RAIL-baseed data analysis
projects. 


RAIL is a flexible open-source software library providing tools to produce at-scale photometric redshift data products, including uncertainties and summary statistics, and stress-test them under realistically complex systematics.

RAIL serves as the infrastructure supporting many extragalactic applications of `the Legacy Survey of Space and Time (LSST) <https://www.lsst.org/>`_ on `the Vera C. Rubin Observatory <https://rubinobservatory.org/>`_, including Rubin-wide commissioning activities. 
RAIL was initiated by the Photometric Redshifts (PZ) Working Group (WG) of the `LSST Dark Energy Science Collaboration (DESC) <https://lsstdesc.org/>`_ as a result of the lessons learned from the `Data Challenge 1 (DC1) experiment <https://academic.oup.com/mnras/article/499/2/1587/5905416>`_ to enable the PZ WG Deliverables in the `LSST-DESC Science Roadmap (see Sec. 5.18) <https://lsstdesc.org/assets/pdf/docs/DESC_SRM_latest.pdf>`_, aiming to guide the selection and implementation of redshift estimators in DESC analysis pipelines.

RAIL is developed and maintained by a diverse team comprising DESC Pipeline Scientists (PSs), international in-kind contributors, LSST Interdisciplinary Collaboration for Computing (LINCC) Frameworks software engineers, and other volunteers, but all are welcome to join the team regardless of LSST data rights. 
To get involved, chime in on the issues in any of the RAIL repositories described in the Overview section.

See `guideline for citing RAIL
<https://rail-hub.readthedocs.io/en/latest/source/citing.html>`_ for
guidance on citing RAIL and the underlying algorithms.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   source/overview
   source/installation
  
.. toctree::
   :maxdepth: 2
   :caption: Concepts

   source/rail_project
   source/pipelines
   source/flavors
   source/components

   
.. toctree::
   :maxdepth: 2
   :caption: Details

   source/analysis_components
   source/factories


.. toctree::
   :maxdepth: 2
   :caption: Usage

   source/rail_project_cli
   source/rail_plot_cli
   source/interactive
   
.. toctree::
   :maxdepth: 2
   :caption: Contributing

   source/contributing
   source/fix_an_issue
   source/new_dataset
   source/new_plotter
   source/new_dataset_holder

.. toctree::
   :maxdepth: 2
   :caption: Demonstrations
   
   demos

.. toctree::
   :maxdepth: 4
   :caption: API

   api/rail



