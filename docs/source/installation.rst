************
Installation
************


`hpmcm` can be requires the Rubin software stack.  We assume that
you have installed the stack into a conda environment.



=======================
Production Installation
=======================

Here we will be installing ``hpmcm`` into an existing conda environment "[env]".

.. code-block:: bash

    conda activate [env]
    pip install hpmcm


======================
Developer Installation
======================

Here we will be installing the source code from `HPMCM
<https://github.com/KIPAC/HPMCM>`_ to be able to develop
the source code.


.. tabs::

   .. group-tab:: General

      .. code-block:: bash

	  conda activate [env]
          git clone https://github.com/KIPAC/HPMCM.git
          cd rail_pz_service
          pip install -e .[dev]


   .. group-tab:: zsh (e.g., Mac M1+ default)

      .. code-block:: bash

	  conda activate [env]
          git clone https://github.com/KIPAC/HPMCM.git
          cd rail_pz_service
          pip install -e '.[dev]'

