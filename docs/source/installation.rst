************
Installation
************


There are two ways you might choose to install `HPMCM`

1. `Production Installation`_: Just install `HPMCM` in an
   existing an existing conda environment using pip.
2. `Developer Installation`_: Download the `HPMCM` source
   code and example notebooks and install from the local version using
   pip in "editable" mode.



=======================	  
Production Installation
=======================   

Here we will be installing ``HPMCM`` into an existing conda environment "[env]".

.. code-block:: bash

    conda activate [env]
    pip install 	


======================	  
Developer Installation
======================   

.. tabs::

   .. group-tab:: General

      .. code-block:: bash

	  conda activate [env]
          git clone https://github.com/KIPAC/HPMCM.git
          cd HPMCM
          pip install -e .[dev]


   .. group-tab:: zsh (e.g., Mac M1+ default)

      .. code-block:: bash

	  conda activate [env]
          git clone https://github.com/KIPAC/HPMCM.git
          cd rail_pz_service
          pip install -e '.[dev]'


=============================
Adding your kernel to jupyter
=============================
If you want to use the kernel that you have just created to run example demos, then you may need to explicitly add an ipython kernel.  You may need to first install ipykernel with `conda install ipykernel`.  You can do then add your kernel with the following command, making sure that you have the conda environment that you wish to add activated.  From your environment, execute the command:
`python -m ipykernel install --user --name [nametocallnewkernel]`
(you may or may not need to prepend `sudo` depending on your permissions).  When you next start up Jupyter you should see a kernel with your new name as an option, including using the Jupyter interface at NERSC.
>>>>>>> 6908ec0 (docs)
