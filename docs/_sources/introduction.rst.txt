Introduction
============

Description
-----------
Fast routines for seismic backprojection/beamforming for both CPU and GPU architectures.

:py:data:`beampower` (BP) implements the beamformed network response introduced in *Frank et al.* (2014) in C and CUDA-C. This library can be used for event detection and location, but also for the imaging of large fault ruptures.

If you use BNR in research to be published, please reference the following articles:

* Frank, William B., Nikolaï M. Shapiro, Allen L. Husker, Vladimir Kostoglodov, Alexey Romanenko, and Michel Campillo. (2014). Using systematically characterized low‐frequency earthquakes as a fault probe in Guerrero, Mexico. *Journal of Geophysical Research: Solid Earth*, doi: `10.1002/2014JB011457 <https://doi.org/10.1002/2014JB011457>`_
* Beaucé, Eric, William B. Frank, Anne Paul, Michel Campillo, and Robert D. van der Hilst (2019). Systematic detection of clustered seismicity beneath the Southwestern Alps. *Journal of Geophysical Research: Solid Earth*, doi: `10.1029/2019JB018110 <https://doi.org/10.1029/2019JB018110>`_

:py:data:`beampower` is available at `https://github.com/ebeauce/beampower <https://github.com/ebeauce/beampower>`_ and can be downloaded with:

.. code-block:: console

    $ git clone https://github.com/ebeauce/beampower.git

Installation
-------------

You may need to edit the Makefile according to your OS (instructions in the Makefile's comments).

After cloning or downloading :py:data:`beampower`, go to the root directory and run:

.. code-block:: console

    $ pip install .

or simply (but without the possibility to edit the Makefile):

.. code-block:: console

    $ pip install git+https://github.com/ebeauce/beampower


Required software/hardware:
---------------------------
* A C compiler that supports OpenMP (clang with Mac computers is fine but must be used with specific flags, see Makefile), 
* CPU version:  Python 3.x, 
* GPU version: Python 3.x and a discrete Nvidia graphics card that supports CUDA C with CUDA toolkit installed. 

:py:data:`beampower` is known to run well with gcc 6.2.X, and cuda 10.X.
