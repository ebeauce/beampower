# beamnetresponse

Package for beamforming or backprojection of seismic data. The Python wrapper
can call the C (CPU) or CUDA-C (GPU, work-in-progress, coming soon!) implementation.

## Reference

There is no publication yet for this repository, but if you use it, please, acknowledge
it in the Data and Resources or Acknowledgements section of your manuscript.

## Installation

### Method 1)

From the root directory:

    python setup.py build_ext
    pip install .

### Method 2)

From anywhere:

    pip install git+https://github.com/ebeauce/beamnetresponse

## Tutorial

The repository comes with a "scripts" folder where a number of programs can be ran
in the following order:
- 0_download_data.py: Download the data used in the tutorial (region = North Anatolian Fault).
- 1_make_asdf_file.py: Format the data in the ASDF format.
- 2_preprocessing.py: Preprocess the data.
- 3_build_tt_table.py: Compute the grid of travel-times.
- 4_compute_cnr_NAF.py: Compute the composite network response.
- 5_show_results.ipynb: A notebook to visualize the detection results.

This tutorial requires many more packages than the beamnetresponse package itself.
First, create a new environment running python 3.8. Then, install the following packages:
- pyasdf [https://seismicdata.github.io/pyasdf/installation.html](https://seismicdata.github.io/pyasdf/installation.html)
- pykonal [https://github.com/malcolmw/pykonal](https://github.com/malcolmw/pykonal)
- obspy
- jupyter
- numpy
- scipy
- matplotlib
