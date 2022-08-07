# Welcome to the __beampower__ repository!

![](scripts/figures/logo.png)<br><br><br>

`beampower` is a package for beamforming (or backprojection) of seismic signal for event detection and location. The Python wrapper
can call the C (CPU) or CUDA-C (GPU) implementation. See the documentation at [https://ebeauce.github.io/beampower/](https://ebeauce.github.io/beampower/).

# How to cite this package

There is no publication (yet) for this repository, but if you use it, please acknowledge it in your manuscript's _Data and Resources_ or _Acknowledgements_ section.

# Installation (manual build)

Download the repository on your computer at any location with the following command or with another GitHub repository manager

    git clone https://github.com/ebeauce/beamnetresponse.git

Then, from the root of the repository, run the following commands:

    cd beamnetresponse
    python setup.py build_ext
    pip install .

# Installation (via build `pip`)

From anywhere, run:

    pip install git+https://github.com/ebeauce/beampower


# Documentation and tutorials

See the [documentation](https://ebeauce.github.io/beampower/) on how to use `beampower` to detect and locate earthquakes. The package also comes with several tutorial notebooks (included also in the doc): 

- [Download data](notebooks/0_download.ipynb)
- [Pre-process data](notebooks/1_preprocess.ipynb)
- [Calculate travel times](notebooks/2_travel_times.ipynb)
- [Locate events](notebooks/3_localization.ipynb)

These notebooks require to install the following packages to be ran properly:

- `xarray`
- `obspy>=1.3.0`  
- `matplotlib`  
- `tqdm`  
- `pykonal`



