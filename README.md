# Welcome to the __beampower__ repository!

<p align="center">
<img src="data/beampower_logo.png" width=500>
</p><br><br><br><br>


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![](https://img.shields.io/github/commit-activity/w/ebeauce/beampower)
![](https://img.shields.io/github/last-commit/ebeauce/beampower)
![](https://img.shields.io/github/stars/ebeauce/beampower?style=social)

### About the package
`beampower` is a package for beamforming (or backprojection) of seismic signal for event detection and location. The Python wrapper
can call the C (CPU) or CUDA-C (GPU) implementation. See the documentation at [https://ebeauce.github.io/beampower/](https://ebeauce.github.io/beampower/).

### How to cite this package

There is no publication (yet) for this repository, but if you use it, please acknowledge it in your manuscript's _Data and Resources_ or _Acknowledgements_ section.

## Installation 

### Option 1: manual build

Download the repository on your computer at any location with the following command or with another GitHub repository manager

    git clone https://github.com/ebeauce/beampower.git

Then, from the root of the repository, run the following commands:

    python setup.py build_ext
    pip install .

### Option 2: via `pip`

From anywhere, run:

    pip install git+https://github.com/ebeauce/beampower


## Documentation and tutorials

See the [documentation](https://ebeauce.github.io/beampower/) on how to use `beampower` to detect and locate earthquakes. The package also comes with several tutorial notebooks (included also in the doc): 

- [Download data](notebooks/0_download.ipynb)
- [Pre-process data](notebooks/1_preprocess.ipynb)
- [Calculate travel times](notebooks/2_travel_times.ipynb)
- [Locate events](notebooks/3_localization.ipynb)

These notebooks require to install the following packages to be ran properly:

- `obspy>=1.3.0`  
- `matplotlib`  
- `tqdm`  
- `pykonal`

