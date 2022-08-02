# beampower

Package for beamforming or backprojection of seismic data. The Python wrapper
can call the C (CPU) or CUDA-C (GPU) implementation. See the documentation at [https://ebeauce.github.io/beampower/](https://ebeauce.github.io/beampower/).

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

    pip install git+https://github.com/ebeauce/beampower

## Tutorial

See the [documentation](https://ebeauce.github.io/beampower/) for a complete tutorial on how to use `beampower` to detect earthquakes.
