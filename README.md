# beamnetresponse

Package for beamforming or backprojection of seismic data. The Python wrapper
can call the C (CPU) or CUDA-C (GPU) implementation. See the documentation at [https://ebeauce.github.io/beamnetresponse/](https://ebeauce.github.io/beamnetresponse/).

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

See the [documentation](https://ebeauce.github.io/beamnetresponse/) for a complete tutorial on how to use `beamnetresponse` to detect earthquakes.
