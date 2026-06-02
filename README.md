# Welcome to the __beampower__ repository!

<p align="center">
<img src="data/beampower_logo.png" width=350>
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

The simplest way to use BeamPower:

```bash
pip install beampower
```

That's it! No compilation needed. GPU support is included if available.

## Learn More

- **Documentation**: https://ebeauce.github.io/beampower/
- **Tutorials**: See the notebooks in the `notebooks/` folder
  - Download seismic data
  - Pre-process waveforms
  - Calculate travel times
  - Locate events

## For Developers

**Clone and install locally:**

```bash
git clone https://github.com/ebeauce/beampower.git
cd beampower
pip install .
```

**If you modify C or CUDA code:**

```bash
# For C code changes:
make clean
make python_CPU

# For CUDA code changes:
make clean
make python_GPU

# For both:
make clean
make

# Then test the wheel:
./build_wheels.sh cpu
```

## Publishing a New Release

See [RELEASE.md](RELEASE.md) for instructions on tagging and publishing.

