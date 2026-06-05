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


## Installation

The simplest way to use BeamPower:

```bash
pip install beampower
```

GPU support is included if available.

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
# option 1:
pip install .
# option 2 (check name of wheel in dist/):
python -m build
pip install dist/beampower-XXXX.whl 
```

**If you modify C or CUDA code:**

```bash
# option 1:
pip install . --no-cache-dir --force-reinstall
# option 2:
# Clean up previous Python and CMake build artifacts
rm -rf build/ dist/
# Re-run the build from scratch
python -m build
# Then reinstall to pick up changes:
pip install --force-reinstall dist/beampower-XXXX.whl
```

## Reference
Please, if you use this package for your research, cite:

Beaucé, Eric and Frank, William B. and Seydoux, Léonard and Poli, Piero and Groebner, Nathan
and van der Hilst, Robert D. and Campillo, Michel (2023). BPMF: A Backprojection and Matched‐Filtering Workflow for Automated Earthquake Detection and Location. *Seismological Research Letters*. DOI: [https://doi.org/10.1785/0220230230](https://doi.org/10.1785/0220230230).
