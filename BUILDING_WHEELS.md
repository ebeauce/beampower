# Building and Distributing BeamPower Wheels

This guide explains how to build, test, and distribute pre-compiled binary wheels for the `beampower` package so users don't need to compile the C/CUDA code.

## Overview

Instead of requiring users to have compilers and the CUDA toolkit installed, `beampower` is now distributed as pre-built binary wheels. Users can simply run:

```bash
pip install beampower
```

## Building Wheels Locally

### Prerequisites

- Python 3.8+
- `build` package: `pip install build wheel`
- For CPU wheels: GCC/Clang compiler, OpenMP
- For GPU wheels: NVIDIA CUDA Toolkit, NVIDIA cuDNN

### Build CPU Wheel

```bash
# Build the CPU library first
make clean
make python_CPU

# Create the wheel
python -m build --wheel
```

### Build GPU Wheel

```bash
# Build the GPU library (requires CUDA)
make clean
make python_GPU

# Create the wheel
python -m build --wheel
```

### Build Both CPU and GPU in the Same Wheel

```bash
make clean
make  # builds both CPU and GPU

python -m build --wheel
```

The resulting `.whl` file will be in the `dist/` directory.

## Automated Building with GitHub Actions (Recommended)

For cross-platform, multi-architecture builds, use the included GitHub Actions workflow.

### Setup

1. Ensure you have a GitHub repository for beampower
2. Add the `.github/workflows/build-wheels.yml` workflow file (see template below)
3. Push to GitHub

### Workflow Template

Create `.github/workflows/build-wheels.yml`:

```yaml
name: Build and publish wheels

on:
  push:
    tags:
      - 'v*'  # Build on version tags
  workflow_dispatch:  # Allow manual builds

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build wheel
        # For Ubuntu: sudo apt-get install gcc make libomp-dev
        # For macOS: brew install libomp (if needed)
    
    - name: Build CPU library
      run: |
        make clean
        make python_CPU
    
    - name: Build wheel
      run: python -m build --wheel
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        path: ./dist/*.whl
    
    - name: Publish to PyPI
      if: startsWith(github.ref, 'refs/tags/')
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        pip install twine
        twine upload dist/*.whl --skip-existing

  build_wheels_gpu:
    name: Build GPU wheels (CUDA)
    runs-on: ubuntu-latest
    container: nvidia/cuda:11.8.0-devel-ubuntu22.04
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install dependencies
      run: |
        apt-get update
        apt-get install -y python3 python3-pip make gcc g++ git
        python3 -m pip install --upgrade pip build wheel
    
    - name: Build GPU library
      run: |
        make clean
        make python_GPU
    
    - name: Build wheel
      run: python3 -m build --wheel
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        path: ./dist/*.whl
    
    - name: Publish to PyPI
      if: startsWith(github.ref, 'refs/tags/')
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        python3 -m pip install twine
        twine upload dist/*.whl --skip-existing
```

### Using cibuildwheel (Advanced Alternative)

For maximum compatibility across Python versions and architectures, consider using `cibuildwheel`:

```yaml
name: Build wheels

on:
  push:
    tags: ['v*']

jobs:
  build_wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]

    steps:
      - uses: actions/checkout@v3
      - uses: PyO3/maturin-action@v1
        with:
          command: build
          args: --release --out dist
      - name: Upload wheels
        uses: actions/upload-artifact@v3
        with:
          path: dist/*.whl
```

## Publishing to PyPI

### Option 1: Automated Publishing via GitHub Actions

1. Create a PyPI API token at https://pypi.org/account/
2. Add it to GitHub secrets as `PYPI_API_TOKEN`
3. Push a version tag: `git tag v1.0.3 && git push --tags`
4. GitHub Actions will automatically build and publish wheels

### Option 2: Manual Publishing

```bash
# Build all wheels locally
make clean
make python_CPU
python -m build --wheel

# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*
```

### Option 3: Test PyPI First

```bash
twine upload --repository testpypi dist/*
```

Then test installation:

```bash
pip install --index-url https://test.pypi.org/simple/ beampower
```

## Platform-Specific Considerations

### Linux (CPU)
- Uses `gcc`, `make`, and OpenMP
- `.so` files target Linux x86_64

### Linux (GPU)
- Requires NVIDIA CUDA Toolkit
- Builds `.so` files with CUDA support
- Docker/container recommended for reproducibility

### macOS
- Use `clang` (Apple Silicon) or `gcc` (Intel)
- Generates `.so` or `.dylib` files
- May need `brew install libomp`

### Windows
- Requires MinGW or Microsoft Visual C++
- Generates `.pyd` files
- More complex; recommend focusing on Linux/macOS first

## Verifying Wheels

After building, verify the wheel contents:

```bash
# List contents
unzip -l dist/beampower-1.0.3-py3-none-any.whl

# Should include:
# - beampower/__init__.py
# - beampower/beampower.py
# - beampower/core.py
# - beampower/lib/beamform_cpu.so (or .pyd, .dylib)
# - beampower/lib/beamform_gpu.so (optional)
```

Test installation:

```bash
pip install --force-reinstall dist/beampower-1.0.3-py3-none-any.whl
python -c "import beampower; print(beampower.__version__)"
```

## Troubleshooting

### Missing `.so` files in wheel

Ensure `MANIFEST.in` includes:
```
recursive-include beampower/lib *.so
recursive-include beampower/lib *.pyd
recursive-include beampower/lib *.dylib
```

And run `make` before `python -m build --wheel`.

### "ModuleNotFoundError: No module named 'beampower.lib'"

The `.so` files weren't included in the wheel. Rebuild using the correct MANIFEST.in.

### CUDA not found when building GPU

Ensure `nvcc` is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
which nvcc
```

### Different Python versions

Build wheels for each supported Python version:

```bash
for python_version in 3.8 3.9 3.10 3.11 3.12; do
    python${python_version} -m build --wheel
done
```

## Distribution Channels

1. **PyPI (Python Package Index)** - Recommended for public packages
2. **GitHub Releases** - Good for alpha/beta versions
3. **Private PyPI Server** - For internal/proprietary packages
4. **Conda-Forge** - For conda users
5. **Direct downloads** - For enterprise customers

## Post-Release

After publishing:

1. Tag the release: `git tag v1.0.3`
2. Create GitHub Release with notes
3. Announce on relevant channels (email, forums, etc.)
4. Update documentation links
5. Monitor for installation issues

## References

- [Python Packaging Guide](https://packaging.python.org/)
- [Wheel Documentation](https://wheel.readthedocs.io/)
- [setuptools Documentation](https://setuptools.pypa.io/)
- [cibuildwheel Documentation](https://cibuildwheel.readthedocs.io/)
