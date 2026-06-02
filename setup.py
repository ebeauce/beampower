from setuptools import setup, find_packages

# This setup.py assumes pre-built .so files are in beampower/lib/
# For building wheels from source, see: BUILDING_WHEELS.md

setup(
    packages=['beampower'],
    include_package_data=True,
    zip_safe=False,
)
