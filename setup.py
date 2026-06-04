from setuptools import setup, find_packages
from setuptools.dist import Distribution

# This setup.py assumes pre-built .so files are in beampower/lib/
# For building wheels from source, see: BUILDING_WHEELS.md

class BinaryDistribution(Distribution):
    """Forces the wheel to be recognized as platform-specific (not none-any)"""
    def has_ext_modules(self):
        return True
    
setup(
    packages=['beampower'],
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
)
