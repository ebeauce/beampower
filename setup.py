"""
Minimal setup file for the beampower library for Python packaging.
:copyright:
    Eric Beauce, William B. Frank, Leonard Seydoux
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

from __future__ import print_function
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_original
from subprocess import call


class BPExtension(Extension):
    def __init__(self, name):
        # Don't run the default setup-tools build commands, use the custom one
        Extension.__init__(self, name=name, sources=[])


# Define a new build command
class BeamPowerBuild(build_ext_original):
    def run(self):
        # Build the Python libraries via Makefile
        cpu_make = ['make', 'python_CPU']
        gpu_make = ['make', 'python_GPU']

        gpu_built = False
        cpu_built = False

        ret = call(cpu_make)
        if ret == 0:
            cpu_built = True
        ret = call(gpu_make)
        if ret == 0:
            gpu_built = True
        if gpu_built is False:
            print("Could not build GPU code")
        if cpu_built is False:
            raise OSError("Could not build cpu code")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="beampower",
    version="1.0.1",
    author="Eric Beauce, William B. Frank, Leonard Seydoux",
    author_email="ebeauce@ldeo.columbia.edu",
    description="Package for beamforming/backprojection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebeauce/beampower",
    project_urls={
        "Bug Tracker": "https://github.com/ebeauce/beampower/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 2.7, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    license="GPL",
    packages=['beampower'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy"
        ],
    python_requires=">=2.7",
    cmdclass={
        'build_ext': BeamPowerBuild},
    ext_modules=[BPExtension('beampower.lib.beamform_cpu'),
                 BPExtension('beampower.lib.beamform_gpu')]
)
