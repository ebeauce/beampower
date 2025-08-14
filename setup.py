from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext as build_ext_original
from subprocess import call

class BPExtension(Extension):
    def __init__(self, name):
        # No sources, since Makefile handles compilation
        super().__init__(name=name, sources=[])


# Define a new build command
class BeamPowerBuild(build_ext_original):
    def run(self):
        cpu_make = ['make', 'python_CPU']
        gpu_make = ['make', 'python_GPU']

        gpu_built = call(gpu_make) == 0
        cpu_built = call(cpu_make) == 0

        if not gpu_built:
            print("Could not build GPU code")
        if not cpu_built:
            raise OSError("Could not build CPU code")



setup(
    packages=['beampower'],
    include_package_data=True,
    zip_safe=False,
    cmdclass={'build_ext': BeamPowerBuild},
    ext_modules=[
        BPExtension('beampower.lib.beamform_cpu'),
        BPExtension('beampower.lib.beamform_gpu')
                 ]
)
