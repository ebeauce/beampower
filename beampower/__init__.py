"""
beampower is a Python package wrapping a C and CUDA-C implementation
of beamforming, sometimes also called back-projection.  

If you have any question, don't hesitate contacting me at: 
ebeauce@ldeo.columbia.edu
"""

__all__ = ["load_library", "beamform"]

from .core import load_library
from .beampower import beamform

__version__ = "1.0.1"
