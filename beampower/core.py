# coding: utf-8

import os
import ctypes as ct


DIRPATH_LIBRARIES = os.path.join(os.path.dirname(__file__), "lib")

LIBRARIES = {
    "cpu": {
        "filepath_library": os.path.join(DIRPATH_LIBRARIES, "beamform_cpu.so"),
        "is_loaded": False,
        "lib": None,
        "beamform_argtypes": [
            ct.POINTER(ct.c_float),  # waveform_features
            ct.POINTER(ct.c_int),  # time_delays
            ct.POINTER(ct.c_float),  # weights_sources
            ct.c_size_t,  # n_samples
            ct.c_size_t,  # n_sources
            ct.c_size_t,  # n_stations
            ct.c_size_t,  # n_phases
            ct.c_int,  # out_of_bounds
            ct.POINTER(ct.c_float),  # beam
        ],
        "beamform_differential_argtypes": [
            ct.POINTER(ct.c_float),  # waveform_features
            ct.POINTER(ct.c_int),  # time_delays
            ct.POINTER(ct.c_float),  # weights_sources
            ct.c_size_t,  # n_samples
            ct.c_size_t,  # n_sources
            ct.c_size_t,  # n_stations
            ct.c_size_t,  # n_phases
            ct.POINTER(ct.c_float),  # beam
        ],
        "beamform_argmax_argtypes": [
            ct.POINTER(ct.c_float),  # waveform_features
            ct.POINTER(ct.c_int),  # time_delays
            ct.POINTER(ct.c_float),  # weights_sources
            ct.c_size_t,  # n_samples
            ct.c_size_t,  # n_sources
            ct.c_size_t,  # n_stations
            ct.c_size_t,  # n_phases
            ct.POINTER(ct.c_float),  # beam_max
            ct.POINTER(ct.c_int),  # beam_argmax
        ],
        "prestack_waveform_features_argtypes": [
            ct.POINTER(ct.c_float),  # waveform_features
            ct.POINTER(ct.c_float),  # weights_phases
            ct.c_size_t,  # n_sources
            ct.c_size_t,  # n_stations
            ct.c_size_t,  # n_channels
            ct.c_size_t,  # n_phases
            ct.POINTER(ct.c_float),  # prestacked_traces
        ],
    },
    "gpu": {
        "filepath_library": os.path.join(DIRPATH_LIBRARIES, "beamform_gpu.so"),
        "is_loaded": False,
        "lib": None,
        "beamform_argtypes": [
            ct.POINTER(ct.c_float),  # waveform_features
            ct.POINTER(ct.c_int),  # time_delays
            ct.POINTER(ct.c_float),  # weights_sources
            ct.c_size_t,  # n_samples
            ct.c_size_t,  # n_sources
            ct.c_size_t,  # n_stations
            ct.c_size_t,  # n_phases
            ct.POINTER(ct.c_float),  # beam
        ],
        "beamform_argmax_argtypes": [
            ct.POINTER(ct.c_float),  # waveform_features
            ct.POINTER(ct.c_int),  # time_delays
            ct.POINTER(ct.c_float),  # weights_sources
            ct.c_size_t,  # n_samples
            ct.c_size_t,  # n_sources
            ct.c_size_t,  # n_stations
            ct.c_size_t,  # n_phases
            ct.POINTER(ct.c_float),  # beam_max
            ct.POINTER(ct.c_int),  # beam_argmax
        ],
    },
}


def load_library(device="cpu"):
    """Load library for device.

    This function loads the library only once, that is, if the library is
    successfully loaded a first time, it will be stored as a persistent
    variable in the LIBRARIES dictionary.

    Parameters
    ----------
    device: str, optional
        Device-compilated library, either "cpu" or "gpu".
    """
    # Get device name
    device_name = device.lower()

    # Check device name
    if device.lower() not in LIBRARIES:
        raise NameError(f"Device should be cpu or gpu, not {device.lower()}")

    # Check if shared object exsits
    library_info = LIBRARIES[device_name]
    filepath_library = library_info["filepath_library"]
    if os.path.exists(filepath_library):

        # If library was previously loaded, return it
        if library_info["is_loaded"] is True:
            return library_info["lib"]

        # Otherwise load it
        else:

            # Load
            lib = ct.cdll.LoadLibrary(filepath_library)

            # Declare types
            lib.beamform.argtypes = library_info["beamform_argtypes"]
            lib.beamform_max.argtypes = library_info["beamform_argmax_argtypes"]
            if device_name == "cpu":
                lib.prestack_waveform_features.argtypes = library_info[
                    "prestack_waveform_features_argtypes"
                ]
                lib.beamform_differential.argtypes = library_info[
                    "beamform_differential_argtypes"
                ]

            # Store pre-loaded library
            LIBRARIES[device_name]["is_loaded"] = True
            LIBRARIES[device_name]["lib"] = lib

            return lib

    else:
        raise OSError(
            f"Library for {device_name.upper()} was not compiled. The shared object {filepath_library} does not exist. \nYou may want to check the output of the command: \n\n \t python setup.py build_ext"
        )
