from __future__ import print_function

import os

import ctypes as ct
import numpy as np

path = os.path.join(os.path.dirname(__file__), 'lib')
CPU_LOADED = False
GPU_LOADED = False

try:
    _libCPU = ct.cdll.LoadLibrary(os.path.join(path, 'beamformed_nr_CPU.so'))
    _libCPU.network_response.argtypes = [
            ct.POINTER(ct.c_float),   # detection_traces
            ct.POINTER(ct.c_int),     # moveouts
            ct.POINTER(ct.c_float),   # weights_sources
            ct.c_size_t,              # n_samples
            ct.c_size_t,              # n_sources
            ct.c_size_t,              # n_stations
            ct.c_size_t,              # n_phases
            ct.POINTER(ct.c_float)]   # nr
    _libCPU.composite_network_response.argtypes = [
            ct.POINTER(ct.c_float),   # detection_traces
            ct.POINTER(ct.c_int),     # moveouts
            ct.POINTER(ct.c_float),   # weights_sources
            ct.c_size_t,              # n_samples
            ct.c_size_t,              # n_sources
            ct.c_size_t,              # n_stations
            ct.c_size_t,              # n_phases
            ct.POINTER(ct.c_float),   # nr
            ct.POINTER(ct.c_int)]     # source_index_nr
    _libCPU.prestack_detection_traces.argtypes = [
            ct.POINTER(ct.c_float),   # detection_traces
            ct.POINTER(ct.c_float),   # weights_phases
            ct.c_size_t,              # n_sources
            ct.c_size_t,              # n_stations
            ct.c_size_t,              # n_channels
            ct.c_size_t,              # n_phases
            ct.POINTER(ct.c_float)]   # prestacked_traces
    CPU_LOADED = True
except OSError:
    print('beamnetresponse CPU library is not compiled!'
          ' Build the CPU library first in order to use the CPU routines.')

try:
    _libGPU = ct.cdll.LoadLibrary(os.path.join(path, 'beamformed_nr_GPU.so'))
    _libGPU.network_response.argtypes = [
            ct.POINTER(ct.c_float),   # detection_traces
            ct.POINTER(ct.c_int),     # moveouts
            ct.POINTER(ct.c_float),   # weights_sources
            ct.c_size_t,              # n_samples
            ct.c_size_t,              # n_sources
            ct.c_size_t,              # n_stations
            ct.c_size_t,              # n_phases
            ct.POINTER(ct.c_float)]   # nr
    _libGPU.composite_network_response.argtypes = [
            ct.POINTER(ct.c_float),   # detection_traces
            ct.POINTER(ct.c_int),     # moveouts
            ct.POINTER(ct.c_float),   # weights_sources
            ct.c_size_t,              # n_samples
            ct.c_size_t,              # n_sources
            ct.c_size_t,              # n_stations
            ct.c_size_t,              # n_phases
            ct.POINTER(ct.c_float),   # nr
            ct.POINTER(ct.c_int)]     # source_index_nr
    #_libGPU.prestack_detection_traces.argtypes = [
    #        ct.POINTER(ct.c_float),   # detection_traces
    #        ct.POINTER(ct.c_float),   # weights_phases
    #        ct.c_size_t,              # n_sources
    #        ct.c_size_t,              # n_stations
    #        ct.c_size_t,              # n_channels
    #        ct.c_size_t,              # n_phases
    #        ct.POINTER(ct.c_float)]   # prestacked_traces
    GPU_LOADED = True
except OSError:
    print('beamnetresponse GPU library is not compiled!'
          ' Build the GPU library first in order to use the GPU routines.')

def network_response(detection_traces, moveouts, weights_phases,
                     weights_sources, device='cpu'):
    """Compute the beamformed network response.  

    This routine computes and returns the whole network response over the
    entire duration of `detection_traces`. Thus, if the input source grid
    is large, the output might be considerably memory-consuming. Therefore,
    this routine is more appropriate for small scale studies such as the
    rupture imaging of an earthquake.

    Parameters
    ------------
    detection_traces: (n_stations, n_channels, n_samples) numpy.ndarray, float
        Any characterization function computed from the continuous seismograms.
    moveouts: (n_sources, n_stations, n_phases) numpy.ndarray, int
        Moveouts, in samples, from each of the `n_sources` theoretical sources
        to each of the `n_stations` seismic stations and for the `n_phases`
        back-projected seismic phases.
    weights_phases: (n_stations, n_channels, n_phases) numpy.ndarray, float
        Weight given to each station and channel for a given phase. For
        example, horizontal components might be given a small or zero
        weight for the P-wave stacking.
    weights_sources: (n_sources, n_stations) numpy.ndarray, float
        Source-receiver-specific weights. For example, based on the
        source-receiver distance.
    device: string, default to 'cpu'
        Either 'cpu' or 'gpu', depending on the available hardware and
        user's preferences.

    Returns
    --------
    nr: (n_sources, n_samples) numpy.ndarray, float
        Full network response with the `n_sources` network responses
        at each time step.
    """

    n_stations, n_channels, n_samples = detection_traces.shape
    n_sources, _, n_phases = moveouts.shape

    # prestack detection traces
    detection_traces = prestack_traces(
            detection_traces, weights_phases, device='cpu')

    detection_traces = np.float32(detection_traces.flatten())
    moveouts = np.int32(moveouts.flatten())
    weights_sources = np.float32(weights_sources.flatten())

    nr = np.zeros(n_sources*n_samples, dtype=np.float32)

    if device.lower() == 'cpu':
        _libCPU.network_response(
                detection_traces.ctypes.data_as(ct.POINTER(ct.c_float)),
                moveouts.ctypes.data_as(ct.POINTER(ct.c_int)),
                weights_sources.ctypes.data_as(ct.POINTER(ct.c_float)),
                n_samples,
                n_sources,
                n_stations,
                n_phases,
                nr.ctypes.data_as(ct.POINTER(ct.c_float)))
    elif device.lower() == 'gpu':
        _libGPU.network_response(
                detection_traces.ctypes.data_as(ct.POINTER(ct.c_float)),
                moveouts.ctypes.data_as(ct.POINTER(ct.c_int)),
                weights_sources.ctypes.data_as(ct.POINTER(ct.c_float)),
                n_samples,
                n_sources,
                n_stations,
                n_phases,
                nr.ctypes.data_as(ct.POINTER(ct.c_float)))
    else:
        print('device should cpu or gpu')
        return
    return nr.reshape(n_sources, n_samples)

def composite_network_response(detection_traces, moveouts, weights_phases,
                               weights_sources, device='cpu'):
    """Compute the composite beamformed network response.  

    This routine only keeps the highest network response and its associated
    source index across the grid at every time step. This memory efficient
    beamforming is well suited for continuous detection of seismic events.

    Parameters
    ------------
    detection_traces: (n_stations, n_channels, n_samples) numpy.ndarray, float
        Any characterization function computed from the continuous seismograms.
    moveouts: (n_sources, n_stations, n_phases) numpy.ndarray, int
        Moveouts, in samples, from each of the `n_sources` theoretical sources
        to each of the `n_stations` seismic stations and for the `n_phases`
        back-projected seismic phases.
    weights_phases: (n_stations, n_channels, n_phases) numpy.ndarray, float
        Weight given to each station and channel for a given phase. For
        example, horizontal components might be given a small or zero
        weight for the P-wave stacking.
    weights_sources: (n_sources, n_stations) numpy.ndarray, float
        Source-receiver-specific weights. For example, based on the
        source-receiver distance.
    device: string, default to 'cpu'
        Either 'cpu' or 'gpu', depending on the available hardware and
        user's preferences.

    Returns
    --------
    nr: (n_samples,) numpy.ndarray, float
        Composite network response, that is the largest network response
        across the source grid at each time step.
    source_index_nr: (n_samples,) numpy.ndarray, int
        Source indexes associated with the composite network response.
        These give the location of the most likely seismic source at
        a given time.
    """
    n_stations, n_channels, n_samples = detection_traces.shape
    n_sources, _, n_phases = moveouts.shape

    # prestack detection traces
    detection_traces = prestack_traces(
            detection_traces, weights_phases, device='cpu')

    detection_traces = np.float32(detection_traces.flatten())
    moveouts = np.int32(moveouts.flatten())
    weights_sources = np.float32(weights_sources.flatten())

    nr = np.zeros(n_samples, dtype=np.float32)
    source_index_nr = np.zeros(n_samples, dtype=np.int32)

    if device.lower() == 'cpu':
        _libCPU.composite_network_response(
                detection_traces.ctypes.data_as(ct.POINTER(ct.c_float)),
                moveouts.ctypes.data_as(ct.POINTER(ct.c_int)),
                weights_sources.ctypes.data_as(ct.POINTER(ct.c_float)),
                n_samples,
                n_sources,
                n_stations,
                n_phases,
                nr.ctypes.data_as(ct.POINTER(ct.c_float)),
                source_index_nr.ctypes.data_as(ct.POINTER(ct.c_int)))
    elif device.lower() == 'gpu':
        _libGPU.composite_network_response(
                detection_traces.ctypes.data_as(ct.POINTER(ct.c_float)),
                moveouts.ctypes.data_as(ct.POINTER(ct.c_int)),
                weights_sources.ctypes.data_as(ct.POINTER(ct.c_float)),
                n_samples,
                n_sources,
                n_stations,
                n_phases,
                nr.ctypes.data_as(ct.POINTER(ct.c_float)),
                source_index_nr.ctypes.data_as(ct.POINTER(ct.c_int)))
    else:
        print('device should cpu or gpu')
        return
    return nr, source_index_nr

def prestack_traces(detection_traces, weights_phases, device='cpu'):
    """Prestack the detection traces ahead of the beamforming.  

    Channel-wise stacking for each target seismic phase can be done
    once and for all at the beginning of the computation.

    Parameters
    -----------
    detection_traces: (n_stations, n_channels, n_stations) numpy.ndarray, float
        Any characterization function computed from the continuous seismograms.
    weights_phases: (n_stations, n_channels, n_phases) numpy.ndarray, float
        Weight given to each station and channel for a given phase. For
        example, horizontal components might be given a small or zero
        weight for the P-wave stacking.
    device: string, default to 'cpu'
        Either 'cpu' or 'gpu', depending on the available hardware and
        user's preferences.

    Returns
    ----------
    prestacked_traces: (n_stations, n_samples, n_phases) numpy.ndarray, float
        Channel-wise stacked detection traces, optimally formatted for the C
        CUDA-C routines.
    """

    n_stations, n_channels, n_samples = detection_traces.shape
    _, _, n_phases = weights_phases.shape
    prestacked_traces = np.zeros(
            (n_stations*n_samples*n_phases), dtype=np.float32)

    detection_traces = np.float32(detection_traces.flatten())
    weights_phases = np.float32(weights_phases.flatten())

    if device.lower() == 'cpu':
        _libCPU.prestack_detection_traces(
                detection_traces.ctypes.data_as(ct.POINTER(ct.c_float)),
                weights_phases.ctypes.data_as(ct.POINTER(ct.c_float)),
                n_samples,
                n_stations,
                n_channels,
                n_phases,
                prestacked_traces.ctypes.data_as(ct.POINTER(ct.c_float)))
    #if device.lower() == 'gpu':
    #    _libGPU.prestack_detection_traces(
    #            detection_traces.ctypes.data_as(ct.POINTER(ct.c_float)),
    #            weights_phases.ctypes.data_as(ct.POINTER(ct.c_float)),
    #            n_samples,
    #            n_stations,
    #            n_channels,
    #            n_phases,
    #            prestacked_traces.ctypes.data_as(ct.POINTER(ct.c_float)))
    return prestacked_traces.reshape((n_stations, n_samples, n_phases))
