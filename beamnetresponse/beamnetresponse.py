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
    print('Beamformed Network Response CPU is not compiled!'
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
    _libGPU.prestack_detection_traces.argtypes = [
            ct.POINTER(ct.c_float),   # detection_traces
            ct.POINTER(ct.c_float),   # weights_phases
            ct.c_size_t,              # n_sources
            ct.c_size_t,              # n_stations
            ct.c_size_t,              # n_channels
            ct.c_size_t,              # n_phases
            ct.POINTER(ct.c_float)]   # prestacked_traces
    GPU_LOADED = True
except OSError:
    print('Beamformed Network Response GPU is not compiled!'
          ' Build the GPU library first in order to use the GPU routines.')

def network_response(detection_traces, moveouts, weights, device='cpu'):
    n_stations, n_channels, n_samples = detection_traces.shape
    n_sources, _, n_phases = moveouts.shape

    detection_traces = np.float32(detection_traces.flatten())
    moveouts = np.int32(moveouts.flatten())
    weights = np.float32(weights.flatten())

    nr = np.zeros(n_samples*n_sources, dtype=np.float32)

    if device.lower() == 'cpu':
        _libCPU.network_response(
                detection_traces.ctypes.data_as(ct.POINTER(ct.c_float)),
                moveouts.ctypes.data_as(ct.POINTER(ct.c_int)),
                weights.ctypes.data_as(ct.POINTER(ct.c_float)),
                n_samples,
                n_sources,
                n_stations,
                n_channels,
                n_phases,
                nr.ctypes.data_as(ct.POINTER(ct.c_float)))
    elif device.lower() == 'gpu':
        _libGPU.network_response(
                detection_traces.ctypes.data_as(ct.POINTER(ct.c_float)),
                moveouts.ctypes.data_as(ct.POINTER(ct.c_int)),
                weights.ctypes.data_as(ct.POINTER(ct.c_float)),
                n_samples,
                n_sources,
                n_stations,
                n_channels,
                n_phases,
                nr.ctypes.data_as(ct.POINTER(ct.c_float)))
    else:
        print('device should cpu or gpu')
        return
    return nr.reshape(n_samples, n_sources)

def composite_network_response(detection_traces, moveouts, weights_phases,
                               weights_sources, device='cpu'):
    n_stations, n_channels, n_samples = detection_traces.shape
    n_sources, _, n_phases = moveouts.shape

    # prestack detection traces
    detection_traces = _libCPU.prestack_traces(
            detection_traces, weights_phases, device=device)

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
