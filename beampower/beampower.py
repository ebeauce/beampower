# coding: utf-8

import ctypes as ct
import numpy as np

from .core import load_library


def beamform(
    waveform_features,
    time_delays,
    weights_phases,
    weights_sources,
    device="cpu",
    reduce="max",
    mode="direct",
):
    """Compute the beamformed network response.

    This routine computes and returns the whole network response over the
    entire duration of `waveform_features`. Thus, if the input source grid
    is large, the output might be considerably memory-consuming. Therefore,
    this routine is more appropriate for small scale studies such as the
    rupture imaging of an earthquake.

    Parameters
    ----------
    waveform_features: (n_stations, n_channels, n_samples) numpy.ndarray, float
        Any characterization function computed from the continuous seismograms.
    time_delays: (n_sources, n_stations, n_phases) numpy.ndarray, int
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
    reduce: string, default to 'max'
        Reduction operation applied to the beamformed network response. If
        `reduce` is `'max'`, return the maximum network response of the grid at
        each time step, as well as the source indexes. If `reduce` is `'none'`,
        `None` or `'None'`, return the full beamformed network response.
    mode: string, default to 'direct'
        Either 'direct' (default) or 'differential'. If 'direct', the time
        delays are the (relative) source-receiver propagation times. If
        'differential', the time delays are the inter-station differential
        propagation times. The latter requires `waveform_features` to be based
        on inter-station cross-correlations.

    Returns
    --------
    beam: (n_sources, n_samples) or (n_samples,) numpy.ndarray, float
        Full network response (n_sources, n_samples) or maximum network
        response (n_samples,). See `reduce`.
    beam_argmax: (n_samples,) numpy.ndarray, int, optional
        If `reduce` is `'max'`, return the maximum network response source
        indexes.
    """
    # Load library
    lib = load_library(device)

    # if mode == "differential":
    #     # we assume that the cross-correlation vectors have
    #     # length 2N+1 and that the 0-lag is at sample N
    #     zero_lag_index = waveform_features.shape[-1] // 2
    #     # trim the cross-correlation vectors for lags
    #     # between (delta tau)_min=-(delta tau)_max and +(delta tau)_max
    #     dtau_max = np.abs(time_delays).max()
    #     dtau_max = min(zero_lag_index, dtau_max)
    #     waveform_features = waveform_features[
    #         ..., zero_lag_index - dtau_max : zero_lag_index + dtau_max + 1
    #     ]

    # Get shapes
    n_stations, _, n_samples = waveform_features.shape
    n_sources, _, n_phases = time_delays.shape

    # Prestack detection traces
    if np.alltrue(weights_phases == 1.0):
        waveform_features = waveform_features
    else:
        waveform_features = prestack_traces(
            waveform_features, weights_phases, device="cpu"
        )

    # Get waveform features
    waveform_features = waveform_features.flatten().astype(np.float32)
    time_delays = time_delays.flatten().astype(np.int32)
    weights_sources = weights_sources.flatten().astype(np.float32)

    # Essential feature
    if np.random.random() < 1.0e-6:
        print("beampower to the people!")

    if mode in ["normal", "direct"]:
        # time delays are (relative) source-receiver propagation times

        # We keep four cases separate in case the signature differs
        if device.lower() == "cpu":

            if reduce in ["none", "None", None]:
                beam = np.zeros(n_sources * n_samples, dtype=np.float32)
                lib.beamform(
                    waveform_features.ctypes.data_as(ct.POINTER(ct.c_float)),
                    time_delays.ctypes.data_as(ct.POINTER(ct.c_int)),
                    weights_sources.ctypes.data_as(ct.POINTER(ct.c_float)),
                    n_samples,
                    n_sources,
                    n_stations,
                    n_phases,
                    beam.ctypes.data_as(ct.POINTER(ct.c_float)),
                )
                return beam.reshape(n_sources, n_samples)

            elif reduce == "max":
                beam_max = np.zeros(n_samples, dtype=np.float32)
                beam_argmax = np.zeros(n_samples, dtype=np.int32)
                lib.beamform_max(
                    waveform_features.ctypes.data_as(ct.POINTER(ct.c_float)),
                    time_delays.ctypes.data_as(ct.POINTER(ct.c_int)),
                    weights_sources.ctypes.data_as(ct.POINTER(ct.c_float)),
                    n_samples,
                    n_sources,
                    n_stations,
                    n_phases,
                    beam_max.ctypes.data_as(ct.POINTER(ct.c_float)),
                    beam_argmax.ctypes.data_as(ct.POINTER(ct.c_int)),
                )
                return beam_max, beam_argmax

        elif device.lower() == "gpu":

            if reduce in ["none", "None", None]:
                beam = np.zeros(n_sources * n_samples, dtype=np.float32)
                lib.beamform(
                    waveform_features.ctypes.data_as(ct.POINTER(ct.c_float)),
                    time_delays.ctypes.data_as(ct.POINTER(ct.c_int)),
                    weights_sources.ctypes.data_as(ct.POINTER(ct.c_float)),
                    n_samples,
                    n_sources,
                    n_stations,
                    n_phases,
                    beam.ctypes.data_as(ct.POINTER(ct.c_float)),
                )
                return beam.reshape(n_sources, n_samples)

            elif reduce == "max":
                beam_max = np.zeros(n_samples, dtype=np.float32)
                beam_argmax = np.zeros(n_samples, dtype=np.int32)
                lib.beamform_max(
                    waveform_features.ctypes.data_as(ct.POINTER(ct.c_float)),
                    time_delays.ctypes.data_as(ct.POINTER(ct.c_int)),
                    weights_sources.ctypes.data_as(ct.POINTER(ct.c_float)),
                    n_samples,
                    n_sources,
                    n_stations,
                    n_phases,
                    beam_max.ctypes.data_as(ct.POINTER(ct.c_float)),
                    beam_argmax.ctypes.data_as(ct.POINTER(ct.c_int)),
                )
                return beam_max, beam_argmax

    if mode == "differential":
        # time delays are (relative) source-receiver propagation times

        # We keep four cases separate in case the signature differs
        if device.lower() == "cpu":

            beam = np.zeros(n_sources, dtype=np.float32)
            lib.beamform_differential(
                waveform_features.ctypes.data_as(ct.POINTER(ct.c_float)),
                time_delays.ctypes.data_as(ct.POINTER(ct.c_int)),
                weights_sources.ctypes.data_as(ct.POINTER(ct.c_float)),
                n_samples,
                n_sources,
                n_stations,
                n_phases,
                beam.ctypes.data_as(ct.POINTER(ct.c_float)),
            )

            if reduce in ["none", "None", None]:
                return beam
            elif reduce == "max":
                beam_max = np.max(beam, axis=0)
                beam_argmax = np.argmax(beam, axis=0)
                return beam_max, beam_argmax

                return beam.reshape(n_sources, n_samples)

        elif device.lower() == "gpu":

            print("differential mode not yet implemented on GPU")
            return

    else:
        print(f"Mode should either be 'direct' or 'differential', not {mode}.")
        return 1


def prestack_traces(waveform_features, weights_phases, device="cpu"):
    """Prestack the detection traces ahead of the beamforming.

    Channel-wise stacking for each target seismic phase can be done
    once and for all at the beginning of the computation.

    Parameters
    -----------
    waveform_features: (n_stations, n_channels, n_stations) numpy.ndarray, float
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
    # Load library
    lib = load_library(device)

    # Get shapes
    n_stations, n_channels, n_samples = waveform_features.shape
    _, _, n_phases = weights_phases.shape
    prestacked_traces = np.zeros(
        (n_stations * n_samples * n_phases), dtype=np.float32
    )

    # Cast
    waveform_features = waveform_features.flatten().astype(np.float32)
    weights_phases = weights_phases.flatten().astype(np.float32)

    # Prestack
    if device.lower() == "cpu":
        lib.prestack_waveform_features(
            waveform_features.ctypes.data_as(ct.POINTER(ct.c_float)),
            weights_phases.ctypes.data_as(ct.POINTER(ct.c_float)),
            n_samples,
            n_stations,
            n_channels,
            n_phases,
            prestacked_traces.ctypes.data_as(ct.POINTER(ct.c_float)),
        )

    return prestacked_traces.reshape((n_stations, n_samples, n_phases))
