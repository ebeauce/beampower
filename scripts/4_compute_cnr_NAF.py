import os
import sys

path_script = os.path.dirname(__file__)
#path_module = os.path.join(path_script, os.pardir, 'beamnetresponse')
sys.path.append(path_script)
#sys.path.append(path_module)

import utils
import read_asdf
import beamnetresponse as bnr

try:
    from scipy.stats import median_abs_deviation as scimad
except ImportError:
    from scipy.stats import median_absolute_deviation as scimad

import numpy as np
import h5py as h5
from time import time as give_time

path_data = os.path.join(path_script, os.pardir, 'data')

# I/O variables
tt_filename = 'tts.h5'
data_filename = 'data_20120726.h5'

# computation architecture
device = 'cpu'

# use PhaseNet in the example or not (default = use envelopes)
use_phasenet = False

if use_phasenet:
    from phasenet import wrapper as PN
    from scipy.ndimage import gaussian_filter1d
    output_filename = 'cnr_NAF_PN.h5'
    smoothing = 5
else:
    output_filename = 'cnr_NAF_env.h5'

# ------------------------------------------------------
#    load travel-times, network metadata and waveforms
# ------------------------------------------------------
tts = utils.load_travel_times(
        os.path.join(path_data, tt_filename), phases=['P', 'S'])
metadata = read_asdf.get_asdf_metadata(
        os.path.join(path_data, data_filename))
# these data are already preprocessed: bandpass filtered at 2-12Hz
# and downsampled at 25Hz
data = read_asdf.read_asdf_data(
        os.path.join(path_data, data_filename), metadata.index, tag='preprocessed_2_12')

# get data into an nd array 
components = ['N', 'E', 'Z']
data_arr = read_asdf.get_np_array(data, metadata['station_code'], components)

# define the phases used in the stacking
phases = ['P', 'S']

# convert travel-times into relative moveouts, in samples
sr = data[0].stats.sampling_rate
moveouts = utils.get_moveout_array(tts, metadata['station_code'], phases)
# express moveouts in terms of relative times
moveouts -= np.min(moveouts, axis=(1, 2), keepdims=True)
moveouts = utils.sec_to_samp(moveouts, sr=sr)

# ------------------------------------------------------
#         define the detection traces
# ------------------------------------------------------
if use_phasenet:
    # -- use PhaseNet as detection traces
    PN_probas, _ = PN.automatic_picking(
            data_arr[np.newaxis, ...], metadata['station_code'], '.',
            'test', mini_batch_size=16)
    detection_traces = np.swapaxes(PN_probas.squeeze(), 2, 1)
    # smooth PhaseNet's output so that it's easier to find good alignments
    # with limited resolution moveouts
    detection_traces = gaussian_filter1d(detection_traces, smoothing, axis=-1)
else:
    # -- use envelopes as detection traces
    norm = scimad(data_arr, axis=-1)[..., np.newaxis]
    norm[norm == 0.] = 1.
    detection_traces = utils.envelope_parallel(data_arr/norm)
    # clip envelopes so that spikes don't dominate the beam
    scale = scimad(detection_traces, axis=-1)[..., np.newaxis]
    detection_traces = np.clip(
            detection_traces, np.min(detection_traces, axis=-1, keepdims=True),
            10.**5.*scale)

# ------------------------------------------------------
#         define the weight matrices
# ------------------------------------------------------
# -- weights_phases: used to pre-stack the channels for each phase
if use_phasenet:
    # ---- weights for PhaseNet detection traces
    weights_phases = np.ones((detection_traces.shape[:-1]) + (2,), dtype=np.float32)
    weights_phases[:, 0, 1] = 0. # S-wave weights to zero for channel 0
    weights_phases[:, 1, 0] = 0. # P-wave weights to zero for channel 1
else:
    # ---- weights for envelope detection traces
    weights_phases = np.ones(detection_traces.shape[:-1] + (len(phases),), dtype=np.float32)
    weights_phases[:, :2, 0] = 0. # P-wave weight = 0 for horizontal components
    weights_phases[:, 2, 1] = 0. # S-wave weight = 0 for vertical components

# -- weights_sources: source-specific weights, for example to limit
#                     the contribution of remote stations
weights_sources = np.ones(moveouts.shape[:-1], dtype=np.float32)

# ------------------------------------------------------
#      compute the composite network response
# ------------------------------------------------------
print('The seismic wavefield was recorded on {:d} stations and {:d} channels.'.
        format(detection_traces.shape[0], detection_traces.shape[1]))
print('We backproject the wavefield onto {:d} theoretical sources.'.
        format(moveouts.shape[0]))
t1 = give_time()
print('Start computing the composite network response...')
cnr, source_index = bnr.beamformed_nr.composite_network_response(
        detection_traces, moveouts, weights_phases,
        weights_sources, device=device)
t2 = give_time()
print(f'Done! In {t2-t1:.2f}s.')

# ------------------------------------------------------
#                save outputs
# ------------------------------------------------------
with h5.File(os.path.join(path_data, output_filename), mode='w') as f:
    f.create_dataset('cnr', data=cnr)
    f.create_dataset('source_indexes', data=source_index)

