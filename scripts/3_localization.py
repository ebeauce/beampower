# coding: utf-8

"""
3. Beamforming for location

We now calculate the beamforming. This consists of calculating waveform
features first (envelopes, kurtosis, etc.) and to shift-and-stack them
according to the points in the 3D source-search grid previously calculated.
"""

import beampower as bp
import glob
import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from matplotlib import pyplot as plt
from scipy import signal, stats
from obspy import read, read_inventory, UTCDateTime
from obspy.geodetics.base import locations2degrees, degrees2kilometers

DIRPATH_INVENTORY = "../data/processed/*.xml"
DIRPATH_WAVEFORMS = "../data/processed/*.mseed"
MIN_DETECTION_INTERVAL_SEC = 30
COLORS = {"E": "k", "N": "k", "Z": "C0"}

# Calculate waveform features

# We first transform the waveforms in a representation that is best suited for
# beamforming. In this notebook, we consider the envelope calculated from the
# Hilbert transform. Other transforms can also be better depending on the data
# and task.


def envelope(x):
    """Envelope transformation.

    Calculate the envelope of the input one-dimensional signal with the Hilbert
    transform. The data is normalized by the mean absolute deviation over the
    entire signal window first. The envelope is clipped at a maximum of 10^5
    times the mad of the envelope.

    Arguments
    ---------
    x: array-like
        The input one-dimensional signal.

    Returns
    -------
    array-like
        The envelope of x with same shape than x.
    """
    # Normalization
    x_mad = stats.median_abs_deviation(x)
    x_mad = 1.0 if x_mad == 0.0 else x_mad

    # Envelope
    x = np.abs(signal.hilbert(x / x_mad))

    # Clip
    x_max = 10.0 ** (5.0 * stats.median_abs_deviation(x))
    return x.clip(None, x_max)


# Header
filepaths_waveforms = glob.glob(DIRPATH_WAVEFORMS)
headers = read(DIRPATH_WAVEFORMS, headonly=True)

# Initialize
waveform_features = xr.DataArray(
    dims=["station", "channel", "time"],
    coords={
        "channel": list(set([header.stats.channel[-1] for header in headers])),
        "station": list(set([header.stats.station for header in headers])),
        "time": pd.to_datetime(headers[0].times("timestamp"), unit="s"),
    },
)

# Transform each file
for trace in tqdm.tqdm(read(DIRPATH_WAVEFORMS), desc="Envelopes"):
    info = trace.stats
    index = {"station": info.station, "channel": info.channel[-1]}
    waveform_features.loc[index] = envelope(trace.data)

# Initialize beamforming Get the travel times obtained from our notebook on
# travel time calculation.

# Reshape
ravel_shape = ["latitude", "longitude", "depth"]
sampling_rate = headers[0].stats.sampling_rate

# Time delays
travel_times = xr.load_dataarray("../data/travel_times.nc")
time_delays = travel_times.stack(source=ravel_shape)
time_delays = time_delays.transpose("source", "station", "phase")
time_delays = np.round(sampling_rate * time_delays)

# Phase weights
weights_phase = xr.ones_like(waveform_features.isel(time=0).drop("time"))
weights_phase = weights_phase.expand_dims({"phase": 2})
weights_phase = weights_phase.assign_coords({"phase": list("PS")})
weights_phase = weights_phase.copy()
weights_phase.loc[{"channel": ["E", "N"], "phase": "P"}] = 0.0
weights_phase.loc[{"channel": "Z", "phase": "S"}] = 0.0
weights_phase = weights_phase.transpose("station", "channel", "phase")

# Sources weights
weights_sources = np.ones(time_delays.shape[:-1])


# Calculate beamforming
# We extract the beamforming maximum at every time stamp in the hole grid.
beam_max, beam_argmax = bp.beamform(
    waveform_features.to_numpy(),
    time_delays.to_numpy(),
    weights_phase.to_numpy(),
    weights_sources,
    device="cpu",
    reduce="max",
)

# Aggregate beam outputs

# The output of the beamforming operation are assembled in a `Dataarray` in
# order to simplify array manipulation and detection.

# Get arrays
beam = xr.DataArray(data=beam_max, coords=waveform_features["time"].coords)
beam = beam.to_dataset(name="beam_max")

# Source coordinates
for dim in ravel_shape:
    coordinates = time_delays[dim].to_numpy()[beam_argmax]
    beam = beam.assign({dim: ("time", coordinates)})

# Detect

# We detect the maxima with find peaks and use a threshold criterion to keep
# only the prominent peaks (or events).

# Detect
min_detection_interval = int(MIN_DETECTION_INTERVAL_SEC * sampling_rate)
peaks = signal.find_peaks(beam.beam_max, distance=min_detection_interval)
events = beam.isel(time=peaks[0])

# Threshold
window_length = int(1800 * sampling_rate)
threshold = 2 * beam.beam_max.rolling(time=window_length, center=True).median()
beam = beam.assign({"threshold": ("time", threshold.to_numpy())})
events = events.isel(time=events.beam_max > threshold)

# Show
beam.beam_max.plot()
threshold.plot()
events.beam_max.plot(marker=".", ls="none", c="C3")
plt.semilogy()
plt.ylim(bottom=10)
plt.ylabel("Beam power")
plt.grid()
plt.title(f"Dection of {len(events.time)} events")
plt.savefig("figures/detection.png", dpi=200)
plt.close()

# Show detection

# We detect the maxima with find peaks and use a threshold criterion to keep only the prominent peaks (or events).

select = slice("2013-04-23 15:00", "2013-04-23 16:00")
inventory = read_inventory(DIRPATH_INVENTORY)
detector_zoom = beam.sel(time=select)
events_zoom = events.sel(time=select)

# Show
detector_zoom.beam_max.plot()
detector_zoom.threshold.plot()
events_zoom.beam_max.plot(marker=".", ls="none", c="C1")
plt.semilogy()
plt.grid()
plt.ylabel("Beam power")

# Watch peak
event_watch = events_zoom.isel(time=events_zoom.beam_max.argmax())
plt.plot(event_watch.time, event_watch.beam_max, "oC3", ms=13, mfc="none")
plt.savefig("figures/detection_zoom")
plt.close()

# str(event_watch.time.data)
date = UTCDateTime(str(event_watch.time.data))

# Get waveform
fig, ax = plt.subplots(1, figsize=(6, 6))
for index, trace in enumerate(read(DIRPATH_WAVEFORMS)):

    # Get trace and info
    trace.trim(date - 1, date + 15)
    trace.filter(type="lowpass", freq=5)
    times = pd.to_datetime(trace.times("timestamp"), unit="s")
    data = trace.data / np.abs(trace.data).max()

    coords = inventory.get_coordinates(trace.id)
    p1 = event_watch.latitude.data, event_watch.longitude.data
    p2 = [coords[dim] for dim in ["latitude", "longitude"]]
    distance = degrees2kilometers(locations2degrees(*p1, *p2))

    # Plot trace
    ax.plot(times, data + distance, color=COLORS[trace.stats.channel[-1]])

# Labels
ax.set_ylabel("Epicentral distance (km)")
ax.axvline(date, color="C3")
ax.legend([key for key in COLORS])
plt.savefig("figures/event")
plt.close()

# Show all detections
plt.plot(events.longitude, events.depth, ".r")
plt.xlabel("Longitude (degrees)")
plt.ylabel("Depth (km)")
plt.gca().invert_yaxis()
plt.grid()
plt.savefig("figures/catalog")
plt.close()
