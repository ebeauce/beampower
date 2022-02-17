import os
import sys

import numpy as np
import h5py as h5

def get_moveout_array(moveouts, stations, phases):
    n_stations = len(stations)
    n_phases = len(phases)
    moveout_arr = np.array([moveouts[ph][sta] for sta in stations for ph in phases]).T
    return moveout_arr.reshape(-1, n_stations, n_phases)

def load_travel_times(path, phases):
    tts = {}
    with h5.File(path, mode='r') as f:
        for ph in phases:
            tts[ph] = {}
            for sta in f[f'tt_{ph}'].keys():
                # flatten the lon/lat/dep grid as we work with 
                # flat source indexes
                tts[ph][sta] = f[f'tt_{ph}'][sta][()].flatten()
    return tts
    
def sec_to_samp(t, sr, epsilon=0.2):
    # we add epsilon so that we fall onto the right
    # integer number even if there is a small precision
    # error in the floating point number
    sign = np.sign(t)
    t_samp_float = abs(t*sr) + epsilon
    # round and restore sign
    t_samp_int = np.int32(sign*np.int32(t_samp_float))
    return t_samp_int


# --------------------------------------------------------
#               characteristic functions
# --------------------------------------------------------
def envelope_parallel(traces):
    """Compute the envelope of traces.  

    The envelope is defined as the modulus of the complex
    analytical signal (a signal whose Fourier transform only has
    energy in positive frequencies).

    Parameters
    -------------
    traces: (n_stations, n_channels, n_samples) numpy.ndarray, float
        The input time series.

    Returns
    -------------
    envelopes: (n_stations, n_channels, n_samples) numpy.ndarray, float
        The moduli of the analytical signal of the input traces.
    """
    import concurrent.futures
    traces_reshaped = traces.reshape(-1, traces.shape[-1])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        envelopes = np.float32(list(executor.map(
           envelope, traces_reshaped)))
    return envelopes.reshape(traces.shape)

def envelope(trace):
    """Compute the envelope of trace.  

    The envelope is defined as the modulus of the complex
    analytical signal (a signal whose Fourier transform only has
    energy in positive frequencies).

    Parameters
    -------------
    trace: (n_samples) numpy.ndarray, float
        The input time series.

    Returns
    -------------
    envelope: (n_samples) numpy.ndarray, float
        The modulus of the analytical signal of the input traces.
    """
    from scipy.signal import hilbert
    return np.float32(np.abs(hilbert(trace)))


# --------------------------------------------------------
#          Post-process network response
# --------------------------------------------------------

def time_dependent_threshold(network_response, window,
                             overlap=0.75, CNR_threshold=10.):
    """Compute a time-dependent detection threshold.  


    Parameters
    -----------
    network_response: (n_samples,) numpy.ndarray, float
        Composite network response on which we calculate
        the detection threshold.
    window: scalar, integer
        Length of the sliding window, in samples, over
        which we calculate the running statistics used
        in the detection threshold.
    overlap: scalar, float, default to 0.75
        Ratio of overlap between two contiguous windows.
    CNR_threshold: scalar, float, default to 10
        Number of running MADs above running median that
        defines the detection threshold.
    Returns
    --------
    detection_threshold: (n_samples,) numpy.ndarray
        Detection threshold on the network response.
    """
    try:
        from scipy.stats import median_abs_deviation as scimad
    except ImportError:
        from scipy.stats import median_absolute_deviation as scimad
    from scipy.interpolate import interp1d

    # calculate n_windows given window
    # and overlap
    shift = int((1.-overlap)*window)
    n_windows = int((len(network_response)-window)//shift)+1
    mad_ = np.zeros(n_windows+2, dtype=np.float32)
    med_ = np.zeros(n_windows+2, dtype=np.float32)
    time = np.zeros(n_windows+2, dtype=np.float32)
    for i in range(1, n_windows+1):
        i1 = i*shift
        i2 = min(network_response.size, i1+window)
        cnr_window = network_response[i1:i2]
        #non_zero = cnr_window != 0
        #if sum(non_zero) < 3:
        #    # won't be possible to calculate median
        #    # and mad on that few samples
        #    continue
        #med_[i] = np.median(cnr_window[non_zero])
        #mad_[i] = scimad(cnr_window[non_zero])
        med_[i] = np.median(cnr_window)
        mad_[i] = scimad(cnr_window)
        time[i] = (i1+i2)/2.
    # add boundary cases manually
    time[0] = 0.
    mad_[0] = mad_[1]
    med_[0] = med_[1]
    time[-1] = len(network_response)
    mad_[-1] = mad_[-2]
    med_[-1] = med_[-2]
    threshold = med_ + CNR_threshold * mad_
    interpolator = interp1d(
            time, threshold, kind='slinear',
            fill_value=(threshold[0], threshold[-1]),
            bounds_error=False)
    full_time = np.arange(0, len(network_response))
    threshold = interpolator(full_time)
    return threshold

def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                  kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1]:http://nbviewer.ipython.org/github/demotu/BMC/blob/master/
        notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan,
                                                    indnan - 1, indnan + 1))),
                          invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]),
                    axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot_peaks(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def find_detections(data, cnr, cnr_source_indexes,
                    moveouts, detection_threshold,
                    station_names, component_names,
                    search_win, duration, offset_start,
                    max_n_stations=None):
    """
    Analyze the composite network response to get candidate earthquakes
    from the data.

    Parameters
    -----------
    data: obspy.Stream
        Continuous recordings of the seismic wavefields.
    cnr: (n_samples,) numpy.ndarray, float
        The composite network response.
    cnr: (n_samples,) numpy.ndarray, integer
        The source indexes corresponding to each time sample of `cnr`.
    moveouts: (n_sources, n_stations, n_channels) numpy.ndarray, integer
        Moveouts used for the composite network response and to use
        on each station/channel to extract the waveforms of interest.
    detection_threshold: scalar or (n_samples,) numpy.ndarray, float
        The number of running MADs taken above the running median
        to define detections.
    station_names: list of strings
        Names of the stations in the same order as for moveouts.
    component_names: list of strings
        Names of the components in the same order as for moveouts.
    search_win: scalar, int
        The shortest duration, in samples, allowed between two
        consecutive detections.
    duration: scalar, float
        Duration of the extracted windows, in seconds.
    offset_start: (n_channels,) list of numpy.ndarray, float
        Times, in seconds, taken before the picks.
    max_n_stations: integer, default to None
        If not None and if smaller than the total number of stations in the
        network, only extract the `max_n_stations` closest stations for
        each theoretical source.
        
    Returns
    -----------
    detections: dictionary,
        Dictionary with data and metadata of the detected earthquakes.
    """
    from obspy import Stream

    sr = data[0].stats.sampling_rate

    # select peaks
    peaks = _detect_peaks(cnr, mpd=search_win)
    # only keep peaks above detection threshold
    peaks = peaks[cnr[peaks] > detection_threshold[peaks]]

    # keep the largest peak for grouped detection
    for i in range(len(peaks)):
        idx = np.int32(np.arange(max(0, peaks[i] - search_win/2),
                                 min(peaks[i] + search_win/2, len(cnr))))
        idx_to_update = np.where(peaks == peaks[i])[0]
        peaks[idx_to_update] = np.argmax(cnr[idx]) + idx[0]

    peaks, idx = np.unique(peaks, return_index=True)

    peaks = np.asarray(peaks)
    sources = cnr_source_indexes[peaks]

    # extract waveforms
    detections = []
    for i in range(len(peaks)):
        event = Stream()
        t0_i = data[0].stats.starttime + peaks[i]/sr
        mv = moveouts[sources[i], ...]
        if max_n_stations is not None:
            # use moveouts as a proxy for distance
            # keep only the max_n_stations closest stations
            mv_max = np.sort(mv[:, 0])[max_n_stations-1]
        else:
            mv_max = np.finfo(np.float32).max
        for s, sta in enumerate(station_names):
            if mv[s, 0] > mv_max:
                continue
            for c, cp in enumerate(component_names):
                t_sc = t0_i + mv[s, c]/sr - offset_start[c]
                event += data.select(station=sta, component=cp)[0]\
                            .slice(starttime=t_sc, endtime=t_sc+duration)
        detections.append(event)
    
    print(f'Extracted {len(detections):d} events.')

    return detections, peaks, sources

