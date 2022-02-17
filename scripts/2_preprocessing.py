import os
import sys

path_script = os.path.dirname(__file__)
sys.path.append(path_script)

import utils

import numpy as np

from time import time as give_time

from obspy import Stream
from obspy import UTCDateTime as udt

def lowpass_chebyshev_II(X, freqmax, sampling_rate,
                         order=10, min_attenuation_dB=40., zerophase=False):
    from scipy.signal import cheby2, sosfilt

    nyquist = sampling_rate/2.

    sos = cheby2(order,
                 min_attenuation_dB,
                 #freqmax/nyquist,
                 freqmax,
                 analog=False,
                 fs=sampling_rate,
                 btype='lowpass',
                 output='sos')

    X = sosfilt(sos, X)
    if zerophase:
        X = sosfilt(sos, X[::-1])[::-1]
    return X


def preprocess(st, inv, freqmin=None, freqmax=None,
               target_SR=None, remove_response=False,
               remove_sensitivity=False, plot_resp=False,
               target_duration=None, verbose=True, unit='VEL',
               target_starttime=None, target_endtime=None,
               **kwargs):
    preprocessed_st = Stream()
    if len(st) == 0:
        print('Input data is empty!')
        return preprocessed_st
    if len(st) > 10:
        print('Too many gaps!')
        return preprocessed_st
    # ----------------------------------
    # start by cleaning the gaps if there are any
    # ----------------------------------
    for tr in st:
        if np.isnan(tr.data.max()):
            print(f'Problem with {tr.id} (detected NaNs)!')
            continue
        # split will lose information about start and end times
        # if the start or the end is masked
        t1 = udt(tr.stats.starttime.timestamp)
        t2 = udt(tr.stats.endtime.timestamp)
        tr = tr.split()
        tr.detrend('constant')
        tr.detrend('linear')
        tr.taper(0.05, type='cosine')
        # it's now safe to fill the gaps with zeros
        tr = tr.merge(fill_value=0.)
        tr.trim(starttime=t1, endtime=t2, pad=True, fill_value=0.)
        preprocessed_st += tr
    # if the trace came as separated segments without masked elements,
    # it is necessary to merge the stream
    preprocessed_st = preprocessed_st.merge(fill_value=0.)
    if target_starttime is not None:
        preprocessed_st.trim(starttime=target_starttime, pad=True, fill_value=0.)
    if target_endtime is not None:
        preprocessed_st.trim(endtime=target_endtime, pad=True, fill_value=0.)
    # delete the original data to save memory
    del st
    # ----------------------------------
    #      resample if necessary
    # ----------------------------------
    for tr in preprocessed_st:
        if target_SR is None:
            continue
        sr_ratio = tr.stats.sampling_rate/target_SR
        if sr_ratio > 1:
            tr.data = lowpass_chebyshev_II(
                       tr.data, 0.48*target_SR, tr.stats.sampling_rate,
                       order=10, min_attenuation_dB=40., zerophase=True)
            if np.round(sr_ratio, decimals=0) == sr_ratio:
                # tr's sampling rate is an integer
                # multiple of target_SR
                tr.decimate(int(sr_ratio), no_filter=True)
            else:
                tr.resample(target_SR, no_filter=True)
        elif sr_ratio < 1:
            if verbose:
                print('Requested sampling rate is too high!')
                print(tr)
            preprocessed_st.remove(tr)
            continue
        else:
            pass
    # ----------------------------------
    #    adjust length if requested
    # ----------------------------------
    if target_duration is not None:
        for i in range(len(preprocessed_st)):
            n_samples = utils.sec_to_samp(
                    target_duration, sr=preprocessed_st[i].stats.sampling_rate)
            preprocessed_st[i].data = preprocessed_st[i].data[:n_samples]
    # ----------------------------------
    #   remove response if requested
    # ----------------------------------
    # -- attach the metadata
    preprocessed_st.attach_response(inv)
    if remove_response:
        for tr in preprocessed_st:
            # assume that the instrument response
            # is already attached to the trace
            T_max = tr.stats.npts*tr.stats.delta
            T_min = tr.stats.delta
            f_min = 1./T_max
            f_max = 1./(2.*T_min)
            pre_filt = [f_min, 3.*f_min, 0.90*f_max, 0.97*f_max]
            tr.remove_response(pre_filt=pre_filt, output=unit, plot=plot_resp)
    elif remove_sensitivity:
        for tr in preprocessed_st:
            tr.remove_sensitivity()
    # ----------------------------------
    #            filter
    # ----------------------------------
    preprocessed_st.detrend('constant')
    preprocessed_st.detrend('linear')
    preprocessed_st.taper(0.02, type='cosine')
    if freqmin is None and freqmax is None:
        # no filtering
        pass
    elif freqmin is None:
        # lowpass filtering
        preprocessed_st.filter('lowpass', freq=freqmax, zerophase=True)
    elif freqmax is None:
        # highpass filtering
        preprocessed_st.filter('highpass', freq=freqmin, zerophase=True)
    else:
        # bandpass filtering
        preprocessed_st.filter('bandpass', freqmin=freqmin,
                               freqmax=freqmax, zerophase=True)
    return preprocessed_st


def preprocess_asdf(ds, tag_name, output_filename, path_data='.', **kwargs):
    from functools import partial
    _preprocess = partial(preprocess, **kwargs)

    # tag map defining which traces are taken as input (raw)
    # and what is the tag name of the output (tag_name)
    tag_map = {'raw': tag_name}

    # full output filename
    output_filename = os.path.join(
            path_data, f'{os.path.splitext(output_filename)[0]}.h5')

    ds.process(_preprocess, output_filename, tag_map)

    # this is necessary if using the MPI version of pyasdf
    del ds

if __name__ == "__main__":
    from pyasdf import ASDFDataSet
    from datetime import timedelta

    # I/O variables:
    path_data = os.path.join(path_script, os.pardir, 'data')
    data_filename = 'data_20120726_raw.h5'
    output_filename = 'data_20120726'

    # preprocessing parameters
    tag_name = 'preprocessed_2_12'
    target_SR = 25. # H
    target_starttime = udt('2012-07-26')
    target_endtime = target_starttime + timedelta(days=1)
    freqmin = 2.0 # Hz
    freqmax = 12.0 # Hz
    remove_response = True

    # remove the target file if already existing
    target = os.path.join(path_data, output_filename + '.h5')
    if os.path.isfile(target):
        print(f'Overwrite the existing {target}')
        os.remove(target)

    # load the raw data
    ds = ASDFDataSet(os.path.join(path_data, data_filename))

    # start preprocessing
    print(f'Start preprocessing {os.path.join(path_data, data_filename)}...')
    preprocess_asdf(
            ds, tag_name, output_filename, path_data=path_data,
            target_SR=target_SR, freqmin=freqmin, freqmax=freqmax,
            target_starttime=target_starttime, target_endtime=target_endtime,
            remove_response=remove_response)
