import pyasdf
import pandas as pd
import numpy as np

def get_asdf_metadata(path):
    ds = pyasdf.ASDFDataSet(path)
    metadata = pd.DataFrame(ds.get_all_coordinates()).transpose()
    net_sta = [code.split(sep='.') for code in metadata.index]
    networks, stations = [[net for net, sta in net_sta],
                          [sta for net, sta in net_sta]]
    metadata['network_code'] = networks
    metadata['station_code'] = stations
    metadata.rename(columns={'elevation_in_m': 'elevation'}, inplace=True)
    return metadata

def read_asdf_data(path, trace_ids, tag='processed_2_12', duration=24.*3600.):
    from obspy import Stream
    from obspy import UTCDateTime as udt
    from datetime import timedelta
    ds = pyasdf.ASDFDataSet(path)
    data = Stream()
    for tr in trace_ids:
        data += ds.waveforms[tr][tag]
    del ds
    date = udt(udt((data[0].stats.starttime.timestamp
        + data[0].stats.endtime.timestamp)/2.).strftime('%Y-%m-%d'))
    data = data.trim(starttime=date, endtime=date+timedelta(seconds=duration))
    return data

def get_np_array(stream, stations, components, verbose=True):
    n_stations = len(stations)
    n_components = len(components)
    if len(stream) == 0:
        print('The input data stream is empty!')
        return
    n_samples = stream[0].data.size
    data = np.zeros((n_stations, n_components, n_samples),
                    dtype=np.float32)
    for s, sta in enumerate(stations):
        for c, cp in enumerate(components):
            channel = stream.select(station=sta, component=cp)
            if len(channel) > 0:
                data[s, c, :] = channel[0].data[:n_samples]
    return data

