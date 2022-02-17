import os

path_script = os.path.dirname(__file__)

from obspy.clients.fdsn import Client
from obspy import UTCDateTime as udt

path_data = os.path.join(path_script, os.pardir, 'data')
if not os.path.isdir(path_data):
    print(f'Creating a new data folder at {path_data}')
    os.mkdir(path_data)

client = Client("IRIS")

network = 'YH'
loc = '*'
channel = 'BH*,HH*'

starttime = udt('2012-07-26')
endtime = udt('2012-07-27')

station_list = ['SAUV', 'SPNC', 'DC08', 'DC07', 'DC06', 'DD06', 'DE07', 'DE08']

for sta in station_list:
    print(f'Trying to download data from {sta}...')
    st = client.get_waveforms(
            network, sta, loc, channel, starttime, endtime)
    for tr in st:
        inv = client.get_stations(
                network=network, station=sta, starttime=starttime, endtime=endtime,
                channel=tr.stats.channel, level='response')
        tr.write(os.path.join(path_data, tr.id+'.mseed'), format='mseed')
        inv.write(os.path.join(path_data, tr.id+'.xml'), format='STATIONXML')
