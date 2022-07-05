import os

path_script = os.path.dirname(__file__)

from obspy.clients.fdsn.mass_downloader import RectangularDomain,\
        Restrictions, MassDownloader
from obspy import UTCDateTime as udt

path_data = os.path.join(path_script, os.pardir, 'data')
if not os.path.isdir(path_data):
    print(f'Creating a new data folder at {path_data}')
    os.mkdir(path_data)

provider = 'IRIS'

network = 'YH'
loc = '*'
channel = 'BH*,HH*'

starttime = udt('2012-07-26')
endtime = udt('2012-07-27')

station_list = ['SAUV', 'SPNC', 'DC08', 'DC07', 'DC06', 'DD06', 'DE07', 'DE08']

domain = RectangularDomain(
        minlatitude=40.60, maxlatitude=40.76,
        minlongitude=30.20, maxlongitude=30.44
        )

restrictions = Restrictions(
        starttime=starttime,
        endtime=endtime,
        chunklength_in_sec=86400.,
        network=network,
        location=loc,
        channel=channel,
        station=station_list,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=100.0
        )


mdl = MassDownloader(providers=[provider])
mdl.download(
        domain, restrictions, mseed_storage=path_data,
        stationxml_storage=path_data)

