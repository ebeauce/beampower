import os

import glob
from pyasdf import ASDFDataSet

path_script = os.path.dirname(__file__)

# I/O variables
path_data = os.path.join(path_script, os.pardir, 'data')
output_filename = 'data_20120726_raw.h5'

# fetch file names
data_files = glob.glob(os.path.join(path_data, '*.mseed'))
metadata_files = glob.glob(os.path.join(path_data, '*.xml'))

# initialize ASDF data set
if os.path.isfile(os.path.join(path_data, output_filename)):
    os.remove(os.path.join(path_data, output_filename))
ds = ASDFDataSet(os.path.join(path_data, output_filename))

# add data
for data_fn in data_files:
    print(f'Adding {data_fn} to the data set.')
    ds.add_waveforms(data_fn, tag='raw')

# add metadata
for metadata_fn in metadata_files:
    print(f'Adding {metadata_fn} to the data set.')
    ds.add_stationxml(metadata_fn)

print(f'Done! You can access the ASDF data set at {os.path.join(path_data, output_filename)}.')
