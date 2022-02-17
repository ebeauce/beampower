import os
import sys

import pathlib
script_folder = pathlib.Path(__file__).parent.resolve()
sys.path.append(script_folder)
root_folder = os.path.join(script_folder, os.pardir)

# utility functions to handle asdf data
import read_asdf
# python package for computing travel times
# see at: https://github.com/malcolmw/pykonal
# please, acknowledge White et al. (2020) if using pykonal
import pykonal
import pandas as pd
import h5py as h5
import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from time import time as give_time

# define functions necessary for computing the travel times
# for a given grid of theoretical seismic sources and array
# of seismic stations, and a given velocity model
def load_velocity_model(path):
    return pd.read_csv(path, index_col=0)

def grid_velocity_model(longitude_min, longitude_max,
                        latitude_min, latitude_max,
                        depth_min, depth_max,
                        path_vel_model,
                        d_longitude=0.01,
                        d_latitude=0.01,
                        d_depth=0.5):
    """
    Resolution of the grid is controlled by d_longitude (degrees),
    d_latitude (degrees) and d_depth (km). The default value of 0.01
    is approximately equivalent to 1km.
    """
    
    model = load_velocity_model(path_vel_model)
    print(model)
    
    # build the interpolating function giving the velocity at any depth
    
    # create interpolator
    # convert m to km
    model['z_top'] /= 1000.
    model['Vp'] /= 1000.
    model['Vs'] /= 1000.
    new_depths = model['z_top'][1:] - 0.00001
    new_vP = model['Vp'][:-1]
    interpolator_P = interp1d(np.hstack( (model['z_top'], new_depths) ),
                              np.hstack( (model['Vp'], new_vP) ))
    new_vS = model['Vs'][:-1]
    interpolator_S = interp1d(np.hstack( (model['z_top'], new_depths) ),
                              np.hstack( (model['Vs'], new_vS) ))
    
    # --------------------
    # define the spacing in each direction
    deg_to_rad = np.pi / 180.
    longitudes = np.arange(longitude_min, longitude_max, d_longitude)
    #latitudes = np.arange(latitude_min, latitude_max, d_latitude)
    # pykonal uses co-latitude in its spherical coordinates
    # therefore it generates ranges from lat_max to lat_min,
    # the last point of the vector being lat_min-d_latitude
    # We need to do the same to make the grids match
    latitudes = np.arange(latitude_max, latitude_min, -d_latitude)
    depths = np.arange(depth_min, depth_max, d_depth)
    #depths = np.arange(depth_max, depth_min, -d_depth)
    # get the number of points in each direction
    n_longitudes = longitudes.size
    n_latitudes = latitudes.size
    n_depths = depths.size
    # convert degrees to radians
    d_longitude *= deg_to_rad
    d_latitude *= deg_to_rad
    
    # -------------------------------
    #    Make the velocity arrays
    # -------------------------------
    t1 = give_time()
    vP_grid = np.zeros((n_depths, n_latitudes, n_longitudes), dtype=np.float32)
    vS_grid = np.zeros((n_depths, n_latitudes, n_longitudes), dtype=np.float32)
    ## pykonal makes computations in spherical coordinates:
    ## so we order the vP and vS models for decreasing depths (increasing radius),
    ## decreasing latitudes (increasing polar angle), and increasing longitudes
    ## (increasing azimuthal angle)
    depths = depths[::-1]
    #latitudes = latitudes[::-1]
    print(depths)
    for i in range(n_depths):
        vP_grid[i, :, :] = interpolator_P(depths[i])
        vS_grid[i, :, :] = interpolator_S(depths[i])
    t2 = give_time()
    print(f'{t2-t1:.2f}s to make the grids of vP and vS')
    
    depths_g, latitudes_g, longitudes_g = np.meshgrid(
            depths, latitudes, longitudes, indexing='ij')

    # store the outputs in a dictionary
    grid = {}
    grid['vP'] = vP_grid
    grid['vS'] = vS_grid
    grid['longitude'] = longitudes_g
    grid['latitude'] = latitudes_g
    grid['depth'] = depths_g
    return grid

def compute_tts(network_metadata,
                longitude_min, longitude_max,
                latitude_min, latitude_max,
                depth_min, depth_max,
                path_vel_model,
                d_longitude=0.01,
                d_latitude=0.01,
                d_depth=0.5):

    grid = grid_velocity_model(longitude_min, longitude_max,
                               latitude_min, latitude_max,
                               depth_min, depth_max,
                               path_vel_model,
                               d_longitude=d_longitude,
                               d_latitude=d_latitude,
                               d_depth=d_depth)

    latitudes = np.unique(grid['latitude'])
    longitudes = np.unique(grid['longitude'])
    depths = np.unique(grid['depth'])
    # ---------------------------------------------
    n_latitudes = latitudes.size
    n_longitudes = longitudes.size
    n_depths = depths.size
    # ---------------------------------------------
    print('Latitude: {:.2f} - {:.2f}, {:d} points with {:.2f} spacing.'.
            format(latitude_min, latitude_max, n_latitudes, d_latitude))
    print('Longitude: {:.2f} - {:.2f}, {:d} points with {:.2f} spacing.'.
            format(longitude_min, longitude_max, n_longitudes, d_longitude))
    print('Depth: {:.2f} km - {:.2f} km, {:d} points with {:.2f} km spacing.'.
            format(depth_min, depth_max, n_depths, d_depth))

    # convert degrees to radians
    dtr = np.pi / 180.
    d_latitude *= dtr
    d_longitude *= dtr

    n_stations = len(network_metadata)

    # first, adjust the grid of the velocity model to match the grid of the solver
    rho_min, theta_min, lbd_min = pykonal.transformations.geo2sph(
            [latitude_max, longitude_min, depth_max-d_depth])
 
    ## initialize the solver
    #solver = pykonal.solver.PointSourceSolver(coord_sys="spherical")
    ## define the computational grid
    #solver.velocity.min_coords = rho_min, theta_min, lbd_min
    #solver.velocity.node_intervals = d_depth, d_latitude, d_longitude
    #solver.velocity.npts = n_depths, n_latitudes, n_longitudes
    ## get coordinates
    #solver_coords = pykonal.transformations.sph2geo(solver.velocity.nodes)
    #breakpoint()
    ## adjust velocity model
    #adjusted_velocity = {}
    #for phase in ['P', 'S']:
    #    interpolator = LinearNDInterpolator(
    #            np.stack([grid['latitude'].flatten(),
    #                      grid['longitude'].flatten(),
    #                      grid['depth'].flatten()], axis=-1),
    #            grid[f'v{phase}'].flatten())
    #    adjusted_velocity[phase] = interpolator(solver_coords.reshape(-1, 3))\
    #            .reshape(solver_coords.shape[:-1])
    #del solver


    # store travel times in a dictionary
    tts = {'tt_P': {}, 'tt_S': {}}
    for s, (row_idx, network_sta) in enumerate(network_metadata.iterrows()):
        print(f'Computing station {s+1}/{n_stations}...')
        for phase in ['P', 'S']:
            # initialize travel time arrays
            tts[f'tt_{phase}'][network_sta.station_code] = np.zeros(
                    grid[f'v{phase}'].shape, dtype=np.float32)

            ##### Initialize the solver and build the grid
            
            # ! theta is colatitude, so we need to input latitude_max
            rho_min, theta_min, lbd_min = pykonal.transformations.geo2sph(
                    [grid['latitude'].max(), grid['longitude'].min(), grid['depth'].max()])
            print('Min radius: {:.0f} km, min co-latitude: {:.2f} rad, min azimuthal angle: {:.2f} rad'.
                  format(rho_min, theta_min, lbd_min))
         
            # initialize the solver
            solver = pykonal.solver.PointSourceSolver(coord_sys="spherical")
            # define the computational grid
            solver.velocity.min_coords = rho_min, theta_min, lbd_min
            solver.velocity.node_intervals = d_depth, d_latitude, d_longitude
            solver.velocity.npts = n_depths, n_latitudes, n_longitudes
            solver.velocity.values = grid[f'v{phase}']
            #solver.velocity.values = adjusted_velocity[phase]

            # define station s as a source to compute the travel time
            # field inside the whole grid
            source_latitude = network_sta.latitude
            source_longitude = network_sta.longitude
            source_depth = (-1.*network_sta.elevation)/1000. # elevation = -depth, and convert to km
            # convert the geographical coordinates to spherical coordinates
            rho_source, theta_source, lbd_source = pykonal.transformations.geo2sph(
                    [source_latitude, source_longitude, source_depth])
            print('Spherical coordinates of the source: {:.2f} km, {:.2f} polar angle, {:.2f} azimuthal angle'.
                 format(rho_source, theta_source/dtr, lbd_source/dtr))
            solver.src_loc = np.array([rho_source, theta_source, lbd_source])
            
            # solve the Eikonal equation
            t1 = give_time()
            solver.solve()
            t2 = give_time()
            print(f'{t2-t1:.2f}sec to compute the {phase}-wave travel time field '
                  f'on station {network_sta.station_code}')

            tts[f'tt_{phase}'][network_sta.station_code] = solver.tt.values[...]
            # just checking:
            print(f'{tts[f"tt_{phase}"][network_sta.station_code].min():.3f}',
                  f'{tts[f"tt_{phase}"][network_sta.station_code].max():.3f}')

    # attach source coordinates to tts:
    tts['source_coordinates'] = {}
    src_coords = pykonal.transformations.sph2geo(solver.velocity.nodes)
    tts['source_coordinates']['latitude'] = src_coords[..., 0]
    tts['source_coordinates']['longitude'] = src_coords[..., 1]
    tts['source_coordinates']['depth'] = src_coords[..., 2]
    return tts

if __name__ == "__main__":
    # define some I/O variables
    path_data = os.path.join(root_folder, 'data')
    data_filename = 'data_20120726.h5'
    vel_model_filename = 'velocity_model_Karabulut2011.csv' # cite Karabulut et al. (2011)
    path_vel_model = os.path.join(path_data, vel_model_filename)
    tt_filename = 'tts.h5'
    path_tts = os.path.join(path_data, tt_filename)

    # define the geographical extent of the grid of
    # theoretical seismic sources
    lon_min, lon_max = 30.20, 30.45
    lat_min, lat_max = 40.60, 40.76
    dep_min, dep_max = -2., 30. # in km
    # define the spacing (in degrees for lon and lat, in km for depths)
    d_lat = 0.01 # about 1 km
    d_lon = 0.01 # about 1 km

    d_dep = 0.5 # km

    # load the seismic station network's metadata
    network_metadata = read_asdf.get_asdf_metadata(
            os.path.join(path_data, data_filename))

    # compute the travel-times
    tts = compute_tts(
            network_metadata, lon_min, lon_max, lat_min, lat_max,
            dep_min, dep_max, path_vel_model, d_longitude=d_lon,
            d_latitude=d_lat, d_depth=d_dep)

    print('Done! Writing travel-times to hdf5 file...')
    # write travel-times in hdf5 file
    with h5.File(path_tts, mode='w') as f:
        for key1 in tts.keys():
            f.create_group(key1)
            for key2 in tts[key1].keys():
                f[key1].create_dataset(
                        key2, data=tts[key1][key2], compression='gzip')
