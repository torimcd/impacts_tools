import xarray as xr
import h5py
import netCDF4
import numpy as np
import scipy
from pyproj import Proj
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter, gaussian_filter1d

def xsection_grid(x_grid, y_grid):
    '''
    Adds an extra element in the time/distance and height dimension to conform to newer mpl pcolormesh() function.
    Useful for plotting ER-2 radar cross sections.
    
    Parameters
    ----------
    x_grid: 2D time or distance field created from er2read() or resample() subroutines.
    y_grid: 2D height field created from er2read() or resample() subroutines.
    '''
    
    # Work with x coordinate (time/distance) first
    xdelta = x_grid[0, -1] - x_grid[0, -2] # should work on time and float dtypes
    vals_to_append = np.atleast_2d(np.tile(x_grid[0,-1] + xdelta, x_grid.shape[0])).T
    x_regrid = np.hstack((x_grid, vals_to_append)) # add column
    x_regrid = np.vstack((np.atleast_2d(x_regrid[0,:]), x_regrid)) # add row
    
    # Now do the y coordinate (height)
    ydelta = y_grid[0, :] - y_grid[1, :] # height difference between first and second gates
    vals_to_append = np.atleast_2d(y_grid[0,:] + ydelta)
    y_regrid = np.vstack((vals_to_append, y_grid)) # add row
    y_regrid = np.hstack((y_regrid, np.atleast_2d(y_regrid[:,-1]).T))
    
    return x_regrid, y_regrid

def despeckle(er2data, sigma=1.):
    temp = np.copy(er2data)
    temp_mask = gaussian_filter(temp, sigma)
    temp_mask = np.ma.masked_invalid(temp_mask)
    er2data = np.ma.masked_where(np.ma.getmask(temp_mask), er2data)
    
    return er2data

def ice_atten(crs_object, hiwrap_object, hrrr_hiwrap_object):
    '''
    Correct W-band reflectivity from atten4uation due to ice scattering.
    Uses mean Z_ku-k_w relationship from Fig. 7 in Kulie et al. (2014; https://doi.org/10.1175/JAMC-D-13-066.1).
    
    Parameters
    ----------
    crs_object: CRS dict object from er2read() funtion, optionally resampled from resample() function
    hiwrap_object: HIWRAP dict object from er2read() funtion, optionally resampled from resample() function
    hrrr_hiwrap_object: xarray dataset object containing HRRR fields interpolated to HIWRAP grid
    
    Execution
    ----------
    crs_object = er2_radar.er2read(crs_filename, **args)
    hiwrap_object = er2_radar.er2read(hiwrap_filename, **args)
    hrrr_hiwrap_object = xr.open_dataset(hrrr_hiwrap_filename)
    '''
    
    # Get start/end times from HIWRAP object (e.g., start/end of entire flight or of leg)
    start_time, end_time = [np.datetime_as_string(hiwrap_object['time'][0]), np.datetime_as_string(hiwrap_object['time'][-1])]
    
    # Resample radar data if needed
    if len(crs_object['time'])!=len(hiwrap_object['time']):
        print('Attempting to resample the CRS data before correction.')
        hiwrap_sub, crs_sub = resample(hiwrap_object, start_time, end_time, crs_object=crs_object)
        crs_object = crs_sub
        hiwrap_object = hiwrap_sub
        del crs_sub, hiwrap_sub
        
    # Fit Kulie et al. (2014) relationship to a Z_ku-dependent func
    dbz_ku_lin = np.array([0., 2000., 4000., 6000., 8000.]) # mm**6 / m**-3
    ks_w_coeff = np.array([0., 7.5, 15.5, 23.75, 31.5]) # db / km
    ks_w_func = np.poly1d(np.polyfit(dbz_ku_lin, ks_w_coeff, deg=1)) # slope, intercept coeffs
    
    # Build mask for T > 0C
    temp_hrrr = hrrr_hiwrap_object['temperature'].values[:, np.where((hrrr_hiwrap_object['time']>=np.datetime64(start_time)) & (hrrr_hiwrap_object['time']<=np.datetime64(end_time)))[0]]
    temp_mask = np.zeros(temp_hrrr.shape, dtype=bool)
    temp_inds = np.where(temp_hrrr>=0.)
    if len(temp_inds[0])>0:
        for beamnum in range(len(temp_inds[0])):
            temp_mask[temp_inds[0][beamnum]:, temp_inds[1][beamnum]] = True

    # Calculate 2-way PIA for each resampled CRS gate
    ks_w = ks_w_func(10.**(hiwrap_object['dbz_Ku']/10.))
    ks_w = ks_w * 0.0265 # convert to db/gate
    ks_w[np.isnan(ks_w)] = 0.; ks_w[ks_w<0.] = 0.
    ks_w = np.ma.array(2. * np.cumsum(ks_w, axis=(0))) # calc the 2-way attenuation
    ks_w = np.ma.masked_where(temp_mask, ks_w) # mask values where T > 0C
    
    # Correct W-band reflectivity
    crs_object['dbz_W'] = crs_object['dbz_W'] + ks_w
    
    return crs_object

def resample(hiwrap_object, start_time, end_time, crs_object=None, exrad_object=None):
    '''
    Resample CRS and/or EXRAD nadir beam data to the HIWRAP grid, which is the coarsest of the 3 radars.
    Only works on portions of a flight to speed up performance.
    INPUTS:
        hiwrap_object: HIWRAP object obtained from er2_radar.er2read() method
        start_time: Start time in YYYY-mm-ddTHH:MM:SS string format
        end_time: End time in YYYY-mm-ddTHH:MM:SS string format
        crs_object: CRS object obtained from er2_radar.er2read() method
        exrad_object: EXRAD object obtained from er2_radar.er2read() method
    OUTPUTS:
        hiwrap_resampled: Trimmed HIWRAP object based on the start/end times
        crs_resampled [optional]: Trimmed CRS object with the same shape as `hiwrap_resampled`
        exrad_resampled [optional]: Trimmed EXRAD object with the same shape as `hiwrap_resampled`
    '''
    
    # Trim HIWRAP data for processing
    time_inds_hiwrap = np.where((hiwrap_object['time']>=np.datetime64(start_time)) & (hiwrap_object['time']<=np.datetime64(end_time)))[0]
    hiwrap_resampled = {}
    hiwrap_resampled['time'] = hiwrap_object['time'][time_inds_hiwrap]
    hiwrap_resampled['nomdist'] = hiwrap_object['nomdist'][time_inds_hiwrap] - hiwrap_object['nomdist'][time_inds_hiwrap][0] # reset to 0 for period
    hiwrap_resampled['time_gate'] = hiwrap_object['time_gate'][:, time_inds_hiwrap]
    hiwrap_resampled['alt_gate'] = hiwrap_object['alt_gate'][:, time_inds_hiwrap]
    hiwrap_resampled['lon_gate'] = hiwrap_object['lon_gate'][:, time_inds_hiwrap]
    hiwrap_resampled['lat_gate'] = hiwrap_object['lat_gate'][:, time_inds_hiwrap]
    hiwrap_resampled['dbz_Ka'] = hiwrap_object['dbz_Ka'][:, time_inds_hiwrap]
    hiwrap_resampled['ldr_Ka'] = hiwrap_object['ldr_Ka'][:, time_inds_hiwrap]
    hiwrap_resampled['vel_Ka'] = hiwrap_object['vel_Ka'][:, time_inds_hiwrap]
    hiwrap_resampled['width_Ka'] = hiwrap_object['width_Ka'][:, time_inds_hiwrap]
    hiwrap_resampled['dbz_Ku'] = hiwrap_object['dbz_Ku'][:, time_inds_hiwrap]
    hiwrap_resampled['ldr_Ku'] = hiwrap_object['ldr_Ku'][:, time_inds_hiwrap]
    hiwrap_resampled['vel_Ku'] = hiwrap_object['vel_Ku'][:, time_inds_hiwrap]
    hiwrap_resampled['width_Ku'] = hiwrap_object['width_Ku'][:, time_inds_hiwrap]
    
    
    # Set reference point (currently Albany, NY)
    lat_0 = 42.6526
    lon_0 = -73.7562

    # Define a map projection to calculate cartesian distances
    p = Proj(proj='laea', zone=10, ellps='WGS84', lat_0=lat_0, lon_0=lon_0)
    
    # Get HIWRAP cartesian points
    lon_hiwrap = hiwrap_object['lon_gate'][0, time_inds_hiwrap] # only need first gate coordinate as the rest in each beam are the same
    lat_hiwrap = hiwrap_object['lat_gate'][0, time_inds_hiwrap] # only need first gate coordinate as the rest in each beam are the same
    hiwrap_x, hiwrap_y = p(lon_hiwrap, lat_hiwrap)
    
    # Resample CRS data if specified
    if crs_object is not None:
        time_inds_crs = np.where((crs_object['time']>=np.datetime64(start_time)) & (crs_object['time']<=np.datetime64(end_time)))[0]
        hiwrap_x_tile = np.tile(np.reshape(hiwrap_x, (len(hiwrap_x), 1)), (1, len(time_inds_crs)))
        hiwrap_y_tile = np.tile(np.reshape(hiwrap_y, (len(hiwrap_y), 1)), (1, len(time_inds_crs)))
        
        # Get CRS cartesian points
        crs_lon = crs_object['lon_gate'][0, time_inds_crs]
        crs_lat = crs_object['lat_gate'][0, time_inds_crs]
        crs_x, crs_y = p(crs_lon, crs_lat)
        crs_x_tile = np.tile(np.reshape(crs_x, (1, len(crs_x))), (len(time_inds_hiwrap), 1))
        crs_y_tile = np.tile(np.reshape(crs_y, (1, len(crs_y))), (len(time_inds_hiwrap), 1))
        
        # Get CRS beam indices and save some variables to a dictionary
        dists = np.sqrt((hiwrap_x_tile - crs_x_tile)**2. + (hiwrap_y_tile-crs_y_tile)**2.)
        crs_beam_inds = np.argmin(dists, axis=1)
        
        crs_resampled = {}
        crs_resampled['time'] = hiwrap_resampled['time']
        crs_resampled['nomdist'] = hiwrap_resampled['nomdist']

        # Loop through beams and determine nearest CRS gate to each HIWRAP gate
        dbz_w = np.ma.zeros(hiwrap_object['dbz_Ku'][:, time_inds_hiwrap].shape)
        ldr_w = np.ma.zeros(hiwrap_object['ldr_Ku'][:, time_inds_hiwrap].shape)
        vel_w = np.ma.zeros(hiwrap_object['vel_Ku'][:, time_inds_hiwrap].shape)
        width_w = np.ma.zeros(hiwrap_object['width_Ku'][:, time_inds_hiwrap].shape)

        for time_ind in range(len(crs_beam_inds)):
            alt_beam_hiwrap = hiwrap_object['alt_gate'][:, time_inds_hiwrap[time_ind]]
            alt_beam_crs = crs_object['alt_gate'][:, time_inds_crs[crs_beam_inds[time_ind]]]
            alt_beam_hiwrap = np.tile(np.reshape(alt_beam_hiwrap, (len(alt_beam_hiwrap), 1)), (1, len(alt_beam_crs)))
            alt_beam_crs = np.tile(np.reshape(alt_beam_crs, (1, len(alt_beam_crs))), (alt_beam_hiwrap.shape[0], 1))

            crs_gate_inds = np.argmin(np.abs(alt_beam_hiwrap - alt_beam_crs), axis=1)
            dbz_w[:, time_ind] = crs_object['dbz_W'][crs_gate_inds, time_inds_crs[crs_beam_inds[time_ind]]]
            ldr_w[:, time_ind] = crs_object['ldr_W'][crs_gate_inds, time_inds_crs[crs_beam_inds[time_ind]]]
            vel_w[:, time_ind] = crs_object['vel_W'][crs_gate_inds, time_inds_crs[crs_beam_inds[time_ind]]]
            width_w[:, time_ind] = crs_object['width_W'][crs_gate_inds, time_inds_crs[crs_beam_inds[time_ind]]]
            
        # Assign variables to dictionary
        crs_resampled['time_gate'] = hiwrap_resampled['time_gate']
        crs_resampled['alt_gate'] = hiwrap_resampled['alt_gate']
        crs_resampled['lon_gate'] = hiwrap_resampled['lon_gate']
        crs_resampled['lat_gate'] = hiwrap_resampled['lat_gate']
        crs_resampled['dbz_W'] = dbz_w
        crs_resampled['ldr_W'] = ldr_w
        crs_resampled['vel_W'] = vel_w
        crs_resampled['width_W'] = width_w
        
    # Resample EXRAD data if specified
    if exrad_object is not None:
        time_inds_exrad = np.where((exrad_object['time']>=np.datetime64(start_time)) & (exrad_object['time']<=np.datetime64(end_time)))[0]
        hiwrap_x_tile = np.tile(np.reshape(hiwrap_x, (len(hiwrap_x), 1)), (1, len(time_inds_exrad)))
        hiwrap_y_tile = np.tile(np.reshape(hiwrap_y, (len(hiwrap_y), 1)), (1, len(time_inds_exrad)))
        
        # Get EXRAD cartesian points
        exrad_lon = exrad_object['lon_gate'][0, time_inds_exrad]
        exrad_lat = exrad_object['lat_gate'][0, time_inds_exrad]
        exrad_x, exrad_y = p(exrad_lon, exrad_lat)
        exrad_x_tile = np.tile(np.reshape(exrad_x, (1, len(exrad_x))), (len(time_inds_hiwrap), 1))
        exrad_y_tile = np.tile(np.reshape(exrad_y, (1, len(exrad_y))), (len(time_inds_hiwrap), 1))
        
        # Get EXRAD beam indices and save some variables to a dictionary
        dists = np.sqrt((hiwrap_x_tile - exrad_x_tile)**2. + (hiwrap_y_tile-exrad_y_tile)**2.)
        exrad_beam_inds = np.argmin(dists, axis=1)
        
        exrad_resampled = {}
        exrad_resampled['time'] = hiwrap_resampled['time']
        exrad_resampled['nomdist'] = hiwrap_resampled['nomdist']

        # Loop through beams and determine nearest EXRAD gate to each HIWRAP gate
        dbz_x = np.ma.zeros(hiwrap_object['dbz_Ku'][:, time_inds_hiwrap].shape)
        vel_x = np.ma.zeros(hiwrap_object['vel_Ku'][:, time_inds_hiwrap].shape)
        width_x = np.ma.zeros(hiwrap_object['width_Ku'][:, time_inds_hiwrap].shape)

        for time_ind in range(len(exrad_beam_inds)):
            alt_beam_hiwrap = hiwrap_object['alt_gate'][:, time_inds_hiwrap[time_ind]]
            alt_beam_exrad = exrad_object['alt_gate'][:, time_inds_exrad[exrad_beam_inds[time_ind]]]
            alt_beam_hiwrap = np.tile(np.reshape(alt_beam_hiwrap, (len(alt_beam_hiwrap), 1)), (1, len(alt_beam_exrad)))
            alt_beam_exrad = np.tile(np.reshape(alt_beam_exrad, (1, len(alt_beam_exrad))), (alt_beam_hiwrap.shape[0], 1))

            exrad_gate_inds = np.argmin(np.abs(alt_beam_hiwrap - alt_beam_exrad), axis=1)
            dbz_x[:, time_ind] = exrad_object['dbz_X'][exrad_gate_inds, time_inds_exrad[exrad_beam_inds[time_ind]]]
            vel_x[:, time_ind] = exrad_object['vel_X'][exrad_gate_inds, time_inds_exrad[exrad_beam_inds[time_ind]]]
            width_x[:, time_ind] = exrad_object['width_X'][exrad_gate_inds, time_inds_exrad[exrad_beam_inds[time_ind]]]
            
        # Assign variables to dictionary
        exrad_resampled['time_gate'] = hiwrap_resampled['time_gate']
        exrad_resampled['alt_gate'] = hiwrap_resampled['alt_gate']
        exrad_resampled['lon_gate'] = hiwrap_resampled['lon_gate']
        exrad_resampled['lat_gate'] = hiwrap_resampled['lat_gate']
        exrad_resampled['dbz_X'] = dbz_x
        exrad_resampled['vel_X'] = vel_x
        exrad_resampled['width_X'] = width_x
    
    # Save out dictionaries
    if crs_object is None: # save out HIWRAP and EXRAD data
        return hiwrap_resampled, exrad_resampled
    if exrad_object is None: # save out HIWRAP and EXRAD data
        return hiwrap_resampled, crs_resampled
    else: # save out all radar data
        return hiwrap_resampled, crs_resampled, exrad_resampled

def er2read(er2file, beam='nadir', atten_file=None, max_roll=None, dbz_sigma=None, ldr_sigma=None, vel_sigma=None, width_sigma=None,
            dbz_min=None, ldr_min=None, vel_min=None, width_min=None):
    '''
    Parses ER-2 radar data and performs QC as requested.
    INPUTS:
        er2file: Path to the ER-2 radar dataset
        beam: 'nadir' or 'scanning' (currently only supports nadir beam data)
        atten_file: None or path to file containing gridded attenuation due to atmospheric gases
        max_roll: None or float value where data masked above threshold [deg]
        dbz_sigma: None or float value where data masked above threshold using a Gaussian filter
        ldr_sigma: None or float value where data masked above threshold using a Gaussian filter
        vel_sigma: None or float value where data masked above threshold using a Gaussian filter
        width_sigma: None or float value where data masked above threshold using a Gaussian filter
        dbz_min: None or float value where data masked below threshold [dBZ]
        ldr_min: None or float value where data masked below threshold [dB]
        vel_min: None or float value where data masked below threshold [m/s]
        width_min: None or float value where data masked below threshold [m/s]
    OUTPUTS:
        er2rad: Dictionary object with select navigation and radar variables
    '''

    er2rad = {}
    hdf = h5py.File(er2file, 'r')

    radname = hdf['Information']['RadarName'][0].decode('UTF-8')

    # Aircraft nav information
    alt_plane = hdf['Navigation']['Data']['Height'][:]
    lat = hdf['Navigation']['Data']['Latitude'][:]
    lon = hdf['Navigation']['Data']['Longitude'][:]
    heading = hdf['Navigation']['Data']['Heading'][:] # deg from north (==0 for northward, ==90 for eastward, ==-90 for westward)
    roll = hdf['Navigation']['Data']['Roll'][:]
    pitch = hdf['Navigation']['Data']['Pitch'][:]
    drift = hdf['Navigation']['Data']['Drift'][:]
    nomdist = hdf['Navigation']['Data']['NominalDistance'][:]

    # Time information
    time_raw = hdf['Time']['Data']['TimeUTC'][:]
    time_dt = [datetime(1970, 1, 1)+timedelta(seconds=time_raw[i]) for i in range(len(time_raw))] # Python datetime object
    time_dt64 = np.array(time_dt, dtype='datetime64[ms]') # Numpy datetime64 object (e.g., for plotting)

    # Radar information
    rg = hdf['Products']['Information']['Range'][:]
    if radname=='CRS':
        radar_dbz = hdf['Products']['Data']['dBZe'][:].T
        radar_ldr = hdf['Products']['Data']['LDR'][:].T
        radar_vel = hdf['Products']['Data']['Velocity_corrected'][:].T
        radar_width = hdf['Products']['Data']['SpectrumWidth'][:].T
        if atten_file is not None: # Correct for 2-way path integrated attenuation
            print('Correcting for attenuation at W-band due to atmospheric gases and LWC.')
            atten_data = xr.open_dataset(atten_file)
            radar_dbz = radar_dbz + atten_data['k_w'].values + atten_data['k_w_liquid'].values
    elif radname=='HIWRAP':
        radar_dbz = hdf['Products']['Ku']['Combined']['Data']['dBZe'][:].T
        radar_ldr = hdf['Products']['Ku']['Combined']['Data']['LDR'][:].T
        radar_vel = hdf['Products']['Ku']['Combined']['Data']['Velocity_corrected'][:].T
        radar_width = hdf['Products']['Ku']['Combined']['Data']['SpectrumWidth'][:].T
        radar2_dbz = hdf['Products']['Ka']['Combined']['Data']['dBZe'][:].T
        radar2_ldr = hdf['Products']['Ka']['Combined']['Data']['LDR'][:].T
        radar2_vel = hdf['Products']['Ka']['Combined']['Data']['Velocity_corrected'][:].T
        radar2_width = hdf['Products']['Ka']['Combined']['Data']['SpectrumWidth'][:].T
        if atten_file is not None: # Correct for 2-way path integrated attenuation
            print('Correcting for attenuation at Ka- and Ku-band due to atmospheric gases.')
            atten_data = xr.open_dataset(atten_file)
            radar_dbz = radar_dbz + atten_data['k_ku'].values
            radar2_dbz = radar2_dbz + atten_data['k_ka'].values
    elif radname=='EXRAD':
        radar_dbz = hdf['Products']['Data']['dBZe'][:].T
        radar_ldr = -999. * np.ones(radar_dbz.shape) # dummy values as variable does not exist
        if 'Velocity_corrected' in list(hdf['Products']['Data'].keys()):
            radar_vel = hdf['Products']['Data']['Velocity_corrected'][:].T # for NUBF correction
        else:
            radar_vel = hdf['Products']['Data']['Velocity'][:].T
        radar_width = np.ma.masked_invalid(hdf['Products']['Data']['SpectrumWidth'][:].T)
        if atten_file is not None: # Correct for 2-way path integrated attenuation
            print('Correcting for attenuation at X-band due to atmospheric gases.')
            atten_data = xr.open_dataset(atten_file)
            radar_dbz = radar_dbz + atten_data['k_x'].values
    else:
        print('Error: Unsupported radar')

    # Make some 1D variables 2D
    time2d = np.tile(time_dt64[np.newaxis, :], (len(rg), 1))
    [alt2d_plane, rg2d] = np.meshgrid(alt_plane, rg)
    alt_gate = alt2d_plane - rg2d # compute the altitude of each gate
    lat2d = np.tile(lat[np.newaxis, :], (len(rg), 1))
    lon2d = np.tile(lon[np.newaxis, :], (len(rg), 1))
    roll2d = np.tile(roll[np.newaxis, :], (len(rg), 1))

    # === QC data if user specifies it ===
    # Remove if aircraft roll exceeds 10 deg
    if max_roll is not None:
        radar_dbz = np.ma.masked_where(np.abs(roll2d) > max_roll, radar_dbz)
        radar_ldr = np.ma.masked_where(np.abs(roll2d) > max_roll, radar_ldr)
        radar_vel = np.ma.masked_where(np.abs(roll2d) > max_roll, radar_vel)
        radar_width = np.ma.masked_where(np.abs(roll2d) > max_roll, radar_width)
        if radname=='HIWRAP': # Ka-band
            radar2_dbz = np.ma.masked_where(np.abs(roll2d) > max_roll, radar2_dbz)
            radar2_ldr = np.ma.masked_where(np.abs(roll2d) > max_roll, radar2_ldr)
            radar2_vel = np.ma.masked_where(np.abs(roll2d) > max_roll, radar2_vel)
            radar2_width = np.ma.masked_where(np.abs(roll2d) > max_roll, radar2_width)

    # Remove if below a specified value/threshold
    if dbz_min is not None:
        radar_dbz = np.ma.masked_where(radar_dbz < dbz_min, radar_dbz)
        if radname=='HIWRAP': # Ka-band
            radar2_dbz = np.ma.masked_where(radar2_dbz < dbz_min, radar2_dbz)
    if ldr_min is not None:
        radar_ldr = np.ma.masked_where(radar_ldr < ldr_min, radar_ldr)
        if radname=='HIWRAP': # Ka-band
            radar2_ldr = np.ma.masked_where(radar2_ldr < ldr_min, radar2_ldr)
    if vel_min is not None:
        radar_vel = np.ma.masked_where(radar_vel < vel_min, radar_vel)
        if radname=='HIWRAP': # Ka-band
            radar2_vel = np.ma.masked_where(radar2_vel < vel_min, radar2_vel)
    if width_min is not None:
        radar_width = np.ma.masked_where(radar_width < width_min, radar_width)
        if radname=='HIWRAP': # Ka-band
            radar2_width = np.ma.masked_where(radar2_width < width_min, radar2_width)

    # Despeckle
    if dbz_sigma is not None:
        radar_dbz = despeckle(radar_dbz, dbz_sigma)
        if radname=='HIWRAP': # Ka-band
            radar2_dbz = despeckle(radar2_dbz, dbz_sigma)
    if ldr_sigma is not None:
        radar_ldr = despeckle(radar_ldr, ldr_sigma)
        if radname=='HIWRAP': # Ka-band
            radar2_ldr = despeckle(radar2_ldr, ldr_sigma)
    if vel_sigma is not None:
        radar_vel = despeckle(radar_vel, vel_sigma)
        if radname=='HIWRAP': # Ka-band
            radar2_vel = despeckle(radar2_vel, vel_sigma)
    if width_sigma is not None:
        radar_width = despeckle(radar_width, width_sigma)
        if radname=='HIWRAP': # Ka-band
            radar2_width = despeckle(radar2_width, width_sigma)

    # Assign values to the dictionary
    er2rad['time'] = time_dt64
    er2rad['time_gate'] = time2d
    er2rad['alt_plane'] = alt_plane
    er2rad['alt_gate'] = alt_gate
    er2rad['lat'] = lat
    er2rad['lon'] = lon
    er2rad['lat_gate'] = lat2d
    er2rad['lon_gate'] = lon2d
    er2rad['heading'] = heading
    er2rad['roll'] = roll
    er2rad['pitch'] = pitch
    er2rad['drift'] = drift
    er2rad['nomdist'] = nomdist
    if radname=='CRS':
        er2rad['dbz_W'] = radar_dbz
        er2rad['ldr_W'] = radar_ldr
        er2rad['vel_W'] = radar_vel
        er2rad['width_W'] = radar_width
    elif radname=='HIWRAP':
        er2rad['dbz_Ka'] = radar2_dbz
        er2rad['ldr_Ka'] = radar2_ldr
        er2rad['vel_Ka'] = radar2_vel
        er2rad['width_Ka'] = radar2_width
        er2rad['dbz_Ku'] = radar_dbz
        er2rad['ldr_Ku'] = radar_ldr
        er2rad['vel_Ku'] = radar_vel
        er2rad['width_Ku'] = radar_width
    elif radname=='EXRAD':
        er2rad['dbz_X'] = radar_dbz
        er2rad['vel_X'] = radar_vel
        er2rad['width_X'] = radar_width
        
    hdf.close() # close the HDF5 object
        
    return er2rad