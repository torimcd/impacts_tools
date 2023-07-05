"""
Classes for IMPACTS ER2 Instruments
"""

import h5py
import h5netcdf
import julian
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from scipy.ndimage import gaussian_filter, gaussian_filter1d


class Er2(object):
    """
    A class to represent the ER2 aircraft during the IMPACTS field campaign

    Variables
    ---------
    time : array(datetime)
        the time in UTC
    lat : array(float)
        the latitude of the ER2
    lon : array(float)
        the longitude of the ER2
    alt : array(float)
        the altitude of the ER2
    pitch : array(float)
        the pitch of the aircraft
    roll : array(float)
        the roll of the aircraft
    drift : array(float)
        the drift of the aircraft
    flight_legs: array(datetime)
        an array containing the start and end time of each flight leg during a storm

    """

    def __init__(self, date, filepath):
        """
       Parameters
        ----------
        date : datetime
            the date of the storm
        filepath : str, optional
            the filepath to the ER2 nav data file that will be read to populate the ER2 attributes
        """
        self.readfile(filepath)


    @classmethod
    def from_instrument(cls, er2params):
        """
        Alternative constructor to initialize the ER2 object directly from an instrument data file
        

        Parameters
        ----------
        er2params : xarray.DataSet()
            a dataset of attributes to populate the ER2 object if it's initialized from an instrument
        """
        pass

    def readfile(self, filepath):
       # this should read the ER2 nav files in the DAAC rather than an instrument file
        pass




class Radar(ABC):
    """ 
    A class to represent the various radars flown on the ER-2 aircraft during IMAPCTS.
    Radar is an Abstract Base Class - meaning we always require a more specific class 
    to be instantiated - ie you have to call Exrad() or Hiwrap(), you can't just call Radar()

    Parameters
    ----------
    data : xarray.Dataset()
        Radar data and attributes

    """

    @abstractmethod     # this stops you from being able to make a new generic radar
    def __init__(self):
        """
        This is an abstract method since only inherited classes will be used to instantiate Radar objects.
        """
        self.name = None
        self.data = None


    @abstractmethod
    def readfile(self, filepath):
        """
        This is an abstract method since only inherited classes will be used to read radar data into objects.
        """
        pass


    def mask_roll(self, max_roll):
        """
        Mask values in the dataset where the roll angle of the ER2 is greater
        than the maximum value provided

        Parameters
        ----------
        max_roll : float
            The maximum roll angle of the aircraft in degrees
        """

        # retain where the plane is not rolling more than the max provided
        return self.data.where(abs(self.data['er2_roll']) < max_roll)


    def despeckle(self, data_array=None, sigma=1.):
        """
        Despeckle the radar data by applying a Gaussion filter with a given standard deviation

        Parameters
        ----------
        data_array : xarray.DataArray()
            The field to despeckle
        sigma : float
            The standard deviation to apply to the gaussian filter
        """
        temp_datacopy =  data_array.copy()

        # run the data array through a gaussian filter
        temp_datafiltered = gaussian_filter(temp_datacopy, sigma)

        # np.isfinite returns values that are not NAN or INF
        return data_array.where(np.isfinite(temp_datafiltered))
    
    def resample(self, data_reference):
        """
        Resample the CRS or EXRAD cross section to the HIWRAP time-range grid
        
        Parameters
        ----------
        data_reference : impacts_tools.er2.Radar()
            The radar object to use as the time-range reference grid (e.g., er2.Hiwrap())
        """
        # remove beams (time dim) that aren't in HIWRAP data (typically every other beam)
        self.data = self.data.sel(time=data_reference.data.time)
        
        # find nearest CRS/EXRAD gate corresponding to HIWRAP range gates
        # CRS/EXRAD height may be offset by 20-25 m (vertical sampling resolution)
        # not a big deal as native vertical resolution much larger (> 100 m )
        return self.data.interp(
            range=data_reference.data.range, method='nearest', assume_sorted=True
        )
    
    def calc_cfad(self, vel_bins=None, alt_bins=None, band=None):
        """
        Calculates the frequency of radial velocity values per altitude level
        for making a contour-by-frequency (CFAD) plot 
        Parameters
        ----------
        vel_bins : np.ndarray or None
            the velocity bins to use. Defaults to np.linspace(-5., 5., num=101)
        alt_bins : np.ndarray or None
            The altitude bins to use. Defaults to alt_bins = np.linspace(100., 10000., num=45)
        band : str
            For HIWRAP, whether to use ka- or ku-band velocity
        Returns
        -------
        cfad : [data, X, Y] 
            The calculated velocity frequency as well as the X, Y meshgrid to plot it on
        """
        if type(vel_bins) == np.ndarray:
            pass
        else:
            vel_bins = np.linspace(-5., 5., num=101)

        if type(alt_bins) == np.ndarray:
            pass
        else:
            alt_bins = np.linspace(100., 10000., num=45)

        if band is not None:
            vel = self.data['vel_' + band.lower()]
            
        else:
            vel = self.data['vel']

        vel_flat = np.ma.masked_invalid(vel.values.flatten())

        hght_flat = self.data['height'].values.flatten()
        vel_fallspeedsremoved_flat = np.zeros_like(vel_flat)

        vel_cfad = np.zeros((len(alt_bins)-1, len(vel_bins)-1))
        medians = np.zeros(len(alt_bins) -1)

        for a in range(len(alt_bins)-1):
            # get the indices for this altitude range
            hght_inds = np.where((hght_flat>=alt_bins[a]) & (hght_flat<alt_bins[a+1]))[0]

            # make sure there's at least some velocity data for this alt bin, loop through vel bins
            if (len(hght_inds)>0) and (vel_flat[hght_inds].mask.sum() < len(hght_inds)): # 
        
                # total number of valid velocity gates for this altitude bin
                num_valid_vel = len(hght_inds) - vel_flat[hght_inds].mask.sum() 
        
                for v in range(len(vel_bins)-1):
                    vel_inds = np.where((hght_flat>=alt_bins[a]) & (hght_flat<alt_bins[a+1]) & (vel_flat>=vel_bins[v]) & (vel_flat<vel_bins[v+1]))[0]

                    # there's at least some velocity data for this alt-vel bin, compute the frequency
                    if (len(vel_inds)>0) and (vel_flat[vel_inds].mask.sum()<len(vel_inds)): 
                
                        # frequency of valid velocity gates for this alt-vel bin
                        vel_cfad[a, v] = (len(vel_inds) - vel_flat[vel_inds].mask.sum()) / num_valid_vel 
        
        
                #vel_max_ind = np.where(vel_cfad[a,:]==vel_cfad[a,:].max())[0]

                # subtract the velocity value of the max freq
                #vel_fallspeedsremoved_flat[hght_inds] = vel_flat[hght_inds] - vel_bins[vel_max_ind[0]]


                # correct the velocity with the median vel at each altitude (Rosenow et al 2014)
                med = np.ma.median(vel_flat[hght_inds])

                vel_fallspeedsremoved_flat[hght_inds] = vel_flat[hght_inds] - med

                # save for plotting median contour
                medians[a] = med

        cfad = np.ma.masked_where(vel_cfad==0.00, vel_cfad)
        [X, Y] = np.meshgrid(vel_bins[:-1]+np.diff(vel_bins)/2, alt_bins[:-1]+np.diff(alt_bins)/2)

        vel_corrected = np.reshape(vel_fallspeedsremoved_flat, vel[:,:].values.shape)

        return [cfad, X, Y, vel_corrected, medians]
        

    def get_flight_legs(self):
        """
        Get the start and end time stamps for each leg of this flight.

        Returns
        -------
        flight_leg : list of tuples
            Tuples of the start and end time of each flight leg.
        """
        pass # not yet implemented
    
class Lidar(ABC):
    """ 
    A class to represent the lidar flown on the ER-2 aircraft during IMAPCTS.
    Lidar is an Abstract Base Class - meaning we always require a more specific class 
    to be instantiated - ie you have to call Crs(), you can't just call Lidar()

    Parameters
    ----------
    data : xarray.Dataset()
        Lidar data and attributes

    """

    @abstractmethod     # this stops you from being able to make a new generic lidar
    def __init__(self):
        """
        This is an abstract method since only inherited classes will be used to instantiate Lidar objects.
        """
        self.name = None
        self.data = None

    def mask_roll(self, max_roll):
        """
        Mask values in the dataset where the roll angle of the ER2 is greater
        than the maximum value provided

        Parameters
        ----------
        max_roll : float
            The maximum roll angle of the aircraft in degrees
        """

        # retain where the plane is not rolling more than the max provided
        return self.data.where(abs(self.data['er2_roll']) < max_roll)


    def despeckle(self, data_array=None, sigma=1.):
        """
        Despeckle the lidar data by applying a Gaussion filter with a given standard deviation

        Parameters
        ----------
        data_array : xarray.DataArray()
            The field to despeckle
        sigma : float
            The standard deviation to apply to the gaussian filter
        """
        temp_datacopy =  data_array.copy()

        # run the data array through a gaussian filter
        temp_datafiltered = gaussian_filter(temp_datacopy, sigma)

        # np.isfinite returns values that are not NAN or INF
        return data_array.where(np.isfinite(temp_datafiltered))
    
    def trim_time_bounds(self, start_time=None, end_time=None, tres='1S'):
        """
        Put the dataset into the specified time bounds.
        
        Parameters
        ----------
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        Returns
        -------
        data : xarray.Dataset
            The reindexed dataset
        """
        
        if (start_time is not None) or (end_time is not None):      
            # compute dataset timedelta
            td_ds = pd.to_timedelta(
                self.data['time'][1].values - self.data['time'][0].values
            )
            
            # format start and end times
            if start_time is None:
                start_time = self.data['time'][0].values
                
            if end_time is None:
                end_time = self.data['time'][-1].values

            # generate trimmed dataset based on time bounds
            if pd.Timedelta(tres) != pd.Timedelta(1, 's'):
                end_time -= td_ds
            ds_sub = self.data.sel(time=slice(start_time, end_time))
            
            # compute along track distance and add to dataset
            from pyproj import Proj, transform, Geod
            geod = Geod(ellps='WGS84')

            dist_delta = np.zeros(len(ds_sub['lat']))
            for i in range(len(dist_delta) - 1):
                _, _, dist_delta[i + 1] = geod.inv(
                    ds_sub['lon'][i].values, ds_sub['lat'][i].values,
                    ds_sub['lon'][i + 1].values, ds_sub['lat'][i + 1].values
                )
            dist = xr.DataArray(
                data = np.cumsum(dist_delta),
                dims = 'time',
                coords = dict(
                    time = ds_sub.time,
                    lat = ds_sub.lat,
                    lon = ds_sub.lon),
                attrs = dict(
                    description='Distance along flight path (pyproj inverse transform)',
                    units='m'
                )
            )

            return ds_sub.assign_coords(distance=dist)
        
    def get_cloudtop_properties(self, cpl_layer_object):
        """
        Get the CPL properties at cloud top as defined by the L2 layer product.
        """
        # drop times with no/masked (e.g., roll > 10) layers
        cpl_layer_data = cpl_layer_object.data.where(
            ~np.isnan(cpl_layer_object.data.cloud_top_altitude), drop=True
        )
        
        # interpolate cloud top altitude and temperature to dataset (e.g., w/ L1B)
        cta = cpl_layer_data.cloud_top_altitude.interp_like(
            self.data.time, method='nearest', kwargs={'fill_value': 'extrapolate'}
        )
        ctt = cpl_layer_data.cloud_top_temperature.interp_like(
            self.data.time, method='nearest', kwargs={'fill_value': 'extrapolate'}
        )
        
        # compute absolute distance from cloud top for each gate
        gate_ind = np.fabs(self.data.height - cta).argmin(dim='gate')
        
        # build bool matrix where lidar gates above cloud top are ignored
        mask = np.ones(self.data.height.shape, dtype=bool) # true = mask data
        for beam in range(self.data.height.shape[1]):
            if ~np.isnan(gate_ind[beam]):
                mask[int(gate_ind[beam]) + 1:, beam] = False
                
        # find cloud top values of 2D products
        data_vars = {}
        data_vars['altitude_top'] = xr.DataArray(
            data = cta.values,
            dims = ['time'],
            coords = dict(
                time = self.data.time,
                lat = self.data.lat,
                lon = self.data.lon),
            attrs = dict(
                description='Cloud top altitude derived from uppermost layer top altitude',
                units='m'
            )
        )
        data_vars['temperature_top'] = xr.DataArray(
            data = ctt.values,
            dims = ['time'],
            coords = dict(
                time = self.data.time,
                lat = self.data.lat,
                lon = self.data.lon),
            attrs = dict(
                description='Cloud top temperature derived from uppermost layer top temperature',
                units='degrees_Celsius'
            )
        )
        for var in list(self.data.data_vars):
            if self.data[var].ndim == 2:
                if 'atb' in var:
                    description = f'Cloud top attenuated backscatter at {var.split("_")[-1]} nm'
                    units = 'km**-1 sr**-1'
                elif var == 'dpol_1064':
                    description = 'Cloud top depolarization ratio at 1064 nm'
                    units = '#'
                elif 'ext' in var:
                    description = f'Cloud top extinction at {var.split("_")[-1]} nm'
                    units = 'km**-1'
                data_2d  = np.ma.masked_where(mask, self.data[var].values)
                data_2d = np.ma.masked_invalid(data_2d)

                data_top = np.nan * np.empty(self.data.height.shape[1])
                for beam in range(self.data.height.shape[1]):
                    if len(data_2d[:, beam].compressed()) > 0:
                        data_top[beam] = np.mean(data_2d[
                            :, beam].compressed()[:3]) # get mean of first 3 good obs below cloud top
                data_vars[f'{var}_top'] = xr.DataArray(
                    data = np.ma.masked_invalid(data_top),
                    dims = 'time',
                    coords = dict(time = self.data.time),
                    attrs = dict(
                        description = description,
                        units = units
                    )
                )
                
        # add cloud top properties to L1B or L2 profile dataset
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {'time': self.data.time}
        )

        ds_merged = xr.merge(
            [self.data, ds], combine_attrs='drop_conflicts'
        )
        
        return ds_merged
        
    def get_flight_legs(self):
        """
        Get the start and end time stamps for each leg of this flight.

        Returns
        -------
        flight_leg : list of tuples
            Tuples of the start and end time of each flight leg.
        """
        pass # not yet implemented
        


# ====================================== #
# CRS
# ====================================== #
class Crs(Radar):
    """
    A class to represent the CRS nadir pointing radar flown on the ER2 during the IMPACTS field campaign.
    Inherits from Radar()
    
    Parameters
    ----------
    filepath: str
        File path to the CRS data file
    start_time: np.datetime64 or None
        The initial time of interest eg. if looking at a single flight leg
    end_time: np.datetime64 or None
        The final time of interest eg. if looking at a single flight leg
    atten_file: str or None
        Path to file containing gridded attenuation due to atmospheric gases
    max_roll : float or None
        The maximum roll angle of the aircraft in degrees, used to mask times with the aircraft is turning and the radars are off-nadir
    dbz_sigma: float or None
        The standard deviation to use in despeckling the radar reflectivity. Data above threshold is masked using a Gaussian filter
    vel_sigma: float or None
        The standard deviation to use in despeckling the radar Doppler velocity. Data above threshold is masked using a Gaussian filter
    width_sigma: float or None
        The standard deviation to use in despeckling the radar spectrum width. Data above threshold is masked using a Gaussian filter
    ldr_sigma: float or None
        The standard deviation to use in despeckling the radar linear depolarization ratio. Data above threshold is masked using a Gaussian filter
    dbz_min: float or None
        The minimum radar reflectivity value
    vel_min: float or None
        The minimum Doppler velocity value of interest. Data will be masked below threshold (m/s)
    width_min: float or None
        The minimum spectrum width value of interest. Data will be masked below threshold (m/s)

 
    """

    def __init__(self, filepath, start_time=None, end_time=None, atten_file=None, dataset='2020', max_roll=None, 
        dbz_sigma=None, vel_sigma=None, width_sigma=None, ldr_sigma=None, dbz_min=None, vel_min=None, width_min=None,
        resample_ref=None, temperature=None):

    
        self.name = 'CRS'

        # read the raw data
        self.data = self.readfile(filepath, start_time, end_time, dataset)
        """
        xarray.Dataset of radar variables and attributes

        Dimensions:
            - range: xarray.DataArray(float) - The radar range gate  
            - time: np.array(np.datetime64[ns]) - The UTC time stamp

        Coordinates:
            - range (range): xarray.DataArray(float) - The radar range gate  
            - height (range, time): xarray.DataArray(float) - Altitude of each range gate (m)  
            - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
            - distance (time): xarray.DataArray(float) - Along-track distance from the flight origin (m)
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)

        Variables:
            - dbz (range, time) : xarray.DataArray(float) - Equivalent reflectivity factor (db) with 1-sigma noise threshold applied
            - vel (range, time) : xarray.DataArray(float) - Doppler velocity corrected to account for intrusion of horizontal reanalysis winds and nonuniform beam filling (m/s)
            - width (range, time) : xarray.DataArray(float) - Doppler velocity spectrum width estimate including aircraft motion. 1-sigma noise threshold applied (m/s)
            - ldr (range, time) : xarray.DataArray(float) - Linear depolarization ratio thresholded at 3-sigma (dB)
            - vel_horiz_offset (range, time) : xarray.DataArray(float) - The horizontal wind offset used to correct Ka-band Doppler velocity [Vel_corr = Vel_uncorr - nubf_offset - horizwind_offset] (m/s)
            - vel_nubf_offset (range, time) : xarray.DataArray(float) - The nonuniform beam filling (NUBF) offset used to correct Ku-band Doppler velocity (m/s)
            - channel_mask (range, time): xarray.DataArray(int) - Composite image channel mask. 0: No signal, 1: Low-resolution pulse, 2: High-resolution pulse, 3: Chirp
            - horizontal_resolution_ku (range) : xarray.DataArray(float) - Approximate horizontal resolution defined as width of spatial weighting after averaging as a function of radar range (m)
            - dxdr (time) : xarray.DataArray(float) - Data cross-track distance from aircraft per radar range. Positive is starboard direction (m/m)
            - dydr (time) : xarray.DataArray(float) - Data along-track distance from aircraft per radar range. Positive is forward direction (m/m)
            - dzdr (time) : xarray.DataArray(float) - Data vertical distance from aircraft per radar range. Positive is upward direction (m/m)
            - er2_altitude (time) : xarray.DataArray(float) - Aircraft height above sea level (m)
            - er2_heading (time) : xarray.DataArray(float) - Aircraft heading in degrees from north. 90 degrees is eastward pointing (degrees)
            - er2_pitch (time) : xarray.DataArray(float) - Aircraft pitch (degrees)
            - er2_roll (time) : xarray.DataArray(float) - Aircraft roll (degrees)
            - er2_drift (time) : xarray.DataArray(float) - Distance between track and heading (degrees)
            - er2_EastVel (time) : xarray.DataArray(float) - Eastward component of velocity (m/s)
            - er2_NorthVel (time) : xarray.DataArray(float) - Northward component of velocity (m/s)
            - er2_upVel (time) : xarray.DataArray(float) - Upward velocity (m/s)
            - er2_track (time) : xarray.DataArray(float) - Direction from motion in degrees from north. 90 degrees is eastward motion (degrees)
            - er2_motion (time) : xarray.DataArray(float) - Estimated aircraft motion notmal to the beam, subtracted from Doppler estimate. Smoothed to a 2 second average motion (m/s)

        Attribute Information:
            Experiment, Date, Aircraft, Radar Name, Data Contact, Instrument PI, Mission PI, Antenna Size, 
            Antenna one-way 3dB beamwidth (degrees), Number of pulses, Radar transmit frequency (Hz), Radar transmit wavelength (m),
            Range gate spacing (m), Nominal antenna pointing, PRI, vertical resolution
        """

        # correct for 2-way path integrated attenuation
        if atten_file is not None: 
            self.correct_attenuation(atten_file)

        # mask values when aircraft is rolling
        if max_roll is not None:
            self.data = self.mask_roll(max_roll)

        # despeckle
        if dbz_sigma is not None:
            self.data['dbz'] = self.despeckle(self.data['dbz'], dbz_sigma)

        if vel_sigma is not None:
            self.data['vel'] = self.despeckle(self.data['vel'], vel_sigma)
        
        if width_sigma is not None:
            self.data['width'] = self.despeckle(self.data['width'], width_sigma)
        
        if ldr_sigma is not None:
            self.data['ldr'] = self.despeckle(self.data['ldr'], ldr_sigma)
            
        if (resample_ref is not None) and (temperature is not None):
            self.data = self.resample(resample_ref)
            self.data = self.correct_ice_atten(resample_ref, temperature)
        elif resample_ref is not None:
            self.data = self.resample(resample_ref)



    def readfile(self, filepath, start_time=None, end_time=None, dataset='2020'):
        """
        Reads the CRS data file and unpacks the fields into an xarray.Dataset

        Parameters
        ----------

        filepath : str
            Path to the data file
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        dataset  : str
            Four-digit deployment year for handling radar variables/metadata
        
        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """

        # open the file
        hdf = h5py.File(filepath, 'r')

        # Time information -- this is the first dimension in nav coords and products
        time_raw = hdf['Time']['Data']['TimeUTC'][:]
        time_dt = [datetime(1970, 1, 1) + timedelta(seconds=time_raw[i]) for i in range(len(time_raw))] # Python datetime object
        time_dt64 = np.array(time_dt, dtype='datetime64[ms]') # Numpy datetime64 object (e.g., for plotting)

        if start_time is not None:
            time_inds = np.where((time_dt64>=start_time) & (time_dt64<=end_time))[0]
        else:
            time_inds = np.where((time_dt64 != None))[0]


        # Aircraft nav information
        if dataset=='2020': # metadata key misspelled in 2020 dataset
            description_str = 'NominalDistance_desciption'
        else:# metadata key spelling fixed in 2022 dataset
            description_str = 'NominalDistance_description'
        nomdist = xr.DataArray(
            data = hdf['Navigation']['Data']['NominalDistance'][:],
            dims = ["time"],
            coords = dict(time=time_dt64),
            attrs = dict(
                description=hdf['Navigation']['Information'][description_str][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['NominalDistance_units'][0].decode('UTF-8')
            )
        )
        lat = xr.DataArray(
            data = hdf['Navigation']['Data']['Latitude'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description = hdf['Navigation']['Information']['Latitude_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Latitude_units'][0].decode('UTF-8')
            )
        )
        lon = xr.DataArray(
            data = hdf['Navigation']['Data']['Longitude'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Longitude_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Longitude_units'][0].decode('UTF-8')
            )
        )
        altitude = xr.DataArray(
            data = hdf['Navigation']['Data']['Height'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Height_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Height_units'][0].decode('UTF-8')
            )
        )
        heading = xr.DataArray(
            data = hdf['Navigation']['Data']['Heading'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Heading_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Heading_units'][0].decode('UTF-8')
            )
        )
        roll = xr.DataArray(
            data = hdf['Navigation']['Data']['Roll'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Roll_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Roll_units'][0].decode('UTF-8')
            )
        )
        pitch = xr.DataArray(
            data = hdf['Navigation']['Data']['Pitch'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Pitch_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Pitch_units'][0].decode('UTF-8')
            )
        )
        drift = xr.DataArray(
            data = hdf['Navigation']['Data']['Drift'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Drift_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Drift_units'][0].decode('UTF-8')
            )
        )
        eastVel = xr.DataArray(
            data = hdf['Navigation']['Data']['EastVelocity'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['EastVelocity_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['EastVelocity_units'][0].decode('UTF-8')
            )
        )
        northVel = xr.DataArray(
            data = hdf['Navigation']['Data']['NorthVelocity'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['NorthVelocity_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['NorthVelocity_units'][0].decode('UTF-8')
            )
        )
        track = xr.DataArray(
            data = hdf['Navigation']['Data']['Track'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Track_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Track_units'][0].decode('UTF-8')
            )
        )
        upvel = xr.DataArray(
            data = hdf['Navigation']['Data']['UpVelocity'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['UpVelocity_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['UpVelocity_units'][0].decode('UTF-8')
            )
        )
        dxdr = xr.DataArray(
            data = hdf['Navigation']['Data']['dxdr'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['dxdr_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['dxdr_units'][0].decode('UTF-8')
            )
        )
        dydr = xr.DataArray(
            data = hdf['Navigation']['Data']['dydr'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['dydr_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['dydr_units'][0].decode('UTF-8')
            )
        )
        dzdr = xr.DataArray(
            data = hdf['Navigation']['Data']['dzdr'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['dzdr_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['dzdr_units'][0].decode('UTF-8')
            )
        )

        # Radar information
        radar_range = hdf['Products']['Information']['Range'][:] # this is the second dimension on product data

        [alt2d, radar_range2d] = np.meshgrid(altitude, radar_range)
        hght = alt2d - radar_range2d

        height = xr.DataArray(
            data = hght[:],
            dims = ['range', 'time'],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Height of each radar range gate',
                units='m'
            )
        )

        dbz = xr.DataArray(
            data = hdf['Products']['Data']['dBZe'][:].T,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['dBZe_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['dBZe_units'][0].decode('UTF-8')
            )   
        )

        width = xr.DataArray(
            data = np.ma.masked_invalid(hdf['Products']['Data']['SpectrumWidth'][:].T),
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['SpectrumWidth_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['SpectrumWidth_units'][0].decode('UTF-8')
            ) 
        )

        if 'Velocity_corrected' in list(hdf['Products']['Data'].keys()):
            # for NUBF correction
            vel = xr.DataArray(
                data = hdf['Products']['Data']['Velocity_corrected'][:].T,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description=hdf['Products']['Information']['Velocity_corrected_description'][0].decode('UTF-8'),
                    units = hdf['Products']['Information']['Velocity_corrected_units'][0].decode('UTF-8')
                ) 
            )
        else:
            vel = xr.DataArray(
                data = hdf['Products']['Data']['Velocity'][:].T,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description=hdf['Products']['Information']['Velocity_description'][0].decode('UTF-8'),
                    units = hdf['Products']['Information']['Velocity_units'][0].decode('UTF-8')
                ) 
            )
        
        if hdf['Products']['Information']['AircraftMotion'][:].ndim > 1: # this was a 2D var in some 2020 data
            ac_mot_data = hdf['Products']['Information']['AircraftMotion'][:,0]
        else: # this is a 1D var for most datasets
            ac_mot_data = hdf['Products']['Information']['AircraftMotion'][:]
            
        aircraft_motion = xr.DataArray(
            data = ac_mot_data,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['AircraftMotion_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['AircraftMotion_units'][0].decode('UTF-8')
            ) 
        )

        mask_copol = xr.DataArray(
            data = hdf['Products']['Information']['MaskCoPol'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = hdf['Products']['Information']['MaskCoPol_description'][0].decode('UTF-8'),
            ) 
        )
        horiz_resolution = xr.DataArray(
            data = hdf['Products']['Information']['ResolutionHorizontal6dB'][:],
            dims = ["range"],
            coords = dict(
                range = radar_range),
            attrs = dict(
                description=hdf['Products']['Information']['ResolutionHorizontal6dB_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['ResolutionHorizontal6dB_units'][0].decode('UTF-8')
            ) 
        )
        vel_horizwind_offset = xr.DataArray(
            data = hdf['Products']['Information']['Velocity_horizwind_offset'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['Velocity_horizwind_offset_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['Velocity_horizwind_offset_units'][0].decode('UTF-8')
            ) 
        )

        # get meta data for attributes
        aircraft = hdf['Information']['Aircraft'][0].decode('UTF-8')
        dataContact = hdf['Information']['DataContact'][0].decode('UTF-8')
        experiment = hdf['Information']['ExperimentName'][0].decode('UTF-8')
        date = hdf['Information']['FlightDate'][0].decode('UTF-8')
        instrumentPI = hdf['Information']['InstrumentPI'][0].decode('UTF-8')
        L1A_ProcessDate = hdf['Information']['L1A_ProcessDate'][0].decode('UTF-8')
        L1B_ProcessDate = hdf['Information']['L1B_ProcessDate'][0].decode('UTF-8')
        L1B_Revision = hdf['Information']['L1B_Revision'][0].decode('UTF-8')
        if isinstance(hdf['Information']['L1B_Revision_Note'][0], np.float64): # for nan as notes
            L1B_Revision_Note = str(hdf['Information']['L1B_Revision_Note'][0])
        else: # for str as notes
            L1B_Revision_Note = hdf['Information']['L1B_Revision_Note'][0].decode('UTF-8')
        missionPI = hdf['Information']['MissionPI'][0].decode('UTF-8')
        radar_name = hdf['Information']['RadarName'][0].decode('UTF-8')
        antenna_beamwidth = hdf['Products']['Information']['AntennaBeamwidth'][0]
        antenna_size = hdf['Products']['Information']['AntennaSize'][0]
        avg_pulses = hdf['Products']['Information']['AveragedPulses'][0]
        freq = hdf['Products']['Information']['Frequency'][0]
        gatespacing = hdf['Products']['Information']['GateSpacing'][0]
        antenna_pointing = hdf['Products']['Information']['NominalAntennaPointing'][0].decode('UTF-8')
        pri = hdf['Products']['Information']['PRI'][0].decode('UTF-8')
        wavelength = hdf['Products']['Information']['Wavelength'][0]
        vert_resolution = hdf['Products']['Information']['ResolutionVertical6dB'][0]
        
        # close the file
        hdf.close()

        # put everything together into an XArray DataSet
        ds = xr.Dataset(
            data_vars={
                "dbz": dbz[:, time_inds],
                "vel": vel[:, time_inds],
                "width": width[:, time_inds],
                "vel_horiz_offset": vel_horizwind_offset[:, time_inds],
                "mask_copol": mask_copol[:, time_inds],
                "horizontal_resolution": horiz_resolution[:],
                "dxdr": dxdr[time_inds],
                "dydr": dydr[time_inds],
                "dzdr": dzdr[time_inds],
                "er2_altitude": altitude[time_inds],
                "er2_heading": heading[time_inds],
                "er2_pitch": pitch[time_inds],
                "er2_roll": roll[time_inds],
                "er2_drift": drift[time_inds],
                "er2_EastVel": eastVel[time_inds],
                "er2_NorthVel": northVel[time_inds],
                "er2_upVel": upvel[time_inds],
                "er2_track": track[time_inds],
                "er2_motion": aircraft_motion[time_inds]
            },
            coords={
                "range": radar_range[:],
                "height": height[:,time_inds],
                "time": time_dt64[time_inds],
                "lon": lon[time_inds],
                "lat": lat[time_inds],
                "distance": nomdist[time_inds]
            },
            attrs = {
                "Experiment": experiment,
                "Date": date,
                "Aircraft": aircraft,
                "Radar Name": radar_name,
                "Data Contact": dataContact,
                "Instrument PI": instrumentPI,
                "Mission PI": missionPI,
                "L1A Process Date": L1A_ProcessDate,
                "L1B Process Date": L1B_ProcessDate,
                "L1B Revision": L1B_Revision,
                "L1B Revision Note": L1B_Revision_Note,
                "Antenna Size (m)": antenna_size,
                "Antenna one-way 3dB beamwidth (degrees)": antenna_beamwidth,
                "Number of pulses averaged per profile": avg_pulses,
                "Radar Transmit Frequency (Hz)": freq,
                "Radar Transmit Wavelength (m)": wavelength,
                "Range Gate Spacing (m)": gatespacing,
                "Nominal Antenna Pointing": antenna_pointing,
                "PRI": pri,
                "vertical_resolution": vert_resolution
            }
        )

        return ds

    def correct_ice_atten(self, hiwrap, data_temperature=None):
        """
        Compute the 2-way path integrated attenuation due to ice scattering and
        correct the W-band reflectivity.
        
        Parameters
        ----------
        hiwrap       : impacts_tools.er2.Hiwrap() object
            The object obtained by running impacts_tools.er2.Hiwrap()
        data_temperature : xr.DataArray() or None
            The temperature field to optionally ignore correction >= 0C [deg C]
        
        Returns
        -------
        self.data['kw_ice'] : xr.DataArray()
            Two-way path integrated attenuation due to ice scattering [dB]
        """
        # fit Kulie et al. (2014) relationship to a Z_ku-dependent func
        dbz_ku_lin = np.array([0., 2000., 4000., 6000., 8000.]) # mm**6 / m**-3
        ks_w_coeff = np.array([0., 7.5, 15.5, 23.75, 31.5]) # db / km
        ks_w_func = np.poly1d(
            np.polyfit(dbz_ku_lin, ks_w_coeff, deg=1)
        ) # slope, intercept coeffs
        
        # calculate attenuation (k) per gate and clean up data
        data_dbz_ku = hiwrap.data.dbz_ku.values
        ks_w = ks_w_func(10. ** (data_dbz_ku / 10.)) # db / km
        ks_w = ks_w * 0.0265 # convert to db per gate
        if data_temperature is None: # clean up k values except for T > 0C
            ks_w[(np.isnan(data_dbz_ku)) | (data_dbz_ku > 40.)] = 0.
        else: # clean up k values, including when T > 0C
            ks_w[
                (np.isnan(data_dbz_ku)) | (data_dbz_ku > 40.) |
                (data_temperature >= 0.)] = 0.
        ks_w[(np.isnan(ks_w)) | (ks_w < 0.)] = 0. # change nan and negative vals to 0

        # calculate two-way path integrated attenuation from the per gate k
        ks_w = np.ma.array(2. * np.cumsum(ks_w, axis=(0)))
        
        # add attenuation to dataset (diagnostic)
        kw_ice = xr.full_like(self.data.dbz, 0.)
        kw_ice.data = ks_w
        kw_ice.attrs['description'] = (
            '2-way path integrated attenuation from ice scattering '
            'using k-Z_Ku relationship'
        )
        kw_ice.attrs['units'] = 'dB'
        self.data['kw_ice'] = kw_ice

        # correct W-band Z
        self.data['dbz'].values += self.data['kw_ice'].values
        
        return self.data

    def correct_attenuation(self, atten_file):
        """
        Correct reflectivity for attenuation at W-band due to atmospheric gases and LWC.

        Parameters
        ----------

        atten_file : str
            filepath to attenuation data

        """
        # open the attentuation file and trim to match radar beams
        atten_data = xr.open_dataset(atten_file).sel(time=self.data.time)

        # add the correction to the reflectivity values
        self.data['dbz'].values += atten_data['k_w'].values # gases
        self.data['dbz'].values += atten_data['k_w_l'].values # LWC
        
        #return self.data['dbz'].values



# ====================================== #
# HIWRAP
# ====================================== #

class Hiwrap(Radar):
    """
    A class to represent the HIWRAP nadir pointing radar flown on the ER2 during the IMPACTS field campaign.
    Inherits from Radar()

    Parameters
    ----------
    filepath(s): str
        File path to the HIWRAP data file (files for 2022 deployment)
    start_time: np.datetime64 or None
        The initial time of interest eg. if looking at a single flight leg
    end_time: np.datetime64 or None
        The final time of interest eg. if looking at a single flight leg
    atten_file: str or None
        Path to file containing gridded attenuation due to atmospheric gases
    max_roll : float or None
        The maximum roll angle of the aircraft in degrees, used to mask times with the aircraft is turning and the radars are off-nadir
    dbz_sigma: float or None
        The standard deviation to use in despeckling the radar reflectivity. Data above threshold is masked using a Gaussian filter
    vel_sigma: float or None
        The standard deviation to use in despeckling the radar Doppler velocity. Data above threshold is masked using a Gaussian filter
    width_sigma: float or None
        The standard deviation to use in despeckling the radar spectrum width. Data above threshold is masked using a Gaussian filter
    ldr_sigma: float or None
        The standard deviation to use in despeckling the radar linear depolarization ratio. Data above threshold is masked using a Gaussian filter
    dbz_min: float or None
        The minimum radar reflectivity value
    vel_min: float or None
        The minimum Doppler velocity value of interest. Data will be masked below threshold (m/s)
    width_min: float or None
        The minimum spectrum width value of interest. Data will be masked below threshold (m/s)

                
    """

    def __init__(self, filepath, start_time=None, end_time=None, atten_file=None, dataset='2020', max_roll=None, 
                dbz_sigma=None, vel_sigma=None, width_sigma=None, 
                dbz_min=None, vel_min=None, width_min=None):
    
        self.name = 'HIWRAP'

        # create a dataset with both ka- and ku-band data
        self.data = self.readfile(filepath, start_time, end_time, dataset)
        """
        xarray.Dataset of radar variables and attributes

        Dimensions:
            - range: xarray.DataArray(float) - The radar range gate  
            - time: np.array(np.datetime64[ns]) - The UTC time stamp

        Coordinates:
            - range (range): xarray.DataArray(float) - The radar range gate  
            - height (range, time): xarray.DataArray(float) - Altitude of each range gate (m)  
            - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
            - distance (time): xarray.DataArray(float) - Along-track distance from the flight origin (m)
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)

        Variables:
            - dbz_ka (range, time) : xarray.DataArray(float) - Ka-band equivalent reflectivity factor (db) with 1-sigma noise threshold applied
            - dbz_ku (range, time) : xarray.DataArray(float) - Ku-band equivalent reflectivity factor (db) with 1-sigma noise threshold applied
            - vel_ka (range, time) : xarray.DataArray(float) - Ka-band Doppler velocity corrected to account for intrusion of horizontal reanalysis winds and nonuniform beam filling (m/s)
            - vel_ku (range, time) : xarray.DataArray(float) - Ku-band Doppler velocity corrected to account for intrusion of horizontal reanalysis winds and nonuniform beam filling (m/s)
            - width_ka (range, time) : xarray.DataArray(float) - Ka-band Doppler velocity spectrum width estimate including aircraft motion. 1-sigma noise threshold applied (m/s)
            - width_ku (range, time) : xarray.DataArray(float) - Ku-band Doppler velocity spectrum width estimate including aircraft motion. 1-sigma noise threshold applied (m/s)
            - ldr_ka (range, time) : xarray.DataArray(float) - Ka-band linear depolarization ratio thresholded at 3-sigma (dB)
            - ldr_ku (range, time) : xarray.DataArray(float) - Ku-band linear depolarization ratio thresholded at 3-sigma (dB)
            - vel_horiz_offset_ka (range, time) : xarray.DataArray(float) - The horizontal wind offset used to correct Ka-band Doppler velocity [Vel_corr = Vel_uncorr - nubf_offset - horizwind_offset] (m/s)
            - vel_horiz_offset_ku (range, time) : xarray.DataArray(float) - The horizontal wind offset used to correct Ku-band Doppler velocity [Vel_corr = Vel_uncorr - nubf_offset - horizwind_offset] (m/s)
            - vel_nubf_offset_ku (range, time) : xarray.DataArray(float) - The nonuniform beam filling (NUBF) offset used to correct Ku-band Doppler velocity (m/s)
            - channel_mask_ka (range, time): xarray.DataArray(int) - Composite Ka-band image channel mask. 0: No signal, 1: Low-resolution pulse, 2: High-resolution pulse, 3: Chirp
            - channel_mask_ku (range, time): xarray.DataArray(int) - Composite Ku-band image channel mask. 0: No signal, 1: Low-resolution pulse, 2: High-resolution pulse, 3: Chirp
            - horizontal_resolution_ka (range) : xarray.DataArray(float) - Approximate horizontal resolution defined as width of spatial weighting after averaging as a function of radar range (m)
            - horizontal_resolution_ku (range) : xarray.DataArray(float) - Approximate horizontal resolution defined as width of spatial weighting after averaging as a function of radar range (m)
            - dxdr (time) : xarray.DataArray(float) - Data cross-track distance from aircraft per radar range. Positive is starboard direction (m/m)
            - dydr (time) : xarray.DataArray(float) - Data along-track distance from aircraft per radar range. Positive is forward direction (m/m)
            - dzdr (time) : xarray.DataArray(float) - Data vertical distance from aircraft per radar range. Positive is upward direction (m/m)
            - er2_altitude (time) : xarray.DataArray(float) - Aircraft height above sea level (m)
            - er2_heading (time) : xarray.DataArray(float) - Aircraft heading in degrees from north. 90 degrees is eastward pointing (degrees)
            - er2_pitch (time) : xarray.DataArray(float) - Aircraft pitch (degrees)
            - er2_roll (time) : xarray.DataArray(float) - Aircraft roll (degrees)
            - er2_drift (time) : xarray.DataArray(float) - Distance between track and heading (degrees)
            - er2_EastVel (time) : xarray.DataArray(float) - Eastward component of velocity (m/s)
            - er2_NorthVel (time) : xarray.DataArray(float) - Northward component of velocity (m/s)
            - er2_upVel (time) : xarray.DataArray(float) - Upward velocity (m/s)
            - er2_track (time) : xarray.DataArray(float) - Direction from motion in degrees from north. 90 degrees is eastward motion (degrees)
            - er2_motion (time) : xarray.DataArray(float) - Estimated aircraft motion notmal to the beam, subtracted from Doppler estimate. Smoothed to a 2 second average motion (m/s)

        Attribute Information:
            Experiment, Date, Aircraft, Radar Name, Data Contact, Instrument PI, Mission PI, Antenna Size, 
            Antenna one-way 3dB beamwidth (degrees), Number of pulses, Radar transmit frequency (Hz), Radar transmit wavelength (m),
            Range gate spacing (m), Nominal antenna pointing, PRI, vertical resolution
        """

        if dbz_sigma is not None:
            self.data['dbz_ka'] = self.despeckle(self.data['dbz_ka'], dbz_sigma)
            self.data['dbz_ku'] = self.despeckle(self.data['dbz_ku'], dbz_sigma)

        if vel_sigma is not None:
            self.data['vel_ka'] = self.despeckle(self.data['vel_ka'], vel_sigma)
            self.data['vel_ku'] = self.despeckle(self.data['vel_ku'], vel_sigma)
        
        if width_sigma is not None:
            self.data['width_ka'] = self.despeckle(self.data['width_ka'], width_sigma)
            self.data['width_ku'] = self.despeckle(self.data['width_ku'], width_sigma)


        # correct for 2-way path integrated attenuation
        if atten_file is not None: 
            self.correct_attenuation(atten_file)

        # mask values when aircraft is rolling
        if max_roll is not None:
            self.data = self.mask_roll(max_roll)


    def readfile(self, filepath, start_time=None, end_time=None, dataset='2020'):
        """
        Reads the HIWRAP data file and unpacks the fields into an xarray.Dataset

        Parameters
        ----------

        filepath : str
            Path to the data file (tuple of str for 2022 HIWRAP Ku- and Ka-bands)
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        dataset  : str
            Four-digit deployment year for handling radar variables/metadata
        
        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """

        # open the file
        if dataset=='2020': # Ku- and Ka-band products in one file
            hdf = h5py.File(filepath, 'r')
        else: # Ku- and Ka-band products in two files
            hdf = h5py.File(filepath[0], 'r') # Ku-band
            hdf2 = h5py.File(filepath[1], 'r') # Ka-band

        # Time information -- this is the first dimension in nav coords and products
        time_raw = hdf['Time']['Data']['TimeUTC'][:]
        time_dt = [datetime(1970, 1, 1) + timedelta(seconds=time_raw[i]) for i in range(len(time_raw))] # Python datetime object
        time_dt64 = np.array(time_dt, dtype='datetime64[ms]') # Numpy datetime64 object (e.g., for plotting)

        if start_time is not None:
            time_inds = np.where((time_dt64>=start_time) & (time_dt64<=end_time))[0]
        else:
            time_inds = np.where((time_dt64 != None))[0]
            
        # Time information for Ka-band data from 2022 deployment (separate file)
        if dataset=='2022':
            time2_raw = hdf2['Time']['Data']['TimeUTC'][:]
            time2_dt = [
                datetime(1970, 1, 1) + timedelta(seconds=time2_raw[i])
                for i in range(len(time2_raw))
            ] # Python datetime object
            time2_dt64 = np.array(
                time2_dt, dtype='datetime64[ms]'
            ) # Numpy datetime64 object (e.g., for plotting)
            ka_inds = np.isin(time2_dt64, time_dt64) # find HIWRAP Ka-band time indices common to Ku-band

        # Aircraft nav information
        if dataset=='2020': # metadata key misspelled in 2020 dataset
            description_str = 'NominalDistance_desciption'
        else:# metadata key spelling fixed in 2022 dataset
            description_str = 'NominalDistance_description'
        nomdist = xr.DataArray(
            data = hdf['Navigation']['Data']['NominalDistance'][:],
            dims = ["time"],
            coords = dict(time=time_dt64),
            attrs = dict(
                description=hdf['Navigation']['Information'][description_str][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['NominalDistance_units'][0].decode('UTF-8')
            )
        )
        lat = xr.DataArray(
            data = hdf['Navigation']['Data']['Latitude'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description = hdf['Navigation']['Information']['Latitude_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Latitude_units'][0].decode('UTF-8')
            )
        )
        lon = xr.DataArray(
            data = hdf['Navigation']['Data']['Longitude'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Longitude_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Longitude_units'][0].decode('UTF-8')
            )
        )
        altitude = xr.DataArray(
            data = hdf['Navigation']['Data']['Height'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Height_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Height_units'][0].decode('UTF-8')
            )
        )
        heading = xr.DataArray(
            data = hdf['Navigation']['Data']['Heading'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Heading_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Heading_units'][0].decode('UTF-8')
            )
        )
        roll = xr.DataArray(
            data = hdf['Navigation']['Data']['Roll'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Roll_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Roll_units'][0].decode('UTF-8')
            )
        )
        pitch = xr.DataArray(
            data = hdf['Navigation']['Data']['Pitch'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Pitch_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Pitch_units'][0].decode('UTF-8')
            )
        )
        drift = xr.DataArray(
            data = hdf['Navigation']['Data']['Drift'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Drift_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Drift_units'][0].decode('UTF-8')
            )
        )
        eastVel = xr.DataArray(
            data = hdf['Navigation']['Data']['EastVelocity'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['EastVelocity_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['EastVelocity_units'][0].decode('UTF-8')
            )
        )
        northVel = xr.DataArray(
            data = hdf['Navigation']['Data']['NorthVelocity'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['NorthVelocity_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['NorthVelocity_units'][0].decode('UTF-8')
            )
        )
        track = xr.DataArray(
            data = hdf['Navigation']['Data']['Track'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Track_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Track_units'][0].decode('UTF-8')
            )
        )
        upvel = xr.DataArray(
            data = hdf['Navigation']['Data']['UpVelocity'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['UpVelocity_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['UpVelocity_units'][0].decode('UTF-8')
            )
        )
        dxdr = xr.DataArray(
            data = hdf['Navigation']['Data']['dxdr'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['dxdr_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['dxdr_units'][0].decode('UTF-8')
            )
        )
        dydr = xr.DataArray(
            data = hdf['Navigation']['Data']['dydr'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['dydr_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['dydr_units'][0].decode('UTF-8')
            )
        )
        dzdr = xr.DataArray(
            data = hdf['Navigation']['Data']['dzdr'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['dzdr_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['dzdr_units'][0].decode('UTF-8')
            )
        )

        # Radar information
        radar_range = hdf['Products']['Information']['Range'][:] # this is the second dimension on product data

        [alt2d, radar_range2d] = np.meshgrid(altitude, radar_range)
        hght = alt2d - radar_range2d

        height = xr.DataArray(
            data = hght[:],
            dims = ['range', 'time'],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Height of each radar range gate',
                units='m'
            )
        )

        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = hdf['Products']['Ka']['Combined']['Data']['dBZe'][:].T
            description_var = hdf['Products']['Ka']['Combined']['Information']['dBZe_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ka']['Combined']['Information']['dBZe_units'][0].decode('UTF-8')
        else:
            data_var = hdf2['Products']['Combined']['Data']['dBZe'][ka_inds, :].T
            description_var = hdf2['Products']['Combined']['Information']['dBZe_description'][0].decode('UTF-8')
            units_var = hdf2['Products']['Combined']['Information']['dBZe_units'][0].decode('UTF-8')
        dbz_ka = xr.DataArray(
            data = data_var,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
                units = units_var
            )   
        )

        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = hdf['Products']['Ku']['Combined']['Data']['dBZe'][:].T
            description_var = hdf['Products']['Ku']['Combined']['Information']['dBZe_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ku']['Combined']['Information']['dBZe_units'][0].decode('UTF-8')
        else:
            data_var = hdf['Products']['Combined']['Data']['dBZe'][:].T
            description_var = hdf['Products']['Combined']['Information']['dBZe_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Combined']['Information']['dBZe_units'][0].decode('UTF-8')
        dbz_ku = xr.DataArray(
            data = data_var,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
                units = units_var
            )   
        )

        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = np.ma.masked_invalid(hdf['Products']['Ka']['Combined']['Data']['SpectrumWidth'][:].T)
            description_var = hdf['Products']['Ka']['Combined']['Information']['SpectrumWidth_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ka']['Combined']['Information']['SpectrumWidth_units'][0].decode('UTF-8')
        else:
            data_var = np.ma.masked_invalid(hdf2['Products']['Combined']['Data']['SpectrumWidth'][ka_inds, :].T)
            description_var = hdf2['Products']['Combined']['Information']['SpectrumWidth_description'][0].decode('UTF-8')
            units_var = hdf2['Products']['Combined']['Information']['SpectrumWidth_units'][0].decode('UTF-8')
        width_ka = xr.DataArray(
            data = data_var,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
                units = units_var
            ) 
        )
        
        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = np.ma.masked_invalid(hdf['Products']['Ku']['Combined']['Data']['SpectrumWidth'][:].T)
            description_var = hdf['Products']['Ku']['Combined']['Information']['SpectrumWidth_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ku']['Combined']['Information']['SpectrumWidth_units'][0].decode('UTF-8')
        else:
            data_var = np.ma.masked_invalid(hdf['Products']['Combined']['Data']['SpectrumWidth'][:].T)
            description_var = hdf['Products']['Combined']['Information']['SpectrumWidth_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Combined']['Information']['SpectrumWidth_units'][0].decode('UTF-8')
        width_ku = xr.DataArray(
            data = data_var,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
                units = units_var
            ) 
        )

        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = np.ma.masked_invalid(hdf['Products']['Ka']['Combined']['Data']['LDR'][:].T)
            description_var = hdf['Products']['Ka']['Combined']['Information']['LDR_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ka']['Combined']['Information']['LDR_units'][0].decode('UTF-8')
        else:
            data_var = np.ma.masked_invalid(hdf2['Products']['Combined']['Data']['LDR'][ka_inds, :].T)
            description_var = hdf2['Products']['Combined']['Information']['LDR_description'][0].decode('UTF-8')
            units_var = hdf2['Products']['Combined']['Information']['LDR_units'][0].decode('UTF-8')
        ldr_ka = xr.DataArray(
            data = data_var,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
                units = units_var
            ) 
        )
        
        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = np.ma.masked_invalid(hdf['Products']['Ku']['Combined']['Data']['LDR'][:].T)
            description_var = hdf['Products']['Ku']['Combined']['Information']['LDR_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ku']['Combined']['Information']['LDR_units'][0].decode('UTF-8')
        else:
            data_var = np.ma.masked_invalid(hdf['Products']['Combined']['Data']['LDR'][:].T)
            description_var = hdf['Products']['Combined']['Information']['LDR_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Combined']['Information']['LDR_units'][0].decode('UTF-8')
        ldr_ku = xr.DataArray(
            data = data_var,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
                units = units_var
            ) 
        )

        # get list of Ku- and Ka- variables
        if dataset=='2020':
            ka_prods = list(hdf['Products']['Ka']['Combined']['Data'].keys())
            ku_prods = list(hdf['Products']['Ku']['Combined']['Data'].keys())
        else:
            ka_prods = list(hdf2['Products']['Combined']['Data'].keys())
            ku_prods = list(hdf['Products']['Combined']['Data'].keys())
            
        if 'Velocity_corrected' in ka_prods:
            # for NUBF correction
            if dataset=='2020': # for files with combined Ku- and Ka-band data
                data_var = hdf['Products']['Ka']['Combined']['Data']['Velocity_corrected'][:].T
                description_var = hdf['Products']['Ka']['Combined']['Information']['Velocity_corrected_description'][0].decode('UTF-8')
                units_var = hdf['Products']['Ka']['Combined']['Information']['Velocity_corrected_units'][0].decode('UTF-8')
            else:
                data_var = hdf2['Products']['Combined']['Data']['Velocity_corrected'][ka_inds, :].T
                description_var = hdf2['Products']['Combined']['Information']['Velocity_corrected_description'][0].decode('UTF-8')
                units_var = hdf2['Products']['Combined']['Information']['Velocity_corrected_units'][0].decode('UTF-8')
            vel_ka = xr.DataArray(
                data = data_var,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description = description_var,
                    units = units_var
                ) 
            )
        else:
            if dataset=='2020': # for files with combined Ku- and Ka-band data
                data_var = hdf['Products']['Ka']['Combined']['Data']['Velocity'][:].T
                description_var = hdf['Products']['Ka']['Combined']['Information']['Velocity_description'][0].decode('UTF-8')
                units_var = hdf['Products']['Ka']['Combined']['Information']['Velocity_units'][0].decode('UTF-8')
            else:
                data_var = hdf2['Products']['Combined']['Data']['Velocity'][ka_inds, :].T
                description_var = hdf2['Products']['Combined']['Information']['Velocity_description'][0].decode('UTF-8')
                units_var = hdf2['Products']['Combined']['Information']['Velocity_units'][0].decode('UTF-8')
            vel_ka = xr.DataArray(
                data = data_var,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description = description_var,
                    units = units_var
                ) 
            )

        if 'Velocity_corrected' in ku_prods:
            # for NUBF correction
            if dataset=='2020': # for files with combined Ku- and Ka-band data
                data_var = hdf['Products']['Ku']['Combined']['Data']['Velocity_corrected'][:].T
                description_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_corrected_description'][0].decode('UTF-8')
                units_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_corrected_units'][0].decode('UTF-8')
            else:
                data_var = hdf['Products']['Combined']['Data']['Velocity_corrected'][:].T
                description_var = hdf['Products']['Combined']['Information']['Velocity_corrected_description'][0].decode('UTF-8')
                units_var = hdf['Products']['Combined']['Information']['Velocity_corrected_units'][0].decode('UTF-8')
            vel_ku = xr.DataArray(
                data = data_var,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description = description_var,
                    units = units_var
                ) 
            )
        else:
            if dataset=='2020': # for files with combined Ku- and Ka-band data
                data_var = hdf['Products']['Ku']['Combined']['Data']['Velocity'][:].T
                description_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_description'][0].decode('UTF-8')
                units_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_units'][0].decode('UTF-8')
            else:
                data_var = hdf['Products']['Combined']['Data']['Velocity'][:].T
                description_var = hdf['Products']['Combined']['Information']['Velocity_description'][0].decode('UTF-8')
                units_var = hdf['Products']['Combined']['Information']['Velocity_units'][0].decode('UTF-8')
            vel_ku = xr.DataArray(
                data = data_var,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description = description_var,
                    units = units_var
                ) 
            )
        
        if hdf['Products']['Information']['AircraftMotion'][:].ndim > 1: # this was a 2D var in some 2020 data
            ac_mot_data = hdf['Products']['Information']['AircraftMotion'][:,0]
        else: # this is a 1D var for most datasets
            ac_mot_data = hdf['Products']['Information']['AircraftMotion'][:]
        aircraft_motion = xr.DataArray(
            data = ac_mot_data,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['AircraftMotion_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['AircraftMotion_units'][0].decode('UTF-8')
            ) 
        )

        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = hdf['Products']['Ka']['Combined']['Information']['ChannelMask'][:].T
            description_var = hdf['Products']['Ka']['Combined']['Information']['ChannelMask_description'][0].decode('UTF-8')
        else:
            data_var = hdf2['Products']['Combined']['Information']['ChannelMask'][ka_inds, :].T
            description_var = hdf2['Products']['Combined']['Information']['ChannelMask_description'][0].decode('UTF-8')
        channel_mask_ka = xr.DataArray(
            data = data_var,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
            ) 
        )
        
        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = hdf['Products']['Ku']['Combined']['Information']['ChannelMask'][:].T
            description_var = hdf['Products']['Ku']['Combined']['Information']['ChannelMask_description'][0].decode('UTF-8')
        else:
            data_var = hdf['Products']['Combined']['Information']['ChannelMask'][:].T
            description_var = hdf['Products']['Combined']['Information']['ChannelMask_description'][0].decode('UTF-8')
        channel_mask_ku = xr.DataArray(
            data = data_var,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
            ) 
        )

        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = hdf['Products']['Ka']['Information']['ResolutionHorizontal6dB'][:]
            description_var = hdf['Products']['Ka']['Information']['ResolutionHorizontal6dB_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ka']['Information']['ResolutionHorizontal6dB_units'][0].decode('UTF-8')
        else:
            data_var = hdf2['Products']['Information']['ResolutionHorizontal6dB'][:]
            description_var = hdf2['Products']['Information']['ResolutionHorizontal6dB_description'][0].decode('UTF-8')
            units_var = hdf2['Products']['Information']['ResolutionHorizontal6dB_units'][0].decode('UTF-8')
        horiz_resolution_ka = xr.DataArray(
            data = data_var,
            dims = ["range"],
            coords = dict(
                range = radar_range),
            attrs = dict(
                description = description_var,
                units = units_var
            ) 
        )

        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = hdf['Products']['Ku']['Information']['ResolutionHorizontal6dB'][:]
            description_var = hdf['Products']['Ku']['Information']['ResolutionHorizontal6dB_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ku']['Information']['ResolutionHorizontal6dB_units'][0].decode('UTF-8')
        else:
            data_var = hdf['Products']['Information']['ResolutionHorizontal6dB'][:]
            description_var = hdf['Products']['Information']['ResolutionHorizontal6dB_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Information']['ResolutionHorizontal6dB_units'][0].decode('UTF-8')
        horiz_resolution_ku = xr.DataArray(
            data = data_var,
            dims = ["range"],
            coords = dict(
                range = radar_range),
            attrs = dict(
                description = description_var,
                units = units_var
            ) 
        )
        
        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = hdf['Products']['Ka']['Combined']['Information']['Velocity_horizwind_offset'][:].T
            description_var = hdf['Products']['Ka']['Combined']['Information']['Velocity_horizwind_offset_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ka']['Combined']['Information']['Velocity_horizwind_offset_units'][0].decode('UTF-8')
        else:
            data_var = hdf2['Products']['Combined']['Information']['Velocity_horizwind_offset'][ka_inds, :].T
            description_var = hdf2['Products']['Combined']['Information']['Velocity_horizwind_offset_description'][0].decode('UTF-8')
            units_var = hdf2['Products']['Combined']['Information']['Velocity_horizwind_offset_units'][0].decode('UTF-8')
        vel_horizwind_offset_ka = xr.DataArray(
            data = data_var,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
                units = units_var
            ) 
        )
        
        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_horizwind_offset'][:].T
            description_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_horizwind_offset_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_horizwind_offset_units'][0].decode('UTF-8')
        else:
            data_var = hdf['Products']['Combined']['Information']['Velocity_horizwind_offset'][:].T
            description_var = hdf['Products']['Combined']['Information']['Velocity_horizwind_offset_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Combined']['Information']['Velocity_horizwind_offset_units'][0].decode('UTF-8')
        vel_horizwind_offset_ku = xr.DataArray(
            data = data_var,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
                units = units_var
            ) 
        )
        
        if dataset=='2020': # for files with combined Ku- and Ka-band data
            data_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_nubf_offset'][:].T
            description_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_nubf_offset_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Ku']['Combined']['Information']['Velocity_nubf_offset_units'][0].decode('UTF-8')
        else:
            data_var = hdf['Products']['Combined']['Information']['Velocity_nubf_offset'][:].T
            description_var = hdf['Products']['Combined']['Information']['Velocity_nubf_offset_description'][0].decode('UTF-8')
            units_var = hdf['Products']['Combined']['Information']['Velocity_nubf_offset_units'][0].decode('UTF-8')
        vel_nubf_offset_ku = xr.DataArray(
            data = data_var,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = description_var,
                units = units_var
            ) 
        )

        # get meta data for attributes
        aircraft = hdf['Information']['Aircraft'][0].decode('UTF-8')
        dataContact = hdf['Information']['DataContact'][0].decode('UTF-8')
        experiment = hdf['Information']['ExperimentName'][0].decode('UTF-8')
        date = hdf['Information']['FlightDate'][0].decode('UTF-8')
        instrumentPI = hdf['Information']['InstrumentPI'][0].decode('UTF-8')
        L1A_ProcessDate = hdf['Information']['L1A_ProcessDate'][0].decode('UTF-8')
        L1B_ProcessDate = hdf['Information']['L1B_ProcessDate'][0].decode('UTF-8')
        L1B_Revision = hdf['Information']['L1B_Revision'][0].decode('UTF-8')
        if isinstance(hdf['Information']['L1B_Revision_Note'][0], np.float64): # for nan as notes
            L1B_Revision_Note = str(hdf['Information']['L1B_Revision_Note'][0])
        else: # for str as notes
            L1B_Revision_Note = hdf['Information']['L1B_Revision_Note'][0].decode('UTF-8')
        missionPI = hdf['Information']['MissionPI'][0].decode('UTF-8')
        radar_name = hdf['Information']['RadarName'][0].decode('UTF-8')
        antenna_size = hdf['Products']['Information']['AntennaSize'][0]
        gatespacing = hdf['Products']['Information']['GateSpacing'][0]
        antenna_pointing = hdf['Products']['Information']['NominalAntennaPointing'][0].decode('UTF-8')
        pri = hdf['Products']['Information']['PRI'][0].decode('UTF-8')
        if dataset=='2020': # for Ku- and Ka- band metadata in same file
            antenna_beamwidth_ka = hdf['Products']['Ka']['Information']['AntennaBeamwidth'][0]
            antenna_beamwidth_ku = hdf['Products']['Ku']['Information']['AntennaBeamwidth'][0]
            avg_pulses_ka = hdf['Products']['Ka']['Information']['AveragedPulses'][0]
            avg_pulses_ku = hdf['Products']['Ku']['Information']['AveragedPulses'][0]
            freq_ka = hdf['Products']['Ka']['Information']['Frequency'][0]
            freq_ku = hdf['Products']['Ku']['Information']['Frequency'][0]
            wavelength_ka = hdf['Products']['Ka']['Information']['Wavelength'][0]
            wavelength_ku = hdf['Products']['Ku']['Information']['Wavelength'][0]
        else:
            antenna_beamwidth_ka = hdf2['Products']['Information']['AntennaBeamwidth'][0]
            antenna_beamwidth_ku = hdf['Products']['Information']['AntennaBeamwidth'][0]
            avg_pulses_ka = hdf2['Products']['Information']['AveragedPulses'][0]
            avg_pulses_ku = hdf['Products']['Information']['AveragedPulses'][0]
            freq_ka = hdf2['Products']['Information']['Frequency'][0]
            freq_ku = hdf['Products']['Information']['Frequency'][0]
            wavelength_ka = hdf2['Products']['Information']['Wavelength'][0]
            wavelength_ku = hdf['Products']['Information']['Wavelength'][0]
        
        # close the file
        hdf.close()
        if dataset!='2020': # for deploymnets where Ku- and Ka-band split in 2 files
            hdf2.close() 

        # put everything together into an XArray DataSet
        ds = xr.Dataset(
            data_vars={
                "dbz_ka": dbz_ka[:,time_inds],
                "dbz_ku": dbz_ku[:,time_inds],
                "vel_ka": vel_ka[:,time_inds],
                "vel_ku": vel_ku[:,time_inds],
                "width_ka": width_ka[:,time_inds],
                "width_ku": width_ku[:,time_inds],
                "ldr_ka": ldr_ka[:,time_inds],
                "ldr_ku": ldr_ku[:,time_inds],
                "vel_horiz_offset_ka": vel_horizwind_offset_ka[:,time_inds],
                "vel_horiz_offset_ku": vel_horizwind_offset_ku[:,time_inds],
                "vel_nubf_offset_ku": vel_nubf_offset_ku[:,time_inds],
                "channel_mask_ka": channel_mask_ka[:,time_inds],
                "channel_mask_ku": channel_mask_ku[:,time_inds],
                "horizontal_resolution_ka": horiz_resolution_ka[:],
                "horizontal_resolution_ku": horiz_resolution_ku[:],
                "dxdr": dxdr[time_inds],
                "dydr": dydr[time_inds],
                "dzdr": dzdr[time_inds],
                "er2_altitude": altitude[time_inds],
                "er2_heading": heading[time_inds],
                "er2_pitch": pitch[time_inds],
                "er2_roll": roll[time_inds],
                "er2_drift": drift[time_inds],
                "er2_EastVel": eastVel[time_inds],
                "er2_NorthVel": northVel[time_inds],
                "er2_upVel": upvel[time_inds],
                "er2_track": track[time_inds],
                "er2_motion": aircraft_motion[time_inds]
            },
            coords={
                "range": radar_range[:],
                "height": height[:,time_inds],
                "time": time_dt64[time_inds],
                "lon": lon[time_inds],
                "lat": lat[time_inds],
                "distance": nomdist[time_inds]
            },
            attrs = {
                "Experiment": experiment,
                "Date": date,
                "Aircraft": aircraft,
                "Radar Name": radar_name,
                "Data Contact": dataContact,
                "Instrument PI": instrumentPI,
                "Mission PI": missionPI,
                "L1A Process Date": L1A_ProcessDate,
                "L1B Process Date": L1B_ProcessDate,
                "L1B Revision": L1B_Revision,
                "L1B Revision Note": L1B_Revision_Note,
                "Antenna Size (m)": antenna_size,
                "Antenna one-way 3dB Ka-Band beamwidth (degrees)": antenna_beamwidth_ka,
                "Antenna one-way 3dB Ku-Band beamwidth (degrees)": antenna_beamwidth_ku,
                "Number of pulses averaged per profile, Ka-band": avg_pulses_ka,
                "Number of pulses averaged per profile, Ku-band": avg_pulses_ku,
                "Radar Transmit Ka-Band Frequency (Hz)": freq_ka,
                "Radar Transmit Ku-Band Frequency (Hz)":freq_ku,
                "Radar Transmit Ka-Band Wavelength (m)": wavelength_ka,
                "Radar Transmit Ku-Band Wavelength (m)": wavelength_ku,
                "Range Gate Spacing (m)": gatespacing,
                "Nominal Antenna Pointing": antenna_pointing,
                "PRI": pri,
                "Bands": "Ka, Ku"
            }
        )

        return ds



    def correct_attenuation(self, atten_file):
        """
        Correct reflectivity for attenuation at Ku- and Ka-band due to atmospheric gases.

        Parameters
        ----------

        atten_file : str
            filepath to attenuation data

        """
        # open the attentuation file and trim to match radar beams
        atten_data = xr.open_dataset(atten_file).sel(time=self.data.time)

        # add the correction to the reflectivity values
        self.data['dbz_ka'].values += atten_data['k_ka'].values
        self.data['dbz_ku'].values += atten_data['k_ku'].values
        
        return self.data





# ====================================== #
# EXRAD
# ====================================== #


class Exrad(Radar):
    """
    A class to represent the EXRAD nadir pointing radar flown on the ER2 during the IMPACTS field campaign.
    Inherits from Radar()
    
    Parameters
    ----------
    filepath: str
        File path to the CRS data file
    start_time: np.datetime64 or None
        The initial time of interest eg. if looking at a single flight leg
    end_time: np.datetime64 or None
        The final time of interest eg. if looking at a single flight leg
    atten_file: str or None
        Path to file containing gridded attenuation due to atmospheric gases
    max_roll: float or None
        The maximum roll angle of the aircraft in degrees, used to mask times with the aircraft is turning and the radars are off-nadir
    dbz_sigma: float or None
        The standard deviation to use in despeckling the radar reflectivity. Data above threshold is masked using a Gaussian filter
    vel_sigma: float or None
        The standard deviation to use in despeckling the radar Doppler velocity. Data above threshold is masked using a Gaussian filter
    width_sigma: float or None
        The standard deviation to use in despeckling the radar spectrum width. Data above threshold is masked using a Gaussian filter
    dbz_min: float or None
        The minimum radar reflectivity value
    vel_min: float or None
        The minimum Doppler velocity value of interest. Data will be masked below threshold (m/s)
    width_min: float or None
        The minimum spectrum width value of interest. Data will be masked below threshold (m/s)
 
    """

    def __init__(self, filepath, start_time=None, end_time=None, atten_file=None, dataset='2020', max_roll=None, 
        dbz_sigma=None, vel_sigma=None, width_sigma=None, dbz_min=None, vel_min=None, width_min=None):
        
        self.name = 'EXRAD'

        # read the raw data
        self.data = self.readfile(filepath, start_time, end_time, dataset)
        """
        xarray.Dataset of radar variables and attributes

        Dimensions:
            - range: xarray.DataArray(float) - The radar range gate  
            - time: np.array(np.datetime64[ns]) - The UTC time stamp

        Coordinates:
            - range (range): xarray.DataArray(float) - The radar range gate  
            - height (range, time): xarray.DataArray(float) - Altitude of each range gate (m)  
            - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
            - distance (time): xarray.DataArray(float) - Along-track distance from the flight origin (m)
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)

        Variables:
            - dbz (range, time) : xarray.DataArray(float) - Equivalent reflectivity factor (db) with 1-sigma noise threshold applied
            - vel (range, time) : xarray.DataArray(float) - Doppler velocity corrected to account for intrusion of horizontal reanalysis winds and nonuniform beam filling (m/s)
            - width (range, time) : xarray.DataArray(float) - Doppler velocity spectrum width estimate including aircraft motion. 1-sigma noise threshold applied (m/s)
            - vel_horiz_offset (range, time) : xarray.DataArray(float) - The horizontal wind offset used to correct Ka-band Doppler velocity [Vel_corr = Vel_uncorr - nubf_offset - horizwind_offset] (m/s)
            - vel_nubf_offset (range, time) : xarray.DataArray(float) - The nonuniform beam filling (NUBF) offset used to correct Ku-band Doppler velocity (m/s)
            - channel_mask (range, time): xarray.DataArray(int) - Composite image channel mask. 0: No signal, 1: Low-resolution pulse, 2: High-resolution pulse, 3: Chirp
            - horizontal_resolution_ku (range) : xarray.DataArray(float) - Approximate horizontal resolution defined as width of spatial weighting after averaging as a function of radar range (m)
            - dxdr (time) : xarray.DataArray(float) - Data cross-track distance from aircraft per radar range. Positive is starboard direction (m/m)
            - dydr (time) : xarray.DataArray(float) - Data along-track distance from aircraft per radar range. Positive is forward direction (m/m)
            - dzdr (time) : xarray.DataArray(float) - Data vertical distance from aircraft per radar range. Positive is upward direction (m/m)
            - er2_altitude (time) : xarray.DataArray(float) - Aircraft height above sea level (m)
            - er2_heading (time) : xarray.DataArray(float) - Aircraft heading in degrees from north. 90 degrees is eastward pointing (degrees)
            - er2_pitch (time) : xarray.DataArray(float) - Aircraft pitch (degrees)
            - er2_roll (time) : xarray.DataArray(float) - Aircraft roll (degrees)
            - er2_drift (time) : xarray.DataArray(float) - Distance between track and heading (degrees)
            - er2_EastVel (time) : xarray.DataArray(float) - Eastward component of velocity (m/s)
            - er2_NorthVel (time) : xarray.DataArray(float) - Northward component of velocity (m/s)
            - er2_upVel (time) : xarray.DataArray(float) - Upward velocity (m/s)
            - er2_track (time) : xarray.DataArray(float) - Direction from motion in degrees from north. 90 degrees is eastward motion (degrees)
            - er2_motion (time) : xarray.DataArray(float) - Estimated aircraft motion notmal to the beam, subtracted from Doppler estimate. Smoothed to a 2 second average motion (m/s)

        Attribute Information:
            Experiment, Date, Aircraft, Radar Name, Data Contact, Instrument PI, Mission PI, Antenna Size, 
            Antenna one-way 3dB beamwidth (degrees), Number of pulses, Radar transmit frequency (Hz), Radar transmit wavelength (m),
            Range gate spacing (m), Nominal antenna pointing, PRI, vertical resolution
        """

        # correct for 2-way path integrated attenuation
        if atten_file is not None: 
            self.correct_attenuation(atten_file)

        # mask values when aircraft is rolling
        if max_roll is not None:
            self.data = self.mask_roll(max_roll)

        if dbz_sigma is not None:
            self.data['dbz'] = self.despeckle(self.data['dbz'], dbz_sigma)

        if vel_sigma is not None:
            self.data['vel'] = self.despeckle(self.data['vel'], vel_sigma)
        
        if width_sigma is not None:
            self.data['width'] = self.despeckle(self.data['width'], width_sigma)
        
        if resample_ref is not None:
            self.data = self.resample(self.data, resample_ref)
        
        
   

    def readfile(self, filepath, start_time=None, end_time=None, dataset='2020'):
        """
        Reads the EXRAD data file and unpacks the fields into an xarray.Dataset

        Parameters
        ----------

        filepath : str
            Path to the data file
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        dataset  : str
            Four-digit deployment year for handling radar variables/metadata
        
        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """

        # open the file
        hdf = h5py.File(filepath, 'r')

        # Time information -- this is the first dimension in nav coords and products
        time_raw = hdf['Time']['Data']['TimeUTC'][:]
        time_dt = [datetime(1970, 1, 1) + timedelta(seconds=time_raw[i]) for i in range(len(time_raw))] # Python datetime object
        time_dt64 = np.array(time_dt, dtype='datetime64[ms]') # Numpy datetime64 object (e.g., for plotting)

        if start_time is not None:
            time_inds = np.where((time_dt64>=start_time) & (time_dt64<=end_time))[0]
        else:
            time_inds = np.where((time_dt64 != None))[0]

        # Aircraft nav information
        if dataset=='2020': # metadata key misspelled in 2020 dataset
            description_str = 'NominalDistance_desciption'
        else:# metadata key spelling fixed in 2022 dataset
            description_str = 'NominalDistance_description'
        nomdist = xr.DataArray(
            data = hdf['Navigation']['Data']['NominalDistance'][:],
            dims = ["time"],
            coords = dict(time=time_dt64),
            attrs = dict(
                description=hdf['Navigation']['Information'][description_str][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['NominalDistance_units'][0].decode('UTF-8')
            )
        )
        lat = xr.DataArray(
            data = hdf['Navigation']['Data']['Latitude'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description = hdf['Navigation']['Information']['Latitude_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Latitude_units'][0].decode('UTF-8')
            )
        )
        lon = xr.DataArray(
            data = hdf['Navigation']['Data']['Longitude'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Longitude_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Longitude_units'][0].decode('UTF-8')
            )
        )
        altitude = xr.DataArray(
            data = hdf['Navigation']['Data']['Height'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Height_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Height_units'][0].decode('UTF-8')
            )
        )
        heading = xr.DataArray(
            data = hdf['Navigation']['Data']['Heading'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Heading_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Heading_units'][0].decode('UTF-8')
            )
        )
        roll = xr.DataArray(
            data = hdf['Navigation']['Data']['Roll'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Roll_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Roll_units'][0].decode('UTF-8')
            )
        )
        pitch = xr.DataArray(
            data = hdf['Navigation']['Data']['Pitch'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Pitch_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Pitch_units'][0].decode('UTF-8')
            )
        )
        drift = xr.DataArray(
            data = hdf['Navigation']['Data']['Drift'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Drift_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Drift_units'][0].decode('UTF-8')
            )
        )
        eastVel = xr.DataArray(
            data = hdf['Navigation']['Data']['EastVelocity'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['EastVelocity_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['EastVelocity_units'][0].decode('UTF-8')
            )
        )
        northVel = xr.DataArray(
            data = hdf['Navigation']['Data']['NorthVelocity'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['NorthVelocity_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['NorthVelocity_units'][0].decode('UTF-8')
            )
        )
        track = xr.DataArray(
            data = hdf['Navigation']['Data']['Track'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['Track_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['Track_units'][0].decode('UTF-8')
            )
        )
        upvel = xr.DataArray(
            data = hdf['Navigation']['Data']['UpVelocity'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['UpVelocity_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['UpVelocity_units'][0].decode('UTF-8')
            )
        )
        dxdr = xr.DataArray(
            data = hdf['Navigation']['Data']['dxdr'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['dxdr_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['dxdr_units'][0].decode('UTF-8')
            )
        )
        dydr = xr.DataArray(
            data = hdf['Navigation']['Data']['dydr'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['dydr_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['dydr_units'][0].decode('UTF-8')
            )
        )
        dzdr = xr.DataArray(
            data = hdf['Navigation']['Data']['dzdr'][:],
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist),
            attrs = dict(
                description=hdf['Navigation']['Information']['dzdr_description'][0].decode('UTF-8'),
                units = hdf['Navigation']['Information']['dzdr_units'][0].decode('UTF-8')
            )
        )

        # Radar information
        radar_range = hdf['Products']['Information']['Range'][:] # this is the second dimension on product data


        [alt2d, radar_range2d] = np.meshgrid(altitude, radar_range)
        hght = alt2d - radar_range2d

        height = xr.DataArray(
            data = hght[:],
            dims = ['range', 'time'],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Height of each radar range gate',
                units='m'
            )
        )


        dbz = xr.DataArray(
            data = hdf['Products']['Data']['dBZe'][:].T,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['dBZe_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['dBZe_units'][0].decode('UTF-8')
            )   
        )

        width = xr.DataArray(
            data = np.ma.masked_invalid(hdf['Products']['Data']['SpectrumWidth'][:].T),
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['SpectrumWidth_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['SpectrumWidth_units'][0].decode('UTF-8')
            ) 
        )

        if 'Velocity_corrected' in list(hdf['Products']['Data'].keys()):
            # for NUBF correction
            vel = xr.DataArray(
                data = hdf['Products']['Data']['Velocity_corrected'][:].T,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description=hdf['Products']['Information']['Velocity_corrected_description'][0].decode('UTF-8'),
                    units = hdf['Products']['Information']['Velocity_corrected_units'][0].decode('UTF-8')
                ) 
            )
        else:
            vel = xr.DataArray(
                data = hdf['Products']['Data']['Velocity'][:].T,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description=hdf['Products']['Information']['Velocity_description'][0].decode('UTF-8'),
                    units = hdf['Products']['Information']['Velocity_units'][0].decode('UTF-8')
                ) 
            )
        
        if hdf['Products']['Information']['AircraftMotion'][:].ndim > 1: # this was a 2D var in some 2020 data
            ac_mot_data = hdf['Products']['Information']['AircraftMotion'][:,0]
        else: # this is a 1D var for most datasets
            ac_mot_data = hdf['Products']['Information']['AircraftMotion'][:]
        aircraft_motion = xr.DataArray(
            data = ac_mot_data,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['AircraftMotion_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['AircraftMotion_units'][0].decode('UTF-8')
            ) 
        )

        mask_copol = xr.DataArray(
            data = hdf['Products']['Information']['MaskCoPol'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = hdf['Products']['Information']['MaskCoPol_description'][0].decode('UTF-8'),
            ) 
        )
        horiz_resolution = xr.DataArray(
            data = hdf['Products']['Information']['ResolutionHorizontal6dB'][:],
            dims = ["range"],
            coords = dict(
                range = radar_range),
            attrs = dict(
                description=hdf['Products']['Information']['ResolutionHorizontal6dB_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['ResolutionHorizontal6dB_units'][0].decode('UTF-8')
            ) 
        )
        vel_horizwind_offset = xr.DataArray(
            data = hdf['Products']['Information']['Velocity_horizwind_offset'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['Velocity_horizwind_offset_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['Velocity_horizwind_offset_units'][0].decode('UTF-8')
            ) 
        )
        vel_nubf_offset = xr.DataArray(
            data = hdf['Products']['Information']['Velocity_nubf_offset'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Information']['Velocity_nubf_offset_description'][0].decode('UTF-8'),
                units = hdf['Products']['Information']['Velocity_nubf_offset_units'][0].decode('UTF-8')
            ) 
        )

        # get meta data for attributes
        aircraft = hdf['Information']['Aircraft'][0].decode('UTF-8')
        dataContact = hdf['Information']['DataContact'][0].decode('UTF-8')
        experiment = hdf['Information']['ExperimentName'][0].decode('UTF-8')
        date = hdf['Information']['FlightDate'][0].decode('UTF-8')
        instrumentPI = hdf['Information']['InstrumentPI'][0].decode('UTF-8')
        L1A_ProcessDate = hdf['Information']['L1A_ProcessDate'][0].decode('UTF-8')
        L1B_ProcessDate = hdf['Information']['L1B_ProcessDate'][0].decode('UTF-8')
        L1B_Revision = hdf['Information']['L1B_Revision'][0].decode('UTF-8')
        if isinstance(hdf['Information']['L1B_Revision_Note'][0], np.float64): # for nan as notes
            L1B_Revision_Note = str(hdf['Information']['L1B_Revision_Note'][0])
        else: # for str as notes
            L1B_Revision_Note = hdf['Information']['L1B_Revision_Note'][0].decode('UTF-8')
        missionPI = hdf['Information']['MissionPI'][0].decode('UTF-8')
        radar_name = hdf['Information']['RadarName'][0].decode('UTF-8')
        antenna_beamwidth = hdf['Products']['Information']['AntennaBeamwidth'][0]
        antenna_size = hdf['Products']['Information']['AntennaSize'][0]
        avg_pulses = hdf['Products']['Information']['AveragedPulses'][0]
        freq = hdf['Products']['Information']['Frequency'][0]
        gatespacing = hdf['Products']['Information']['GateSpacing'][0]
        antenna_pointing = hdf['Products']['Information']['NominalAntennaPointing'][0].decode('UTF-8')
        pri = hdf['Products']['Information']['PRI'][0].decode('UTF-8')
        wavelength = hdf['Products']['Information']['Wavelength'][0]
        vert_resolution = hdf['Products']['Information']['ResolutionVertical6dB'][0]
        
        # close the file
        hdf.close()

        # put everything together into an XArray DataSet
        ds = xr.Dataset(
            data_vars={
                "dbz": dbz[:,time_inds],
                "vel": vel[:,time_inds],
                "width": width[:,time_inds],
                "vel_horiz_offset": vel_horizwind_offset[:,time_inds],
                "vel_nubf_offset": vel_nubf_offset[:,time_inds],
                "mask_copol": mask_copol[:,time_inds],
                "horizontal_resolution": horiz_resolution,
                "dxdr": dxdr[time_inds],
                "dydr": dydr[time_inds],
                "dzdr": dzdr[time_inds],
                "er2_altitude": altitude[time_inds],
                "er2_heading": heading[time_inds],
                "er2_pitch": pitch[time_inds],
                "er2_roll": roll[time_inds],
                "er2_drift": drift[time_inds],
                "er2_EastVel": eastVel[time_inds],
                "er2_NorthVel": northVel[time_inds],
                "er2_upVel": upvel[time_inds],
                "er2_track": track[time_inds],
                "er2_motion": aircraft_motion[time_inds]
            },
            coords={
                "range": radar_range,
                "height": height[:,time_inds],
                "time": time_dt64[time_inds],
                "lon": lon[time_inds],
                "lat": lat[time_inds],
                "distance": nomdist[time_inds]
            },
            attrs = {
                "Experiment": experiment,
                "Date": date,
                "Aircraft": aircraft,
                "Radar Name": radar_name,
                "Data Contact": dataContact,
                "Instrument PI": instrumentPI,
                "Mission PI": missionPI,
                "L1A Process Date": L1A_ProcessDate,
                "L1B Process Date": L1B_ProcessDate,
                "L1B Revision": L1B_Revision,
                "L1B Revision Note": L1B_Revision_Note,
                "Antenna Size (m)": antenna_size,
                "Antenna one-way 3dB beamwidth (degrees)": antenna_beamwidth,
                "Number of pulses averaged per profile": avg_pulses,
                "Radar Transmit Frequency (Hz)": freq,
                "Radar Transmit Wavelength (m)": wavelength,
                "Range Gate Spacing (m)": gatespacing,
                "Nominal Antenna Pointing": antenna_pointing,
                "PRI": pri,
                "vertical_resolution": vert_resolution
            }
        )
        
        return ds


    def correct_attenuation(self, atten_file):
        """
        Correct reflectivity for attenuation at X-band due to atmospheric gases.

        Parameters
        ----------

        atten_file : str
            filepath to attenuation data

        """
        # open the attentuation file and trim to match radar beams
        atten_data = xr.open_dataset(atten_file).sel(time=self.data.time)

        # add the correction to the reflectivity values
        self.data['dbz'].values += atten_data['k_x'].values
        
        return self.data

# ====================================== #
# CPL
# ====================================== #

class Cpl(Lidar):
    """
    A class to represent the CPL lidar flown on the ER2 during the IMPACTS field campaign.
    Inherits from Lidar()

    Parameters
    ----------
    filepath(s): str
        File path to the CPL data file
    start_time: np.datetime64 or None
        The initial time of interest eg. if looking at a single flight leg
    end_time: np.datetime64 or None
        The final time of interest eg. if looking at a single flight leg
    max_roll : float or None
        The maximum roll angle of the aircraft in degrees, used to mask times with the aircraft is turning and the radars are off-nadir
    atb_sigma: float or None
        The standard deviation to use in despeckling the total attenuated backscatter (L1B data, all wavelengths).
        Data above threshold is masked using a Gaussian filter.
    """

    def __init__(self, filepath, start_time=None, end_time=None, max_roll=None,
                 atb_sigma=None, l1b_trim_ref=None, l2_cloudtop_ref=None):
        # read the raw data
        if 'ATB' in filepath:
            self.name = 'CPL ATB'
            self.data = self.readfile_atb(filepath, start_time, end_time)
            """
            xarray.Dataset of lidar variables and attributes

            Dimensions:
                - gate: xarray.DataArray(float) - The lidar gate number  
                - time: np.array(np.datetime64[ns]) - The UTC time stamp

            Coordinates:
                - gate (gate): xarray.DataArray(float) - The lidar gate number  
                - height (range, time): xarray.DataArray(float) - Altitude of each range gate (m)  
                - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
                - distance (time): xarray.DataArray(float) - Along-track distance from the flight origin (m)
                - lat (time): xarray.DataArray(float) - Latitude (degrees)
                - lon (time): xarray.DataArray(float) - Longitude (degrees)

            Variables:
                - atb_1064 (gate, time) : xarray.DataArray(float) - Attenuated total backscatter (km**-1 sr**-1) profile at 1064 nm for each record
                - atb_532 (gate, time) : xarray.DataArray(float) - Attenuated total backscatter (km**-1 sr**-1) profile at 532 nm for each record
                - atb_355 (gate, time) : xarray.DataArray(float) - Attenuated total backscatter (km**-1 sr**-1) profile at 355 nm for each record

            Attribute Information:
                Experiment, Date, Aircraft, Lidar Name, Data Contact, Instrument PI, Mission PI,
                L1B Calibration Source, vertical_resolution
            """
            
            if atb_sigma is not None:
                for var in ['atb_1064', 'atb_532', 'atb_355']:
                    self.data[var] = self.despeckle(self.data[var], atb_sigma)
        elif ('L2' in filepath) and ('Pro' in filepath):
            self.name = 'CPL L2 Profiles'
            self.data = self.readfile_l2pro(filepath, start_time, end_time)
            """
                xarray.Dataset of lidar variables and attributes

                Dimensions:
                    - gate: xarray.DataArray(float) - The lidar gate number  
                    - time: np.array(np.datetime64[ns]) - The UTC time stamp

                Coordinates:
                    - gate (gate): xarray.DataArray(float) - The lidar gate number  
                    - height (range, time): xarray.DataArray(float) - Altitude of each range gate (m)  
                    - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
                    - lat (time): xarray.DataArray(float) - Latitude (degrees)
                    - lon (time): xarray.DataArray(float) - Longitude (degrees)

                Variables:
                    - dpol_1064 (gate, time) : xarray.DataArray(float) - Total depolarization ratio profile at 1064 nm for each record
                    - ext_[1064, 532, 355] (gate, time) : xarray.DataArray(float) - Extinction coefficient (km**-1) profile at [1064, 532, 355] nm for each record
                    - cod_[1064, 532, 355] (gate, time) : xarray.DataArray(float) - Integrated cloud optical depth at [1064, 532, 355] nm

                Attribute Information:
                    Experiment, Date, Aircraft, Lidar Name, Data Contact, Instrument PI, Mission PI,
                    L2 version number, vertical_resolution
            """
        elif ('L2' in filepath) and ('Lay' in filepath):
            self.name = 'CPL L2 Layers'
            self.data = self.readfile_l2lay(filepath, start_time, end_time)
            """
                xarray.Dataset of lidar variables and attributes

                Dimensions:
                    - layer: xarray.DataArray(float) - The derived layer number  
                    - time: np.array(np.datetime64[ns]) - The UTC time stamp

                Coordinates:
                    - layer (layer): xarray.DataArray(float) - The derived layer number  
                    - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
                    - lat (time): xarray.DataArray(float) - Latitude (degrees)
                    - lon (time): xarray.DataArray(float) - Longitude (degrees)

                Variables:
                    - layer_top_altitude (layer, time) : xarray.DataArray(float) - Altitude (n) for the top portion of each identified layer
                    - layer_top_temperature (layer, time) : xarray.DataArray(float) - Temperature (C) for the top portion of each identified layer
                    - cloud_top_altitude (time) : xarray.DataArray(float) - Cloud top altitude (m) derived from uppermost layer top altitude
                    - cloud_top_temperature (time) : xarray.DataArray(float) - Cloud top temperature (C) derived from uppermost layer top temperature
                    - number_layers (time) : xarray.DataArray(float) - Number of layers in 5-s profile

                Attribute Information:
                    Experiment, Date, Aircraft, Lidar Name, Data Contact, Instrument PI, Mission PI,
                    L2 version number
            """

        # trim dataset based on specified start/end
        if (start_time is not None) or (end_time is not None):
            self.data = self.trim_time_bounds(start_time, end_time)
        elif ('CPL L2' in self.name) and (l1b_trim_ref is not None):
            self.data = self.trim_l2_to_l1b(l1b_trim_ref)
            
        # compute cloudtop properties for L1B and L2 profile data (optional)
        if (l2_cloudtop_ref is not None) and (
                    (self.name == 'CPL ATB') or (self.name == 'CPL L2 Profiles')):
                self.data = self.get_cloudtop_properties(l2_cloudtop_ref)
            
        # mask values when aircraft is rolling
        if (max_roll is not None) and ('er2_roll' in self.data.data_vars):
            self.data = self.mask_roll(max_roll)


    def readfile_atb(self, filepath, start_time=None, end_time=None):
        """
        Reads the CPL L1B data file and unpacks the fields into an xarray.Dataset

        Parameters
        ----------

        filepath : str
            Path to the data file
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        
        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """  
        hdf = h5py.File(filepath, 'r')
        
        # construct the time array
        try:
            y_str = hdf['Date'][()].decode('UTF-8').split()[-1]
        except:
            dby_str = [chr(hdf['Date'][i]) for i in range(len(hdf['Date'][()]))]
            y_str = ''.join(dby_str).split()[-1]
        date = pd.to_datetime(f'{y_str}', format='%Y').to_pydatetime()
        dt = np.array(
            [date + timedelta(
                days=int(hdf['Dec_JDay'][i]) - 1, hours=int(hdf['Hour'][i]),
                minutes=int(hdf['Minute'][i]), seconds=int(hdf['Second'][i])
            ) for i in range(hdf['Hour'].shape[0])], dtype='datetime64[ms]'
        )
        
        # aircraft nav information
        lat = xr.DataArray(
            data = hdf['Latitude'][:],
            dims = ['time'],
            coords = dict(time=dt),
            attrs = dict(
                description='Latitude',
                units = 'degrees'
            )
        )
        lon = xr.DataArray(
            data = hdf['Longitude'][:],
            dims = ['time'],
            coords = dict(time=dt),
            attrs = dict(
                description='Latitude',
                units = 'degrees'
            )
        )
        if len(hdf['Plane_Alt'][()]) == len(dt):
            alt_data = 1000. * hdf['Plane_Alt'][()]
        else: # take the midpoint
            alt_data = 1000.* (hdf['Plane_Alt'][:-1] + hdf['Plane_Alt'][1:]) / 2.
        altitude = xr.DataArray(
            data = alt_data,
            dims = ['time'],
            coords = dict(time=dt),
            attrs = dict(
                description='Height of the aircraft above mean sea level',
                units = 'm'
            )
        )
        if len(hdf['Plane_Heading'][()]) == len(dt):
            heading_data = hdf['Plane_Heading'][()]
        else: # take the midpoint
            heading_data = (hdf['Plane_Heading'][:-1] + hdf['Plane_Heading'][1:]) / 2.
        heading = xr.DataArray(
            data = heading_data,
            dims = ['time'],
            coords = dict(time=dt),
            attrs = dict(
                description='Plane heading (clockwise from North)',
                units = 'degrees'
            )
        )
        if len(hdf['Plane_Roll'][()]) == len(dt):
            roll_data = hdf['Plane_Roll'][()]
        else: # take the midpoint
            roll_data = (hdf['Plane_Roll'][:-1] + hdf['Plane_Roll'][1:]) / 2.
        roll = xr.DataArray(
            data = roll_data,
            dims = ['time'],
            coords = dict(time=dt),
            attrs = dict(
                description='Plane roll (right turn is positive)',
                units = 'degrees'
            )
        )
        
        # lidar information
        hght1d = 1000. * hdf['Bin_Alt'][:] # this is the second dimension on product data
        hght2d = np.tile(np.atleast_2d(hght1d).T, (1, len(dt)))

        height = xr.DataArray(
            data = hght2d,
            dims = ['gate', 'time'],
            coords = dict(
                gate = np.arange(len(hght1d)),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Height of each lidar range gate',
                units='m'
            )
        )

        atb1064 = xr.DataArray(
            data = np.ma.masked_where(hdf['ATB_1064'][:].T <= 0., hdf['ATB_1064'][:].T),
            dims = ['gate', 'time'],
            coords = dict(
                gate = np.arange(len(hght1d)),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Attenuated total backscatter profile at 1064 nm for each record',
                units='km**-1 sr**-1'
            )
        )

        atb532 = xr.DataArray(
            data = np.ma.masked_where(hdf['ATB_532'][:].T <= 0., hdf['ATB_532'][:].T),
            dims = ['gate', 'time'],
            coords = dict(
                gate = np.arange(len(hght1d)),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Attenuated total backscatter profile at 532 nm for each record',
                units='km**-1 sr**-1'
            )
        )

        atb355 = xr.DataArray(
            data = np.ma.masked_where(hdf['ATB_355'][:].T <= 0., hdf['ATB_355'][:].T),
            dims = ['gate', 'time'],
            coords = dict(
                gate = np.arange(len(hght1d)),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Attenuated total backscatter profile at 355 nm for each record',
                units='km**-1 sr**-1'
            )
        )
        
        # metadata for attributes
        try:
            experiment = hdf['Project'][()].decode('UTF-8')
        except:
            experiment = ''.join(
                [chr(hdf['Project'][i]) for i in range(len(hdf['Project'][()]))]
            )
        try:
            L1B_Version = hdf['L1B_Version'][()].decode('UTF-8')
        except:
            L1B_Version = ''.join(
                [chr(hdf['L1B_Version'][i]) for i in range(len(hdf['L1B_Version'][()]))]
            )
        vert_resolution = hdf['Bin_Width'][()]
        hdf.close()
        
        # construct the dataset
        ds = xr.Dataset(
            data_vars={
                'atb_1064': atb1064,
                'atb_532': atb532,
                'atb_355': atb355,
                'er2_altitude': altitude,
                'er2_heading': heading,
                'er2_roll': roll
            },
            coords={
                'gate': np.arange(len(hght1d)),
                'height': height,
                'time': dt,
                'lon': lon,
                'lat': lat,
            },
            attrs = {
                'Experiment': experiment,
                'Date': datetime.strftime(date, '%Y-%m-%d'),
                'Aircraft': 'ER-2',
                'Lidar Name': 'CPL',
                'Data Contact': 'Patrick Selmer',
                'Instrument PI': 'Matthew McGill',
                'Mission PI': 'Lynn McMurdie',
                'L1B Calibration Source': L1B_Version,
                'vertical_resolution': f'{vert_resolution:.2f} m'
            }
        )
        
        return ds.drop_duplicates('time')
    
    def readfile_l2pro(self, filepath, start_time=None, end_time=None):
        """
        Reads the CPL L2 profile data file and unpacks the fields into an xarray.Dataset

        Parameters
        ----------

        filepath : str
            Path to the data file
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        
        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """  
        hdf = h5py.File(filepath, 'r')

        # construct the time array
        y_str = hdf['metadata_parameters']['File_Year'][()].decode('UTF-8').split()[0]
        jday = hdf['profile']['Profile_Decimal_Julian_Day'][()]
        date = pd.to_datetime(f'{y_str}', format='%Y').to_pydatetime()
        dt = np.array(
            [date + timedelta(
                seconds=int((86400. * (jday[i, 1] - 1)).round())
            ) for i in range(jday.shape[0])], dtype='datetime64[ms]'
        )
        dt, tind = np.unique(dt, return_index=True) # ignore duplicate times
        dt_start = xr.DataArray(
            data=np.array(
                [date + timedelta(
                    seconds=int((86400. * (jday[i, 0] - 1)).round())
                ) for i in tind], dtype='datetime64[ms]'
            ),
            dims = ['time'],
            attrs = dict(
                description='UTC start time of profile'
            )
        )
        dt_end = xr.DataArray(
            data=np.array(
                [date + timedelta(
                    seconds=int((86400. * (jday[i, 2] - 1)).round())
                ) for i in tind], dtype='datetime64[ms]'
            ),
            dims = ['time'],
            attrs = dict(
                description='UTC end time of profile'
            )
        )

        # aircraft nav information
        lat = xr.DataArray(
            data = hdf['geolocation']['CPL_Latitude'][tind, 1],
            dims = ['time'],
            coords = dict(time=dt),
            attrs = dict(
                description='Latitude',
                units = 'degrees'
            )
        )
        lon = xr.DataArray(
            data = hdf['geolocation']['CPL_Longitude'][tind ,1],
            dims = ['time'],
            coords = dict(time=dt),
            attrs = dict(
                description='Longitude',
                units = 'degrees'
            )
        )

        # lidar information
        hght1d = 1000. * hdf['metadata_parameters']['Bin_Altitude_Array'][:] # second dim on product data
        hght2d = np.tile(np.atleast_2d(hght1d).T, (1, len(dt)))
        height = xr.DataArray(
            data = hght2d,
            dims = ['gate', 'time'],
            coords = dict(
                gate = np.arange(len(hght1d)),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Height of each lidar range gate',
                units='m'
            )
        )
        ext1064 = xr.DataArray(
            data = np.ma.masked_where(
                hdf['profile']['Extinction_Coefficient_1064'][:, tind] <= 0.,
                hdf['profile']['Extinction_Coefficient_1064'][:, tind]
            ),
            dims = ['gate', 'time'],
            coords = dict(
                gate = np.arange(len(hght1d)),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Extinction coefficient profile at 1064 nm for each record',
                units='km**-1'
            )
        )
        ext532 = xr.DataArray(
            data = np.ma.masked_where(
                hdf['profile']['Extinction_Coefficient_532'][:, tind] <= 0.,
                hdf['profile']['Extinction_Coefficient_532'][:, tind]
            ),
            dims = ['gate', 'time'],
            coords = dict(
                gate = np.arange(len(hght1d)),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Extinction coefficient profile at 532 nm for each record',
                units='km**-1'
            )
        )
        ext355 = xr.DataArray(
            data = np.ma.masked_where(
                hdf['profile']['Extinction_Coefficient_355'][:, tind] <= 0.,
                hdf['profile']['Extinction_Coefficient_355'][:, tind]
            ),
            dims = ['gate', 'time'],
            coords = dict(
                gate = np.arange(len(hght1d)),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Extinction coefficient profile at 355 nm for each record',
                units='km**-1'
            )
        )
        dpol_data = np.ma.masked_where(
            np.isnan(ext1064.values),
            hdf['profile']['Total_Depolarization_Ratio_1064'][:, tind]
        )
        dpol_data = np.ma.masked_where(dpol_data < 0., dpol_data)
        dpol1064 = xr.DataArray(
            data = dpol_data,
            dims = ['gate', 'time'],
            coords = dict(
                gate = np.arange(len(hght1d)),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Total depolarization ratio profile at 1064 nm for each record',
                units='km**-1 sr**-1'
            )
        )
        cod1064 = xr.DataArray(
            data = np.ma.masked_where(
                hdf['profile']['Cloud_Optical_Depth_1064'][tind] < -1.,
                hdf['profile']['Cloud_Optical_Depth_1064'][tind]
            ),
            dims = ['time'],
            coords = dict(
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Integrated cloud optical depth at 1064 nm for each record',
                units='#'
            )
        )
        cod532 = xr.DataArray(
            data = np.ma.masked_where(
                hdf['profile']['Cloud_Optical_Depth_532'][tind] < -1.,
                hdf['profile']['Cloud_Optical_Depth_532'][tind]
            ),
            dims = ['time'],
            coords = dict(
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Cloud optical depth at 532 nm for each record',
                units='#'
            )
        )
        cod355 = xr.DataArray(
            data = np.ma.masked_where(
                hdf['profile']['Cloud_Optical_Depth_355'][tind] < -1.,
                hdf['profile']['Cloud_Optical_Depth_355'][tind]
            ),
            dims = ['time'],
            coords = dict(
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Cloud optical depth at 355 nm for each record',
                units='#'
            )
        )

        # metadata for attributes
        try:
            L2_Version = hdf['metadata_parameters']['Product_Version_Number'][()].decode('UTF-8')
        except:
            L2_Version = ''.join(
                [chr(hdf['metadata_parameters']['Product_Version_Number'][i]) for i in range(
                    len(hdf['metadata_parameters']['Product_Version_Number'][()]))]
            )
        vert_resolution = 1000. * hdf['metadata_parameters']['Bin_Size'][()]
        hdf.close()

        # construct the dataset
        ds = xr.Dataset(
            data_vars={
                'dpol_1064': dpol1064,
                'ext_1064': ext1064,
                'ext_532': ext532,
                'ext_355': ext355,
                'cod_1064': cod1064,
                'cod_532': cod532,
                'cod_355': cod355
            },
            coords={
                'gate': np.arange(len(hght1d)),
                'height': height,
                'time': dt,
                'time_start': dt_start,
                'time_end': dt_end,
                'lon': lon,
                'lat': lat,
            },
            attrs = {
                'Experiment': 'IMPACTS',
                'Date': datetime.strftime(date, '%Y-%m-%d'),
                'Aircraft': 'ER-2',
                'Lidar Name': 'CPL',
                'Data Contact': 'Patrick Selmer',
                'Instrument PI': 'Matthew McGill',
                'Mission PI': 'Lynn McMurdie',
                'L2 Product Version Number': L2_Version,
                'vertical_resolution': f'{vert_resolution:.2f} m'
            }
        )

        return ds
    
    def readfile_l2lay(self, filepath, start_time=None, end_time=None):
        """
        Reads the CPL L2 layer data file and unpacks the fields into an xarray.Dataset

        Parameters
        ----------

        filepath : str
            Path to the data file
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        
        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """  
        hdf = h5py.File(filepath, 'r')

        # construct the time array
        y_str = hdf['metadata_parameters']['File_Year'][()].decode('UTF-8').split()[0]
        jday = hdf['layer_descriptor']['Profile_Decimal_Julian_Day'][()]
        date = pd.to_datetime(f'{y_str}', format='%Y').to_pydatetime()
        dt = np.array(
            [date + timedelta(
                seconds=int((86400. * (jday[i, 1] - 1)).round())
            ) for i in range(jday.shape[0])], dtype='datetime64[ms]'
        )
        dt, tind = np.unique(dt, return_index=True) # ignore duplicate times
        
        # aircraft nav information
        lat = xr.DataArray(
            data = hdf['geolocation']['CPL_Latitude'][tind, 1],
            dims = ['time'],
            coords = dict(time=dt),
            attrs = dict(
                description='Latitude',
                units = 'degrees'
            )
        )
        lon = xr.DataArray(
            data = hdf['geolocation']['CPL_Longitude'][tind ,1],
            dims = ['time'],
            coords = dict(time=dt),
            attrs = dict(
                description='Longitude',
                units = 'degrees'
            )
        )
        
        # lidar information
        lta = xr.DataArray(
            data = np.ma.masked_where(
                hdf['layer_descriptor']['Layer_Top_Altitude'][tind, :].T <= -999.,
                1000. * hdf['layer_descriptor']['Layer_Top_Altitude'][tind, :].T
            ),
            dims = ['layer', 'time'],
            coords = dict(
                layer = np.arange(10),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Altitude for the top portion of each identified layer',
                units='m'
            )
        )
        ltt = xr.DataArray(
            data = np.ma.masked_where(
                hdf['layer_descriptor']['Layer_Top_Temperature'][tind, :].T <= -999.,
                hdf['layer_descriptor']['Layer_Top_Temperature'][tind, :].T
            ),
            dims = ['layer', 'time'],
            coords = dict(
                layer = np.arange(10),
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Temperature for the top portion of each identified layer',
                units='degrees_Celsius'
            )
        )
        cta = xr.DataArray(
            data = lta[0, tind].values,
            dims = ['time'],
            coords = dict(
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Cloud top altitude derived from uppermost layer top altitude',
                units='m'
            )
        )
        ctt = xr.DataArray(
            data = ltt[0, tind].values,
            dims = ['time'],
            coords = dict(
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Cloud top temperature derived from uppermost layer top temperature',
                units='degrees_Celsius'
            )
        )
        nlay = xr.DataArray(
            data = hdf['layer_descriptor']['Number_Layers'][tind],
            dims = ['time'],
            coords = dict(
                time = dt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description='Number of layers in 5-s profile',
                units='#'
            )
        )
        
        # metadata for attributes
        try:
            L2_Version = hdf['metadata_parameters']['Product_Version_Number'][()].decode('UTF-8')
        except:
            L2_Version = ''.join(
                [chr(hdf['metadata_parameters']['Product_Version_Number'][i]) for i in range(
                    len(hdf['metadata_parameters']['Product_Version_Number'][()]))]
            )
        hdf.close()
        
        # construct the dataset
        ds = xr.Dataset(
            data_vars={
                'layer_top_altitude': lta,
                'layer_top_temperature': ltt,
                'cloud_top_altitude': cta,
                'cloud_top_temperature': ctt,
                'number_layers': nlay
            },
            coords={
                'layer': np.arange(10),
                'time': dt,
                'lon': lon,
                'lat': lat,
            },
            attrs = {
                'Experiment': 'IMPACTS',
                'Date': datetime.strftime(date, '%Y-%m-%d'),
                'Aircraft': 'ER-2',
                'Lidar Name': 'CPL',
                'Data Contact': 'Patrick Selmer',
                'Instrument PI': 'Matthew McGill',
                'Mission PI': 'Lynn McMurdie',
                'L2 Product Version Number': L2_Version
            }
        )
        
        return ds.where(~np.isnan(ds.cloud_top_altitude), drop=True) # drop times with no layers
    
    def trim_l2_to_l1b(self, l1b_object):
        """
        Put the dataset into the same time bounds as the CPL L1B 1-Hz ATB profile data.
        
        Parameters
        ----------
        l1b_object: impacts_tools.er2.CPL()
            CPL L1B 1-Hz ATB object to optionally constrain times and navigation data

        Returns
        -------
        data : xarray.Dataset
            The reindexed dataset with er2_altitude, er2_heading, er2_roll, distance vars
        """
        
        # first remove L1B periods with duplicate times
        l1b_data = l1b_object.data.drop_duplicates('time').drop_vars(
            ['lat', 'lon']
        )
        
        # trim the L2 data
        l2_sub = self.data.sel(
            time=slice(l1b_data.time[0], l1b_data.time[-1])
        )
        
        # extract L1B data valid at the L2 profile midpoints
        l1b_sub = l1b_data[['er2_altitude', 'er2_heading', 'er2_roll']].reindex(
            time=l2_sub.time, method='nearest', tolerance='5S'
        )
        
        return xr.merge(
            [l2_sub, l1b_sub], compat='override', combine_attrs='override'
        )
        

class VAD(object):
    """
    A class to represent the Velocity-Azimuth Display (VAD) wind product derived from the EXRAD conical scans during the IMPACTS field campaign.

    :param data: dataset containing VAD data and attributes
    :type data: xarray.Dataset
    """
    def __init__(self, filepath, start_time=None, end_time=None):
        self.name = 'VAD'
        self.data = self.readfile(filepath, start_time, end_time)

    def readfile(self, filepath, start_time, end_time):
        """
        Reads the VAD data file and unpacks the fields into an xarray.Dataset

        Parameters
        ----------

        filepath : str
            Path to the data file
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        
        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """
        vad = xr.open_dataset(filepath)

        # Time information - this is a Julian date
        time_raw = vad.time.values

        time_dt = [julian.from_jd(time_raw[i], fmt='jd') for i in range(len(time_raw))] # Python datetime object
        time_dt64 = np.array(time_dt, dtype='datetime64[ms]') # Numpy datetime64 object (e.g., for plotting)

        radar_range = vad['range'].values

        histbins=vad['histdim'].values

        yt = xr.DataArray(
            data = vad['yt'].values,
            dims=["time"],
            coords = dict(
                time = time_dt64
            ),
            attrs = dict(
                description = vad['yt'].long_name,
                units = vad['yt'].units
            )
        )
        lat = xr.DataArray(
            data = vad['lat'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt),
            attrs = dict(
                description = vad['lat'].long_name,
                units = vad['lat'].units
            )
        )
        lon = xr.DataArray(
            data = vad['lon'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt),
            attrs = dict(
                description = vad['lon'].long_name,
                units = vad['lon'].units
            )
        )

        height = xr.DataArray(
            data = vad['hght'].values,
            dims=["range", "time"],
            coords = dict(
                range = radar_range,
                time=time_dt64
            ),
            attrs = dict(
                description = vad['hght'].long_name,
                units = vad['hght'].units
            )
        )

        npoints_valid = xr.DataArray(
            data = vad['npoints_valid'].values,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                lat = lat,
                lon = lon,
                distance = yt),
            attrs = dict(
                description=vad['npoints_valid'].long_name
            ) 
        )

        npoints_total = xr.DataArray(
            data = vad['npoints_total'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt, 
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['npoints_total'].long_name
            ) 
        )

        elapsed_time = xr.DataArray(
            data = vad['elapsed_time'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['elapsed_time'].long_name,
                units=vad['elapsed_time'].units
            ) 
        )
        zt = xr.DataArray(
            data = vad['zt'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['zt'].long_name,
                units=vad['zt'].units
            ) 
        )


        uvel = xr.DataArray(
            data = vad['uvel'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['uvel'].long_name,
                units=vad['uvel'].units
            ) 
        )

        vvel = xr.DataArray(
            data = vad['vvel'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['vvel'].long_name,
                units=vad['vvel'].units
            ) 
        )

        avel = xr.DataArray(
            data = vad['avel'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['avel'].long_name,
                units=vad['avel'].units
            ) 
        )

        xvel = xr.DataArray(
            data = vad['xvel'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['xvel'].long_name,
                units=vad['xvel'].units
            ) 
        )

        dshr = xr.DataArray(
            data = vad['dshr'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['dshr'].long_name,
                units=vad['dshr'].units
            ) 
        )

        dstr = xr.DataArray(
            data = vad['dstr'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['dstr'].long_name,
                units=vad['dstr'].units
            ) 
        )

        refl = xr.DataArray(
            data = vad['refl'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['refl'].long_name,
                units=vad['refl'].units
            ) 
        )

        refl_max = xr.DataArray(
            data = vad['refl_max'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['refl_max'].long_name,
                units=vad['refl_max'].units
            ) 
        )
        refl_std = xr.DataArray(
            data = vad['refl_std'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['refl_std'].long_name,
                units=vad['refl_std'].units
            ) 
        )

        footprint_time = xr.DataArray(
            data = vad['footprint_time'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['footprint_time'].long_name,
                units=vad['footprint_time'].units
            ) 
        )

        cor = xr.DataArray(
            data = vad['cor'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['cor'].long_name
            ) 
        )
        c0 = xr.DataArray(
            data = vad['c0'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['c0'].long_name
            ) 
        )
        c1 = xr.DataArray(
            data = vad['c1'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                distance = yt,
                height = height,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['c1'].long_name
            ) 
        )
        c2 = xr.DataArray(
            data = vad['c2'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['c2'].long_name
            ) 
        )
        d1 = xr.DataArray(
            data = vad['d1'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['d1'].long_name
            ) 
        )
        d2 = xr.DataArray(
            data = vad['d2'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['d2'].long_name
            ) 
        )
        '''azhist = xr.DataArray(
            data = vad['azihist'].values,
            dims = ["bins", "range","time"],
            coords = dict(
                bins=histbins,
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['azihist'].long_name
            ) 
        )'''
        qc1 = xr.DataArray(
            data = vad['qc1'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                distance = yt,
                height = height,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['qc1'].long_name
            ) 
        )
        qc2 = xr.DataArray(
            data = vad['qc2'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['qc2'].long_name
            ) 
        )
        qc3 = xr.DataArray(
            data = vad['qc3'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['qc3'].long_name
            ) 
        )
        qc4 = xr.DataArray(
            data = vad['qc4'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['qc4'].long_name
            ) 
        )
        qc5 = xr.DataArray(
            data = vad['qc5'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                height = height,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['qc5'].long_name
            ) 
        )

        delta_time = xr.DataArray(
            data = vad['delta_time'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['delta_time'].long_name,
                units=vad['delta_time'].units
            ) 
        )

        delta_time_std = xr.DataArray(
            data = vad['delta_time_std'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['delta_time_std'].long_name,
                units=vad['delta_time_std'].units
            ) 
        )

        delta_azimuth = xr.DataArray(
            data = vad['delta_azimuth'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['delta_azimuth'].long_name,
                units=vad['delta_azimuth'].units
            ) 
        )

        delta_azimuth_std = xr.DataArray(
            data = vad['delta_azimuth_std'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['delta_azimuth_std'].long_name,
                units=vad['delta_azimuth_std'].units
            ) 
        )


        er2_altitude = xr.DataArray(
            data = vad['ac_alt'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_alt'].long_name,
                units=vad['ac_alt'].units
            ) 
        )

        er2_altitude_std = xr.DataArray(
            data = vad['ac_alt_std'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_alt_std'].long_name,
                units=vad['ac_alt_std'].units
            ) 
        )

        er2_track = xr.DataArray(
            data = vad['ac_track'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_track'].long_name,
                units=vad['ac_track'].units
            ) 
        )

        er2_track_std = xr.DataArray(
            data = vad['ac_track_std'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_track_std'].long_name,
                units=vad['ac_track_std'].units
            ) 
        )

        er2_roll = xr.DataArray(
            data = vad['ac_roll'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_roll'].long_name,
                units=vad['ac_roll'].units
            ) 
        )

        er2_roll_std = xr.DataArray(
            data = vad['ac_roll_std'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_roll_std'].long_name,
                units=vad['ac_roll_std'].units
            ) 
        )

        er2_pitch = xr.DataArray(
            data = vad['ac_pitch'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_pitch'].long_name,
                units=vad['ac_pitch'].units
            ) 
        )

        er2_pitch_std = xr.DataArray(
            data = vad['ac_pitch_std'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_pitch_std'].long_name,
                units=vad['ac_pitch_std'].units
            ) 
        )

        er2_heading = xr.DataArray(
            data = vad['ac_heading'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_heading'].long_name,
                units=vad['ac_heading'].units
            ) 
        )

        er2_heading_std = xr.DataArray(
            data = vad['ac_heading_std'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_heading_std'].long_name,
                units=vad['ac_heading_std'].units
            ) 
        )

        er2_groundspeed = xr.DataArray(
            data = vad['ac_gspd'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_gspd'].long_name,
                units=vad['ac_gspd'].units
            ) 
        )
        er2_groundspeed_std = xr.DataArray(
            data = vad['ac_gspd_std'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['ac_gspd_std'].long_name,
                units=vad['ac_gspd_std'].units
            ) 
        )

        vertical_resolution = xr.DataArray(
            data = vad['vertical_resolution'].values,
            dims = ["range"],
            coords = dict(
                range = radar_range),
            attrs = dict(
                description=vad['vertical_resolution'].long_name,
                units=vad['vertical_resolution'].units
            ) 
        )

        antenna_rotdir = xr.DataArray(
            data = vad['antenna_rotdir'].values,
            dims = ["time"],
            coords = dict(
                time = time_dt64,
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['antenna_rotdir'].long_name
            ) 
        )

        ds = xr.Dataset(
            data_vars= {
                "npoints_valid": npoints_valid,
                "npoints_total": npoints_total,
                "elapsed_time": elapsed_time,
                "yt": yt,
                "zt": zt,
                "uvel": uvel,
                "vvel": vvel,
                "avel": avel,
                "xvel": xvel,
                "dshr": dshr,
                "dstr": dstr,
                "refl": refl,
                "refl_max": refl_max,
                "refl_std": refl_std,
                "footprint_time": footprint_time,
                "delta_time": delta_time,
                "delta_time_std": delta_time_std,
                "delta_azimuth": delta_azimuth,
                "delta_azimuth_std": delta_azimuth_std,
                "er2_altitude": er2_altitude,
                "er2_altitude_std":er2_altitude_std,
                "er2_heading": er2_heading,
                "er2_heading_std": er2_heading_std,
                "er2_pitch": er2_pitch,
                "er2_pitch_std": er2_pitch_std,
                "er2_roll": er2_roll,
                "er2_roll_std": er2_roll_std,
                "er2_groundspeed": er2_groundspeed,
                "er2_groundspeed_std": er2_groundspeed_std,
                "er2_track": er2_track,
                "er2_track_std": er2_track_std,
                "vertical_resolution": vertical_resolution,
                "antenna_rotdir": antenna_rotdir,
                "cor": cor,
                "c0": c0,
                "c1": c1,
                "c2": c2,
                "d1": d1,
                "d2": d2,
                #"azhist": azhist,
                "qc1": qc1,
                "qc2": qc2,
                "qc3": qc3,
                "qc4": qc4,
                "qc5": qc5
            },

            coords={
                "bins": 12,
                "range": radar_range,
                "height": height,
                "time": time_dt64,
                "lon": lon,
                "lat": lat,
                "distance": yt
            },

            attrs = vad.attrs
        )
        
        return ds

    def regrid_to_radar(self, radar):
        """
        Regrids the VAD data to a nadir radar object, EXRAD, HIWRAP, or CRS

        Parameters
        ----------

        radar : er2.Radar
            the radar object to regrid to
        
        Returns
        -------
        regridded_data_2 : xarray.Dataset
            the VAD data set interpolated to the Radar grid
        """

        # check that the radar to regrid to is initialized
        try: radar.data          
        except: NameError: radar.data = None

        if radar.data is None:
            print("Radar object to regrid to does not exist. Try initializing the radar first")
        
        else:
            #regridded_data_1 = self.data.interp_like(radar.data['time'])
            #regridded_data_2 = regridded_data_1.interp_like(radar.data['range'])
        
            regridded_data_2 = self.data.interp(
                    {"range": radar.data['range'], 
                    "time": radar.data['time']
                    }
                )

        return regridded_data_2
