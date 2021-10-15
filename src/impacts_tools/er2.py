"""
Classes for IMPACTS ER2 Instruments
"""

import h5py
import h5netcdf
import julian
import numpy as np
import xarray as xr
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
        if vel_bins == None:
            vel_bins = np.linspace(-5., 5., num=101)

        if alt_bins == None:
            alt_bins = np.linspace(100., 10000., num=45)

        if band is not None:
            vel_flat = np.ma.masked_invalid(self.data['vel_' + band.lower()].values.flatten())
        else:
            vel_flat = np.ma.masked_invalid(self.data['vel'].values.flatten())
        
        hght_flat = self.data['height'].values.flatten()
        fall_speeds_flat = np.zeros_like(vel_flat)

        vel_cfad = np.zeros((len(alt_bins)-1, len(vel_bins)-1))

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
        
        
                vel_max_ind = np.where(vel_cfad[a,:]==vel_cfad[a,:].max())[0]

                # subtract the velocity value of the max freq
                fall_speeds_flat[hght_inds] = vel_flat[hght_inds] - vel_bins[vel_max_ind[0]]


        cfad = np.ma.masked_where(vel_cfad==0.00, vel_cfad)
        [X, Y] = np.meshgrid(vel_bins[:-1]+np.diff(vel_bins)/2, alt_bins[:-1]+np.diff(alt_bins)/2)

        return [cfad, X, Y]

        

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

    def __init__(self, filepath, start_time=None, end_time=None, atten_file=None, max_roll=None, 
        dbz_sigma=None, vel_sigma=None, width_sigma=None, ldr_sigma=None, dbz_min=None, vel_min=None, width_min=None):

    
        self.name = 'CRS'

        # read the raw data
        self.data = self.readfile(filepath, start_time, end_time)
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



    def readfile(self, filepath, start_time=None, end_time=None):
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
        nomdist = xr.DataArray(
            data = hdf['Navigation']['Data']['NominalDistance'][:],
            dims = ["time"],
            coords = dict(time=time_dt64),
            attrs = dict(
                description=hdf['Navigation']['Information']['NominalDistance_desciption'][0].decode('UTF-8'),
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
        
        aircraft_motion = xr.DataArray(
            data = hdf['Products']['Information']['AircraftMotion'][:,0],
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


    def correct_ice_atten(self, hiwrap, hrrr):
        pass

    def correct_attenuation(self, atten_file):
        pass



# ====================================== #
# HIWRAP
# ====================================== #

class Hiwrap(Radar):
    """
    A class to represent the HIWRAP nadir pointing radar flown on the ER2 during the IMPACTS field campaign.
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

    def __init__(self, filepath, start_time=None, end_time=None, atten_file=None, max_roll=None, 
                dbz_sigma=None, vel_sigma=None, width_sigma=None, 
                dbz_min=None, vel_min=None, width_min=None):
    
        self.name = 'HIWRAP'

        # create a dataset with both ka- and ku-band data
        self.data = self.readfile(filepath, start_time=start_time, end_time=end_time)
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
            self.mask_roll(max_roll)


    def readfile(self, filepath, start_time=None, end_time=None):
        """
        Reads the HIWRAP data file and unpacks the fields into an xarray.Dataset

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
        nomdist = xr.DataArray(
            data = hdf['Navigation']['Data']['NominalDistance'][:],
            dims = ["time"],
            coords = dict(time=time_dt64),
            attrs = dict(
                description=hdf['Navigation']['Information']['NominalDistance_desciption'][0].decode('UTF-8'),
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

        dbz_ka = xr.DataArray(
            data = hdf['Products']['Ka']['Combined']['Data']['dBZe'][:].T,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Ka']['Combined']['Information']['dBZe_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ka']['Combined']['Information']['dBZe_units'][0].decode('UTF-8')
            )   
        )

        dbz_ku = xr.DataArray(
            data = hdf['Products']['Ku']['Combined']['Data']['dBZe'][:].T,
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Ku']['Combined']['Information']['dBZe_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ku']['Combined']['Information']['dBZe_units'][0].decode('UTF-8')
            )   
        )

        width_ka = xr.DataArray(
            data = np.ma.masked_invalid(hdf['Products']['Ka']['Combined']['Data']['SpectrumWidth'][:].T),
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Ka']['Combined']['Information']['SpectrumWidth_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ka']['Combined']['Information']['SpectrumWidth_units'][0].decode('UTF-8')
            ) 
        )
        width_ku = xr.DataArray(
            data = np.ma.masked_invalid(hdf['Products']['Ku']['Combined']['Data']['SpectrumWidth'][:].T),
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Ku']['Combined']['Information']['SpectrumWidth_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ku']['Combined']['Information']['SpectrumWidth_units'][0].decode('UTF-8')
            ) 
        )

        ldr_ka = xr.DataArray(
            data = np.ma.masked_invalid(hdf['Products']['Ka']['Combined']['Data']['LDR'][:].T),
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Ka']['Combined']['Information']['LDR_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ka']['Combined']['Information']['LDR_units'][0].decode('UTF-8')
            ) 
        )
        ldr_ku = xr.DataArray(
            data = np.ma.masked_invalid(hdf['Products']['Ku']['Combined']['Data']['LDR'][:].T),
            dims = ["range", "time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Ku']['Combined']['Information']['LDR_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ku']['Combined']['Information']['LDR_units'][0].decode('UTF-8')
            ) 
        )

        if 'Velocity_corrected' in list(hdf['Products']['Ka']['Combined']['Data'].keys()):
            # for NUBF correction
            vel_ka = xr.DataArray(
                data = hdf['Products']['Ka']['Combined']['Data']['Velocity_corrected'][:].T,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description=hdf['Products']['Ka']['Combined']['Information']['Velocity_corrected_description'][0].decode('UTF-8'),
                    units = hdf['Products']['Ka']['Combined']['Information']['Velocity_corrected_units'][0].decode('UTF-8')
                ) 
            )
        else:
            vel_ka = xr.DataArray(
                data = hdf['Products']['Ka']['Combined']['Data']['Velocity'][:].T,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description=hdf['Products']['Ka']['Combined']['Information']['Velocity_description'][0].decode('UTF-8'),
                    units = hdf['Products']['Ka']['Combined']['Information']['Velocity_units'][0].decode('UTF-8')
                ) 
            )

        if 'Velocity_corrected' in list(hdf['Products']['Ku']['Combined']['Data'].keys()):
            # for NUBF correction
            vel_ku = xr.DataArray(
                data = hdf['Products']['Ku']['Combined']['Data']['Velocity_corrected'][:].T,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description=hdf['Products']['Ku']['Combined']['Information']['Velocity_corrected_description'][0].decode('UTF-8'),
                    units = hdf['Products']['Ku']['Combined']['Information']['Velocity_corrected_units'][0].decode('UTF-8')
                ) 
            )
        else:
            vel_ku = xr.DataArray(
                data = hdf['Products']['Ku']['Combined']['Data']['Velocity'][:].T,
                dims = ["range", "time"],
                coords = dict(
                    range = radar_range,
                    height = height,
                    time = time_dt64,
                    distance = nomdist,
                    lat = lat,
                    lon = lon),
                attrs = dict(
                    description=hdf['Products']['Ku']['Combined']['Information']['Velocity_description'][0].decode('UTF-8'),
                    units = hdf['Products']['Ku']['Combined']['Information']['Velocity_units'][0].decode('UTF-8')
                ) 
            )
        
        aircraft_motion = xr.DataArray(
            data = hdf['Products']['Information']['AircraftMotion'][:],
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

        channel_mask_ka = xr.DataArray(
            data = hdf['Products']['Ka']['Combined']['Information']['ChannelMask'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = hdf['Products']['Ka']['Combined']['Information']['ChannelMask_description'][0].decode('UTF-8'),
            ) 
        )
        channel_mask_ku = xr.DataArray(
            data = hdf['Products']['Ku']['Combined']['Information']['ChannelMask'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description = hdf['Products']['Ku']['Combined']['Information']['ChannelMask_description'][0].decode('UTF-8'),
            ) 
        )

        horiz_resolution_ka = xr.DataArray(
            data = hdf['Products']['Ka']['Information']['ResolutionHorizontal6dB'][:],
            dims = ["range"],
            coords = dict(
                range = radar_range),
            attrs = dict(
                description=hdf['Products']['Ka']['Information']['ResolutionHorizontal6dB_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ka']['Information']['ResolutionHorizontal6dB_units'][0].decode('UTF-8')
            ) 
        )

        horiz_resolution_ku = xr.DataArray(
            data = hdf['Products']['Ku']['Information']['ResolutionHorizontal6dB'][:],
            dims = ["range"],
            coords = dict(
                range = radar_range),
            attrs = dict(
                description=hdf['Products']['Ku']['Information']['ResolutionHorizontal6dB_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ku']['Information']['ResolutionHorizontal6dB_units'][0].decode('UTF-8')
            ) 
        )
        vel_horizwind_offset_ka = xr.DataArray(
            data = hdf['Products']['Ka']['Combined']['Information']['Velocity_horizwind_offset'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Ka']['Combined']['Information']['Velocity_horizwind_offset_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ka']['Combined']['Information']['Velocity_horizwind_offset_units'][0].decode('UTF-8')
            ) 
        )
        vel_horizwind_offset_ku = xr.DataArray(
            data = hdf['Products']['Ku']['Combined']['Information']['Velocity_horizwind_offset'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Ku']['Combined']['Information']['Velocity_horizwind_offset_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ku']['Combined']['Information']['Velocity_horizwind_offset_units'][0].decode('UTF-8')
            ) 
        )
        vel_nubf_offset_ku = xr.DataArray(
            data = hdf['Products']['Ku']['Combined']['Information']['Velocity_nubf_offset'][:].T,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                height = height,
                time = time_dt64,
                distance = nomdist,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=hdf['Products']['Ku']['Combined']['Information']['Velocity_nubf_offset_description'][0].decode('UTF-8'),
                units = hdf['Products']['Ku']['Combined']['Information']['Velocity_nubf_offset_units'][0].decode('UTF-8')
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
        L1B_Revision_Note = hdf['Information']['L1B_Revision_Note'][0].decode('UTF-8')
        missionPI = hdf['Information']['MissionPI'][0].decode('UTF-8')
        radar_name = hdf['Information']['RadarName'][0].decode('UTF-8')
        antenna_beamwidth_ka = hdf['Products']['Ka']['Information']['AntennaBeamwidth'][0]
        antenna_beamwidth_ku = hdf['Products']['Ku']['Information']['AntennaBeamwidth'][0]
        antenna_size = hdf['Products']['Information']['AntennaSize'][0]
        avg_pulses_ka = hdf['Products']['Ka']['Information']['AveragedPulses'][0]
        avg_pulses_ku = hdf['Products']['Ku']['Information']['AveragedPulses'][0]
        freq_ka = hdf['Products']['Ka']['Information']['Frequency'][0]
        freq_ku = hdf['Products']['Ku']['Information']['Frequency'][0]
        gatespacing = hdf['Products']['Information']['GateSpacing'][0]
        antenna_pointing = hdf['Products']['Information']['NominalAntennaPointing'][0].decode('UTF-8')
        pri = hdf['Products']['Information']['PRI'][0].decode('UTF-8')
        wavelength_ka = hdf['Products']['Ka']['Information']['Wavelength'][0]
        wavelength_ku = hdf['Products']['Ku']['Information']['Wavelength'][0]
        
        # close the file
        hdf.close()

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
        pass





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

    def __init__(self, filepath, start_time=None, end_time=None, atten_file=None, max_roll=None, 
        dbz_sigma=None, vel_sigma=None, width_sigma=None, dbz_min=None, vel_min=None, width_min=None):
        
        self.name = 'EXRAD'

        # read the raw data
        self.data = self.readfile(filepath, start_time, end_time)
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
        
        
   

    def readfile(self, filepath, start_time=None, end_time=None):
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
        nomdist = xr.DataArray(
            data = hdf['Navigation']['Data']['NominalDistance'][:],
            dims = ["time"],
            coords = dict(time=time_dt64),
            attrs = dict(
                description=hdf['Navigation']['Information']['NominalDistance_desciption'][0].decode('UTF-8'),
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
        
        aircraft_motion = xr.DataArray(
            data = hdf['Products']['Information']['AircraftMotion'][:],
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
        # open the attentuation file
        atten_data = xr.open_dataset(atten_file)

        # add the correction to the reflectivity values
        self.data['dbz'].values += atten_data['k_x'].values

    

class VAD(object):
    """
    A class to represent the Velocity-Azimuth Display (VAD) wind product derived from the EXRAD conical scans during the IMPACTS field campaign.

    :param data: dataset containing VAD data and attributes
    :type data: xarray.Dataset
    """
    def __init__(self, filepath):
        self.name = 'VAD'

        self.data = self.readfile(filepath)

    def readfile(self, filepath, start_time=None, end_time=None):
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
            data = vad['hght'][:,100].values,
            dims=["range"],
            coords = dict(
                range = radar_range
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
                distance = yt,
                lat = lat,
                lon = lon),
            attrs = dict(
                description=vad['cor'].long_name
            ) 
        )
        qc1 = xr.DataArray(
            data = vad['qc1'].values,
            dims = ["range","time"],
            coords = dict(
                range = radar_range,
                time = time_dt64,
                distance = yt,
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
                "qc1": qc1,
                "qc2": qc2,
                "qc3": qc3,
                "qc4": qc4,
                "qc5": qc5
            },

            coords= {
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
