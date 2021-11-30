"""
Classes for IMPACTS P3 Instruments
"""
import xarray as xr
import numpy as np
from pyproj import Proj
from scipy.spatial import cKDTree
from datetime import datetime, timedelta

class P3(object):
    """
    A class to represent the P3 aircraft during the IMPACTS field campaign
    
    Parameters
    ----------
    filepath : str
        File path to the P3 navigation data file
    start_time : np.datetime64 or None
        The initial time of interest eg. if looking at a single flight leg
    end_time : np.datetime64 or None
        The final time of interest eg. if looking at a single flight leg
    """

    def __init__(self, filepath, start_time=None, end_time=None):
        self.name = 'P3nav'

        # read the raw data
        self.data = self.readfile(filepath, start_time, end_time)
        """
        xarray.Dataset of P3 navigation variables and attributes

        Dimensions:
            - time: np.array(np.datetime64[ns]) - The UTC time stamp

        Coordinates:
            - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  

        Variables:
            - Time_Start (time) : 
            - Day_Of_Year (time) : 
            - Latitude (time) : 
            - Longitude (time) : 
            - GPS_Altitude (time) : 
            - Pressure_Altitude (time) : 
            - Radar_Altitude (time) : 
            - Ground_Speed (time) : 
            - True_Air_Speed (time) : 
            - Indicated_Air_Speed (time) : 
            - Mach_Number (time) : 
            - Vertical_Spped (time) : 
            - True_Heading (time) : 
            - Track_Angle (time) : 
            - Drift_Angle (time) : 
            - Pitch_Angle (time) : 
            - Roll_Angle (time) : 
            - Static_Air_Temp (time) : 
            - Potential_Temp (time) : 
            - Dew_Point (time) : 
            - Total_Air_Temp (time) : 
            - IR_Surface_Temp (time) : 
            - Static_Pressure (time) : 
            - Cabin_Pressure (time) : 
            - Wind_Speed (time) : 
            - Wind_Direction (time) : 
            - Solar_Zenith_Angle (time) : 
            - Aircraft_Sun_Elevation (time) : 
            - Sun_Azimuth (time) : 
            - Aircraft_Sun_Azimuth (time) : 
            - Mixing_Ratio (time) : 
            - Part_Press_Water_Vapor (time) : 
            - Sat_Vapor_Press_H2O (time) : 
            - Sat_Vapor_Press_Ice (time) : 
            - Relative_Humidity (time) : 

        
        Attribute Information:
            PI_CONTACT_INFO, Platform, LOCATION, ASSOCIATED_DATA, INSTRUMENT_INFO, DATA_INFO, UNCERTAINTY, 
            ULOD_FLAG, ULOD_VALUE, LLOD_FLAG, LLOD_VALUE, DM_CONTACT_INFO, PROJECT_INFO, STIPULATIONS_ON_USE, 
            OTHER_COMMENTS, REVISION, R0
        
        """


    def readfile(self, filepath, start_time=None, end_time=None):
        """
        Reads the P3 navigation data file and upacks the fields into an xarray.Dataset.
        This assumes the nav file is in NASA Ames FFI format (http://cedadocs.ceda.ac.uk/73/4/FFI-summary.html)
        
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
        ds : xarray.Dataset
            The unpacked dataset
        """

        f = open(filepath, 'r')

        # NLHEAD is Number of header lines, FFI is NASA AMES FFI format number
        NLHEAD, FFI = f.readline().rstrip('\n').split(',')
        
        # Check that the file is indeed NASA AMES 1001
        if FFI.replace(' ', '') != '1001':
            print("Check file type, looks like it's not FFI 1001")
            return


        ONAME = f.readline().rstrip('\n')   # Originator/PI Name
        ORG = f.readline().rstrip('\n')     # Name of organization
        SNAME = f.readline().rstrip('\n')   # Instrument/platform name
        MNAME = f.readline().rstrip('\n')   # Project/mission name
        IVOL, NVOL = f.readline().rstrip('\n').split(',')   # IVOL is Current volume number (almost always 1), NVOL is Number of volumes for data (almost always 1)
        
        yy1, mm1, dd1, yy2, mm2, dd2 = f.readline().split(',')
        datestr = yy1 + '-' + mm1 + '-' + dd1
        DATE = (int(yy1), int(mm1), int(dd1))   # YYYY MM DD UTC begin date
        RDATE = (int(yy2), int(mm2), int(dd2))  # Reduction/revision UTC date 
        
        DX = f.readline().rstrip('\n')          # Interval between successive values (data rate)
        XNAME = f.readline().rstrip('\n')       # Name/Description of DX variable above
        NV = int(f.readline().rstrip('\n'))     # Number of primary variables in file
        
        _vscl = f.readline().split(',')
        VSCAL = [float(x) for x in _vscl]       # Scaling factor for each variable column
        
        _vmiss = f.readline().split(',')
        VMISS = [float(x) for x in _vmiss]      # Missing value for each variable column
        
        VNAME = ['time']     # Name of first variable
        VUNIT = ['seconds since ' + datestr]    # Units of first variable

        # unpack the variable names and units    
        for nvar in range(NV):
            line_buffer = f.readline().rstrip('\n').split(',', 1)
            VNAME.append(line_buffer[0])
            VUNIT.append(line_buffer[1][1:])
        
        NSCOML = int(f.readline().rstrip('\n'))  # Number of special comment lines within header
        SCOM = []       # Special comments about file/data, etc.
        
        # unpack the special comments
        for nscom in range(NSCOML):
            SCOM.append(f.readline().rstrip('\n'))
        
        NNCOML = int(f.readline().rstrip('\n')) # Number of normal comment lines within header
        NCOM = []       # Normal comments
    
        # unpack the normal comments
        for nncom in range(NNCOML):
            NCOM.append(f.readline().rstrip('\n'))
    
        # Insert elements to account for time column
        VSCAL.insert(0, 1)
        VMISS.insert(0, np.nan)

        # close the nav file
        f.close()

        junk = np.genfromtxt(filepath, delimiter=',', skip_header=int(NLHEAD),
                         missing_values=VMISS, usemask=True,
                         filling_values=np.nan)
        


        readfile = {}
        if len(VNAME) != len(VSCAL):
            print("ALL variables must be read in this type of file, "
              "please check name_map to make sure it is the "
              "correct length.")


        for jj, name in enumerate(VNAME):
            if name=='True_Air_Speed' or name=='Indicated_Air_Speed' or name=='Mach_Number':
                VMISS[jj] = -8888.
            if name=='True_Air_Speed' and VUNIT[jj]=='kts': # change TAS to m/s
                VSCAL[jj] = 0.51
                VUNIT[jj] = 'm/s'
            readfile[name] = np.array(junk[:, jj] * VSCAL[jj])
            # Turn missing values to nan
            readfile[name][readfile[name]==VMISS[jj]] = np.nan

        print(readfile)

        time = np.array([np.datetime64(datestr) + np.timedelta64(int(readfile['time'][i]), 's') for i in range(len(readfile['time']))])
        
        attrs = {}
        instrum_info_counter = 1
        for ii, comment in enumerate(NCOM[:-1]): # add global attributes
            parsed_comment = comment.split(':')
            if len(parsed_comment) > 1:
                attrs[parsed_comment[0]] = parsed_comment[1][1:]
            else: # handles multiple instrument info lines in *_R0.ict files
                instrum_info_counter += 1
                attrs[
                    'INSTRUMENT_INFO_'+str(instrum_info_counter)] = parsed_comment[0][1:]

        data_vars = {}
        for jj, name in enumerate(VNAME[2:]):
            da = xr.DataArray(
                data = readfile[name],
                dims = ['time'],
                coords = dict(
                    time = time),
                attrs = dict(
                    description=name,
                    units = VUNIT[jj+2][:]
                )
            )
            data_vars[name] = da


        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                "time": time
            },
            attrs = attrs
        )

        return ds

    def average(self, start_time=None, end_time=None, resolution=5.):
        """
        Average the aircraft state parameters according to the time resolution

        Parameters
        ----------

        start_time: np.dateteim64 or None
            The start time of interest
        end_time: np.datetime64 or None
            The end time of interest
        resolution: Float (s)
            The averaging interval defaults to 5 seconds. If resolution=1, averaging is skipped.

        Returns
        -------
        iwg_avg : dictionary
            The averaged dataset

        """
        if start_time is None:
            start_dt64 = self.data['time'][0]
        else:
            start_dt64 = np.datetime64(start_time)

        if end_time is None:
            end_dt64 = self.data['time'][-1]
        else:
            end_dt64 = np.datetime64(end_time)

        if resolution>1:
            iwg_avg = {}
            iwg_avg['Information'] = self.data.attrs

            dur = (end_dt64 - start_dt64) / np.timedelta64(1, 's') # dataset duration to consider [s]

            # Allocate arrays
            for var in self.data.data_vars:
                if (var!='Information') and (var!='time'):
                    iwg_avg[var] = {}
                    iwg_avg[var]['data'] = np.zeros(int(dur/resolution))
                    iwg_avg[var]['units'] = self.data[var].attrs['units']

            time_subset_begin = start_dt64 # allocate time array of N-sec interval obs
            # get midpoint of N-second interval
            time_subset_mid = start_dt64 + np.timedelta64(int(resolution/2.), 's') + np.timedelta64(int(np.mod(resolution/2., 1)*1000.), 'ms')
            curr_time = start_dt64
            i = 0

            while curr_time<end_dt64:
                if curr_time>start_dt64:
                    time_subset_begin = np.append(time_subset_begin, curr_time)
                    time_subset_mid = np.append(time_subset_mid, curr_time + np.timedelta64(int(resolution/2.), 's') +
                                            np.timedelta64(int(np.mod(resolution/2., 1)*1000.), 'ms'))
                time_inds = np.where((self.data['time']>=curr_time) &
                                 (self.data['time']<curr_time+np.timedelta64(int(resolution), 's')))[0]

                for var in iwg_avg.keys(): # loop through variables
                    if (var=='Latitude') or (var=='Longitude') or (var=='GPS_Altitude') or (var=='Pressure_Altitude'): # these require special interpolation about the time midpoint
                        times = np.arange(int(resolution))
                        coord_array = self.data[var][time_inds]
                        iwg_avg[var]['data'][i] = np.interp(resolution/2., times, coord_array)
                    elif (var!='Information') and (var!='time'): # do a simple mean for all other variables
                        var_array = self.data[var][time_inds]
                        iwg_avg[var]['data'][i] = np.mean(var_array)
                i += 1
                curr_time += np.timedelta64(int(resolution), 's')

            iwg_avg['time'] = {}; iwg_avg['time_midpoint'] = {}
            iwg_avg['time']['data'] = time_subset_begin
            iwg_avg['time']['units'] = 'Start of {}-second period as numpy datetime64 object'.format(str(int(resolution)))
            iwg_avg['time_midpoint']['data'] = time_subset_mid
            iwg_avg['time_midpoint']['units'] = 'Middle of {}-second period as numpy datetime64 object'.format(str(int(resolution)))

            return iwg_avg
