"""
Classes for IMPACTS P3 Instruments
"""

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from scipy.optimize import least_squares

def parse_header(f, date, stream='default'):
        '''
        NLHEAD : Number of header lines
        FFI : NASA AMES FFI format number
        ONAME : Originator/PI Name
        ORG : Name of organization
        SNAME : Instrument/platform name
        MNAME : Project/mission name
        IVOL : Current volume number (almost always 1)
        NVOL : Number of volumes for data (almost always 1)
        DATE : YYYY MM DD UTC begin date
        RDATE : Reduction/revision UTC date
        DX : Interval between successive values (data rate)
        XNAME : Name/Description of DX variable above
        NV : Number of primary variables in file
        VSCL : Scaling factor for each variable column
        VMISS : Missing value for each variable column
        VNAME : Name of first variable
        NSCOML : Number of special comment lines within header
        SCOM : Special comments about file/data, etc.
        NNCOML : Number of normal comment lines within header
        NCOM : Normal comments
        '''
        if stream == 'und': # format exception for UND microphysics files
            delim = ' '
            delim2 = None
        else: # all other NASA AMES formats 
            delim = ','
            delim2 = ','
        hdr = {}
        hdr['NLHEAD'], hdr['FFI'] = f.readline().rstrip('\n').split(delim)

        # Check that the file is indeed NASA AMES 1001
        if hdr['FFI'].replace(' ', '') != '1001':
            print("Check file type, looks like it's not FFI 1001")
            return

        hdr['ONAME'] = f.readline().rstrip('\n')
        hdr['ORG'] = f.readline().rstrip('\n')
        hdr['SNAME'] = f.readline().rstrip('\n')
        hdr['MNAME'] = f.readline().rstrip('\n')
        hdr['IVOL'], hdr['NVOL'] = f.readline().rstrip('\n').split(delim2)
        yy1, mm1, dd1, yy2, mm2, dd2 = f.readline().split(delim2)
        hdr['DATE'] = (int(yy1), int(mm1), int(dd1))
        hdr['RDATE'] = (int(yy2), int(mm2), int(dd2))
        if stream == 'und':
            hdr['DX'] = f.readline().rstrip('\n').lstrip(delim)
        else:
            hdr['DX'] = f.readline().rstrip('\n')
        hdr['XNAME'] = f.readline().rstrip('\n')
        hdr['NV'] = int(f.readline().rstrip('\n'))
        vscl = f.readline().split(delim2)
        hdr['VSCAL'] = [float(x) for x in vscl]
        vmiss = f.readline().split(delim2)
        hdr['VMISS'] = [float(x) for x in vmiss]
        hdr['VNAME'] = ['time']
        hdr['VUNIT'] = ['seconds since ' + date]
        for nvar in range(hdr['NV']):
            if stream == 'und':
                line_buffer = f.readline().rstrip('].\n').split('[', -1)
                hdr['VNAME'].append((line_buffer[0]).split('(')[0].rstrip(' '))
                hdr['VUNIT'].append(line_buffer[-1])
            else:
                line_buffer = f.readline().rstrip('\n').split(',', 1)
                hdr['VNAME'].append(line_buffer[0])
                hdr['VUNIT'].append(line_buffer[1][1:])
        hdr['NSCOML'] = int(f.readline().rstrip('\n'))
        hdr['SCOM'] = []
        for nscom in range(hdr['NSCOML']):
            hdr['SCOM'].append(f.readline().rstrip('\n'))
        hdr['NNCOML'] = int(f.readline().rstrip('\n'))
        hdr['NCOM'] = []
        for nncom in range(hdr['NNCOML']):
            hdr['NCOM'].append(f.readline().rstrip('\n'))
        # Insert elements to account for time column
        hdr['VSCAL'].insert(0, 1)
        hdr['VMISS'].insert(0, np.nan)
        f.close()

        return hdr

class P3():
    """
    A class to represent the P-3 aircraft during the IMPACTS field campaign.
    """

    def __init__(self, filepath, date, start_time=None, end_time=None, tres='1S', fmt='ames'):
        self.name = 'P-3 Met-Nav'
        
        # read the raw data
        self.data = self.readfile(filepath, date, start_time, end_time, tres, fmt)
        """
        xarray.Dataset of P-3 meteorological and navigation variables and attributes
        Dimensions:
            - time: np.array(np.datetime64[ms]) - The UTC time stamp
        Coordinates:
            - time (time): np.array(np.datetime64[ms]) - The UTC time stamp
        Variables:
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)
            - alt_gps (time) : xarray.DataArray(float) - Aircraft GPS altitude (m above mean sea level)
            - alt_pres (time) : xarray.DataArray(float) - Aircraft pressure altitude (ft)
            - alt_radar (time) : xarray.DataArray(float) - Aircraft radar altitude (ft)
            - grnd_spd (time) : xarray.DataArray(float) - Aircraft ground speed (m/s)
            - tas (time) : xarray.DataArray(float) - Aircraft true air speed (m/s)
            - ias (time) : xarray.DataArray(float) - Aircraft indicated air speed (m/s)
            - mach (time) : xarray.DataArray(float) - Aircraft mach number
            - zvel_p3 (time) : xarray.DataArray(float) - Aircraft vertical speed (m/s)
            - heading (time) : xarray.DataArray(float) - Aircraft true heading (deg clockwise from +y)
            - track (time) : xarray.DataArray(float) - Aircraft track angle (deg clockwise from +y)
            - drift (time) : xarray.DataArray(float) - Aircraft drift angle (deg clockwise from +y)
            - pitch (time) : xarray.DataArray(float) - Aircraft pitch angle (deg, positive is up)
            - roll (time) : xarray.DataArray(float) - Aircraft roll angle (deg, positive is right turn)
            - temp (time) : xarray.DataArray(float) - Static (ambient) air temperature (deg C)
            - temp_total (time) : xarray.DataArray(float) - Total air temperature (deg C, static and dynamic)
            - temp_ir (time) : xarray.DataArray(float) - Infrared surface temperature (deg C)
            - temp_pot (time) : xarray.DataArray(float) - Potential temperature (K)
            - dwpt (time) : xarray.DataArray(float) - Dew point temperature (deg C)
            - pres_static (time) : xarray.DataArray(float) - Static air pressure (hPa)
            - pres_cabin (time) : xarray.DataArray(float) - Cabin air pressure (hPa)
            - wspd (time) : xarray.DataArray(float) - Horizontal wind speed (m/s, limited to where roll <= 5 degrees)
            - wdir (time) : xarray.DataArray(float) - Horizontal wind direction (deg clockwise from +y)
            - uwnd (time) : xarray.DataArray(float) - Horizontal U-component wind speed (m/s, not available in 2020 data)
            - vwnd (time) : xarray.DataArray(float) - Horizontal V-component wind speed (m/s, not available in 2020 data)
            - mixrat (time) : xarray.DataArray(float) - Mixing ratio (g/kg)
            - pres_vapor (time) : xarray.DataArray(float) - Partial pressure (hPa) with respect to water vapor
            - svp_h2o (time) : xarray.DataArray(float) - Saturation vapor pressure (hPa) with respect to water
            - svp_ice (time) : xarray.DataArray(float) - Saturation vapor pressure (hPa) with respect to ice
            - rh (time) : xarray.DataArray(float) - Relative humidity with respect to water (percent)
            - zenith (time) : xarray.DataArray(float) - Solar zenith angle (deg)
            - sun_elev_p3 (time) : xarray.DataArray(float) - Aircraft sun elevation (deg)
            - sun_az (time) : xarray.DataArray(float) - Sun azimuth (deg)
            - sun_az_p3 (time) : xarray.DataArray(float) - Aircraft sun azimuth (deg)
            
        Attribute Information:
            [TEXT]
        """


    def parse_header(self, f, date):
        '''
        NLHEAD : Number of header lines
        FFI : NASA AMES FFI format number
        ONAME : Originator/PI Name
        ORG : Name of organization
        SNAME : Instrument/platform name
        MNAME : Project/mission name
        IVOL : Current volume number (almost always 1)
        NVOL : Number of volumes for data (almost always 1)
        DATE : YYYY MM DD UTC begin date
        RDATE : Reduction/revision UTC date
        DX : Interval between successive values (data rate)
        XNAME : Name/Description of DX variable above
        NV : Number of primary variables in file
        VSCL : Scaling factor for each variable column
        VMISS : Missing value for each variable column
        VNAME : Name of first variable
        NSCOML : Number of special comment lines within header
        SCOM : Special comments about file/data, etc.
        NNCOML : Number of normal comment lines within header
        NCOM : Normal comments
        '''
        hdr = {}
        hdr['NLHEAD'], hdr['FFI'] = f.readline().rstrip('\n').split(',')

        # Check that the file is indeed NASA AMES 1001
        if hdr['FFI'].replace(' ', '') != '1001':
            print("Check file type, looks like it's not FFI 1001")
            return

        hdr['ONAME'] = f.readline().rstrip('\n')
        hdr['ORG'] = f.readline().rstrip('\n')
        hdr['SNAME'] = f.readline().rstrip('\n')
        hdr['MNAME'] = f.readline().rstrip('\n')
        hdr['IVOL'], hdr['NVOL'] = f.readline().rstrip('\n').split(',')
        yy1, mm1, dd1, yy2, mm2, dd2 = f.readline().split(',')
        hdr['DATE'] = (int(yy1), int(mm1), int(dd1))
        hdr['RDATE'] = (int(yy2), int(mm2), int(dd2))
        hdr['DX'] = f.readline().rstrip('\n')
        hdr['XNAME'] = f.readline().rstrip('\n')
        hdr['NV'] = int(f.readline().rstrip('\n'))
        vscl = f.readline().split(',')
        hdr['VSCAL'] = [float(x) for x in vscl]
        vmiss = f.readline().split(',')
        hdr['VMISS'] = [float(x) for x in vmiss]
        hdr['VNAME'] = ['time']
        hdr['VUNIT'] = ['seconds since ' + date]
        for nvar in range(hdr['NV']):
            line_buffer = f.readline().rstrip('\n').split(',', 1)
            hdr['VNAME'].append(line_buffer[0])
            hdr['VUNIT'].append(line_buffer[1][1:])
        hdr['NSCOML'] = int(f.readline().rstrip('\n'))
        hdr['SCOM'] = []
        for nscom in range(hdr['NSCOML']):
            hdr['SCOM'].append(f.readline().rstrip('\n'))
        hdr['NNCOML'] = int(f.readline().rstrip('\n'))
        hdr['NCOM'] = []
        for nncom in range(hdr['NNCOML']):
            hdr['NCOM'].append(f.readline().rstrip('\n'))
        # Insert elements to account for time column
        hdr['VSCAL'].insert(0, 1)
        hdr['VMISS'].insert(0, np.nan)
        f.close()

        return hdr
    
    def readfile(self, filepath, date, start_time=None, end_time=None, tres='1S', fmt='ames'):
        """
        Reads the P-3 Met-Nav data file and unpacks the fields into an xarray.Dataset
        
        Parameters
        ----------
        filepath : str
            Path to the data file
        date: str
            Flight start date in YYYY-mm-dd format
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        tres: str
            The time interval to average over (e.g., '5S' for 5 seconds)
        fmt: str
            ames - NASA Ames format; iwg - IWG packet format (no headers)
        
        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """
        if fmt == 'ames':
            # get header info following the NASA AMES format
            header = self.parse_header(open(filepath, 'r'), date)

            # parse the data
            data_raw = np.genfromtxt(
                filepath, delimiter=',', skip_header=int(header['NLHEAD']),
                missing_values=header['VMISS'], usemask=True, filling_values=np.nan
            )

            # construct dictionary of variable data and metadata
            readfile = {}
            if len(header['VNAME']) != len(header['VSCAL']):
                print(
                    'ALL variables must be read in this type of file. '
                    'Please check name_map to make sure it is the correct length.'
                )
            for jj, unit in enumerate(header['VUNIT']):
                header['VUNIT'][jj] = unit.split(',')[0]

            for jj, name in enumerate(header['VNAME']): # fix scaling and missing data flags for some vars
                if (name=='True_Air_Speed' or name=='Indicated_Air_Speed'
                        or name=='Mach_Number'):
                    header['VMISS'][jj] = -8888.
                if name=='True_Air_Speed' and header['VUNIT'][jj]=='kts': # [m/s]
                    header['VMISS'][jj] = -8888. * 0.51
                    header['VSCAL'][jj] = 0.51
                    header['VUNIT'][jj] = 'm/s'
                readfile[name] = np.array(data_raw[:, jj] * header['VSCAL'][jj])
                # turn missing values to nan
                readfile[name][readfile[name]==header['VMISS'][jj]] = np.nan
            readfile['Wind_Speed'][readfile['Wind_Speed']==-8888.] = np.nan # wspd has two missing data flags
            
            # compute time
            time = np.array([
                np.datetime64(date) + np.timedelta64(int(readfile['time'][i]), 's')
                for i in range(len(readfile['time']))], dtype='datetime64[ms]'
            )

            # populate dataset attributes
            p3_attrs = {
                'Experiment': 'IMPACTS',
                'Platform': 'P-3',
                'Mission PI': 'Lynn McMurdie (lynnm@uw.edu)'
            }
            instrum_info_counter = 1
            for ii, comment in enumerate(header['NCOM'][:-1]): # add global attrs
                parsed_comment = comment.split(':')
                if len(parsed_comment) > 1:
                    p3_attrs[parsed_comment[0]] = parsed_comment[1][1:]
                else: # handles multiple instrument info lines in *_R0.ict files
                    instrum_info_counter += 1
                    p3_attrs[
                        'INSTRUMENT_INFO_'+str(instrum_info_counter)] = parsed_comment[0][1:]
        elif fmt == 'iwg':
            names = [
                'fmt', 'time', 'Latitude', 'Longitude', 'GPS_Altitude', 'WGS_84_Alt',
                'Pressure_Altitude', 'Radar_Altitude', 'Ground_Speed', 'True_Air_Speed',
                'Indicated_Air_Speed', 'Mach_Number', 'Vertical_Speed', 'True_Heading',
                'Track_Angle', 'Drift_Angle', 'Pitch_Angle', 'Roll_Angle', 'Side_slip',
                'Angle_of_Attack', 'Static_Air_Temp', 'Dew_Point', 'Total_Air_Temp',
                'Static_Pressure', 'Dynamic_Press', 'Cabin_Pressure', 'Wind_Speed',
                'Wind_Direction', 'Vert_Wind_Spd', 'Solar_Zenith_Angle',
                'Aircraft_Sun_Elevation', 'Sun_Azimuth', 'Aircraft_Sun_Azimuth'
            ]
            dtypes = [
                str, 'datetime64[s]', float, float, float, float, float, float, float, float,
                float, float, float, float, float, float, float, float, float, float, float,
                float, float, float, float, float, float, float, float, float, float, float,
                float,
            ]
            readfile = np.genfromtxt(filepath, delimiter=',', names=names, dtype=dtypes)
            time = readfile['time']
            p3_attrs = {
                'Experiment': 'IMPACTS',
                'Platform': 'P-3',
                'Mission PI': 'Lynn McMurdie (lynnm@uw.edu)'
            }

        # populate data arrays
        lat = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Latitude']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft latitude',
                units='degrees_north')
        )
        lon = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Longitude']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft longitude',
                units='degrees_east')
        )
        alt_gps = xr.DataArray(
            data = np.ma.masked_invalid(readfile['GPS_Altitude']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft GPS altitude (mean sea level)',
                units='meters')
        )
        alt_pres = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Pressure_Altitude']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft pressure altitude',
                units='feet')
        )
        alt_radar = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Radar_Altitude']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft radar altitude',
                units='feet')
        )
        grnd_spd = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Ground_Speed']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft ground speed',
                units='m/s')
        )
        tas = xr.DataArray(
            data = np.ma.masked_invalid(readfile['True_Air_Speed']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft true air speed',
                units='m/s')
        )
        ias = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Indicated_Air_Speed']),
            dims = ['time'], coords = dict(time = time),
            attrs = dict(
                description='Aircraft indicated air speed',
                units='kts')
        )
        mach = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Mach_Number']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft mach number',
                units='mach')
        )
        vert_vel = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Vertical_Speed']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft vertical speed',
                units='m/s')
        )
        heading = xr.DataArray(
            data = np.ma.masked_invalid(readfile['True_Heading']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft true heading (clockwise from +y)',
                units='degrees')
        )
        track = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Track_Angle']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft track angle (clockwise from +y)',
                units='degrees')
        )
        drift = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Drift_Angle']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft drift angle (clockwise from +y)',
                units='degrees')
        )
        pitch = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Pitch_Angle']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft pitch angle (positive is up)',
                units='degrees')
        )
        roll = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Roll_Angle']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft roll angle (positive is right turn)',
                units='degrees')
        )
        t = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Static_Air_Temp']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Static (ambient) air temperature',
                units='degrees_Celsius')
        )
        t_tot = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Total_Air_Temp']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Total air temperature',
                units='degrees_Celsius')
        )
        td = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Dew_Point']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Dew point temperature',
                units='degrees_Celsius')
        )
        pstat = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Static_Pressure']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Static air pressure',
                units='hPa')
        )
        pcab = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Cabin_Pressure']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Cabin air pressure',
                units='hPa')
        )
        wspd = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Wind_Speed']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Horizontal wind speed (limited to where roll <= 5 degrees)',
                units='m/s')
        )
        wdir = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Wind_Direction']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Horizontal wind direction (clockwise from +y)',
                units='degrees')
        )
        zenith = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Solar_Zenith_Angle']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Solar zenith angle',
                units='degrees')
        )
        sun_elev_ac = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft_Sun_Elevation']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft sun elevation',
                units='degrees')
        )
        sun_az = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Sun_Azimuth']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Sun azimuth',
                units='degrees')
        )
        sun_az_ac = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft_Sun_Azimuth']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft sun azimuth',
                units='degrees')
        )
        if (fmt == 'ames') and ('U' in readfile) and ('V' in readfile): # for 2022 AMES data
            uwnd = xr.DataArray(
                data = np.ma.masked_invalid(readfile['U']), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Horizontal U-component wind speed',
                    units='m/s')
            )
            vwnd = xr.DataArray(
                data = np.ma.masked_invalid(readfile['V']), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Horizontal V-component wind speed',
                    units='m/s')
            )
        else: # if no u, v data
            uwnd = xr.DataArray(
                data = np.ma.array(np.zeros(len(time)), mask=True),
                dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Horizontal U-component wind speed',
                    units='m/s')
            )
            vwnd = xr.DataArray(
                data = np.ma.array(np.zeros(len(time)), mask=True),
                dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Horizontal V-component wind speed',
                    units='m/s')
            )
        if fmt == 'ames': # NASA AMES format
            r = xr.DataArray(
                data = np.ma.masked_invalid(readfile['Mixing_Ratio']), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Mixing ratio',
                    units='g/kg')
            )
            pres_vapor = xr.DataArray(
                data = np.ma.masked_invalid(readfile['Part_Press_Water_Vapor']), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Partial pressure with respect to water vapor',
                    units='hPa')
            )
            es_h2o = xr.DataArray(
                data = np.ma.masked_invalid(readfile['Sat_Vapor_Press_H2O']), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Saturation vapor pressure with respect to water',
                    units='hPa')
            )
            es_ice = xr.DataArray(
                data = np.ma.masked_invalid(readfile['Sat_Vapor_Press_Ice']), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Saturation vapor pressure with respect to ice',
                    units='hPa')
            )
            rh = xr.DataArray(
                data = np.ma.masked_invalid(readfile['Relative_Humidity']), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Relative humidity with respect to water',
                    units='percent')
            )
            pt = xr.DataArray(
                data = np.ma.masked_invalid(readfile['Potential_Temp']), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Potential temperature',
                    units='degrees_Kelvin')
            )
            t_ir = xr.DataArray(
                data = np.ma.masked_invalid(readfile['IR_Surf_Temp']), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Infrared surface temperature',
                    units='degrees_Celsius')
            )
        elif fmt == 'iwg': # IWG1 packets don't have these vars
            r = xr.DataArray(
                data = np.ma.array(np.zeros(len(time)), mask=True), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Mixing ratio',
                    units='g/kg')
            )
            pres_vapor = xr.DataArray(
                data = np.ma.array(np.zeros(len(time)), mask=True), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Partial pressure with respect to water vapor',
                    units='hPa')
            )
            es_h2o = xr.DataArray(
                data = np.ma.array(np.zeros(len(time)), mask=True), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Saturation vapor pressure with respect to water',
                    units='hPa')
            )
            es_ice = xr.DataArray(
                data = np.ma.array(np.zeros(len(time)), mask=True), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Saturation vapor pressure with respect to ice',
                    units='hPa')
            )
            rh = xr.DataArray(
                data = np.ma.array(np.zeros(len(time)), mask=True), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Relative humidity with respect to water',
                    units='percent')
            )
            pt = xr.DataArray(
                data = np.ma.array(np.zeros(len(time)), mask=True), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Potential temperature',
                    units='degrees_Kelvin')
            )
            t_ir = xr.DataArray(
                data = np.ma.array(np.zeros(len(time)), mask=True), dims = ['time'],
                coords = dict(time = time),
                attrs = dict(
                    description='Infrared surface temperature',
                    units='degrees_Celsius')
            )
        
        # put everything together into an XArray Dataset
        ds = xr.Dataset(
            data_vars={
                'lon': lon,
                'lat': lat,
                'alt_gps': alt_gps,
                'alt_pres': alt_pres,
                'alt_radar': alt_radar,
                'grnd_spd': grnd_spd,
                'tas': tas,
                'ias': ias,
                'mach': mach,
                'zvel_P3': vert_vel,
                'heading': heading,
                'track': track,
                'drift': drift,
                'pitch': pitch,
                'roll': roll,
                'temp': t,
                'temp_total': t_tot,
                'temp_ir': t_ir,
                'temp_pot': pt,
                'dwpt': td,
                'pres_static': pstat,
                'pres_cabin': pcab,
                'wspd': wspd,
                'wdir': wdir,
                'uwnd': uwnd,
                'vwnd': vwnd,
                'mixrat': r,
                'pres_vapor': pres_vapor,
                'svp_h2o': es_h2o,
                'svp_ice': es_ice,
                'rh': rh,
                'zenith': zenith,
                'sun_elev_P3': sun_elev_ac,
                'sun_az': sun_az,
                'sun_az_P3': sun_az_ac
            },
            coords={
                'time': time
            },
            attrs=p3_attrs
        )
        
        if fmt == 'iwg': # remove duplicate times (bad data)
            ds = ds.drop_duplicates('time')
        
        # trim the dataset if needed
        if (start_time is not None) or (end_time is not None):
            if start_time is None:
                start_time = str(np.datetime_as_string(ds['time'][0]))
            if end_time is None:
                end_time = str(np.datetime_as_string(ds['time'][-1]))
                
            # remove 1 s from end_time if tres > 1 (for slice() function)
            if pd.to_timedelta(tres) > pd.to_timedelta('1S'):
                end_time = str(
                    np.datetime_as_string(
                        np.datetime64(end_time) - np.timedelta64(1, 's')
                    )
                )
            ds = ds.sel(time=slice(start_time, end_time))
                
                
        # resample (average) the dataset if needed
        if pd.to_timedelta(tres) > pd.to_timedelta('1S'):
            ds = ds.resample(time=tres).mean(skipna=True, keep_attrs=True)
        elif pd.to_timedelta(tres) < pd.to_timedelta('1S'):
            print('Upsampling data is not supported at this time.')
            
        return ds
    
class Instrument(ABC):
    """
    A class to represent most P-3 instruments during the IMPACTS field campaign.
    
    Instrument is an Abstract Base Class - meaning we always require a more specific class 
    to be instantiated - ie you have to call Tamms() or Psd(), you can't just call Instrument()
    Parameters
    ----------
    data : xarray.Dataset()
        Instrument data and attributes
    """
    @abstractmethod     # this stops you from being able to make a new generic instrument
    def __init__(self):
        """
        This is an abstract method since only inherited classes will be used to instantiate Instrument objects.
        """
        self.name = None
        self.data = None
    
    def trim_to_p3(self, p3_object):
        """
        Put the dataset into the same time bounds and frequency as the P-3 Met-Nav data.
        
        Parameters
        ----------
        p3_object: impacts_tools.p3.P3()
            P-3 Met-Nav object to optionally constrain times and average data

        Returns
        -------
        data : xarray.Dataset
            The reindexed dataset
        tres: str
            The time interval/frequency
        """
        
        # P-3 Met-Nav timedelta for tweaking the end time bounds
        td_p3 = pd.to_timedelta(
            p3_object.data['time'][1].values - p3_object.data['time'][0].values
        )
        
        # compute dataset timedelta
        if 'time' in list(self.data.coords): # for 1 Hz datasets
            time_dim = 'time'
        else: # for datasets > 1 Hz frequency (e.g., TAMMS)
            time_dim = 'time_raw'
        td_ds = pd.to_timedelta(
            self.data[time_dim][1].values - self.data[time_dim][0].values
        )
            
        # copy P-3 datetimes and upsample based on datset frequency
        if td_p3 == pd.Timedelta(1, 's'):
            end_time = p3_object.data['time'][-1].values
        else:
            end_time = p3_object.data['time'][-1].values + td_p3 - td_ds
        dt_range = pd.date_range(
            start=p3_object.data['time'][0].values, end=end_time, freq=td_ds
        )
        dummy_times = xr.Dataset(
            coords = {time_dim: dt_range}
        )
        
        return (
            self.data.interp_like(dummy_times),
            pd.tseries.frequencies.to_offset(td_p3).freqstr
        )
    
    def trim_time_bounds(self, start_time=None, end_time=None, tres='1S'):
        """
        Put the dataset into the specified time bounds and frequency.
        
        Parameters
        ----------
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        tres: str
            The time interval to average over (e.g., '5S' for 5 seconds)

        Returns
        -------
        data : xarray.Dataset
            The reindexed dataset
        """
        
        if (start_time is not None) or (end_time is not None):      
            # compute dataset timedelta
            if 'time' in list(self.data.coords): # for 1 Hz datasets
                time_dim = 'time'
            else: # for datasets > 1 Hz frequency (e.g., TAMMS)
                time_dim = 'time_raw'
            td_ds = pd.to_timedelta(
                self.data[time_dim][1].values - self.data[time_dim][0].values
            )
            
            # format start and end times
            if start_time is None:
                start_time = self.data[time_dim][0].values
                
            if end_time is None:
                end_time = self.data[time_dim][-1].values

            # generate upsampled datetime array based on specified frequency
            if pd.Timedelta(tres) != pd.Timedelta(1, 's'):
                end_time -= td_ds
            dummy_times = xr.Dataset(
                coords={
                    time_dim: pd.date_range(
                        start=start_time, end=end_time, freq=td_ds
                    )
                }
            )

            return self.data.interp_like(dummy_times)
        
    def downsample(self, tres='1S'):
        """
        Downsample the time series data according to the specified frequency.
        
        Parameters
        ----------
        freq: pandas.to_timedelta().TimedeltaIndex
            The time interval to average over (e.g., '5S' for 5 seconds)
        """
        if self.name == 'UIOOPS PSD': # special resampling of some variables
            td_ds = pd.to_timedelta(
                self.data['time'][1].values - self.data['time'][0].values
            )
            if pd.to_timedelta(tres) > td_ds: # upsampling not supported
                sum_vars = ['count', 'count_habit', 'sv']
                mean_nan_vars = ['ND', 'ND_habit']
                mean_vars = ['area_ratio', 'aspect_ratio']
                if '2DS' in self.instruments:
                    mean_vars.append('active_time_2ds')
                if 'HVPS' in self.instruments:
                    mean_vars.append('active_time_hvps')
                    
                ds_sum_vars = self.data[sum_vars].resample(time=tres).sum(
                    skipna=True, keep_attrs=True)
                ds_mean_nan_vars = self.data[mean_nan_vars].fillna(0.).resample(time=tres).mean(
                    skipna=True, keep_attrs=True)
                ds_mean_nan_vars = ds_mean_nan_vars.where(ds_mean_nan_vars != 0.)
                ds_mean_vars = self.data[mean_vars].resample(time=tres).mean(
                    skipna=True, keep_attrs=True)
                ds_downsampled = xr.merge(
                    [ds_sum_vars, ds_mean_nan_vars, ds_mean_vars]
                ).transpose('habit', 'size', 'time')
                return ds_downsampled
            else:
                return self.data
        elif self.name == 'SODA PSD': # special resampling of some variables
            td_ds = pd.to_timedelta(
                self.data['time'][1].values - self.data['time'][0].values
            )
            if pd.to_timedelta(tres) > td_ds: # upsampling not supported
                # vars to sum along time dimension (not for Merged files)
                if 'count' in self.data.data_vars:
                    sum_vars = ['count', 'sv']
                    ds_sum_vars = self.data[sum_vars].resample(time=tres).sum(
                        skipna=True, keep_attrs=True)
                
                # vars to average along time dimension
                mean_nan_vars = ['ND']
                mean_vars = ['area_ratio', 'aspect_ratio']
                if '2DS' == self.instruments:
                    mean_vars.append('qc_flag_2ds')
                elif 'HVPS' == self.instruments:
                    mean_vars.append('qc_flag_hvps')
                else: # 2D-S + HVPS
                    if 'qc_flag_2ds' in self.data.data_vars: # two files merged
                        mean_vars.append('qc_flag_2ds')
                        mean_vars.append('qc_flag_hvps')
                    else: # MergedHorizontal or MergedVertical
                        mean_vars.append('qc_flag')
                ds_mean_nan_vars = self.data[mean_nan_vars].fillna(0.).resample(time=tres).mean(
                    skipna=True, keep_attrs=True)
                ds_mean_nan_vars = ds_mean_nan_vars.where(ds_mean_nan_vars != 0.)
                ds_mean_vars = self.data[mean_vars].resample(time=tres).mean(
                    skipna=True, keep_attrs=True)
                
                if 'count' in self.data.data_vars:
                    ds_downsampled = xr.merge(
                        [ds_sum_vars, ds_mean_nan_vars, ds_mean_vars]
                    ).transpose('size', 'time')
                else:
                    ds_downsampled = xr.merge(
                        [ds_mean_nan_vars, ds_mean_vars]
                    ).transpose('size', 'time')
                return ds_downsampled
            else:
                return self.data
        else:
            if 'time' in list(self.data.coords): # for 1 Hz datasets
                td_ds = pd.to_timedelta(
                    self.data['time'][1].values - self.data['time'][0].values
                )
                if pd.to_timedelta(tres) > td_ds: # upsampling not supported
                    return self.data.resample(time=tres).mean(skipna=True, keep_attrs=True)
            else: # for datasets > 1 Hz frequency (e.g., TAMMS)
                sum_vars = []
                ds_downsampled = self.data.resample(
                    time_raw=tres).mean(skipna=True, keep_attrs=True)
                return ds_downsampled # return new dataset (keep original resolution)
    
# ====================================== #
# TAMMS
# ====================================== #
class Tamms(Instrument):
    """
    A class to represent the TAMMS flown on the P-3 during the IMPACTS field campaign.
    Inherits from Instrument()
    
    Parameters
    ----------
    filepath: str
        File path to the TAMMS data file
    p3_object: impacts_tools.p3.P3() object or None
        The optional P-3 Met-Nav object to automatically trim and average the TAMMS data
    start_time: np.datetime64 or None
        The initial time of interest eg. if looking at a single flight leg
    end_time: np.datetime64 or None
        The final time of interest eg. if looking at a single flight leg
    tres: str
        The time interval to average over (e.g., '5S' for 5 seconds)
    """

    def __init__(self, filepath, date, p3_object=None, start_time=None, end_time=None, tres='1S'):
        self.name = 'TAMMS'
        
        # read the raw data
        self.data = self.readfile(filepath, date)
        """
        xarray.Dataset of TAMMS variables and attributes
        Dimensions:
            - time_raw: np.array(np.datetime64[ms]) - The UTC time stamp at the native resolution (20 Hz)
            - time: np.array(np.datetime64[ms]) - The UTC time start of the N-s upsampled interval
        Coordinates:
            - time_raw (time_raw): np.array(np.datetime64[ms]) - The UTC time stamp  at the native resolution (20 Hz)
            - time (time): np.array(np.datetime64[ms]) - The UTC time start of the N-s upsampled interval
        Variables:
            - lat_raw (time_raw): xarray.DataArray(float) - Latitude (degrees)
            - lon_raw (time_raw): xarray.DataArray(float) - Longitude (degrees)
            - alt_gps_raw (time_raw) : xarray.DataArray(float) - Aircraft GPS altitude (m above mean sea level)
            - alt_pres_raw (time_raw) : xarray.DataArray(float) - Aircraft pressure altitude (ft)
            - pitch_raw (time_raw) : xarray.DataArray(float) - Aircraft pitch angle (deg, positive is up)
            - roll_raw (time_raw) : xarray.DataArray(float) - Aircraft roll angle (deg, positive is right turn)
            - temp_raw (time_raw) : xarray.DataArray(float) - Static (ambient) air temperature (deg C)
            - wspd_raw (time_raw) : xarray.DataArray(float) - Horizontal wind speed (m/s)
            - wdir_raw (time_raw) : xarray.DataArray(float) - Horizontal wind direction (deg clockwise from +y)
            - uwnd_raw (time_raw) : xarray.DataArray(float) - Horizontal U-component wind speed (m/s)
            - vwnd_raw (time_raw) : xarray.DataArray(float) - Horizontal V-component wind speed (m/s)
            - wwnd_raw (time_raw) : xarray.DataArray(float) - Vertical component wind speed (m/s)
            - wwnd_std (time) : xarray.DataArray(float) - Standard deviation of the vertical component wind speed (m/s)
            * variables without _raw appended are averaged over the time interval specified
        """
        
        # trim dataset to P-3 time bounds or from specified start/end
        if p3_object is not None:
            self.data, tres = self.trim_to_p3(p3_object)
        elif (start_time is not None) or (end_time is not None):
            self.data = self.trim_time_bounds(start_time, end_time, tres)
            
        # downsample data if specified by the P-3 Met-Nav data or tres argument
        ds_downsampled = self.downsample(tres)
        ds_downsampled = ds_downsampled.rename_dims(
            dims_dict={'time_raw': 'time'}
        )
        name_dict = {'time_raw': 'time'}
        for var in list(ds_downsampled.data_vars):
            name_dict[var] = var.split('_raw')[0]
        ds_downsampled = ds_downsampled.rename(name_dict=name_dict)
        
        # compute the vertical motion standard deviation for downsampled data
        wwnd_std = xr.DataArray(
            data = np.ma.masked_invalid(
                self.data['wwnd_raw'].resample(time_raw=tres).std(
                    skipna=False, keep_attrs=True)
            ),
            dims = ['time'], coords = dict(time = ds_downsampled['time']),
            attrs = dict(
                description='Standard deviation of the vertical component wind speed',
                units='m/s'
            )
        )
        ds_downsampled['wwnd_std'] = wwnd_std
        
        # merge the native (*_raw) and downsampled resolution datasets
        self.data = xr.merge([self.data, ds_downsampled])
        
    def readfile(self, filepath, date):#, p3_object=None, start_time=None, end_time=None, tres='1S'):
        """
        Reads the TAMMS data file and unpacks the fields into an xarray.Dataset

        Parameters
        ----------
        filepath : str
            Path to the data file
        date: str
            Flight start date in YYYY-mm-dd format
        p3_object: impacts_tools.p3.P3() or None
            P-3 Met-Nav object to optionally contrain times and average data
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        tres: str
            The time interval to average over (e.g., '5S' for 5 seconds)

        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """

        # get header info following the NASA AMES format
        header = parse_header(open(filepath, 'r'), date)

        # parse the data
        data_raw = np.genfromtxt(
            filepath, delimiter=',', skip_header=int(header['NLHEAD']),
            missing_values=header['VMISS'], usemask=True, filling_values=np.nan
        )

        # construct dictionary of variable data and metadata
        readfile = {}
        for jj, unit in enumerate(header['VUNIT']):
            header['VUNIT'][jj] = unit.split(',')[0]
        for jj, name in enumerate(header['VNAME']):
            readfile[name] = np.array(data_raw[:, jj] * header['VSCAL'][jj])
            readfile[name][readfile[name]==header['VMISS'][jj]] = np.nan

        # populate dataset attributes
        p3_attrs = {
            'Experiment': 'IMPACTS',
            'Platform': 'P-3',
            'Mission PI': 'Lynn McMurdie (lynnm@uw.edu)'}
        instrum_info_counter = 1
        for ii, comment in enumerate(header['NCOM'][:-1]): # add global attrs
            parsed_comment = comment.split(':')
            if len(parsed_comment) > 1:
                p3_attrs[parsed_comment[0]] = parsed_comment[1][1:]
            else: # handles multiple instrument info lines in *_R0.ict files
                instrum_info_counter += 1
                p3_attrs[
                    'INSTRUMENT_INFO_'+str(instrum_info_counter)] = parsed_comment[0][1:]

        # compute time
        sec_frac, sec = np.modf(readfile['time'])
        time = np.array([
            np.datetime64(date) + np.timedelta64(int(sec[i]), 's') +
            np.timedelta64(int(np.round(1000. * sec_frac[i])), 'ms')
            for i in range(len(readfile['time']))], dtype='datetime64[ms]'
        )

        # populate data arrays
        lat = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Latitude_deg']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Aircraft latitude',
                units='degrees_north')
        )
        lon = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Longitude_deg']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Aircraft longitude',
                units='degrees_east')
        )
        alt_gps = xr.DataArray(
            data = np.ma.masked_invalid(readfile['GPS_alt_m']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Aircraft GPS altitude (mean sea level)',
                units='meters')
        )
        alt_pres = xr.DataArray(
            data = np.ma.masked_invalid(readfile['PALT_ft']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Aircraft pressure altitude',
                units='feet')
        )
        pitch = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Pitch_deg']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Aircraft pitch angle (positive is up)',
                units='degrees')
        )
        roll = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Roll_deg']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Aircraft roll angle (positive is right turn)',
                units='degrees')
        )
        t = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Tstat_degC']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Static (ambient) air temperature',
                units='degrees_Celsius')
        )
        wspd = xr.DataArray(
            data = np.ma.masked_invalid(readfile['WSPD_ms-1']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Horizontal wind speed',
                units='m/s')
        )
        wdir = xr.DataArray(
            data = np.ma.masked_invalid(readfile['WDIR_deg']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Horizontal wind direction (clockwise from +y)',
                units='degrees')
        )
        wwnd = xr.DataArray(
            data = np.ma.masked_invalid(readfile['w_ms-1']),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Vertical component wind speed',
                units='m/s')
        )
        wdir_math = wdir - 270. # convert to math-relative dirction
        wdir_math[wdir_math < 0.] += 360. # fix negative values
        uwnd = xr.DataArray(
            data = np.ma.masked_invalid(wspd * np.cos(np.deg2rad(wdir_math))),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Horizontal U-component wind speed',
                units='m/s')
        )
        vwnd = xr.DataArray(
            data = np.ma.masked_invalid(wspd * np.sin(np.deg2rad(wdir_math))),
            dims = ['time_raw'],
            coords = dict(time_raw = time),
            attrs = dict(
                description='Horizontal V-component wind speed',
                units='m/s')
        )

        # put everything together into an XArray Dataset
        ds = xr.Dataset(
            data_vars={
                'lon_raw': lon,
                'lat_raw': lat,
                'alt_gps_raw': alt_gps,
                'alt_pres_raw': alt_pres,
                'pitch_raw': pitch,
                'roll_raw': roll,
                'temp_raw': t,
                'wspd_raw': wspd,
                'wdir_raw': wdir,
                'uwnd_raw': uwnd,
                'vwnd_raw': vwnd,
                'wwnd_raw': wwnd
            },
            coords={
                'time_raw': time
            },
            attrs=p3_attrs
        )

        return ds

#
class Und(Instrument):
    """
    A class to represent the UND instruments summary on the P-3 during the IMPACTS field campaign.
    Inherits from Instrument()
    
    Parameters
    ----------
    filepath: str
        File path to the TAMMS data file
    p3_object: impacts_tools.p3.P3() object or None
        The optional P-3 Met-Nav object to automatically trim and average the TAMMS data
    start_time: np.datetime64 or None
        The initial time of interest eg. if looking at a single flight leg
    end_time: np.datetime64 or None
        The final time of interest eg. if looking at a single flight leg
    tres: str
        The time interval to average over (e.g., '5S' for 5 seconds)
    """

    def __init__(self, filepath, date, p3_object=None, start_time=None, end_time=None, tres='1S'):
        self.name = 'UND Probes Summary'
        
        # read the raw data
        self.data = self.readfile(filepath, date)
        """
        xarray.Dataset of UND summary variables and attributes
        Dimensions:
            - time: np.array(np.datetime64[ms]) - The UTC time start of the N-s upsampled interval
        Coordinates:
            - time (time): np.array(np.datetime64[ms]) - The UTC time start of the N-s upsampled interval
        Variables:
            - wwnd_std (time) : xarray.DataArray(float) - Standard deviation of the vertical component wind speed (m/s)
            * variables without _raw appended are averaged over the time interval specified
        """
        
        # trim dataset to P-3 time bounds or from specified start/end
        if p3_object is not None:
            self.data, tres = self.trim_to_p3(p3_object)
        elif (start_time is not None) or (end_time is not None):
            self.data = self.trim_time_bounds(start_time, end_time, tres)
            
        # downsample data if specified by the P-3 Met-Nav data or tres argument
        self.data = self.downsample(tres)
        
    def readfile(self, filepath, date):
        """
        Reads the UND summary data file and unpacks the fields into an xarray.Dataset

        Parameters
        ----------
        filepath : str
            Path to the data file
        date: str
            Flight start date in YYYY-mm-dd format
        p3_object: impacts_tools.p3.P3() or None
            P-3 Met-Nav object to optionally contrain times and average data
        start_time : np.datetime64 or None
            The initial time of interest
        end_time : np.datetime64 or None
            The final time of interest
        tres: str
            The time interval to average over (e.g., '5S' for 5 seconds)

        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """

        # get header info following the NASA AMES format
        header = parse_header(
            open(filepath, 'r', encoding = 'ISO-8859-1'), date, stream='und'
        )

        # parse the data
        data_raw = np.genfromtxt(
            filepath, skip_header=int(header['NLHEAD']),
            missing_values=header['VMISS'], usemask=True, filling_values=np.nan
        )

        # construct dictionary of variable data and metadata
        readfile = {}
        for jj, unit in enumerate(header['VUNIT']):
            header['VUNIT'][jj] = unit.split(',')[0]
        for jj, name in enumerate(header['VNAME']):
            readfile[name] = np.array(data_raw[:, jj] * header['VSCAL'][jj])
            readfile[name][readfile[name]==header['VMISS'][jj]] = np.nan

        # populate dataset attributes
        p3_attrs = {
            'Experiment': 'IMPACTS',
            'Platform': 'P-3',
            'Mission PI': 'Lynn McMurdie (lynnm@uw.edu)'}
        instrum_info_counter = 1
        for ii, comment in enumerate(header['NCOM'][:-1]): # add global attrs
            parsed_comment = comment.split(':')
            if len(parsed_comment) > 1:
                p3_attrs[parsed_comment[0]] = parsed_comment[1][1:]
            else: # handles multiple instrument info lines in *_R0.ict files
                instrum_info_counter += 1
                p3_attrs[
                    'INSTRUMENT_INFO_'+str(instrum_info_counter)] = parsed_comment[0][1:]

        # compute time
        sec_frac, sec = np.modf(readfile['time'])
        time = np.array([
            np.datetime64(date) + np.timedelta64(int(sec[i]), 's') +
            np.timedelta64(int(np.round(1000. * sec_frac[i])), 'ms')
            for i in range(len(readfile['time']))], dtype='datetime64[ms]'
        )

        # populate data arrays
        temp = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft ambient temperature']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft ambient temperature',
                units = 'degrees_Celsius'
            )
        )
        spress = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft static pressure']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft static pressure',
                units = 'hPa'
            )
        )
        td = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft dew point temperature']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft dew point temperature',
                units = 'degrees_Celsius'
            )
        )
        tas = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft True Air Speed']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft true air speed',
                units = 'm/s'
            )
        )
        palt = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Pressure Altitude']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Pressure altitude',
                units = 'm'
            )
        )
        track = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft Track Angle']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft track angle',
                units = 'degrees'
            )
        )
        ias = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft Indicated Air Speed']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft indicated air speed',
                units = 'm/s'
            )
        )
        grdspd = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft Ground Speed']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft ground speed',
                units = 'm/s'
            )
        )
        palt_ac = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft Presure Altitude']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft pressure altitude',
                units = 'feet'
            )
        )
        alt = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft GPS Altitude MSL']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft GPS altitude (above mean sea level)',
                units = 'm'
            )
        )
        wspd = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft wind speed']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft wind speed',
                units = 'm/s'
            )
        )
        wdir = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft wind direction']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft wind direction',
                units = 'degrees'
            )
        )
        roll = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft Roll Angle']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft roll angle',
                units = 'degrees'
            )
        )
        pitch = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft Pitch Angle']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft pitch angle',
                units = 'degrees'
            )
        )
        head = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft True Heading']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft true heading',
                units = 'degrees'
            )
        )
        w = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft Vertical Velocity']),
            dims = 'time',
            coords = dict(time=time),
            attrs = dict(
                description = 'Aircraft vertical velocity',
                units = 'm/s'
            )
        )
        lat = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft latitude']),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft latitude',
                units='degrees_north')
        )
        lon = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Aircraft longitude']),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Aircraft longitude',
                units='degrees_east')
        )
        lwc_king = xr.DataArray(
            data = np.ma.masked_invalid(
                readfile['Liquid Water Content based on King Probe measurement adjusted']
            ),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Liquid Water Content based on King Probe measurement (adjusted)',
                baseline_threshold = '5.1 cm-3',
                baseline_interval = '30 s',
                baseline_slope = '1.000',
                units='g m-3')
        )
        lwc_cdp = xr.DataArray(
            data = np.ma.masked_invalid(
                readfile['Liquid Water Content Based on the Cloud Droplet Probe']
            ),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Liquid Water Content based on the Cloud Droplet Probe',
                units='g m-3')
        )
        mdd_cdp = xr.DataArray(
            data = np.ma.masked_invalid(
                readfile["Cloud Droplet Probe's Mean Droplet Diameter"]
            ),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Cloud Droplet Probe mean droplet diameter',
                units='um')
        )
        re_cdp = xr.DataArray(
            data = np.ma.masked_invalid(
                readfile["Cloud Droplet Probe's Effective Droplet Radius"]
            ),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Cloud Droplet Probe effective droplet radius',
                units='um')
        )
        n_cdp = xr.DataArray(
            data = np.ma.masked_invalid(
                readfile['Number Concentration of Droplets Based on the Cloud Droplet Probe']
            ),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Number concentration of droplets based on the Cloud Droplet Probe',
                units='cm-3')
        )
        ricefreq = xr.DataArray(
            data = np.ma.masked_invalid(readfile['The current Sensor']),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Frequency from the Rosemont Icing Detector',
                units='Hz')
        )
        twc_wcm = xr.DataArray(
            data = np.ma.masked_invalid(
                readfile['Total Water Content based on the WCM Probe measurement adjusted for baseline offset']
            ),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Total Water Content based on the WCM probe measurement (adjusted)',
                units='g m-3')
        )
        lwc_wcm = xr.DataArray(
            data = np.ma.masked_invalid(
                readfile['Liquid Water Content element 083 based on the WCM Probe measurement adjusted for baseline offset']
            ),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description='Liquid Water Content based on the WCM probe measurement (adjusted)',
                units='g m-3')
        )
        iwc_wcm = xr.DataArray(
            data = np.ma.masked_invalid(
                readfile[
                    'Ice Water Content based on the WCM Probe measurement calculated '
                    'with sampling efficiencies and adjusted for baseline offset'
                ]
            ),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description = (
                    'Ice Water Content based on the WCM probe measurement '
                    '(calculated with sampling efficiencies and adjusted for baseline offset)'
                ),
                units='g m-3')
        )
        flag_cld = xr.DataArray(
            data = np.ma.masked_invalid(
                readfile['Parameter to determine if the WCM Probe is in cloud or not based on CIP']
            ),
            dims = 'time',
            coords = dict(time = time),
            attrs = dict(
                description = (
                    'Parameter to determine if the WCM probe is in cloud or not '
                    '(0: out of cloud; 1 - in cloud)'
                ),
                units='-')
        )

        # put everything together into an XArray Dataset
        ds = xr.Dataset(
            data_vars={
                'lon': lon,
                'lat': lat,
                'alt_gps': alt,
                'alt_pres': palt_ac,
                'grnd_spd': grdspd,
                'tas': tas,
                'ias': ias,
                'zvel': w,
                'heading': head,
                'track': track,
                'pitch': pitch,
                'roll': roll,
                'temp': temp,
                'dwpt': td,
                'pres_static': spress,
                'wspd': wspd,
                'wdir': wdir,
                'lwc_king': lwc_king,
                'lwc_cdp': lwc_cdp,
                'mdd_cdp': mdd_cdp,
                're_cdp': re_cdp,
                'n_cdp': n_cdp,
                'freq_rice': ricefreq,
                'twc_wcm': twc_wcm,
                'lwc_wcm': lwc_wcm,
                'iwc_wcm': iwc_wcm,
                'flag_cloud': flag_cld
            },
            coords={
                'time': time
            },
            attrs=p3_attrs
        )

        return ds
        
# ====================================== #
# PSDs
# ====================================== #
class Psd(Instrument):
    """
    A class to represent the PSDs from optical array probes flown on the P-3 during the IMPACTS field campaign.
    Inherits from Instrument()
    
    Parameters
    ----------
    filepath_2ds: str or None
        File path to the 2D-S/Hawkeye 2D-S PSD data file
    filepath_hvps: str or None
        File path to the HVPS PSD data file
    p3_object: impacts_tools.p3.P3() object or None
        The optional P-3 Met-Nav object to automatically trim and average the PSD data
    start_time: np.datetime64 or None
        The initial time of interest eg. if looking at a single flight leg
    end_time: np.datetime64 or None
        The final time of interest eg. if looking at a single flight leg
    tres: str
        The time interval to average over (e.g., '5S' for 5 seconds)
    software: str
        Processing software ('uioops' or 'soda') for file-specific reading of data
    binlims: tuple of float
        2- or 3-element tuple representing size limits (mm) for the 2D-S and/or HVPS
    ovld_thresh: float (0. - 1.) or None
        Proportion of 1-s interval of allowable dead time (overload) for the PSD
    calc_bulk: bool
        Compute bulk properties (only if True)
    calc_gamma_params: bool
        Calculate gamma fit parameters using DIGF technique
    """

    def __init__(
            self, filepath_2ds, filepath_hvps, date, p3_object=None,
            start_time=None, end_time=None, tres='1S',
            software='UIOOPS', binlims=(0.1, 1.4, 30.), ovld_thresh=0.7,
            calc_bulk=False, calc_gamma_params=False):
        self.name = f'{software} PSD'
        if (filepath_2ds is not None) and (filepath_hvps is not None):
            self.instruments = '2DS and HVPS'
        elif filepath_2ds is not None:
            self.instruments = '2DS'
        elif filepath_hvps is not None:
            self.instruments = 'HVPS'
        
        # read the raw data
        if software == 'UIOOPS':
            self.data = self.readfile_uioops(
                filepath_2ds, filepath_hvps, date, binlims, ovld_thresh
            )
            """
            xarray.Dataset of UIOOPS PSD variables and attributes
            Dimensions:
                - time: np.array(np.datetime64[ns]) - The UTC time start of the N-s upsampled interval
            Coordinates:
                - habit (habit): np.array(dtype=char) - Habits from the Holroyd classification scheme
                - bin_center (size): np.array(np.float64) - Size bin midpoint (mm)
                - bin_left (size): np.array(np.float64) - Size bin left endpoint (mm)
                - bin_right (size): np.array(np.float64) - Size bin right endpoint (mm)
                - bin_width (size): np.array(np.float64) - Size bin width (cm)
                - time (time): np.array(np.datetime64[ns]) - The UTC time start of the N-s upsampled interval
            Variables:
                - count (size, time): xarray.DataArray(float) - Particle count per size bin (#)
                - count_habit (habit, size, time): xarray.DataArray(float) - Particle count per habit category and size bin (#)
                - sv (size, time): xarray.DataArray(float) - Sample volume (cm3)
                - ND (size, time): xarray.DataArray(float) - Number distribution function (cm-4)
                - ND_habit (habit, size, time): xarray.DataArray(float) - Number distribution function per habit category(cm-4)
                - area_ratio (size, time): xarray.DataArray(float) - Mean area ratio per size bin (#)
                - aspect_ratio (size, time): xarray.DataArray(float) - Mean aspect ratio per size bin (#)
                - active_time_<probe> (size, time): xarray.DataArray(float) - Probe active time (s)
                - n (time): xarray.DataArray(float) - Number concentration (L-1)
                - iwc_<mD> (time): xarray.DataArray(float) - Ice water content using specified mass-dimension (m-D) relationship (g m-3)
                - dm_<mD> (time): xarray.DataArray(float) - Mass-weighted mean diam using specified m-D relationship (mm)
                - dmm_<mD> (time): xarray.DataArray(float) - Median mass diam using specified m-D relationship (mm)
                - rhoe_<mD> (time): xarray.DataArray(float) - Effective density using specified m-D relationship (g cm-3)
                - area_ratio_mean_<mD> (time): xarray.DataArray(float) - Mean area ratio using number or mass (m-D) weighting (#)
                - aspect_ratio_mean_<mD> (time): xarray.DataArray(float) - Mean aspect ratio using number or mass (m-D) weighting (#)
            """
        elif software == 'SODA':
            self.data = self.readfile_soda(
                filepath_2ds, filepath_hvps, date, binlims, ovld_thresh
            )
            """
            xarray.Dataset of UIOOPS PSD variables and attributes
            Dimensions:
                - time: np.array(np.datetime64[ns]) - The UTC time start of the N-s upsampled interval
            Coordinates:
                - bin_center (size): np.array(np.float64) - Size bin midpoint (mm)
                - bin_left (size): np.array(np.float64) - Size bin left endpoint (mm)
                - bin_right (size): np.array(np.float64) - Size bin right endpoint (mm)
                - bin_width (size): np.array(np.float64) - Size bin width (cm)
                - time (time): np.array(np.datetime64[ns]) - The UTC time start of the N-s upsampled interval
            Variables:
                - count (size, time): xarray.DataArray(float) - Particle count per size bin (#)
                - sv (size, time): xarray.DataArray(float) - Sample volume (cm3)
                - ND (size, time): xarray.DataArray(float) - Number distribution function (cm-4)
                - area_ratio (size, time): xarray.DataArray(float) - Mean area ratio per size bin (#)
                - aspect_ratio (size, time): xarray.DataArray(float) - Mean aspect ratio per size bin (#)
                - qc_flag_<probe> (size, time): xarray.DataArray(float) - Probe quality flag (0: good, 1: medium, 2: bad)
                - n (time): xarray.DataArray(float) - Number concentration (L-1)
                - iwc_<mD> (time): xarray.DataArray(float) - Ice water content using specified mass-dimension (m-D) relationship (g m-3)
                - dm_<mD> (time): xarray.DataArray(float) - Mass-weighted mean diam using specified m-D relationship (mm)
                - dmm_<mD> (time): xarray.DataArray(float) - Median mass diam using specified m-D relationship (mm)
                - rhoe_<mD> (time): xarray.DataArray(float) - Effective density using specified m-D relationship (g cm-3)
                - area_ratio_mean_<mD> (time): xarray.DataArray(float) - Mean area ratio using number or mass (m-D) weighting (#)
                - aspect_ratio_mean_<mD> (time): xarray.DataArray(float) - Mean aspect ratio using number or mass (m-D) weighting (#)
            """
        
        # trim dataset to P-3 time bounds or from specified start/end
        if p3_object is not None:
            self.data, tres = self.trim_to_p3(p3_object)
        elif (start_time is not None) or (end_time is not None):
            self.data = self.trim_time_bounds(start_time, end_time, tres)

        # downsample data if specified by the P-3 Met-Nav data or tres argument
        self.data = self.downsample(tres)
        
        # compute bulk properties
        if calc_bulk:
            self.data = self.bulk_properties(calc_gamma_params)
        
    def readfile_uioops(
            self, filepath_2ds, filepath_hvps, date,
            binlims=(0.1, 1.4, 30.), ovld_thresh=0.7):
        
        # initialize dataset list to accomodate 2 probe PSDs
        ds_list = []
        
        # load the datasets if available
        for (probe, file) in zip(['2ds', 'hvps'], [filepath_2ds, filepath_hvps]):
            if file is not None:
                data = xr.open_dataset(file)
                time = self.hhmmss2dt(data['time'], date)
                bin_min = data['bin_min'].values
                bin_max = data['bin_max'].values
                bin_width = data['bin_dD'].values / 10. # (cm)
                bin_mid = data['bin_mid'].values
                with np.errstate(divide='ignore', invalid='ignore'):
                    count_temp = data['count'].values.T
                    sv_temp = data['sample_vol'].values.T
                    count_habit_raw = (
                        data['habitsd'].values.T) * np.tile(
                        np.moveaxis(np.atleast_3d(sv_temp), -1, 0), (10, 1, 1)) * np.tile(
                        np.atleast_3d(bin_width), (10, 1, sv_temp.shape[1]))
                    count_habit_temp = np.array([
                        count_habit_raw[3, :, :], count_habit_raw[0, :, :],
                        np.nansum(count_habit_raw[1:3, :, :], axis=0),
                        count_habit_raw[4, :, :], count_habit_raw[5, :, :],
                        count_habit_raw[6, :, :], count_habit_raw[7, :, :],
                        count_habit_raw[8, :, :]])
                    ND_temp = np.ma.masked_where(
                        count_temp==0.,
                        count_temp / sv_temp / np.tile(np.atleast_2d(bin_width).T,
                                             (1, count_temp.shape[1])))
                    ND_habit_temp = np.ma.masked_where(
                        count_habit_temp==0.,
                        count_habit_temp / sv_temp /
                        np.tile(np.atleast_3d(bin_width).T,
                                (count_habit_temp.shape[0], 1,
                                 count_habit_temp.shape[2])
                               )
                    )
                    ar_temp = data['mean_area_ratio'].values.T
                    asr_temp = data['mean_aspect_ratio_ellipse'].values.T
                    active_time_temp = data['sum_IntArr'].values
                
                # establish the data arrays
                habits = [
                    'Tiny', 'Spherical', 'Linear', 'Hexagonal', 'Irregular',
                    'Graupel', 'Dendrite', 'Aggregate']
                bin_mid = xr.DataArray(
                    data=bin_mid, dims = 'size',
                    attrs = dict(
                        description='Particle size bin midpoint',
                        units = 'mm')
                )
                bin_min = xr.DataArray(
                    data=bin_min, dims = 'size',
                    attrs = dict(
                        description='Particle size bin left endpoint',
                        units = 'mm')
                )
                bin_max = xr.DataArray(
                    data=bin_max, dims = 'size',
                    attrs = dict(
                        description='Particle size bin right endpoint',
                        units = 'mm')
                )
                bin_width = xr.DataArray(
                    data=bin_width, dims = 'size',
                    attrs = dict(
                        description='Particle size bin width',
                        units = 'cm')
                )
                count = xr.DataArray(
                    data = count_temp,
                    dims = ['size', 'time'],
                    coords = dict(
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Particle count per size bin',
                        units = '#')
                )
                count_hab = xr.DataArray(
                    data = count_habit_temp,
                    dims = ['habit', 'size', 'time'],
                    coords = dict(
                        habit=habits,
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Particle count per habit category and size bin',
                        units = '#')
                )
                sv = xr.DataArray(
                    data = sv_temp,
                    dims = ['size', 'time'],
                    coords = dict(
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Sample volume per bin over the time interval',
                        units = 'cm3')
                )
                ND = xr.DataArray(
                    data = ND_temp,
                    dims = ['size', 'time'],
                    coords = dict(
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Number distribution function (PSD)',
                        units = 'cm-4')
                )
                ND_hab = xr.DataArray(
                    data = ND_habit_temp,
                    dims = ['habit', 'size', 'time'],
                    coords = dict(
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Number distribution function (PSD) per habit category',
                        units = 'cm-4')
                )
                ar = xr.DataArray(
                    data = ar_temp,
                    dims = ['size', 'time'],
                    coords = dict(
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Mean area ratio (circular fit) per bin',
                        units = '#')
                )
                asr = xr.DataArray(
                    data = asr_temp,
                    dims = ['size', 'time'],
                    coords = dict(
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Mean aspect ratio (elliptical fit) per bin',
                        units = '#')
                )
                at = xr.DataArray(
                    data = active_time_temp,
                    dims = 'time',
                    coords = dict(time=time),
                    attrs = dict(
                        description='Probe active time',
                        deadtime_threshold = f'{ovld_thresh}',
                        units = 's')
                )
                
                # put everything together into an XArray DataSet
                ds = xr.Dataset(
                    data_vars={
                        'count': count,
                        'count_habit': count_hab,
                        'sv': sv,
                        'ND': ND,
                        'ND_habit': ND_hab,
                        'area_ratio': ar,
                        'aspect_ratio': asr,
                        'active_time': at
                    },
                    coords={
                        'habit': habits,
                        'bin_center': bin_mid,
                        'bin_left': bin_min,
                        'bin_right': bin_max,
                        'bin_width': bin_width,
                        'time': time
                    },
                    attrs={
                        'Experiment': 'IMPACTS',
                        'Date': date,
                        'Aircraft': 'P-3',
                        'Data Contact': 'Joseph Finlon (jfinlon@uw.edu)',
                        'Instrument PI': 'David Delene (david.delene@und.edu)',
                        'Mission PI': 'Lynn McMurdie (lynnm@uw.edu)',
                        'L3A Software': data.attrs['Software'],
                        'L3A Revision': data.attrs['Data Version'],
                    }
                )
                
                # trim based on probe size limits
                if probe == '2ds':
                    ds = ds.sel(
                        size=(ds.bin_left >= binlims[0]) & (ds.bin_left < binlims[1])
                    )
                else:
                    ds = ds.sel(
                        size=(ds.bin_left >= binlims[-2]) & (ds.bin_left < binlims[-1])
                    )
                ds_list.append(ds)
                
        # concatonate probe PSDs if applicable
        if len(ds_list) == 1: # no need to merge PSDs
            ds_merged = ds_list[0].drop_vars('active_time')
        else:
            ds_merged = xr.concat(
                [ds_list[0].drop_vars('active_time'),
                 ds_list[1].drop_vars('active_time')
                ], dim='size'
            ) # concatenate, drop active_time for now
            #ds_merged['active_time_2ds'] = ds_list[0]['active_time']
            #ds_merged['active_time_hvps'] = ds_list[1]['active_time']

        # mask periods when dead time exceeds the specified threshold (optional)
        if (ovld_thresh is not None) and (len(ds_list) > 0):
            if len(ds_list) == 1: # only one probe available
                good_times = ((ds_list[0]['active_time'] >= 1. - ovld_thresh))
            else: # both 2D-S and HVPS available, need good data from both probes
                good_times = (
                    (ds_list[0]['active_time'] >= 1. - ovld_thresh) & # 2D-S
                    (ds_list[1]['active_time'] >= 1. - ovld_thresh) # HVPS
                )
            ds_merged = ds_merged.where(good_times) # values for bad times become nan

        # add probe active time to dataset
        if (len(ds_list) == 1) and (filepath_2ds is not None): # 2D-S only
            ds_merged['active_time_2ds'] = ds_list[0]['active_time']
        elif (len(ds_list) == 1) and (filepath_hvps is not None): # HVPS only
            ds_merged['active_time_hvps'] = ds_list[0]['active_time']
        else: # both 2D-S and HVPS available, add the probe active time for each
            ds_merged['active_time_2ds'] = ds_list[0]['active_time']
            ds_merged['active_time_hvps'] = ds_list[1]['active_time']
                        
        return ds_merged
    
    def readfile_soda(
            self, filepath_2ds, filepath_hvps, date,
            binlims=(0.1, 1.4, 30.), qc_thresh=1):
        # initialize dataset list to accomodate 2 probe PSDs
        ds_list = []

        # load the datasets if available
        for (probe, file) in zip(['2ds', 'hvps'], [filepath_2ds, filepath_hvps]):
            if (file is not None) and (
                    (probe == '2ds') or (filepath_2ds != filepath_hvps)):
                data = xr.open_dataset(file)
                time = np.array([
                    np.datetime64(
                        datetime.strptime(data.attrs['FlightDate'], '%m/%d/%Y')
                    ) + np.timedelta64(
                        int(data['time'].values[i]), 's')
                    for i in range(len(data['time']))
                ], dtype='datetime64[s]')
                bin_min = data['CONCENTRATION'].attrs['bin_endpoints'][:-1] / 1000.
                bin_max = data['CONCENTRATION'].attrs['bin_endpoints'][1:] / 1000.
                bin_width = (bin_max - bin_min) / 10. # (cm)
                bin_mid = data['CONCENTRATION'].attrs['bin_midpoints'] / 1000.
                with np.errstate(divide='ignore', invalid='ignore'):
                    if 'COUNTS' in data.data_vars: # only in 2D-S and HVPS files
                        dD_2d = np.tile(bin_width[:, np.newaxis], (1, len(time)))
                        count_temp = data['COUNTS'].values
                        ND_temp = (10. ** -8) * np.ma.masked_where(
                            count_temp == 0., data['CONCENTRATION'].values
                        )
                        sv_temp = count_temp / ND_temp / dD_2d
                    else: # MergedHorizontal and MergedVertical files
                        ND_temp = (10. ** -8) * np.ma.masked_where(
                            data['CONCENTRATION'].values == 0.,
                            data['CONCENTRATION'].values
                        )
                    ar_temp = data['MEAN_AREARATIO'].values
                    asr_temp = data['MEAN_ASPECTRATIO'].values
                    qc_flag = data['PROBE_QC'].values
                
                # establish the data arrays
                bin_mid = xr.DataArray(
                    data=bin_mid, dims = 'size',
                    attrs = dict(
                        description='Particle size bin midpoint',
                        units = 'mm')
                )
                bin_min = xr.DataArray(
                    data=bin_min, dims = 'size',
                    attrs = dict(
                        description='Particle size bin left endpoint',
                        units = 'mm')
                )
                bin_max = xr.DataArray(
                    data=bin_max, dims = 'size',
                    attrs = dict(
                        description='Particle size bin right endpoint',
                        units = 'mm')
                )
                bin_width = xr.DataArray(
                    data=bin_width, dims = 'size',
                    attrs = dict(
                        description='Particle size bin width',
                        units = 'cm')
                )
                if 'COUNTS' in data.data_vars: # only in 2D-S and HVPS files
                    count = xr.DataArray(
                        data = count_temp,
                        dims = ['size', 'time'],
                        coords = dict(
                            bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                            bin_width=bin_width, time=time),
                        attrs = dict(
                            description='Particle count per size bin',
                            units = '#')
                    )
                    sv = xr.DataArray(
                        data = sv_temp,
                        dims = ['size', 'time'],
                        coords = dict(
                            bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                            bin_width=bin_width, time=time),
                        attrs = dict(
                            description='Sample volume per bin over the time interval',
                            units = 'cm3')
                    )
                ND = xr.DataArray(
                    data = ND_temp,
                    dims = ['size', 'time'],
                    coords = dict(
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Number distribution function (PSD)',
                        units = 'cm-4')
                )
                ar = xr.DataArray(
                    data = ar_temp,
                    dims = ['size', 'time'],
                    coords = dict(
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Mean area ratio per bin',
                        units = '#')
                )
                asr = xr.DataArray(
                    data = asr_temp,
                    dims = ['size', 'time'],
                    coords = dict(
                        bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                        bin_width=bin_width, time=time),
                    attrs = dict(
                        description='Mean aspect ratio per bin',
                        units = '#')
                )
                qc = xr.DataArray(
                    data = qc_flag,
                    dims = 'time',
                    coords = dict(time=time),
                    attrs = dict(
                        description='Probe quality flag (0: good, 1: medium, 2: bad)',
                        qc_threshold = f'{qc_thresh}',
                        units = '#')
                )
                
                # put everything together into an XArray DataSet
                if 'COUNTS' in data.data_vars: # only in 2D-S and HVPS files
                    data_vars = {
                        'count': count,
                        'sv': sv,
                        'ND': ND,
                        'area_ratio': ar,
                        'aspect_ratio': asr,
                        'qc_flag': qc
                    }
                else:
                    data_vars = {
                        'ND': ND,
                        'area_ratio': ar,
                        'aspect_ratio': asr,
                        'qc_flag': qc
                    }
                ds = xr.Dataset(
                    data_vars = data_vars,
                    coords = {
                        'bin_center': bin_mid,
                        'bin_left': bin_min,
                        'bin_right': bin_max,
                        'bin_width': bin_width,
                        'time': time
                    },
                    attrs = {
                        'Experiment': 'IMPACTS',
                        'Date': date,
                        'Aircraft': 'P-3',
                        'Data Contact': 'Aaron Bansemer (bansemer@ucar.edu)',
                        'Instrument PI': 'David Delene (david.delene@und.edu)',
                        'Mission PI': 'Lynn McMurdie (lynnm@uw.edu)',
                        'L3A Software': data.attrs['Source'],
                    }
                )
                
                # trim based on probe size limits
                if (probe == '2ds') and (filepath_2ds != filepath_hvps):
                    ds = ds.sel(
                        size=(ds.bin_left >= binlims[0]) & (ds.bin_left < binlims[-2])
                    )
                elif (probe == '2ds') and (filepath_2ds == filepath_hvps): # Merged file
                    ds = ds.sel(
                        size=(ds.bin_left >= binlims[0]) & (ds.bin_left < binlims[-1])
                    )
                else: # HVPS file
                    ds = ds.sel(
                        size=(ds.bin_left >= binlims[1]) & (ds.bin_left < binlims[-1])
                    )
                ds_list.append(ds)
                
        # concatonate probe PSDs if applicable
        if len(ds_list) == 1: # no need to merge PSDs
            ds_merged = ds_list[0].drop_vars('qc_flag')
        else:
            if len(binlims) < 4:
                ds_merged = xr.concat(
                    [ds_list[0].drop_vars('qc_flag'),
                     ds_list[1].drop_vars('qc_flag')
                    ], dim='size'
                ) # concatenate, drop qc_flag for now
            else: # blend probes in transition region
                ds_merged = self.weight_psd(
                    ds_list[0], ds_list[1], qc_thresh, binlims)

        # mask periods when qc flag meets/exceeds the specified threshold (optional)
        if (qc_thresh is not None) and (len(ds_list) > 0):
            if len(ds_list) == 1: # only one probe available
                good_times = ((ds_list[0]['qc_flag'] <= qc_thresh))
            else: # both 2D-S and HVPS available, need good data from both probes
                good_times = (
                    (ds_list[0]['qc_flag'] <= qc_thresh) & # 2D-S
                    (ds_list[1]['qc_flag'] <= qc_thresh) # HVPS
                )
            ds_merged = ds_merged.where(good_times) # values for bad times become nan

        # add probe qc flag to dataset
        if (len(ds_list) == 1) and (filepath_2ds is not None):
            if filepath_2ds != filepath_hvps: # 2D-S only
                ds_merged['qc_flag_2ds'] = ds_list[0]['qc_flag']
            else: # MergedHorizontal or MergedVertical
                ds_merged['qc_flag'] = ds_list[0]['qc_flag']
        elif (len(ds_list) == 1) and (filepath_hvps is not None): # HVPS only
            ds_merged['qc_flag_hvps'] = ds_list[0]['qc_flag']
        else: # both 2D-S and HVPS available, add the qc flag for each
            ds_merged['qc_flag_2ds'] = ds_list[0]['qc_flag']
            ds_merged['qc_flag_hvps'] = ds_list[1]['qc_flag']
                        
        return ds_merged
    
    def weight_psd(self, psd_2ds, psd_hvps, qc_thresh, binlims):
        print(f'Weighting 2D-S and HVPS PSDs in {binlims[1]}-{binlims[2]} mm range')

        bin_mid = xr.DataArray(
            data = np.append(
                psd_2ds['bin_center'].values,
                psd_hvps['bin_center'].values[
                    psd_hvps['bin_center'] >
                    psd_2ds['bin_center'].values[-1]
                ]
            ),
            dims = 'size',
            attrs = psd_2ds['bin_center'].attrs
        )
        bin_min = xr.DataArray(
            data = np.append(
                psd_2ds['bin_left'].values,
                psd_hvps['bin_left'].values[
                    psd_hvps['bin_center'] >
                    psd_2ds['bin_center'].values[-1]
                ]
            ),
            dims = 'size',
            attrs = psd_2ds['bin_left'].attrs
        )
        bin_max = xr.DataArray(
            data = np.append(
                psd_2ds['bin_right'].values,
                psd_hvps['bin_right'].values[
                    psd_hvps['bin_center'] >
                    psd_2ds['bin_center'].values[-1]
                ]
            ),
            dims = 'size',
            attrs = psd_2ds['bin_right'].attrs
        )
        dD = xr.DataArray(
            data = (bin_max.values - bin_min.values) / 10.,
            dims = 'size',
            attrs = psd_2ds['bin_width'].attrs
        )

        # compute weights following Fontaine et al. (2014)
        # doi: 
        w_hvps = (bin_mid - binlims[1]) / (binlims[2] - binlims[1])
        w_hvps = np.clip(w_hvps, 0., 1.) # ensure weights between 0 and 1
        weight_hvps = xr.DataArray(
            data = w_hvps,
            dims = 'size',
            attrs = dict(
                description = 'HVPS weight per bin for the composite PSD',
                units = '# [0-1]'
            )
        )
        weight_2ds = xr.DataArray(
            data = 1 - weight_hvps.values,
            dims = 'size',
            attrs = dict(
                description = '2D-S weight per bin for the composite PSD',
                units = '# [0-1]'
            )
        )

        # interpolate psds (nearest neighbor) to match new bin arrangement
        psd_interp_2ds = psd_2ds.copy(deep=True)
        psd_interp_2ds = psd_2ds.swap_dims({'size': 'bin_center'}).drop_vars(
            ['count', 'sv']
        )
        if qc_thresh is not None:
            psd_interp_2ds.drop_vars('qc_flag')
        psd_interp_2ds = weight_2ds * psd_interp_2ds.interp(
            bin_center=bin_mid, method='nearest'
        )
        psd_interp_hvps = psd_hvps.copy(deep=True)
        psd_interp_hvps = psd_hvps.swap_dims({'size': 'bin_center'}).drop_vars(
            ['count', 'sv']
        )
        if qc_thresh is not None:
            psd_interp_hvps.drop_vars('qc_flag')
        psd_interp_hvps = weight_hvps * psd_interp_hvps.interp(
            bin_center=bin_mid, method='nearest'
        )

        # get bin widths
        dD_2ds = psd_interp_2ds['bin_width'].values
        dD_2ds[np.isnan(dD_2ds)] = np.nan # dummy value for bins outside probe range
        dD_hvps = psd_interp_hvps['bin_width'].values
        dD_hvps[np.isnan(dD_hvps)] = np.nan # dummy value for bins outside probe range

        # normalize N(D) based on new bin widths
        psd_interp_2ds['ND'] = (dD / dD_2ds) * psd_interp_2ds['ND']
        psd_interp_2ds['ND'].values[np.isnan(psd_interp_2ds['ND'])] = 0.
        psd_interp_hvps['ND'] = (dD / dD_hvps) * psd_interp_hvps['ND']
        psd_interp_hvps['ND'].values[np.isnan(psd_interp_hvps['ND'])] = 0.
        ND_temp = psd_interp_2ds['ND'].values + psd_interp_hvps['ND'].values
        ND = xr.DataArray(
            data = np.ma.masked_where(ND_temp == 0., ND_temp),
            dims = ['size', 'time'],
            coords = dict(
                bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                bin_width=dD, time=psd_interp_2ds.time),
            attrs = psd_2ds['ND'].attrs
        )

        # compute weighted mean of aspect and area ratio distributions
        psd_interp_2ds['area_ratio'].values[np.isnan(psd_interp_2ds['area_ratio'])] = 0.
        psd_interp_hvps['area_ratio'].values[np.isnan(psd_interp_hvps['area_ratio'])] = 0.
        ar_temp = psd_interp_2ds['area_ratio'].values + psd_interp_hvps['area_ratio'].values
        ar = xr.DataArray(
            data = np.ma.masked_where(ar_temp == 0., ar_temp),
            dims = ['size', 'time'],
            coords = dict(
                bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                bin_width=dD, time=psd_interp_2ds.time),
            attrs = psd_interp_2ds['area_ratio'].attrs
        )

        psd_interp_2ds['aspect_ratio'].values[np.isnan(psd_interp_2ds['aspect_ratio'])] = 0.
        psd_interp_hvps['aspect_ratio'].values[np.isnan(psd_interp_hvps['aspect_ratio'])] = 0.
        asr_temp = psd_interp_2ds['aspect_ratio'].values + psd_interp_hvps['aspect_ratio'].values
        asr = xr.DataArray(
            data = np.ma.masked_where(asr_temp == 0., asr_temp),
            dims = ['size', 'time'],
            coords = dict(
                bin_center=bin_mid, bin_left=bin_min, bin_right=bin_max,
                bin_width=dD, time=psd_interp_2ds.time),
            attrs = psd_interp_2ds['aspect_ratio'].attrs
        )

        # make the dataset object
        psd_merged = xr.Dataset(
            data_vars={
                'ND': ND,
                'area_ratio': ar,
                'aspect_ratio': asr
            },
            coords={
                'bin_center': bin_mid,
                'bin_left': bin_min,
                'bin_right': bin_max,
                'bin_width': dD,
                'weight_2ds': weight_2ds,
                'weight_hvps': weight_hvps,
                'time': psd_interp_2ds.time
            },
            attrs = psd_interp_2ds.attrs
        )

        return psd_merged
                
    def bulk_properties(self, calc_gamma_params=False, matched_objects=None):
        """
        Compute bulk microphysical properties from the PSD data.
        
        Parameters
        ----------
        calc_gamma_params: boolean
            Optionally compute PSD N0, mu, lambda (takes some time)
        matched_objects : impacts_tools.match.Radar() object or list of objects
            Matched radar data to use in deriving a time-varying m-D relationship.
        """
        with np.errstate(divide='ignore', invalid='ignore'): # suppress divide by zero warnings
            # initialize gamma fitting technique
            x0 = [1.e-1, -1., 5.] # initial guess for N0 (cm-4), mu, lambda (cm-1)
            
            # number concentration (/L)
            nt_temp = 1000. * (self.data['count'] / self.data['sv']).sum(dim='size')
            
            # spherical volume from Chase et al. (2018) [cm**3 / cm**3]
            vol = (np.pi / 6.) * (
                0.6 * ((self.data['bin_center'] / 10.) ** 3.) *
                self.data['count'] / self.data['sv']
            ).sum(dim='size')
            
            # number-weighted mean aspect and area ratio
            ar_nw_temp = (
                self.data['area_ratio'] * self.data['count']
            ).sum(dim='size') / self.data['count'].sum(dim='size')
            asr_nw_temp = (
                self.data['aspect_ratio'] * self.data['count']
            ).sum(dim='size') / self.data['count'].sum(dim='size')
            
            ## ===== Brown and Francis (1995) m-D products =====
            mass_particle = (
                0.00294 / 1.5) * (
                self.data['bin_center'] / 10.
            ) ** 1.9 # particle mass (g)
            mass_bf = mass_particle * self.data['count'] # binned mass (g)
            mass_rel_bf = mass_bf.cumsum(
                dim='size') / mass_bf.cumsum(dim='size')[-1, :] # binned mass fraction
            z_bf_temp = 1.e12 * (0.174 / 0.93) * (6. / np.pi / 0.934) ** 2 * (
                mass_particle ** 2 * self.data['count'] / self.data['sv']
            ).sum(dim='size') # simulated Z (mm^6 m^-3)
            iwc_bf_temp = 10. ** 6 * (mass_bf / self.data['sv']).sum(dim='size') # IWC (g m-3)
            dmm_bf_temp = xr.full_like(nt_temp, np.nan) # allocate array of nan
            dmm_bf_temp[~np.isnan(mass_rel_bf[-1,:])] = self.data['bin_center'][
            	(0.5 - mass_rel_bf[:, ~np.isnan(mass_rel_bf[-1,:])]).argmin(dim='size')] # med mass D (mm)
            dm_bf_temp = 10. * (
                (self.data['bin_center'] / 10.) * mass_bf /
                self.data['sv']).sum(dim='size') / (
                mass_bf / self.data['sv']
            ).sum(dim='size') # mass-weighted mean D from Chase et al. (2020) (mm)
            ar_bf_temp = (
                self.data['area_ratio'] * mass_bf / self.data['sv']).sum(dim='size') / (
                mass_bf / self.data['sv']
            ).sum(dim='size') # mass-weighted mean area ratio
            asr_bf_temp = (
                self.data['aspect_ratio'] * mass_bf / self.data['sv']).sum(dim='size') / (
                mass_bf / self.data['sv']
            ).sum(dim='size') # mass-weighted mean aspect ratio
            rhoe_bf_temp = (iwc_bf_temp / 10. ** 6) / vol # eff density from Chase et al. (2018) (g cm**-3)
            
            # optionally compute gamma fit params for each observation
            if calc_gamma_params:
                print('Computing gamma fit parameters will take some time.')
                N0_bf_temp = -999. * np.ones(len(self.data.time))
                mu_bf_temp = -999. * np.ones(len(self.data.time))
                lam_bf_temp = -999. * np.ones(len(self.data.time))
                for i in range(len(self.data.time)):
                    if iwc_bf_temp[i] > 0: # only compute when there's a PSD
                        sol = least_squares(
                            self.calc_chisquare, x0, method='lm', ftol=1e-9, xtol=1e-9, max_nfev=int(1e6),
                            args=(
                                nt_temp[i], iwc_bf_temp[i], z_bf_temp[i], 0.00294 / 1.5, 1.9
                            )
                        ) # solve the gamma params using least squares minimziation
                        N0_bf_temp[i] = sol.x[0]
                        mu_bf_temp[i] = sol.x[1]
                        lam_bf_temp[i] = sol.x[2]
            
            ## ===== Heymsfield et al. (2010; doi: 10.1175/2010JAS3507.1) m-D products =====
            mass_particle = 0.00528 * (
                self.data['bin_center'] / 10.
            ) ** 2.1 # particle mass (g)
            mass_hy = mass_particle * self.data['count'] # binned mass (g)
            mass_rel_hy = mass_hy.cumsum(
                dim='size') / mass_hy.cumsum(dim='size')[-1, :] # binned mass fraction
            z_hy_temp = 1.e12 * (0.174 / 0.93) * (6. / np.pi / 0.934) ** 2 * (
                mass_particle ** 2 * self.data['count'] / self.data['sv']
            ).sum(dim='size') # simulated Z (mm^6 m^-3)
            iwc_hy_temp = 10. ** 6 * (mass_hy / self.data['sv']).sum(dim='size') # IWC (g m-3)
            dmm_hy_temp = xr.full_like(nt_temp, np.nan) # allocate array of nan
            dmm_hy_temp[~np.isnan(mass_rel_hy[-1,:])] = self.data['bin_center'][
            	(0.5 - mass_rel_hy[:, ~np.isnan(mass_rel_hy[-1,:])]).argmin(dim='size')] # med mass D (mm)
            dm_hy_temp = 10. * (
                (self.data['bin_center'] / 10.) * mass_hy /
                self.data['sv']).sum(dim='size') / (
                mass_hy / self.data['sv']
            ).sum(dim='size') # mass-weighted mean D from Chase et al. (2020) (mm)
            ar_hy_temp = (
                self.data['area_ratio'] * mass_hy / self.data['sv']).sum(dim='size') / (
                mass_hy / self.data['sv']
            ).sum(dim='size') # mass-weighted mean area ratio
            asr_hy_temp = (
                self.data['aspect_ratio'] * mass_hy / self.data['sv']).sum(dim='size') / (
                mass_hy / self.data['sv']
            ).sum(dim='size') # mass-weighted mean aspect ratio
            rhoe_hy_temp = (iwc_hy_temp / 10. ** 6) / vol # eff density from Chase et al. (2018) (g cm**-3)
            
            # optionally compute gamma fit params for each observation
            if calc_gamma_params:
                N0_hy_temp = -999. * np.ones(len(self.data.time))
                mu_hy_temp = -999. * np.ones(len(self.data.time))
                lam_hy_temp = -999. * np.ones(len(self.data.time))
                for i in range(len(self.data.time)):
                    if iwc_hy_temp[i] > 0: # only compute when there's a PSD
                        sol = least_squares(
                            self.calc_chisquare, x0, method='lm', ftol=1e-9, xtol=1e-9, max_nfev=int(1e6),
                            args=(
                                nt_temp[i], iwc_hy_temp[i], z_hy_temp[i], 0.00528, 2.1
                            )
                        ) # solve the gamma params using least squares minimziation
                        N0_hy_temp[i] = sol.x[0]
                        mu_hy_temp[i] = sol.x[1]
                        lam_hy_temp[i] = sol.x[2]
        
        # mask bad gamma parameter values, or set to NaN if skipping
        if calc_gamma_params:
            N0_bf_temp = np.ma.masked_where(N0_bf_temp == -999., N0_bf_temp)
            N0_hy_temp = np.ma.masked_where(N0_hy_temp == -999., N0_hy_temp)
            mu_bf_temp = np.ma.masked_where(mu_bf_temp == -999., mu_bf_temp)
            mu_hy_temp = np.ma.masked_where(mu_hy_temp == -999., mu_hy_temp)
            lam_bf_temp = np.ma.masked_where(lam_bf_temp == -999., lam_bf_temp)
            lam_hy_temp = np.ma.masked_where(lam_hy_temp == -999., lam_hy_temp)
        else:
            N0_bf_temp = np.ma.array(np.zeros(len(self.data.time)), mask=True)
            N0_hy_temp = np.ma.array(np.zeros(len(self.data.time)), mask=True)
            mu_bf_temp = np.ma.array(np.zeros(len(self.data.time)), mask=True)
            mu_hy_temp = np.ma.array(np.zeros(len(self.data.time)), mask=True)
            lam_bf_temp = np.ma.array(np.zeros(len(self.data.time)), mask=True)
            lam_hy_temp = np.ma.array(np.zeros(len(self.data.time)), mask=True)
            
        # add bulk properties to PSD object
        n = xr.DataArray(
            data = nt_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Number concentration',
                units = 'L-1')
        )
        ar_nw = xr.DataArray(
            data = ar_nw_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Number-weighted mean area ratio (elliptical fit)',
                units = '#')
        )
        asr_nw = xr.DataArray(
            data = asr_nw_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Number-weighted mean aspect ratio (elliptical fit)',
                units = '#')
        )
        N0_bf = xr.DataArray(
            data = N0_bf_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='PSD intercept parameter [Baker and Lawson (1995) m-D relationship]',
                relationship='m = 0.00196 * D ** 1.9',
                units = 'cm-4')
        )
        mu_bf = xr.DataArray(
            data = mu_bf_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='PSD shape parameter [Baker and Lawson (1995) m-D relationship]',
                relationship='m = 0.00196 * D ** 1.9',
                units = '#')
        )
        lam_bf = xr.DataArray(
            data = lam_bf_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='PSD slope parameter [Baker and Lawson (1995) m-D relationship]',
                relationship='m = 0.00196 * D ** 1.9',
                units = 'cm-1')
        )
        iwc_bf = xr.DataArray(
            data = iwc_bf_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Ice water content [Baker and Lawson (1995) m-D relationship]',
                relationship='m = 0.00196 * D ** 1.9',
                units = 'g m-3')
        )
        dm_bf = xr.DataArray(
            data = dm_bf_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Mass-weighted mean diameter [Baker and Lawson (1995) m-D relationship]',
                relationship='m = 0.00196 * D ** 1.9',
                units = 'mm')
        )
        dmm_bf = xr.DataArray(
            data = dmm_bf_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Median mass diameter [Baker and Lawson (1995) m-D relationship]',
                relationship='m = 0.00196 * D ** 1.9',
                units = 'mm')
        )
        ar_bf = xr.DataArray(
            data = ar_bf_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Mass-weighted mean area ratio [Baker and Lawson (1995) m-D relationship]',
                relationship='m = 0.00196 * D ** 1.9',
                units = '#')
        )
        asr_bf = xr.DataArray(
            data = asr_bf_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Mass-weighted mean aspect ratio [Baker and Lawson (1995) m-D relationship]',
                relationship='m = 0.00196 * D ** 1.9',
                units = '#')
        )
        rhoe_bf = xr.DataArray(
            data = rhoe_bf_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Effective density [Baker and Lawson (1995) m-D relationship]',
                relationship='m = 0.00196 * D ** 1.9',
                units = 'g cm-3')
        )
        N0_hy = xr.DataArray(
            data = N0_hy_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='PSD intercept parameter [Heymsfield et al. (2010) m-D relationship]',
                relationship='m = 0.00528 * D ** 2.1',
                units = 'cm-4')
        )
        mu_hy = xr.DataArray(
            data = mu_hy_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='PSD shape parameter [Heymsfield et al. (2010) m-D relationship]',
                relationship='m = 0.00528 * D ** 2.1',
                units = '#')
        )
        lam_hy = xr.DataArray(
            data = lam_hy_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='PSD slope parameter [Heymsfield et al. (2010) m-D relationship]',
                relationship='m = 0.00528 * D ** 2.1',
                units = 'cm-1')
        )
        iwc_hy = xr.DataArray(
            data = iwc_hy_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Ice water content [Heymsfield et al. (2010) m-D relationship]',
                relationship='m = 0.00528 * D ** 2.1',
                units = 'g m-3')
        )
        dm_hy = xr.DataArray(
            data = dm_hy_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Mass-weighted mean diameter [Heymsfield et al. (2010) m-D relationship]',
                relationship='m = 0.00528 * D ** 2.1',
                units = 'mm')
        )
        dmm_hy = xr.DataArray(
            data = dmm_hy_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Median mass diameter [Heymsfield et al. (2010) m-D relationship]',
                relationship='m = 0.00528 * D ** 2.1',
                units = 'mm')
        )
        ar_hy = xr.DataArray(
            data = ar_hy_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Mass-weighted mean area ratio [Heymsfield et al. (2010) m-D relationship]',
                relationship='m = 0.00528 * D ** 2.1',
                units = '#')
        )
        asr_hy = xr.DataArray(
            data = asr_hy_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Mass-weighted mean aspect ratio [Heymsfield et al. (2010) m-D relationship]',
                relationship='m = 0.00528 * D ** 2.1',
                units = '#')
        )
        rhoe_hy = xr.DataArray(
            data = rhoe_hy_temp,
            dims = 'time',
            coords = dict(time=self.data.time),
            attrs = dict(
                description='Effective density [Heymsfield et al. (2010) m-D relationship]',
                relationship='m = 0.00528 * D ** 2.1',
                units = 'g cm-3')
        )
        
        # put bulk properties together into an XArray DataSet
        ds = xr.Dataset(
            data_vars={
                'n': n,
                'N0_bf': N0_bf,
                'N0_hy': N0_hy,
                'mu_bf': mu_bf,
                'mu_hy': mu_hy,
                'lambda_bf': lam_bf,
                'lambda_hy': lam_hy,
                'iwc_bf': iwc_bf,
                'iwc_hy': iwc_hy,
                'dm_bf': dm_bf,
                'dm_hy': dm_hy,
                'dmm_bf': dmm_bf,
                'dmm_hy': dmm_hy,
                'rhoe_bf': rhoe_bf,
                'rhoe_hy': rhoe_hy,
                'area_ratio_mean_n': ar_nw,
                'area_ratio_mean_bf': ar_bf,
                'area_ratio_mean_hy': ar_hy,
                'aspect_ratio_mean_n': asr_nw,
                'aspect_ratio_mean_bf': asr_bf,
                'aspect_ratio_mean_hy': asr_hy
            },
            coords={
                'time': self.data.time
            }
        )
        if not calc_gamma_params: # remove gamma params if skipping calculation
            ds = ds.drop_vars(
                ['N0_bf', 'mu_bf', 'lambda_bf', 'N0_hy', 'mu_hy', 'lambda_hy']
            )
        
        ds_merged = xr.merge(
            [self.data, ds], combine_attrs='drop_conflicts'
        ).transpose('habit', 'size', 'time')
        
        return ds_merged
    
    def calc_chisquare(
            self, x, n, iwc, z, a, b, rime_ind=None, exponential=False):
        '''
        Compute gamma fit parameters for the PSD.
        Follows McFarquhar et al. (2015) by finding N0-mu-lambda minimizing zeroth
        (Nt), second (mass), fourth (reflectivity) moments of ice phase PSD.
        Inputs:
            x: N0, mu, lambda to test on the minimization procedure
            n: Measured number concentration (L-1)
            iwc: Measured IWC using an assumed m-D relation (g m-3)
            z: Measured Z following Hogan et al. (2012) using assumed m-D relation (mm6 m-3)
            a: Prefactor component to the assumed m-D reltation [cm**-b]
            b: Exponent component to the assumed m-D reltation
            rime_ind (optional, for LS products only): Riming category index to use for the reflectivity moment
            exponential: Boolean, True if setting mu=0 for the fit (exponential form)
        Outputs:
            chi_square: Chi-square value for the provided N0-mu-lambda configuration
        '''
        Dmax = self.data['bin_center'] / 10. # particle size (cm)
        dD = self.data['bin_width'] # bin width (cm)
        mass_particle = a * Dmax ** b # binned particle mass (g)

        if exponential: # exponential form with mu=0
            ND_fit = x[0] * np.exp(-x[2] * Dmax)
        else: # traditional gamma function with variable mu
            ND_fit = x[0] * Dmax ** x[1] * np.exp(-x[2] * Dmax)

        n_fit = 1000.*np.nansum(ND_fit*dD) # concentration from fit PSD (L-1)
        iwc_fit = 10. ** 6  * np.nansum(
            mass_particle* ND_fit * dD
        ) # IWC from fit PSD (g m**-3)
        if rime_ind is not None:
            Z_fit = forward_Z() #initialize class
            Z_fit.set_PSD(PSD=ND_fit[np.newaxis,:]*10.**8, D=Dmax/100., dD=dD/100., Z_interp=True) # get the PSD in the format to use in the routine (mks units)
            Z_fit.load_split_L15() # Load the leinonen output
            Z_fit.fit_sigmas(Z_interp=True) # Fit the backscatter cross-sections
            Z_fit.calc_Z() # Calculate Z...outputs are Z.Z_x, Z.Z_ku, Z.Z_ka, Z.Z_w for the four radar wavelengths
            z_fit = 10.**(Z_fit.Z_x[0, rime_ind] / 10.) # mm**6 m**-3
        else:
            z_fit = 1.e12 * (0.174 / 0.93) * (6. / np.pi / 0.934) ** 2 * np.nansum(
                mass_particle ** 2 * ND_fit * dD
            ) # Z from fit PSD (mm6 m-3)

        # compute 3-element (1 per moment) chi-square value
        csq_Nt = ((n - n_fit) / np.sqrt(n * n_fit)) ** 2
        csq_iwc = ((iwc - iwc_fit) / np.sqrt(iwc * iwc_fit)) ** 2
        csq_z = ((z - z_fit) / np.sqrt(z * z_fit)) ** 2
        chi_square = [csq_Nt, csq_iwc, csq_z]

        return chi_square

    def hhmmss2dt(self, time_hhmmss: int, date: str):
        """
        If you have date as an integer, use this method to obtain a datetime.date object.

        Parameters
        ----------
        time_hhmmss : int
          Time as a regular integer value in HHMMSS (example: 235959).
        date        : str
          Date of flight start.

        Returns
        -------
        dt          : np.array(dtype='datetime64[s]')
          A datetime object which corresponds to the given value `time_hhmmss`.
        """
        year = np.tile(int(date[:4]), (len(time_hhmmss)))
        month = np.tile(int(date[5:7]), (len(time_hhmmss)))
        day = np.tile(int(date[8:]), (len(time_hhmmss)))
        hour = (time_hhmmss / 10000).astype(int)
        minute = ((time_hhmmss % 10000) / 100).astype(int)
        second = (time_hhmmss % 100).astype(int)

        if len(np.where(np.diff(hour) < 0)[0]) > 0:
            hour[np.where(np.diff(hour) < 0)[0][0] + 1:] += 24

        df = pd.DataFrame(
            {
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': second
            }
        )
        dt = pd.to_datetime(df).to_numpy().astype('datetime64[s]')

        return dt