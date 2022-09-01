"""
Classes for IMPACTS P3 Instruments
"""

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import least_squares

class P3():
    """
    A class to represent the P-3 aircraft during the IMPACTS field campaign.
    """

    def __init__(self, filepath, date, start_time=None, end_time=None, tres='1S'):
        self.name = 'P-3 Met-Nav'
        
        # read the raw data
        self.data = self.readfile(filepath, date, start_time, end_time, tres)
        """
        xarray.Dataset of P-3 meteorological and navigation variables and attributes
        Dimensions:
            - time: np.array(np.datetime64[ns]) - The UTC time stamp
        Coordinates:
            - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)
        Variables:
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
    
    def readfile(self, filepath, date, start_time=None, end_time=None, tres='1S'):
        """
        Reads the CRS data file and unpacks the fields into an xarray.Dataset
        
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
        
        Returns
        -------
        data : xarray.Dataset
            The unpacked dataset
        """
        
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
        time = np.array([
            np.datetime64(date) + np.timedelta64(int(readfile['time'][i]), 's')
            for i in range(len(readfile['time']))], dtype='datetime64[ms]'
        )

        # Populate data arrays
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
        pt = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Potential_Temp']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Potential temperature',
                units='degrees_Kelvin')
        )
        td = xr.DataArray(
            data = np.ma.masked_invalid(readfile['Dew_Point']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Dew point temperature',
                units='degrees_Celsius')
        )
        t_ir = xr.DataArray(
            data = np.ma.masked_invalid(readfile['IR_Surf_Temp']), dims = ['time'],
            coords = dict(time = time),
            attrs = dict(
                description='Infrared surface temperature',
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
        if ('U' in readfile) and ('V' in readfile): # for 2022 data
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