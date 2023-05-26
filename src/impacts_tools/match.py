"""
Classes for matched IMPACTS ER-2 radar instances

Code by Joe Finlon, Univ. Washington, 2022
Follows methodology in Finlon et al. 2022 [doi: 10.1175/JAS-D-21-0311.1]
"""
import xarray as xr
import numpy as np
from pyproj import Proj
from scipy.stats import iqr
from scipy.spatial import cKDTree
from scipy.ndimage import generic_filter
from abc import ABC, abstractmethod

class Match(ABC):
    """
    A class to represent matched radar instances during the IMPACTS field campaign.
    
    Match is an Abstract Base Class - meaning we always require a more specific class 
    to be instantiated - ie you have to call Tamms() or Psd(), you can't just call Instrument()
    Parameters
    ----------
    data : xarray.Dataset()
        Matched radar data and attributes
    """
    @abstractmethod     # this stops you from being able to make a new generic instrument
    def __init__(self):
        """
        This is an abstract method since only inherited classes will be used to instantiate Match objects.
        """
        self.name = None
        self.data = None
        
    def qc_radar(self, radar_object, qc=False):
        """
        xarray.Dataset (1-D) of QC'd radar data, with bad values removed
        """
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            # build mask
            mask = np.zeros(radar_object['dbz'].shape, dtype=bool)
            mask[
                (np.isnan(radar_object['dbz'].values)) |
                (radar_object['height'].values < 500.)] = True
            
            if qc: # additional qc based on Z variability and SPW
                # compute IQR of Z using 9x3 (vertical x horizontal) moving window
                # window should reflect similar distance in horizontal and vertical
                iqr_dbz = generic_filter(
                    radar_object['dbz'].values, iqr,
                    size=(9,3), mode='nearest'
                )

                # compute 5th percentile of spectrum width
                p05_spw = np.nanpercentile(radar_object['width'].values, 5)
                
                # additional conditions for mask
                mask[(iqr_dbz > 5.) & (radar_object['width'].values < p05_spw)] = True
        elif self.name == 'Matched HIWRAP':
            # build mask
            mask = np.zeros(radar_object['dbz_ku'].shape, dtype=bool)
            mask[
                (np.isnan(radar_object['dbz_ku'].values)) |
                (np.isnan(radar_object['dbz_ka'].values)) |
                (radar_object['height'].values < 500.)] = True
            
            if qc: # additional qc based on Z variability and SPW
                # compute IQR of Z using 9x3 (vertical x horizontal) moving window
                # window should reflect similar distance in horizontal and vertical
                iqr_dbz_ka = generic_filter(
                    radar_object['dbz_ka'].values, iqr,
                    size=(9,3), mode='nearest'
                )
                iqr_dbz_ku = generic_filter(
                    radar_object['dbz_ku'].values, iqr,
                    size=(9,3), mode='nearest'
                )

                # compute 5th percentile of spectrum width
                p05_spw_ka = np.nanpercentile(radar_object['width_ka'].values, 5)
                p05_spw_ku = np.nanpercentile(radar_object['width_ku'].values, 5)
                
                # additional conditions for mask
                mask[(iqr_dbz_ka > 5.) & (radar_object['width_ka'].values < p05_spw_ka)] = True
                mask[(iqr_dbz_ku > 5.) & (radar_object['width_ku'].values < p05_spw_ku)] = True
        
        # build 1D dataset with select radar vars
        gate_idx = np.arange(mask.shape[0] * mask.shape[1])
        mask_flat = xr.DataArray(
            data = mask.flatten(),
            dims = 'gate_idx'
        )
        time_flat = xr.DataArray(
            data = np.tile(
                np.atleast_2d(radar_object['time'].values),
                (radar_object.dims['range'], 1)).flatten(),
            dims = 'gate_idx',
            attrs = radar_object['time'].attrs
        )
        hght = xr.DataArray(
            data =  radar_object['height'].values.flatten(),
            dims = 'gate_idx',
            attrs = radar_object['lat'].attrs
        )
        dist_raw = (
            xr.DataArray(np.ones(radar_object.dims['range']), dims=('range')) *
            radar_object['distance']).values.flatten()
        dist = xr.DataArray(
            data =  dist_raw - np.min(dist_raw),
            dims = 'gate_idx',
            attrs = radar_object['distance'].attrs
        )
        lat = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(radar_object.dims['range']), dims=('range')) *
                radar_object['lat']).values.flatten(),
            dims = 'gate_idx',
            attrs = radar_object['lat'].attrs
        )
        lon = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(radar_object.dims['range']), dims=('range')) *
                radar_object['lon']).values.flatten(),
            dims = 'gate_idx',
            attrs = radar_object['lon'].attrs
        )
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            dbz = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(radar_object.dims['range']), dims=('range')) *
                    radar_object['dbz']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_object['dbz'].attrs
            )
            data_vars = {'dbz': dbz}
        elif self.name == 'Matched HIWRAP':
            dbz_ka = xr.DataArray(
                data =  radar_object['dbz_ka'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_object['dbz_ka'].attrs
            )
            dbz_ku = xr.DataArray(
                data =  radar_object['dbz_ku'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_object['dbz_ku'].attrs
            )
            data_vars = {'dbz_ka': dbz_ka, 'dbz_ku': dbz_ku}
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time_flat,
                'height': hght,
                'distance': dist,
                'lat': lat,
                'lon': lon
            }
        )

        # trim dataset based on flattened boolean flag
        ds = ds.where(~mask_flat, drop=True)
        
        return ds
    
    def qc_lidar(self, lidar_object, alt_bounds_p3):
        """
        xarray.Dataset (1-D) of QC'd lidar data, with bad values removed
        
        Parameters
        ----------
        lidar_object: impacts_tools.er2.Cpl(Lidar).Dataset object
        
        alt_bounds_p3: tuple
            Minimum and maximum P-3 altitude (m) for flight segment.
        """
        mask = np.ones(lidar_object[list(lidar_object.data_vars)[0]].shape, dtype=bool)
        
        # mask pixels > 50 m below (above) min (max) P-3 altitude
        if self.name == 'Matched CPL ATB':
            mask[
                (lidar_object['height'].values >= alt_bounds_p3[0] - 50.) &
                (lidar_object['height'].values <= alt_bounds_p3[1] + 50.) &
                (lidar_object['atb_1064'].values >= 5.e-4) &
                (lidar_object['atb_1064'].values >= 5.e-4) &
                (lidar_object['atb_1064'].values >= 5.e-4) &
                (lidar_object['height'].values >= 500.)] = False
        
        # build 1D dataset with select lidar vars
        gate_idx = np.arange(mask.shape[0] * mask.shape[1])
        mask_flat = xr.DataArray(
            data = mask.flatten(),
            dims = 'gate_idx'
        )
        time_flat = xr.DataArray(
            data = np.tile(
                np.atleast_2d(lidar_object['time'].values),
                (lidar_object.dims['gate'], 1)).flatten(),
            dims = 'gate_idx',
            attrs = lidar_object['time'].attrs
        )
        hght = xr.DataArray(
            data =  lidar_object['height'].values.flatten(),
            dims = 'gate_idx',
            attrs = lidar_object['lat'].attrs
        )
        dist_raw = (
            xr.DataArray(np.ones(lidar_object.dims['gate']), dims=('gate')) *
            lidar_object['distance']).values.flatten()
        dist = xr.DataArray(
            data =  dist_raw - np.min(dist_raw),
            dims = 'gate_idx',
            attrs = lidar_object['distance'].attrs
        )
        lat = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(lidar_object.dims['gate']), dims=('gate')) *
               lidar_object['lat']).values.flatten(),
            dims = 'gate_idx',
            attrs = lidar_object['lat'].attrs
        )
        lon = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(lidar_object.dims['gate']), dims=('gate')) *
                lidar_object['lon']).values.flatten(),
            dims = 'gate_idx',
            attrs = lidar_object['lon'].attrs
        )
        if self.name == 'Matched CPL ATB':
            atb1064 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_object.dims['gate']), dims=('gate')) *
                   lidar_object['atb_1064']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = lidar_object['atb_1064'].attrs
            )
            atb532 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_object.dims['gate']), dims=('gate')) *
                    lidar_object['atb_532']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs =lidar_object['atb_532'].attrs
            )
            atb355 = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(lidar_object.dims['gate']), dims=('gate')) *
                    lidar_object['atb_355']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = lidar_object['atb_355'].attrs
            )
            data_vars = {'atb_1064': atb1064, 'atb_532': atb532, 'atb_355': atb355}
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time_flat,
                'height': hght,
                'distance': dist,
                'lat': lat,
                'lon': lon
            }
        )

        # trim dataset based on flattened boolean flag
        ds = ds.where(~mask_flat, drop=True)
        
        return ds
    
    def match_lidar(
            self, lidar_qc, p3_object, query_k, dist_thresh, time_thresh,
            ref_coords, n_workers):
        # define map proj to calculate cartesian distances
        if ref_coords is None:
            ref_coords = (p3_object['lat'].values[0], p3_object['lon'].values[0])
        p = Proj(
            proj='laea', zone=10, ellps='WGS84',
            lat_0=ref_coords[0], lon_0=ref_coords[1]
        )
        
        # use proj to get cartiesian distances between the datasets
        lid_x, lid_y = p(lidar_qc['lon'].values, lidar_qc['lat'].values)
        p3_x, p3_y = p(p3_object['lon'].values, p3_object['lat'].values)
        
        if len(lid_x) == 0: # no available lidar data, skip rest of routine
            return None
        # perform the kdtree search
        kdt = cKDTree(
            list(zip(lid_x, lid_y, lidar_qc['height'].values)), leafsize=16
        )
        prdistance, prind1d = kdt.query(
            list(zip(p3_x, p3_y, p3_object['alt_gps'].values)),
            k=query_k, eps=0, p=2,
            distance_upper_bound=dist_thresh, workers=n_workers
        )
        
        # remove matched data outside of defined bounds
        bad_inds = np.where(prind1d == lidar_qc[list(lidar_qc.data_vars)[0]].shape)
        if ((query_k == 1) and (len(bad_inds[0]) > 0)) or (
                (query_k > 1) and (len(bad_inds[0]) > 0)) or (
                (query_k > 1) and (len(bad_inds[1]) > 0)):
            # mask inds and distances that are outside the search area
            prind1d[bad_inds] = np.ma.masked
            prdistance[bad_inds] = np.ma.masked
            
        # mask data outside distance bounds
        if self.name == 'Matched CPL ATB':
            match_dict = {}
            for wavelength in [1064, 532, 355]:
                match_dict[f'atb_{wavelength}'] = dict(
                    data = np.ma.masked_where(
                        prind1d == 0, lidar_qc[f'atb_{wavelength}'].values[prind1d]
                    ),
                    description = (
                        'Mean attenuated total backscatter profile at '
                        f'{wavelength} nm among matched lidar gates'
                    ),
                    units = 'km**-1 sr**-1'
                )
            
        # perform the matching routine
        if query_k == 1: # nearest neighbor
            return
        else: # more than closest gate considered, use Barnes-weighted mean
            W_d_k = np.ma.array(
                np.exp(-1 * prdistance**2. / dist_thresh**2.)
            ) # distance weights
            
            # L1B ATB
            if self.name == 'Matched CPL ATB':
                for wavelength in [1064, 532, 355]:
                    W_d_k2 = np.ma.masked_where(
                        np.ma.getmask(match_dict[f'atb_{wavelength}']['data']),
                        W_d_k.copy()
                    ) # mask weights where ATB is masked
                    w1 = np.ma.sum(
                        W_d_k2 * match_dict[f'atb_{wavelength}']['data'], axis=1
                    ) # weighted sum of ATB per N-s period
                    w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
                    #dbz_matched_temp = dbz_matched.copy()
                    match_dict[f'atb_{wavelength}']['data'] = w1 / w2 # flatten matched, weighted ATB
                    '''
                    dbz_stdev = np.ma.zeros(dbz_matched.shape[0])
                    for i in range(dbz_matched_temp.shape[0]):
                        square_diff = (
                            dbz_matched_temp[i, :] - dbz_matched[i]
                        )**2. # diff**2 between gates and weighted mean
                        ssd = np.nansum(square_diff) # sum of squared diff between gates and weighted mean
                        if np.isnan(ssd):
                            dbz_stdev[i] = np.nan
                        else:
                            num_goodvals = len(dbz_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                            dbz_stdev[i] = np.sqrt(ssd / num_goodvals)
                    dbz_stdev = np.ma.masked_invalid(dbz_stdev)
                    dbz_matched = np.ma.masked_where(
                        dbz_stdev > 5., dbz_matched).filled(np.nan) # mask suspected skin paint artifact
                    '''
            # time
            time_2d = np.tile(
                np.atleast_2d(p3_object['time'].values).T, (1, query_k)
            )
            tdiff = (
                lidar_qc['time'].values[prind1d] - time_2d
            ) / np.timedelta64(1, 's') # [s]
            W_d_k2 = np.ma.masked_where(np.ma.getmask(tdiff), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * tdiff, axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            tdiff_matched = w1 / w2
            time_lidar_matched = np.array([], dtype='datetime64[ns]')
            for i in range(len(tdiff_matched)):
                time_lidar_matched = np.append(
                    time_lidar_matched,
                    p3_object['time'].values[i] + np.timedelta64(int(tdiff_matched[i]), 's')
                )
            
            # distance difference
            W_d_k2 = np.ma.masked_where(np.ma.getmask(prdistance), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * prdistance, axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            dist_matched = w1 / w2
            
            # altitude
            W_d_k2 = np.ma.masked_where(
                np.ma.getmask(lidar_qc['height'].values[prind1d]),
                W_d_k.copy()
            )
            w1 = np.ma.sum(W_d_k2 * lidar_qc['height'].values[prind1d], axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            alt_lidar_matched = w1 / w2
            
        # along track distance
        nomdist = np.zeros(len(p3_object['time']))
        for i in range(len(p3_object['time'])):
            if ~np.isnan(time_lidar_matched[i]):
                nomdist[i] = lidar_qc['distance'].values[
                    np.argmin(np.abs(lidar_qc['time'].values - time_lidar_matched[i]))
                ]
        
        # mask data
        mask_altdiff = np.abs(
            alt_lidar_matched.data - p3_object['alt_gps'].values
        ) > 100. # mean gate alt > 100 m from P-3 alt
        mask_tdiff = np.abs(tdiff_matched) > time_thresh # time offset too big
        if self.name == 'Matched CPL ATB':
            mask_final = (
                np.isnan(match_dict['atb_1064']['data']) +
                np.isnan(match_dict['atb_532']['data']) +
                np.isnan(match_dict['atb_355']['data']) +
                mask_tdiff.data + mask_altdiff
            )
            
        # establish the data arrays
        time = p3_object['time'].values

        time_lidar = xr.DataArray(
            data = np.ma.masked_where(mask_final, time_lidar_matched),
            dims = 'time',
            attrs = dict(
                description = 'Mean time of the matched lidar gates',
            )
        )
        dist_lidar = xr.DataArray(
            data = np.ma.masked_where(mask_final, nomdist),
            dims = 'time',
            attrs = dict(
                description = ('Along-track distance corresponding to each '
                               'matched observation'),
                units = 'm'
            )
        )
        alt_lidar = xr.DataArray(
            data = np.ma.masked_where(mask_final, alt_lidar_matched),
            dims = 'time',
            attrs = dict(
                description = 'Mean altitdue of the matched lidar gates ',
                units = 'm'
            )
        )
        data_vars = {}
        data_vars['dist_offset'] = xr.DataArray(
            data = np.ma.masked_where(mask_final, dist_matched),
            dims = 'time',
            coords = dict(time = time, time_lidar = time_lidar),
            attrs = dict(
                description = 'Mean distance between matched lidar gates and P-3',
                units = 'm'
            )
        )
        data_vars['time_offset'] = xr.DataArray(
            data = np.ma.masked_where(mask_final, tdiff_matched),
            dims = 'time',
            coords = dict(time = time, time_lidar = time_lidar),
            attrs = dict(
                description = 'Mean time offset between matched lidar gates and P-3',
                units = 's'
            )
        )
        for wavelength in [1064, 532, 355]:
            data_vars[f'atb_{wavelength}'] = xr.DataArray(
                data = np.ma.masked_where(mask_final, match_dict[f'atb_{wavelength}']['data']),
                dims = 'time',
                coords = dict(time = time, time_lidar = time_lidar),
                attrs = dict(
                    description = match_dict[f'atb_{wavelength}']['description'],
                    units = match_dict[f'atb_{wavelength}']['units']
                )
            )
        
        # create dataset
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time,
                'time_lidar': time_lidar,
                'distance': dist_lidar,
                'altitude': alt_lidar
            },
            attrs = {
                'distance_max': f'{dist_thresh:.0f} m',
                'time_diff_max': f'{time_thresh:.0f} s',
                'n_neighbors': f'{query_k}'
            }
        )
        
        return ds
    
    def matchdata(
            self, radar_qc, p3_object, query_k, dist_thresh, time_thresh,
            ref_coords, n_workers):
        # define map proj to calculate cartesian distances
        if ref_coords is None:
            ref_coords = (p3_object['lat'].values[0], p3_object['lon'].values[0])
        p = Proj(
            proj='laea', zone=10, ellps='WGS84',
            lat_0=ref_coords[0], lon_0=ref_coords[1]
        )
        
        # use proj to get cartiesian distances between the datasets
        rad_x, rad_y = p(radar_qc['lon'].values, radar_qc['lat'].values)
        p3_x, p3_y = p(p3_object['lon'].values, p3_object['lat'].values)
        
        # perform the kdtree search
        kdt = cKDTree(
            list(zip(rad_x, rad_y, radar_qc['height'].values)), leafsize=16
        )
        prdistance, prind1d = kdt.query(
            list(zip(p3_x, p3_y, p3_object['alt_gps'].values)),
            k=query_k, eps=0, p=2,
            distance_upper_bound=dist_thresh, workers=n_workers
        )
        
        # remove matched data outside of defined bounds
        if self.name == 'Matched HIWRAP':
            bad_inds = np.where(prind1d == radar_qc['dbz_ka'].shape)
        else:
            bad_inds = np.where(prind1d == radar_qc['dbz'].shape)
        if ((query_k == 1) and (len(bad_inds[0]) > 0)) or (
                (query_k > 1) and (len(bad_inds[0]) > 0)) or (
                (query_k > 1) and (len(bad_inds[1]) > 0)):
            # mask inds and distances that are outside the search area
            prind1d[bad_inds] = np.ma.masked
            prdistance[bad_inds] = np.ma.masked
            
        # mask dbz outside distance bounds
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            dbz_matched = np.ma.masked_where(
                prind1d == 0, radar_qc['dbz'].values[prind1d]
            )
            key_dbz = 'dbz'
        elif self.name == 'Matched HIWRAP':
            dbz_matched = np.ma.masked_where(
                prind1d == 0, radar_qc['dbz_ka'].values[prind1d]
            )
            dbz2_matched = np.ma.masked_where(
                prind1d == 0, radar_qc['dbz_ku'].values[prind1d]
            )
            key_dbz = 'dbz_ka'
        
        # Perform the matching routine
        if query_k == 1: # nearest neighbor
            return
        else: # more than closest gate considered, use Barnes-weighted mean
            # dbz
            W_d_k = np.ma.array(
                np.exp(-1 * prdistance**2. / dist_thresh**2.)
            ) # distance weights
            W_d_k2 = np.ma.masked_where(
                np.ma.getmask(dbz_matched), W_d_k.copy()
            ) # mask weights where dbz is masked
            w1 = np.ma.sum(
                W_d_k2 * 10.**(dbz_matched / 10.), axis=1
            ) # weighted sum of linear Z per N-s period
            w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
            dbz_matched_temp = dbz_matched.copy()
            dbz_matched = 10. * np.ma.log10(w1 / w2) # flatten matched, weighted dbz
            dbz_stdev = np.ma.zeros(dbz_matched.shape[0])
            for i in range(dbz_matched_temp.shape[0]):
                square_diff = (
                    dbz_matched_temp[i, :] - dbz_matched[i]
                )**2. # diff**2 between gates and weighted mean
                ssd = np.nansum(square_diff) # sum of squared diff between gates and weighted mean
                if np.isnan(ssd):
                    dbz_stdev[i] = np.nan
                else:
                    num_goodvals = len(dbz_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                    dbz_stdev[i] = np.sqrt(ssd / num_goodvals)
            dbz_stdev = np.ma.masked_invalid(dbz_stdev)
            dbz_matched = np.ma.masked_where(
                dbz_stdev > 5., dbz_matched).filled(np.nan) # mask suspected skin paint artifact
            
            # dbz2 (HIWRAP Ku)
            if self.name == 'Matched HIWRAP':
                W_d_k2 = np.ma.masked_where(
                    np.ma.getmask(dbz2_matched), W_d_k.copy()
                ) # mask weights where dbz is masked
                w1 = np.ma.sum(
                    W_d_k2 * 10.**(dbz2_matched / 10.), axis=1
                ) # weighted sum of linear Z per N-s period
                w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights per N-s period
                dbz2_matched_temp = dbz2_matched.copy()
                dbz2_matched = 10. * np.ma.log10(w1 / w2) # flatten matched, weighted dbz
                dbz2_stdev = np.ma.zeros(dbz2_matched.shape[0])
                for i in range(dbz2_matched_temp.shape[0]):
                    square_diff = (
                        dbz2_matched_temp[i, :] - dbz2_matched[i]
                    ) ** 2. # diff**2 between gates and weighted mean
                    ssd = np.nansum(square_diff) # sum of squared diff between gates and weighted mean
                    if np.isnan(ssd):
                        dbz2_stdev[i] = np.nan
                    else:
                        num_goodvals = len(dbz2_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                        dbz2_stdev[i] = np.sqrt(ssd / num_goodvals)
                dbz2_stdev = np.ma.masked_invalid(dbz2_stdev)
                dbz2_matched = np.ma.masked_where(
                    dbz2_stdev > 5., dbz2_matched).filled(np.nan) # mask suspected skin paint artifact
                
            # time
            time_2d = np.tile(
                np.atleast_2d(p3_object['time'].values).T, (1, query_k)
            )
            tdiff = (
                radar_qc['time'].values[prind1d] - time_2d
            ) / np.timedelta64(1, 's') # [s]
            W_d_k2 = np.ma.masked_where(np.ma.getmask(tdiff), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * tdiff, axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            tdiff_matched = w1 / w2
            time_radar_matched = np.array([], dtype='datetime64[ns]')
            for i in range(len(tdiff_matched)):
                time_radar_matched = np.append(
                    time_radar_matched,
                    p3_object['time'].values[i] + np.timedelta64(int(tdiff_matched[i]), 's')
                )
            
            # distance difference
            W_d_k2 = np.ma.masked_where(np.ma.getmask(prdistance), W_d_k.copy())
            w1 = np.ma.sum(W_d_k2 * prdistance, axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            dist_matched = w1 / w2
            
            # altitude
            W_d_k2 = np.ma.masked_where(
                np.ma.getmask(radar_qc['height'].values[prind1d]),
                W_d_k.copy()
            )
            w1 = np.ma.sum(W_d_k2 * radar_qc['height'].values[prind1d], axis=1)
            w2 = np.ma.sum(W_d_k2, axis=1)
            alt_radar_matched = w1 / w2
            
        # along track distance
        nomdist = np.zeros(len(p3_object['time']))
        for i in range(len(p3_object['time'])):
            if ~np.isnan(time_radar_matched[i]):
                nomdist[i] = radar_qc['distance'].values[
                    np.argmin(np.abs(radar_qc['time'].values - time_radar_matched[i]))
                ]
        # mask data
        mask_altdiff = np.abs(
            alt_radar_matched.data - p3_object['alt_gps'].values
        ) > 100. # mean gate alt > 100 m from P-3 alt
        mask_tdiff = np.abs(tdiff_matched) > time_thresh # time offset too big
        if self.name == 'Matched HIWRAP':
            mask_final = (
                np.isnan(dbz_matched) + np.isnan(dbz2_matched) +
                mask_altdiff + mask_tdiff.data
            )
        else:
            mask_final = np.isnan(dbz_matched) + mask_altdiff + mask_tdiff.data
            
        # establish the data arrays
        time = p3_object['time'].values
        #neighbor = np.arange(query_k)

        time_radar = xr.DataArray(
            data = np.ma.masked_where(mask_final, time_radar_matched),
            dims = 'time',
            attrs = dict(
                description = 'Mean time of the matched radar gates',
            )
        )
        dist_radar = xr.DataArray(
            data = np.ma.masked_where(mask_final, nomdist),
            dims = 'time',
            attrs = dict(
                description = ('Along-track distance corresponding to each '
                               'matched observation'),
                units = 'm'
            )
        )
        ddiff = xr.DataArray(
            data = np.ma.masked_where(mask_final, dist_matched),
            dims = 'time',
            coords = dict(time = time, time_radar = time_radar),
            attrs = dict(
                description = 'Mean distance between matched radar gates and P-3',
                units = 'm'
            )
        )
        tdiff = xr.DataArray(
            data = np.ma.masked_where(mask_final, tdiff_matched),
            dims = 'time',
            coords = dict(time = time, time_radar = time_radar),
            attrs = dict(
                description = 'Mean time offset between matched radar gates and P-3',
                units = 's'
            )
        )
        dbz = xr.DataArray(
            data = np.ma.masked_where(mask_final, dbz_matched),
            dims = 'time',
            coords = dict(time = time, time_radar = time_radar),
            attrs = dict(
                description = 'Mean reflectivity among matched radar gates',
                units = 's'
            )
        )
        if self.name == 'Matched HIWRAP':
            dbz2 = xr.DataArray(
                data = np.ma.masked_where(mask_final, dbz2_matched),
                dims = 'time',
                coords = dict(time = time, time_radar = time_radar),
                attrs = dict(
                    description = 'Mean reflectivity among matched radar gates',
                    units = 's'
                )
            )
            data_vars = {
                'dist_diff': ddiff,
                'time_diff': tdiff,
                'dbz_ka': dbz,
                'dbz_ku': dbz2
            }
        else:
            data_vars = {
                'dist_diff': ddiff,
                'time_diff': tdiff,
                'dbz': dbz,
            }
            
        # create dataset
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time,
                'time_radar': time_radar,
                'distance': dist_radar
            },
            attrs = {
                'distance_max': f'{dist_thresh:.0f} m',
                'time_diff_max': f'{time_thresh:.0f} s',
                'n_neighbors': f'{query_k}'
            }
        )
        
        return ds

class MatchLidar(ABC):
    """
    A class to represent matched lidar instances during the IMPACTS field campaign.
    
    MatchLidar is an Abstract Base Class - meaning we always require a more specific class 
    to be instantiated - ie you have to call Tamms() or Psd(), you can't just call Instrument()
    Parameters
    ----------
    data : xarray.Dataset()
        Matched radar data and attributes
    """
    @abstractmethod     # this stops you from being able to make a new generic instrument
    def __init__(self):
        """
        This is an abstract method since only inherited classes will be used to instantiate Match objects.
        """
        self.name = None
        self.data = None
        
    def qc_radar(self, radar_object, qc=False):
        """
        xarray.Dataset (1-D) of QC'd radar data, with bad values removed
        """
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            # build mask
            mask = np.zeros(radar_object['dbz'].shape, dtype=bool)
            mask[
                (np.isnan(radar_object['dbz'].values)) |
                (radar_object['height'].values < 500.)] = True
            
            if qc: # additional qc based on Z variability and SPW
                # compute IQR of Z using 9x3 (vertical x horizontal) moving window
                # window should reflect similar distance in horizontal and vertical
                iqr_dbz = generic_filter(
                    radar_object['dbz'].values, iqr,
                    size=(9,3), mode='nearest'
                )

                # compute 5th percentile of spectrum width
                p05_spw = np.nanpercentile(radar_object['width'].values, 5)
                
                # additional conditions for mask
                mask[(iqr_dbz > 5.) & (radar_object['width'].values < p05_spw)] = True
        elif self.name == 'Matched HIWRAP':
            # build mask
            mask = np.zeros(radar_object['dbz_ku'].shape, dtype=bool)
            mask[
                (np.isnan(radar_object['dbz_ku'].values)) |
                (np.isnan(radar_object['dbz_ka'].values)) |
                (radar_object['height'].values < 500.)] = True
            
            if qc: # additional qc based on Z variability and SPW
                # compute IQR of Z using 9x3 (vertical x horizontal) moving window
                # window should reflect similar distance in horizontal and vertical
                iqr_dbz_ka = generic_filter(
                    radar_object['dbz_ka'].values, iqr,
                    size=(9,3), mode='nearest'
                )
                iqr_dbz_ku = generic_filter(
                    radar_object['dbz_ku'].values, iqr,
                    size=(9,3), mode='nearest'
                )

                # compute 5th percentile of spectrum width
                p05_spw_ka = np.nanpercentile(radar_object['width_ka'].values, 5)
                p05_spw_ku = np.nanpercentile(radar_object['width_ku'].values, 5)
                
                # additional conditions for mask
                mask[(iqr_dbz_ka > 5.) & (radar_object['width_ka'].values < p05_spw_ka)] = True
                mask[(iqr_dbz_ku > 5.) & (radar_object['width_ku'].values < p05_spw_ku)] = True
        
        # build 1D dataset with select radar vars
        gate_idx = np.arange(mask.shape[0] * mask.shape[1])
        mask_flat = xr.DataArray(
            data = mask.flatten(),
            dims = 'gate_idx'
        )
        time_flat = xr.DataArray(
            data = np.tile(
                np.atleast_2d(radar_object['time'].values),
                (radar_object.dims['range'], 1)).flatten(),
            dims = 'gate_idx',
            attrs = radar_object['time'].attrs
        )
        hght = xr.DataArray(
            data =  radar_object['height'].values.flatten(),
            dims = 'gate_idx',
            attrs = radar_object['lat'].attrs
        )
        dist_raw = (
            xr.DataArray(np.ones(radar_object.dims['range']), dims=('range')) *
            radar_object['distance']).values.flatten()
        dist = xr.DataArray(
            data =  dist_raw - np.min(dist_raw),
            dims = 'gate_idx',
            attrs = radar_object['distance'].attrs
        )
        lat = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(radar_object.dims['range']), dims=('range')) *
                radar_object['lat']).values.flatten(),
            dims = 'gate_idx',
            attrs = radar_object['lat'].attrs
        )
        lon = xr.DataArray(
            data =  (
                xr.DataArray(np.ones(radar_object.dims['range']), dims=('range')) *
                radar_object['lon']).values.flatten(),
            dims = 'gate_idx',
            attrs = radar_object['lon'].attrs
        )
        if (self.name == 'Matched CRS') or (self.name == 'Matched EXRAD'):
            dbz = xr.DataArray(
                data =  (
                    xr.DataArray(np.ones(radar_object.dims['range']), dims=('range')) *
                    radar_object['dbz']).values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_object['dbz'].attrs
            )
            data_vars = {'dbz': dbz}
        elif self.name == 'Matched HIWRAP':
            dbz_ka = xr.DataArray(
                data =  radar_object['dbz_ka'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_object['dbz_ka'].attrs
            )
            dbz_ku = xr.DataArray(
                data =  radar_object['dbz_ku'].values.flatten(),
                dims = 'gate_idx',
                coords = dict(
                    time = time_flat, height = hght, distance = dist,
                    lat = lat, lon = lon),
                attrs = radar_object['dbz_ku'].attrs
            )
            data_vars = {'dbz_ka': dbz_ka, 'dbz_ku': dbz_ku}
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'time': time_flat,
                'height': hght,
                'distance': dist,
                'lat': lat,
                'lon': lon
            }
        )

        # trim dataset based on flattened boolean flag
        ds = ds.where(~mask_flat, drop=True)
        
        return ds

# ====================================== #
# HIWRAP
# ====================================== #
class Hiwrap(Match):
    """
    A class to represent the TAMMS flown on the P-3 during the IMPACTS field campaign.
    Inherits from Instrument()
    
    Parameters
    ----------
    hiwrap_object: impacts_tools.er2.Hiwrap(Radar) object
        The time-trimmed HIWRAP radar object
    p3_object: impacts_tools.p3.P3() object
        The time-trimmed P-3 Met-Nav object
    query_k: int
        Number of radar gates considered in the average (1 - nearest neighbor)
    dist_thresh: float
        Maximum distance (m) allowed in the kdTree search
    time_thresh: None or float
        Maximum time offset (s) between ER-2 and P-3 allowed for matched instance
    qc: bool
        True - remove gates with high dbz gradient, no dbz, low spw
        False - only remove gates with no dbz
    ref_coords: None or 2-element tuple of float
        Reference lat, lon pair (deg) to project ER-2 and P-3 points to cartesian grid.
        None defaults to first lat, lon point in P-3 object
    """
    
    def __init__(
            self, radar_object, p3_object, query_k=1, dist_thresh=4000.,
            time_thresh=None, qc=False, ref_coords=None, n_workers=1):
        self.name = 'Matched HIWRAP'
        
        # qc radar data (threshold by Z gradient and SW if specified)
        radar_qc = self.qc_radar(radar_object, qc)
        
        # read the raw data
        self.data = self.matchdata(
            radar_qc, p3_object, query_k, dist_thresh, time_thresh,
            ref_coords, n_workers
        )
        """
        xarray.Dataset of matched HIWRAP reflectivity and attributes
        Dimensions:
            - time: np.array(np.datetime64[s]) - The UTC time start of the N-s upsampled interval
        Coordinates:
            - time (time): np.array(np.datetime64[s]) - The UTC time start of the N-s upsampled interval
            - distance (time): xarray.DataArray(float) - Along track ER-2 distance (m)
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)
        Variables:
            - alt_gps (time) : xarray.DataArray(float) - P-3 GPS altitude (m above mean sea level)
            - temp (time) : xarray.DataArray(float) - Static (ambient) air temperature (deg C)
            - dwpt (time) : xarray.DataArray(float) - Dew point temperature (deg C)
            - z_ku (time) : xarray.DataArray(float) - Matched Ku-band reflectivity (dBZ)
            - z_ka (time) : xarray.DataArray(float) - Matched Ka-band reflectivity (dBZ)
            - dfr_ku_ka (time) : xarray.DataArray(float) - Matched dual frequency ratio (dB)
        """
        
# ====================================== #
# CPL
# ====================================== #
class Cpl(Match):
    """
    A class to represent the CPL data matched to the P-3 position during the IMPACTS field campaign.
    Inherits from Match()
    
    Parameters
    ----------
    lidar_object: impacts_tools.er2.Cpl(Lidar).data xarray dataset object
        The time-trimmed CPL lidar object
    p3_object: impacts_tools.p3.P3().data xarray dataset object
        The time-trimmed P-3 Met-Nav object
    query_k: int
        Number of radar gates considered in the average (1 - nearest neighbor)
    dist_thresh: float
        Maximum distance (m) allowed in the kdTree search
    time_thresh: None or float
        Maximum time offset (s) between ER-2 and P-3 allowed for matched instance
    qc: bool
        True - remove gates with high dbz gradient, no dbz, low spw
        False - only remove gates with no dbz
    ref_coords: None or 2-element tuple of float
        Reference lat, lon pair (deg) to project ER-2 and P-3 points to cartesian grid.
        None defaults to first lat, lon point in P-3 object
    """
    
    def __init__(
            self, lidar_object, p3_object, query_k=1, dist_thresh=4000.,
            time_thresh=None, qc=False, ref_coords=None, n_workers=1):
        self.name = 'Matched CPL ATB'
        
        # get P-3 alt bounds
        alt_minP3 = np.nanmin(p3_object['alt_gps'].values)
        alt_maxP3 = np.nanmax(p3_object['alt_gps'].values)
        alt_boundsP3 = (alt_minP3, alt_maxP3)
            
        # qc lidar data
        lidar_qc = self.qc_lidar(lidar_object, alt_boundsP3)
        
        # match the data
        self.data = self.match_lidar(
            lidar_qc, p3_object, query_k, dist_thresh, time_thresh,
            ref_coords, n_workers
        )