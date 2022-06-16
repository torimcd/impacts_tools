import numpy as np
from pyproj import Proj
from scipy.spatial import cKDTree

def match(er2obj, p3obj, radname, sphere_size, start_time, end_time, query_k=1, outlier_method=None, return_indices=False):
    '''
    Get the matched radar data based on the P-3 lat, lon, alt.
    Inputs:
        er2_obj: ER-2 object obtained from the er2read() function
        p3_obj: P-3 object obtained from the iwgread() and iwg_avg() functions
        radname: Radar name ('CRS', 'HIWRAP')
        sphere_size: Maximum distance [int in m] allowed in the kdTree search
        start_time: Start time [str in YYYY-MM-DDTHH:MM:SS format] to consider in matching routine
        end_time: End time [str in YYYY-MM-DDTHH:MM:SS format] to consider in matching routine
        query_k: Number of gates (int) considered in the average (1 == use closest)
        outlier_method: None (no outliers removed) or 'iqr', 'z', 'modz'
        return_indices: True == returns the matched gates in 1d coords; False == does not
    '''
    # Load P-3 info and trim if needed
    p3_time = p3obj['time']['data']
    p3_lat = p3obj['Latitude']['data']
    p3_lon = p3obj['Longitude']['data']
    p3_alt = p3obj['GPS_Altitude']['data']
    
    start_dt64 = np.datetime64(start_time)
    end_dt64 = np.datetime64(end_time)
    
    # Turn radar spatial data into 1-D arrays
    er2_time = np.ravel(er2obj['time_gate'][:, :])
    er2_x = np.ravel(er2obj['lon_gate'][:, :])
    er2_y = np.ravel(er2obj['lat_gate'][:, :])
    er2_alt = np.ravel(er2obj['alt_gate'][:, :])
    
    # Turn radar data into 1-D arrays
    if radname=='CRS':
        radar_dbz = np.ma.ravel(er2obj['dbz_W'][:, :])
        radar_ldr = np.ma.ravel(er2obj['ldr_W'][:, :])
        radar_vel = np.ma.ravel(er2obj['vel_W'][:, :])
        radar_width = np.ma.ravel(er2obj['width_W'][:, :])
    elif radname=='HIWRAP':
        radar_dbz = np.ma.ravel(er2obj['dbz_Ku'][:, :])
        radar_ldr = np.ma.ravel(er2obj['ldr_Ku'][:, :])
        radar_vel = np.ma.ravel(er2obj['vel_Ku'][:, :])
        radar_width = np.ma.ravel(er2obj['width_Ku'][:, :])
        radar2_dbz = np.ma.ravel(er2obj['dbz_Ka'][:, :])
        radar2_ldr = np.ma.ravel(er2obj['ldr_Ka'][:, :])
        radar2_vel = np.ma.ravel(er2obj['vel_Ka'][:, :])
        radar2_width = np.ma.ravel(er2obj['width_Ka'][:, :])
    elif (radname=='EXRAD'): # TODO: accomodate nadir beam argument (also: implement EXRAD-scanning to this?)
        radar_dbz = np.ma.ravel(er2obj['dbz_X'][:, :])
        radar_vel = np.ma.ravel(er2obj['vel_X'][:, :])
        radar_width = np.ma.ravel(er2obj['width_X'][:, :])
        
    # Remove radar gates where dbZ is masked (may decide to do this differently later, esp. if other var values remain masked)
    # Also remove radar gates outside of the P-3 flight times (to only consider matches with that flight segment)
    if radname=='CRS':
        if outlier_method=='w':
            time_inds = np.where((er2_time>=start_dt64) & (er2_time<=end_dt64))[0]
            width_thresh = np.percentile(radar_width.compressed(), 5) # compute the 5th percentile to use as a threshold
            remove_inds = np.logical_or.reduce((radar_dbz.mask, radar_width.data<width_thresh, er2_time<start_dt64, er2_time>end_dt64))
        else:
            remove_inds = np.logical_or.reduce((radar_dbz.mask, er2_time<start_dt64, er2_time>end_dt64))
        radar_dbz = radar_dbz[~remove_inds]
        radar_ldr = radar_ldr[~remove_inds]
        radar_vel = radar_vel[~remove_inds]
        radar_width = radar_width[~remove_inds]
    elif radname=='HIWRAP':
        # @joefinlon: See if the first logical argument in 'remove_inds' should be handled differently
        # Currently requires both Ku- and Ka-band dbz to be masked in order to ignore radar gate 
        if outlier_method=='w':
            time_inds = np.where((er2_time>=start_dt64) & (er2_time<=end_dt64))[0]
            width_thresh = np.percentile(radar_width.compressed(), 5) # compute the 5th percentile to use as a threshold
            width2_thresh = np.percentile(radar2_width.compressed(), 5) # compute the 5th percentile to use as a threshold
            remove_inds = np.logical_or.reduce((radar_dbz.mask+radar2_dbz.mask, radar_width.data<width_thresh, radar2_width.data<width2_thresh, er2_time<start_dt64, er2_time>end_dt64))
        else:
            remove_inds = np.logical_or.reduce((radar_dbz.mask+radar2_dbz.mask, er2_time<start_dt64, er2_time>end_dt64))
        radar_dbz = radar_dbz[~remove_inds]
        radar_ldr = radar_ldr[~remove_inds]
        radar_vel = radar_vel[~remove_inds]
        radar_width = radar_width[~remove_inds]
        radar2_dbz = radar2_dbz[~remove_inds]
        radar2_ldr = radar2_ldr[~remove_inds]
        radar2_vel = radar2_vel[~remove_inds]
        radar2_width = radar2_width[~remove_inds]
    elif radname=='EXRAD':
        if outlier_method=='w':
            time_inds = np.where((er2_time>=start_dt64) & (er2_time<=end_dt64))[0]
            width_thresh = np.percentile(radar_width.compressed(), 5) # compute the 5th percentile to use as a threshold
            remove_inds = np.logical_or.reduce((radar_dbz.mask, radar_width.data<width_thresh, er2_time<start_dt64, er2_time>end_dt64))
        else:
            remove_inds = np.logical_or.reduce((radar_dbz.mask, er2_time<start_dt64, er2_time>end_dt64))
        radar_dbz = radar_dbz[~remove_inds]
        #radar_ldr = radar_ldr[~remove_inds]
        radar_vel = radar_vel[~remove_inds]
        radar_width = radar_width[~remove_inds]
    er2_time = er2_time[~remove_inds]
    er2_x = er2_x[~remove_inds]
    er2_y = er2_y[~remove_inds]
    er2_alt = er2_alt[~remove_inds]

    # Trim P-3 nav data with +/- 1 min buffer on either side of specified period (since P-3 legs differ from the ER-2)
    start_dt64 = start_dt64 - np.timedelta64(1, 'm')
    end_dt64 = end_dt64 + np.timedelta64(1, 'm')
    time_inds = np.where((p3_time>=start_dt64) & (p3_time<=end_dt64))[0]
    if ('time_midpoint' in p3obj.keys()) and (p3_time[time_inds[-1]]==end_dt64): # P-3 data averaged in N-sec intervals...need to remove the last ob in time_inds
        time_inds = time_inds[:-1]
    p3_time = p3_time[time_inds]
    p3_lat = p3_lat[time_inds]
    p3_lon = p3_lon[time_inds]
    p3_alt = p3_alt[time_inds]
    
    # This section may need to be populated to handle masked P-3 nav data (will assume everything is fine for now)

    # Set reference point (currently Albany, NY)
    lat_0 = 42.6526
    lon_0 = -73.7562
    
    # Define a map projection to calculate cartesian distances
    p = Proj(proj='laea', zone=10, ellps='WGS84', lat_0=lat_0, lon_0=lon_0)
    
    # Use a projection to get cartiesian distances between the datasets
    er2_x2, er2_y2 = p(er2_x, er2_y)
    p3_x2, p3_y2 = p(p3_lon, p3_lat)
    
    # Set kdtree parameters
    leafsize = 16
    query_eps = 0
    query_p = 2
    query_distance_upper_bound = sphere_size
    query_n_jobs = 1
    K_d = sphere_size
    
    # Perform the kdtree search
    kdt = cKDTree(list(zip(er2_x2, er2_y2, er2_alt)), leafsize=leafsize)
    prdistance, prind1d = kdt.query(list(zip(p3_x2, p3_y2, p3_alt)), k=query_k, eps=query_eps, p=query_p,
                                    distance_upper_bound=query_distance_upper_bound, n_jobs=query_n_jobs)
    # Perform the matching routine
    if query_k==1: # closest gate approach (more simple)
        # Mask matched data that is outside of the defined bounds
        bad_inds = np.where(prind1d == radar_dbz.shape[0])
        if len(bad_inds[0]) > 0:
            print('Nearest radar gate was outside distance upper bound...eliminating those instances')
            #mask inds and distances that are outside the search area
            prind1d[bad_inds] = np.ma.masked
            prdistance[bad_inds] = np.ma.masked
        
        # Trim radar data to only include valid matched values
        dbz_matched = radar_dbz[prind1d]
        vel_matched = radar_vel[prind1d]
        width_matched = radar_width[prind1d]
        dbz_matched = np.ma.masked_where(prind1d == 0, dbz_matched)
        vel_matched = np.ma.masked_where(prind1d == 0, vel_matched)
        width_matched = np.ma.masked_where(prind1d == 0, width_matched)
        if radname=='CRS':
            ldr_matched = radar_ldr[prind1d]
            ldr_matched = np.ma.masked_where(prind1d == 0, ldr_matched)
        elif radname=='HIWRAP':
            ldr_matched = radar_ldr[prind1d]
            dbz2_matched = radar2_dbz[prind1d]
            vel2_matched = radar2_vel[prind1d]
            width2_matched = radar2_width[prind1d]
            ldr2_matched = radar2_ldr[prind1d]
            ldr_matched = np.ma.masked_where(prind1d == 0, ldr_matched)
            dbz2_matched = np.ma.masked_where(prind1d == 0, dbz2_matched)
            vel2_matched = np.ma.masked_where(prind1d == 0, vel2_matched)
            width2_matched = np.ma.masked_where(prind1d == 0, width2_matched)
            ldr2_matched = np.ma.masked_where(prind1d == 0, ldr2_matched)
            
        # Get the current P-3 lat,lon and alt to save in the matched dictionary - maybe add other P-3 vars to this later
        time_p3_matched = p3_time
        lat_p3_matched = p3_lat
        lon_p3_matched = p3_lon
        alt_p3_matched = p3_alt

        # Compute the time difference between matched radar obs and the P-3
        time_offset_matched = (er2_time[prind1d] - p3_time) / np.timedelta64(1, 's') # [s]
        
        # Get the current ER-2 nav and radar data to save in the matched dictionary - maybe add other vars to this later
        time_er2_matched = er2_time[prind1d]
        lat_er2_matched = er2_y[prind1d]
        lon_er2_matched = er2_x[prind1d]
        alt_er2_matched = er2_alt[prind1d]
        dist_er2_matched = prdistance
        ind_er2_matched = prind1d # TODO: This will be useful var in Barnes-weighted mean for query_k>1
        
    else: # do a Barnes weighted mean of the radar gates
        # Mask matched data that is outside of the defined bounds
        bad_inds = np.where(prind1d == radar_dbz.shape[0])
        if len(bad_inds[0]) > 0 or len(bad_inds[1]) > 0:
            print('Nearest radar gate was outside distance upper bound...eliminating those instances')
            #mask inds and distances that are outside the search area
            prind1d[bad_inds] = np.ma.masked
            prdistance[bad_inds] = np.ma.masked

        # Trim radar data to only include valid matched values
        dbz_matched = radar_dbz[prind1d]
        dbz_matched = np.ma.masked_where(prind1d == 0, dbz_matched)
        # vel_matched = radar_vel[prind1d]
        # vel_matched = np.ma.masked_where(prind1d == 0, vel_matched)
        width_matched = radar_width[prind1d]
        width_matched = np.ma.masked_where(prind1d == 0, width_matched)
        if radname=='CRS':
            ldr_matched = radar_ldr[prind1d]
            ldr_matched = np.ma.masked_where(prind1d == 0, ldr_matched)
        elif radname=='HIWRAP':
            ldr_matched = radar_ldr[prind1d]
            ldr_matched = np.ma.masked_where(prind1d == 0, ldr_matched)
            dbz2_matched = radar2_dbz[prind1d]
            dbz2_matched = np.ma.masked_where(prind1d == 0, dbz2_matched)
            # vel2_matched = radar2_vel[prind1d]
            # vel2_matched = np.ma.masked_where(prind1d == 0, vel2_matched)
            width2_matched = radar2_width[prind1d]
            width2_matched = np.ma.masked_where(prind1d == 0, width2_matched)
            ldr2_matched = radar2_ldr[prind1d]
            ldr2_matched = np.ma.masked_where(prind1d == 0, ldr2_matched)

        # Eliminate observations that are outliers (e.g., skin paints) before averaging the data
        # Follows Chase et al. (2018, JGR; https://github.com/dopplerchase/Chase_et_al_2018/blob/master/apr3tocit_tools.py)
        # See http://colingorrie.github.io/outlier-detection.html for more info
        # dbz
        if outlier_method=='iqr':
            IQR = np.array([])
            for i in range(dbz_matched.shape[0]):
                dbz_matched_sub = dbz_matched[i, :]
                dbz_matched_sub = dbz_matched_sub[~dbz_matched_sub.mask] # remove masked matched values
                if len(dbz_matched_sub)==0:
                    IQR = np.append(IQR, np.nan)
                else: # mask gates where dbz > 1.5*IQR above 75th percentile
                    centiles = np.nanpercentile(dbz_matched_sub, [25, 75])
                    if isinstance(centiles, np.ndarray):
                        IQR = np.append(IQR, centiles[1] - centiles[0])
                        dbz_matched_sub = np.ma.masked_where(dbz_matched_sub > centiles[1]+1.5*IQR[-1], dbz_matched_sub)
                        dbz_matched[i, :] = dbz_matched_sub
            IQR = np.ma.masked_invalid(IQR)
        elif outlier_method=='ldr':
            IQR = np.array([])
            for i in range(dbz_matched.shape[0]):
                dbz_matched_sub = dbz_matched[i, :]
                ldr_matched_sub = ldr_matched[i, :]
                '''
                if len(~dbz_matched_sub.mask)!=len(ldr_matched_sub):
                    print(dbz_matched_sub)
                    print(dbz_matched_sub.mask)
                    print(ldr_matched_sub)
                '''
                #ldr_matched_sub = ldr_matched_sub[~dbz_matched_sub.mask] # remove masked matched values
                #dbz_matched_sub = dbz_matched_sub[~dbz_matched_sub.mask] # remove masked matched values
                if len(dbz_matched_sub)==0:
                    IQR = np.append(IQR, np.nan)
                else:
                    #centiles = np.nanpercentile(dbz_matched_sub, [25, 75])
                    centiles = np.nanpercentile(dbz_matched_sub.compressed(), [25, 75])
                    if isinstance(centiles, np.ndarray):
                        IQR = np.append(IQR, centiles[1] - centiles[0])
                        if (centiles[1]-centiles[0])>5.: # to impose strict LDR criteria, need to ensure we're truly removing a skin paint
                            ldr_thresh = -20. if radname=='CRS' else -40. # use lower (more negative) LDR threshold for Ku-band
                            dbz_matched_sub = np.ma.masked_where(np.ma.masked_where(dbz_matched_sub.mask, ldr_matched_sub)>ldr_thresh, dbz_matched_sub)
                            dbz_matched[i, :] = dbz_matched_sub
            IQR = np.ma.masked_invalid(IQR)
        elif outlier_method=='modz':
            IQR = np.array([])
            for i in range(dbz_matched.shape[0]):
                dbz_matched_sub = dbz_matched[i, :]
                dbz_matched_sub = dbz_matched_sub[~dbz_matched_sub.mask] # remove masked matched values
                if len(dbz_matched_sub)==0:
                    IQR = np.append(IQR, np.nan)
                else:
                    centiles = np.nanpercentile(dbz_matched_sub, [25, 75])
                    if isinstance(centiles, np.ndarray):
                        IQR = np.append(IQR, centiles[1] - centiles[0])
                        zthresh = 3.5
                        mad = np.ma.median(np.abs(dbz_matched_sub - np.ma.median(dbz_matched_sub))) # median absolute difference
                        zscore = 0.6745 * (dbz_matched_sub - np.ma.median(dbz_matched_sub)) / mad # modified z-score
                        dbz_matched_sub = np.ma.masked_where(zscore>zthresh, dbz_matched_sub)
                        dbz_matched[i, :] = dbz_matched_sub
            IQR = np.ma.masked_invalid(IQR)
        elif outlier_method=='w': # spectrum width skin paint detection
            #width_thresh = np.percentile(radar_width.compressed(), 5) # compute the 5th percentile to use as a threshold
            #print(width_thresh)
            IQR = np.array([])
            for i in range(dbz_matched.shape[0]):
                dbz_matched_sub = dbz_matched[i, :]
                #width_matched_sub = width_matched[i, :]
                #width_matched_sub = width_matched_sub[~dbz_matched_sub.mask] # remove masked matched values
                #dbz_matched_sub = dbz_matched_sub[~dbz_matched_sub.mask] # remove masked matched values
                if len(dbz_matched_sub)==0:
                    IQR = np.append(IQR, np.nan)
                else:
                    #centiles = np.nanpercentile(dbz_matched_sub, [25, 75])
                    centiles = np.nanpercentile(dbz_matched_sub.compressed(), [25, 75])
                    if isinstance(centiles, np.ndarray):
                        IQR = np.append(IQR, centiles[1] - centiles[0])
                    
                    #dbz_thresh = 25. # CAUTION: only tested on EXRAD
                    #dbz_matched_sub = np.ma.masked_where(np.ma.masked_where(dbz_matched_sub.mask, width_matched_sub)<width_thresh, dbz_matched_sub)
                    #dbz_matched_sub = np.ma.masked_where((dbz_matched_sub>=dbz_thresh) & (width_matched_sub<width_thresh), dbz_matched_sub)
                    #dbz_matched[i, :] = dbz_matched_sub
            IQR = np.ma.masked_invalid(IQR)
            
        # dbz2 (HIWRAP only)
        if radname=='HIWRAP':
            if outlier_method=='iqr':
                IQR2 = np.array([])
                for i in range(dbz2_matched.shape[0]):
                    dbz2_matched_sub = dbz2_matched[i, :]
                    dbz2_matched_sub = dbz2_matched_sub[~dbz2_matched_sub.mask] # remove masked matched values
                    if len(dbz2_matched_sub)==0:
                        IQR2 = np.append(IQR2, np.nan)
                    else: # mask gates where dbz > 1.5*IQR above 75th percentile
                        centiles = np.nanpercentile(dbz2_matched_sub, [25, 75])
                        if isinstance(centiles, np.ndarray):
                            IQR2 = np.append(IQR2, centiles[1] - centiles[0])
                            dbz2_matched_sub = np.ma.masked_where(dbz2_matched_sub > centiles[1]+1.5*IQR2[-1], dbz2_matched_sub)
                            dbz2_matched[i, :] = dbz2_matched_sub
                IQR2 = np.ma.masked_invalid(IQR2)
            elif outlier_method=='ldr':
                IQR2 = np.array([])
                for i in range(dbz2_matched.shape[0]):
                    dbz2_matched_sub = dbz2_matched[i, :]
                    ldr2_matched_sub = ldr2_matched[i, :]
                    #ldr2_matched_sub = ldr2_matched_sub[~dbz2_matched_sub.mask] # remove masked matched values
                    #dbz2_matched_sub = dbz2_matched_sub[~dbz2_matched_sub.mask] # remove masked matched values
                    if len(dbz2_matched_sub)==0:
                        IQR2 = np.append(IQR2, np.nan)
                    else:
                        #centiles = np.nanpercentile(dbz2_matched_sub, [25, 75])
                        centiles = np.nanpercentile(dbz2_matched_sub.compressed(), [25, 75])
                        if isinstance(centiles, np.ndarray):
                            IQR2 = np.append(IQR2, centiles[1] - centiles[0])
                            if (centiles[1]-centiles[0])>5.: # to impose strict LDR criteria, need to ensure we're truly removing a skin paint
                                ldr_thresh = -20. # for Ka-band
                                dbz2_matched_sub = np.ma.masked_where(np.ma.masked_where(dbz2_matched_sub.mask, ldr2_matched_sub)>ldr_thresh, dbz2_matched_sub)
                                dbz2_matched[i, :] = dbz2_matched_sub
                IQR2 = np.ma.masked_invalid(IQR2)
            elif outlier_method=='modz':
                IQR2 = np.array([])
                for i in range(dbz2_matched.shape[0]):
                    dbz2_matched_sub = dbz2_matched[i, :]
                    dbz2_matched_sub = dbz2_matched_sub[~dbz2_matched_sub.mask] # remove masked matched values
                    if len(dbz2_matched_sub)==0:
                        IQR2 = np.append(IQR2, np.nan)
                    else:
                        centiles = np.nanpercentile(dbz2_matched_sub, [25, 75])
                        if isinstance(centiles, np.ndarray):
                            IQR2 = np.append(IQR2, centiles[1] - centiles[0])
                            zthresh = 3.5
                            mad = np.ma.median(np.abs(dbz2_matched_sub - np.ma.median(dbz2_matched_sub))) # median absolute difference
                            zscore = 0.6745 * (dbz2_matched_sub - np.ma.median(dbz2_matched_sub)) / mad # modified z-score
                            dbz2_matched_sub = np.ma.masked_where(zscore>zthresh, dbz2_matched_sub)
                            dbz2_matched[i, :] = dbz2_matched_sub
                IQR2 = np.ma.masked_invalid(IQR2)
            elif outlier_method=='w': # spectrum width skin paint detection
                #width2_thresh = np.percentile(radar2_width.compressed(), 5) # compute the 5th percentile to use as a threshold
                #print(width2_thresh)
                IQR2 = np.array([])
                for i in range(dbz2_matched.shape[0]):
                    dbz2_matched_sub = dbz2_matched[i, :]
                    #width2_matched_sub = width2_matched[i, :]
                    #width2_matched_sub = width2_matched_sub[~dbz2_matched_sub.mask] # remove masked matched values
                    #dbz2_matched_sub = dbz2_matched_sub[~dbz2_matched_sub.mask] # remove masked matched values
                    if len(dbz2_matched_sub)==0:
                        IQR2 = np.append(IQR2, np.nan)
                    else:
                        #centiles = np.nanpercentile(dbz2_matched_sub, [25, 75])
                        centiles = np.nanpercentile(dbz2_matched_sub.compressed(), [25, 75])
                        if isinstance(centiles, np.ndarray):
                            IQR2 = np.append(IQR2, centiles[1] - centiles[0])

                        #dbz2_thresh = 25. # CAUTION: only tested on EXRAD
                        #dbz2_matched_sub = np.ma.masked_where(np.ma.masked_where(dbz2_matched_sub.mask, width2_matched_sub)<width2_thresh, dbz2_matched_sub)
                        #dbz2_matched[i, :] = dbz2_matched_sub
                IQR2 = np.ma.masked_invalid(IQR2)

        # Barnes-weighted mean and n-gate standard deviation from the mean
        # dbz
        dbz_matched = np.ma.masked_where(np.isnan(dbz_matched), dbz_matched)
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.)) # obtain distance weights
        W_d_k2 = np.ma.masked_where(np.ma.getmask(dbz_matched), W_d_k.copy()) # mask weights where dbz is masked
        w1 = np.ma.sum(W_d_k2 * 10.**(dbz_matched/10.), axis=1) # weighted sum of linear reflectivity (mm^6 m^-3) per matched period
        w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights for each matched period (n-sec interval)
        dbz_matched_temp = dbz_matched.copy()
        dbz_matched = 10. * np.ma.log10(w1 / w2) # matched dbz will now be 1-D array instead of 2 (was nTimes x query_k)
        dbz_stdev = np.ma.zeros(dbz_matched.shape[0])
        for i in range(dbz_matched_temp.shape[0]):
            square_diff = (dbz_matched_temp[i, :] - dbz_matched[i])**2. # squared differences between gates and weighted mean
            ssd = np.nansum(square_diff) # sum of squared differences between gates and weighted mean
            if np.isnan(ssd):
                dbz_stdev[i] = np.nan
            else:
                num_goodvals = len(dbz_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                dbz_stdev[i] = np.sqrt(ssd / num_goodvals)
        dbz_stdev = np.ma.masked_invalid(dbz_stdev)
        dbz_matched = np.ma.masked_where(dbz_stdev>5., dbz_matched) # found to be suspected skin paint artifact

        # dbz2 (HIWRAP only)
        if radname=='HIWRAP':
            dbz2_matched = np.ma.masked_where(np.isnan(dbz2_matched), dbz2_matched)
            W_d_k = np.ma.array(np.exp(-1*  prdistance**2. / K_d**2.)) # obtain distance weights
            W_d_k2 = np.ma.masked_where(np.ma.getmask(dbz2_matched), W_d_k.copy()) # mask weights where dbz is masked
            w1 = np.ma.sum(W_d_k2 * 10.**(dbz2_matched/10.), axis=1) # weighted sum of linear reflectivity (mm^6 m^-3) per matched period
            w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights for each matched period (n-sec interval)
            dbz2_matched_temp = dbz2_matched.copy()
            dbz2_matched = 10. * np.ma.log10(w1 / w2) # matched dbz will now be 1-D array instead of 2 (was nTimes x query_k)
            dbz2_stdev = np.ma.zeros(dbz2_matched.shape[0])
            for i in range(dbz2_matched_temp.shape[0]):
                square_diff = (dbz2_matched_temp[i, :] - dbz2_matched[i])**2. # squared differences between gates and weighted mean
                ssd = np.nansum(square_diff) # sum of squared differences between gates and weighted mean
                if np.isnan(ssd):
                    dbz2_stdev[i] = np.nan
                else:
                    num_goodvals = len(dbz2_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                    dbz2_stdev[i] = np.sqrt(ssd / num_goodvals)
            dbz2_stdev = np.ma.masked_invalid(dbz2_stdev)
            dbz2_matched = np.ma.masked_where(dbz2_stdev>5., dbz2_matched) # found to be suspected skin paint artifact

        # Get the current P-3 lat,lon and alt to save in the matched dictionary - maybe add other P-3 vars to this later
        time_p3_matched = p3_time
        lat_p3_matched = p3_lat
        lon_p3_matched = p3_lon
        alt_p3_matched = p3_alt

        # Compute time difference, using same Barnes weighting technique
        p3_time_tile = np.tile(np.reshape(p3_time, (len(p3_time), 1)), (1, query_k))
        time_offset_tile = (er2_time[prind1d] - p3_time_tile) / np.timedelta64(1, 's') # [s]
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.))
        W_d_k2 = np.ma.masked_where(np.ma.getmask(time_offset_tile), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * time_offset_tile, axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        time_offset_matched = w1 / w2
        time_er2_matched = np.array([], dtype='datetime64[ns]')
        # print(p3_time.shape, time_offset_matched.shape)
        for i in range(len(time_offset_matched)):
            # print(p3_time[i], time_offset_matched[i], p3_time[i]+np.timedelta64(int(time_offset_matched[i]), 's'))
            time_er2_matched = np.append(time_er2_matched, p3_time[i] + np.timedelta64(int(time_offset_matched[i]), 's'))

        # Compute distance between P-3 and ER-2 gates, using same Barnes weighting technique
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.))
        W_d_k2 = np.ma.masked_where(np.ma.getmask(prdistance), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * prdistance, axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        dist_er2_matched = w1 / w2

        # Compute ER-2 matched latitude, longitude, and altitude, using same Barnes weighting technique
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.))

        W_d_k2 = np.ma.masked_where(np.ma.getmask(er2_y[prind1d]), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * er2_y[prind1d], axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        lat_er2_matched = w1 / w2

        W_d_k2 = np.ma.masked_where(np.ma.getmask(er2_x[prind1d]), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * er2_x[prind1d], axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        lon_er2_matched = w1 / w2

        W_d_k2 = np.ma.masked_where(np.ma.getmask(er2_alt[prind1d]), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * er2_alt[prind1d], axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        alt_er2_matched = w1 / w2

    # Create the dictionaries
    match_obj = {}
    
    kdtree = {}
    kdtree['prind1d'] = {}; kdtree['prdistance'] = {}; kdtree['query_k'] = {}
    kdtree['prind1d']['data'] = prind1d
    kdtree['prind1d']['info'] = 'Index in the raveled radar array (after removing masked dbz values) for the matched values'
    kdtree['prdistance']['data'] = dist_er2_matched
    kdtree['prdistance']['info'] = 'Cartesian distance between the P-3 and matched radar gate (Barnes average if query_k greater than 1) [m]'
    kdtree['query_k']['data'] = query_k
    kdtree['query_k']['info'] = 'Number of gates that were considered to be matched'
    
    matched = {}
    matched['time_p3'] = {}; matched['lat_p3'] = {}; matched['lon_p3'] = {}; matched['alt_p3'] = {}
    matched['time_rad'] = {}; matched['lat_rad'] = {}; matched['lon_rad'] = {}; matched['alt_rad'] = {}
    matched['dist'] = {}; matched['time_diff'] = {}
    matched['time_p3']['data'] = time_p3_matched
    matched['time_p3']['info'] = 'Time of the P-3 observation [numpy datetime64]'
    matched['lat_p3']['data'] = lat_p3_matched
    matched['lat_p3']['info'] = 'Latitude of the P-3 aircraft [deg]'
    matched['lon_p3']['data'] = lon_p3_matched
    matched['lon_p3']['info'] = 'Longitude of the P-3 aircraft [deg]'
    matched['alt_p3']['data'] = alt_p3_matched
    matched['alt_p3']['info'] = 'Altitude of the P-3 aircraft [m]'
    matched['time_rad']['data'] = time_er2_matched
    matched['time_rad']['info'] = 'Time of the matched radar observation [numpy datetime64]'
    matched['lat_rad']['data'] = lat_er2_matched
    matched['lat_rad']['info'] = 'Latitude of the center of the matched radar gates [deg]'
    matched['lon_rad']['data'] = lon_er2_matched
    matched['lon_rad']['info'] = 'Longitude of the center of the matched radar gates [deg]'
    matched['alt_rad']['data'] = alt_er2_matched
    matched['alt_rad']['info'] = 'Altitude of the center of the matched radar gates [m ASL]'
    matched['dist']['data'] = dist_er2_matched
    matched['dist']['info'] = 'Cartesian distance between the P-3 and matched radar gate (Barnes average if query_k greater than 1) [m]'
    matched['time_diff']['data'] = time_offset_matched
    matched['time_diff']['info'] = 'Time difference between the matched radar gate and the P-3 observation [s]'
    if radname=='CRS': # Potentially add the other radar vars to the dictionary later
        matched['dbz_W'] = {}
        matched['dbz_W']['data'] = dbz_matched
        matched['dbz_W']['info'] = 'CRS W-band equivalent reflectivity factor matched to the P-3 location [dBZ]'
        if query_k>1:
            if outlier_method is not None:
                matched['dbz_W_IQR'] = {}
                matched['dbz_W_IQR']['data'] = IQR
                matched['dbz_W_IQR']['info'] = 'Interquartile range in reflectivity for n-closest gates, before noise filtering'
            matched['dbz_W_std'] = {}
            matched['dbz_W_std']['data'] = dbz_stdev
            matched['dbz_W_std']['info'] = 'Standard deviation in reflectivity for n-closest gates from the Barnes-weighted mean'
    elif radname=='HIWRAP': # Potentially add the other radar vars to the dictionary later
        matched['dbz_Ku'] = {}; matched['dbz_Ka'] = {}
        matched['dbz_Ku']['data'] = dbz_matched
        matched['dbz_Ku']['info'] = 'HIWRAP Ku-band equivalent reflectivity factor matched to the P-3 location [dBZ]'
        matched['dbz_Ka']['data'] = dbz2_matched
        matched['dbz_Ka']['info'] = 'HIWRAP Ka-band equivalent reflectivity factor matched to the P-3 location [dBZ]'
        if query_k>1:
            if outlier_method is not None:
                matched['dbz_Ku_IQR'] = {}
                matched['dbz_Ku_IQR']['data'] = IQR
                matched['dbz_Ku_IQR']['info'] = 'Interquartile range in reflectivity for n-closest gates, before noise filtering'
            matched['dbz_Ku_std'] = {}
            matched['dbz_Ku_std']['data'] = dbz_stdev
            matched['dbz_Ku_std']['info'] = 'Standard deviation in reflectivity for n-closest gates from the Barnes-weighted mean'
            if outlier_method is not None:
                matched['dbz_Ka_IQR'] = {}
                matched['dbz_Ka_IQR']['data'] = IQR2
                matched['dbz_Ka_IQR']['info'] = 'Interquartile range in reflectivity for n-closest gates, before noise filtering'
            matched['dbz_Ka_std'] = {}
            matched['dbz_Ka_std']['data'] = dbz2_stdev
            matched['dbz_Ka_std']['info'] = 'Standard deviation in reflectivity for n-closest gates from the Barnes-weighted mean'
    elif radname=='EXRAD': # Potentially add the other radar vars to the dictionary later
        matched['dbz_X'] = {}
        matched['dbz_X']['data'] = dbz_matched
        matched['dbz_X']['info'] = 'EXRAD nadir-beam X-band equivalent reflectivity factor matched to the P-3 location [dBZ]'
        if query_k>1:
            matched['dbz_X_IQR'] = {}
            matched['dbz_X_IQR']['data'] = IQR
            matched['dbz_X_IQR']['info'] = 'Interquartile range in reflectivity for n-closest gates, before noise filtering'
            matched['dbz_X_std'] = {}
            matched['dbz_X_std']['data'] = dbz_stdev
            matched['dbz_X_std']['info'] = 'Standard deviation in reflectivity for n-closest gates from the Barnes-weighted mean'
    
    if return_indices:
        matched['prind1d']['data'] = prind1d
        matched['prind1d']['info'] = 'Index in the raveled radar array (after removing masked dbz values) for the matched values'
        
    match_obj['kdtree'] = kdtree
    match_obj['matched'] = matched
        
    return match_obj

def match_nn(er2obj, p3obj, Dm_liquid, Dm_solid, Nw, IWC, sphere_size, start_time, end_time, query_k=1, outlier_method=None, return_indices=False):
    '''
    Get the matched neural network (NN) radar retrieval data based on the P-3 lat, lon, alt.
    Since the NN retrieval can be computationally intensive, Dm_liquid, Dm_solid, Nw, and IWC need to be trimmed inputs.
    Inputs:
        er2_obj: ER-2 HIWRAP object obtained from the er2read() function
        p3_obj: P-3 object obtained from the iwgread() and iwg_avg() functions
        Dm_liquid: Retrieved Dm (liquid; mm) trimmed from start/end times
        Dm_solid: Retrieved Dm (solid; mm) trimmed from start/end times
        Dm_liquid: Retrieved Dm (liquid; mm) trimmed from start/end times
        Nw: Retrieved Nw (m**-4) trimmed from start/end times
        IWC: Retrieved IWC (g m**-3) trimmed from start/end times
        sphere_size: Maximum distance [int in m] allowed in the kdTree search
        start_time: Start time [str in YYYY-MM-DDTHH:MM:SS format] to consider in matching routine
        end_time: End time [str in YYYY-MM-DDTHH:MM:SS format] to consider in matching routine
        query_k: Number of gates (int) considered in the average (1 == use closest)
        return_indices: True == returns the matched gates in 1d coords; False == does not
    '''
    # Load P-3 info and trim if needed
    p3_time = p3obj['time']['data']
    p3_lat = p3obj['Latitude']['data']
    p3_lon = p3obj['Longitude']['data']
    p3_alt = p3obj['GPS_Altitude']['data']
    
    start_dt64 = np.datetime64(start_time)
    end_dt64 = np.datetime64(end_time)
    
    # Trim radar spatial data (to match NN retrieval data) and turn into 1-D arrays
    time_inds = np.where((er2obj['time']>=np.datetime64(start_time)) & (er2obj['time']<=np.datetime64(end_time)))[0]
    er2_time = np.ravel(er2obj['time_gate'][:, time_inds])
    er2_x = np.ravel(er2obj['lon_gate'][:, time_inds])
    er2_y = np.ravel(er2obj['lat_gate'][:, time_inds])
    er2_alt = np.ravel(er2obj['alt_gate'][:, time_inds])

    # Turn NN retrieval data into 1-D arrays
    nn_dm_liq = np.ma.ravel(Dm_liquid[:, :])
    nn_dm_sol = np.ma.ravel(Dm_solid[:, :])
    nn_nw = np.ma.ravel(Nw[:, :])
    nn_iwc = np.ma.ravel(IWC[:, :])

    # Remove NN retrieval values/gates where data are masked
    # Should be ==nn_*.mask if all vars were properly masked outside func
    remove_inds = np.logical_or.reduce((nn_dm_liq.mask, nn_dm_sol.mask, nn_nw.mask, nn_iwc.mask))
    nn_dm_liq = nn_dm_liq[~remove_inds]
    nn_dm_sol = nn_dm_sol[~remove_inds]
    nn_nw = nn_nw[~remove_inds]
    nn_iwc = nn_iwc[~remove_inds]
    er2_time = er2_time[~remove_inds]
    er2_x = er2_x[~remove_inds]
    er2_y = er2_y[~remove_inds]
    er2_alt = er2_alt[~remove_inds]

    # Trim P-3 nav data with +/- 1 min buffer on either side of specified period (since P-3 legs differ from the ER-2)
    start_dt64 = start_dt64 - np.timedelta64(1, 'm')
    end_dt64 = end_dt64 + np.timedelta64(1, 'm')
    time_inds = np.where((p3_time>=start_dt64) & (p3_time<=end_dt64))[0]
    if ('time_midpoint' in p3obj.keys()) and (p3_time[time_inds[-1]]==end_dt64): # P-3 data averaged in N-sec intervals...need to remove the last ob in time_inds
        time_inds = time_inds[:-1]
    p3_time = p3_time[time_inds]
    p3_lat = p3_lat[time_inds]
    p3_lon = p3_lon[time_inds]
    p3_alt = p3_alt[time_inds]
    
    # This section may need to be populated to handle masked P-3 nav data (will assume everything is fine for now)

    # Set reference point (currently Albany, NY)
    lat_0 = 42.6526
    lon_0 = -73.7562
    
    # Define a map projection to calculate cartesian distances
    p = Proj(proj='laea', zone=10, ellps='WGS84', lat_0=lat_0, lon_0=lon_0)
    
    # Use a projection to get cartiesian distances between the datasets
    er2_x2, er2_y2 = p(er2_x, er2_y)
    p3_x2, p3_y2 = p(p3_lon, p3_lat)
    
    # Set kdtree parameters
    leafsize = 16
    query_eps = 0
    query_p = 2
    query_distance_upper_bound = sphere_size
    query_n_jobs = 1
    K_d = sphere_size
    
    # Perform the kdtree search
    kdt = cKDTree(list(zip(er2_x2, er2_y2, er2_alt)), leafsize=leafsize)
    prdistance, prind1d = kdt.query(list(zip(p3_x2, p3_y2, p3_alt)), k=query_k, eps=query_eps, p=query_p,
                                    distance_upper_bound=query_distance_upper_bound, n_jobs=query_n_jobs)
    # Perform the matching routine
    if query_k==1: # closest gate approach (more simple)
        # Mask matched data that is outside of the defined bounds
        bad_inds = np.where(prind1d == nn_dm_liq.shape[0])
        if len(bad_inds[0]) > 0:
            print('Nearest radar gate was outside distance upper bound...eliminating those instances')
            #mask inds and distances that are outside the search area
            prind1d[bad_inds] = np.ma.masked
            prdistance[bad_inds] = np.ma.masked
        
        # Trim NN retrieval data to only include valid matched values
        dm_liq_matched = nn_dm_liq[prind1d]
        dm_sol_matched = nn_dm_sol[prind1d]
        nw_matched = nn_nw[prind1d]
        iwc_matched = nn_iwc[prind1d]
        dm_liq_matched = np.ma.masked_where(prind1d == 0, dm_liq_matched)
        dm_sol_matched = np.ma.masked_where(prind1d == 0, dm_sol_matched)
        nw_matched = np.ma.masked_where(prind1d == 0, nw_matched)
        iwc_matched = np.ma.masked_where(prind1d == 0, iwc_matched)
            
        # Get the current P-3 lat,lon and alt to save in the matched dictionary - maybe add other P-3 vars to this later
        time_p3_matched = p3_time
        lat_p3_matched = p3_lat
        lon_p3_matched = p3_lon
        alt_p3_matched = p3_alt

        # Compute the time difference between matched radar obs and the P-3
        time_offset_matched = (er2_time[prind1d] - p3_time) / np.timedelta64(1, 's') # [s]
        
        # Get the current ER-2 nav and radar data to save in the matched dictionary - maybe add other vars to this later
        time_er2_matched = er2_time[prind1d]
        lat_er2_matched = er2_y[prind1d]
        lon_er2_matched = er2_x[prind1d]
        alt_er2_matched = er2_alt[prind1d]
        dist_er2_matched = prdistance
        ind_er2_matched = prind1d # TODO: This will be useful var in Barnes-weighted mean for query_k>1
        
    else: # do a Barnes weighted mean of the NN retrieval gates
        # Mask matched data that is outside of the defined bounds
        bad_inds = np.where(prind1d == nn_dm_liq.shape[0])
        if len(bad_inds[0]) > 0 or len(bad_inds[1]) > 0:
            print('Nearest radar gate was outside distance upper bound...eliminating those instances')
            #mask inds and distances that are outside the search area
            prind1d[bad_inds] = np.ma.masked
            prdistance[bad_inds] = np.ma.masked

        # Trim NN retrieval data to only include valid matched values
        dm_liq_matched = nn_dm_liq[prind1d]
        dm_sol_matched = nn_dm_sol[prind1d]
        nw_matched = nn_nw[prind1d]
        iwc_matched = nn_iwc[prind1d]
        dm_liq_matched = np.ma.masked_where(prind1d == 0, dm_liq_matched)
        dm_sol_matched = np.ma.masked_where(prind1d == 0, dm_sol_matched)
        nw_matched = np.ma.masked_where(prind1d == 0, nw_matched)
        iwc_matched = np.ma.masked_where(prind1d == 0, iwc_matched)

        # === Barnes-weighted mean and n-gate standard deviation from the mean ===
        # dm_liq
        dm_liq_matched = np.ma.masked_where(np.isnan(dm_liq_matched), dm_liq_matched)
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.)) # obtain distance weights
        W_d_k2 = np.ma.masked_where(np.ma.getmask(dm_liq_matched), W_d_k.copy()) # mask weights where dm is masked
        w1 = np.ma.sum(W_d_k2 * dm_liq_matched, axis=1) # weighted sum of dm per matched period
        w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights for each matched period (n-sec interval)
        dm_liq_matched_temp = dm_liq_matched.copy()
        dm_liq_matched = w1 / w2 # matched dm will now be 1-D array instead of 2 (was nTimes x query_k)
        dm_liq_stdev = np.ma.zeros(dm_liq_matched.shape[0])
        for i in range(dm_liq_matched_temp.shape[0]):
            square_diff = (dm_liq_matched_temp[i, :] - dm_liq_matched[i])**2. # squared differences between gates and weighted mean
            ssd = np.nansum(square_diff) # sum of squared differences between gates and weighted mean
            if np.isnan(ssd):
                dm_liq_stdev[i] = np.nan
            else:
                num_goodvals = len(dm_liq_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                dm_liq_stdev[i] = np.sqrt(ssd / num_goodvals)
        dm_liq_stdev = np.ma.masked_invalid(dm_liq_stdev)
        #dm_liq_matched = np.ma.masked_where(dm_liq_stdev>5., dm_liq_matched) # found to be suspected skin paint artifact
        
        # dm_sol
        dm_sol_matched = np.ma.masked_where(np.isnan(dm_sol_matched), dm_sol_matched)
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.)) # obtain distance weights
        W_d_k2 = np.ma.masked_where(np.ma.getmask(dm_sol_matched), W_d_k.copy()) # mask weights where dm is masked
        w1 = np.ma.sum(W_d_k2 * dm_sol_matched, axis=1) # weighted sum of dm per matched period
        w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights for each matched period (n-sec interval)
        dm_sol_matched_temp = dm_sol_matched.copy()
        dm_sol_matched = w1 / w2 # matched dm will now be 1-D array instead of 2 (was nTimes x query_k)
        dm_sol_stdev = np.ma.zeros(dm_sol_matched.shape[0])
        for i in range(dm_sol_matched_temp.shape[0]):
            square_diff = (dm_sol_matched_temp[i, :] - dm_sol_matched[i])**2. # squared differences between gates and weighted mean
            ssd = np.nansum(square_diff) # sum of squared differences between gates and weighted mean
            if np.isnan(ssd):
                dm_sol_stdev[i] = np.nan
            else:
                num_goodvals = len(dm_sol_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                dm_sol_stdev[i] = np.sqrt(ssd / num_goodvals)
        dm_sol_stdev = np.ma.masked_invalid(dm_sol_stdev)
        #dm_sol_matched = np.ma.masked_where(dm_sol_stdev>5., dm_sol_matched) # found to be suspected skin paint artifact
        
        # nw
        nw_matched = np.ma.masked_where(np.isnan(nw_matched), nw_matched)
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.)) # obtain distance weights
        W_d_k2 = np.ma.masked_where(np.ma.getmask(nw_matched), W_d_k.copy()) # mask weights where dm is masked
        w1 = np.ma.sum(W_d_k2 * nw_matched, axis=1) # weighted sum of nw per matched period
        w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights for each matched period (n-sec interval)
        nw_matched_temp = nw_matched.copy()
        nw_matched = w1 / w2 # matched nw will now be 1-D array instead of 2 (was nTimes x query_k)
        nw_stdev = np.ma.zeros(nw_matched.shape[0])
        for i in range(nw_matched_temp.shape[0]):
            square_diff = (nw_matched_temp[i, :] - nw_matched[i])**2. # squared differences between gates and weighted mean
            ssd = np.nansum(square_diff) # sum of squared differences between gates and weighted mean
            if np.isnan(ssd):
                nw_stdev[i] = np.nan
            else:
                num_goodvals = len(nw_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                nw_stdev[i] = np.sqrt(ssd / num_goodvals)
        nw_stdev = np.ma.masked_invalid(nw_stdev)
        #nw_matched = np.ma.masked_where(nw_stdev>5., nw_matched) # found to be suspected skin paint artifact
        
        # iwc
        iwc_matched = np.ma.masked_where(np.isnan(iwc_matched), iwc_matched)
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.)) # obtain distance weights
        W_d_k2 = np.ma.masked_where(np.ma.getmask(iwc_matched), W_d_k.copy()) # mask weights where dm is masked
        w1 = np.ma.sum(W_d_k2 * iwc_matched, axis=1) # weighted sum of dm per matched period
        w2 = np.ma.sum(W_d_k2, axis=1) # sum of weights for each matched period (n-sec interval)
        iwc_matched_temp = iwc_matched.copy()
        iwc_matched = w1 / w2 # matched dm will now be 1-D array instead of 2 (was nTimes x query_k)
        iwc_stdev = np.ma.zeros(iwc_matched.shape[0])
        for i in range(iwc_matched_temp.shape[0]):
            square_diff = (iwc_matched_temp[i, :] - iwc_matched[i])**2. # squared differences between gates and weighted mean
            ssd = np.nansum(square_diff) # sum of squared differences between gates and weighted mean
            if np.isnan(ssd):
                iwc_stdev[i] = np.nan
            else:
                num_goodvals = len(iwc_matched_temp[i, :]) - np.sum(np.isnan(square_diff))
                iwc_stdev[i] = np.sqrt(ssd / num_goodvals)
        iwc_stdev = np.ma.masked_invalid(iwc_stdev)
        #iwc_matched = np.ma.masked_where(iwc_stdev>5., iwc_matched) # found to be suspected skin paint artifact

        # Get the current P-3 lat,lon and alt to save in the matched dictionary - maybe add other P-3 vars to this later
        time_p3_matched = p3_time
        lat_p3_matched = p3_lat
        lon_p3_matched = p3_lon
        alt_p3_matched = p3_alt

        # Compute time difference, using same Barnes weighting technique
        p3_time_tile = np.tile(np.reshape(p3_time, (len(p3_time), 1)), (1, query_k))
        time_offset_tile = (er2_time[prind1d] - p3_time_tile) / np.timedelta64(1, 's') # [s]
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.))
        W_d_k2 = np.ma.masked_where(np.ma.getmask(time_offset_tile), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * time_offset_tile, axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        time_offset_matched = w1 / w2
        time_er2_matched = np.array([], dtype='datetime64[ns]')
        # print(p3_time.shape, time_offset_matched.shape)
        for i in range(len(time_offset_matched)):
            # print(p3_time[i], time_offset_matched[i], p3_time[i]+np.timedelta64(int(time_offset_matched[i]), 's'))
            time_er2_matched = np.append(time_er2_matched, p3_time[i] + np.timedelta64(int(time_offset_matched[i]), 's'))

        # Compute distance between P-3 and ER-2 gates, using same Barnes weighting technique
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.))
        W_d_k2 = np.ma.masked_where(np.ma.getmask(prdistance), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * prdistance, axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        dist_er2_matched = w1 / w2

        # Compute ER-2 matched latitude, longitude, and altitude, using same Barnes weighting technique
        W_d_k = np.ma.array(np.exp(-1 * prdistance**2. / K_d**2.))

        W_d_k2 = np.ma.masked_where(np.ma.getmask(er2_y[prind1d]), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * er2_y[prind1d], axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        lat_er2_matched = w1 / w2

        W_d_k2 = np.ma.masked_where(np.ma.getmask(er2_x[prind1d]), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * er2_x[prind1d], axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        lon_er2_matched = w1 / w2

        W_d_k2 = np.ma.masked_where(np.ma.getmask(er2_alt[prind1d]), W_d_k.copy())
        w1 = np.ma.sum(W_d_k2 * er2_alt[prind1d], axis=1)
        w2 = np.ma.sum(W_d_k2, axis=1)
        alt_er2_matched = w1 / w2

    # Create the dictionaries
    match_obj = {}
    
    kdtree = {}
    kdtree['prind1d'] = {}; kdtree['prdistance'] = {}; kdtree['query_k'] = {}
    kdtree['prind1d']['data'] = prind1d
    kdtree['prind1d']['info'] = 'Index in the raveled radar array (after removing masked dbz values) for the matched values'
    kdtree['prdistance']['data'] = dist_er2_matched
    kdtree['prdistance']['info'] = 'Cartesian distance between the P-3 and matched radar gate (Barnes average if query_k greater than 1) [m]'
    kdtree['query_k']['data'] = query_k
    kdtree['query_k']['info'] = 'Number of gates that were considered to be matched'
    
    matched = {}
    matched['time_p3'] = {}; matched['lat_p3'] = {}; matched['lon_p3'] = {}; matched['alt_p3'] = {}
    matched['time_rad'] = {}; matched['lat_rad'] = {}; matched['lon_rad'] = {}; matched['alt_rad'] = {}
    matched['dm_liq'] = {}; matched['dm_sol'] = {}; matched['nw'] = {}; matched['iwc'] = {}
    matched['dist'] = {}; matched['time_diff'] = {}
    matched['time_p3']['data'] = time_p3_matched
    matched['time_p3']['info'] = 'Time of the P-3 observation [numpy datetime64]'
    matched['lat_p3']['data'] = lat_p3_matched
    matched['lat_p3']['info'] = 'Latitude of the P-3 aircraft [deg]'
    matched['lon_p3']['data'] = lon_p3_matched
    matched['lon_p3']['info'] = 'Longitude of the P-3 aircraft [deg]'
    matched['alt_p3']['data'] = alt_p3_matched
    matched['alt_p3']['info'] = 'Altitude of the P-3 aircraft [m]'
    matched['time_rad']['data'] = time_er2_matched
    matched['time_rad']['info'] = 'Time of the matched radar observation [numpy datetime64]'
    matched['lat_rad']['data'] = lat_er2_matched
    matched['lat_rad']['info'] = 'Latitude of the center of the matched radar gates [deg]'
    matched['lon_rad']['data'] = lon_er2_matched
    matched['lon_rad']['info'] = 'Longitude of the center of the matched radar gates [deg]'
    matched['alt_rad']['data'] = alt_er2_matched
    matched['alt_rad']['info'] = 'Altitude of the center of the matched radar gates [m ASL]'
    matched['dist']['data'] = dist_er2_matched
    matched['dist']['info'] = 'Cartesian distance between the P-3 and matched radar gate (Barnes average if query_k greater than 1) [m]'
    matched['time_diff']['data'] = time_offset_matched
    matched['time_diff']['info'] = 'Time difference between the matched radar gate and the P-3 observation [s]'
    matched['dm_liq']['data'] = dm_liq_matched
    matched['dm_liq']['info'] = 'Retrieved liquid equivalent mass-weighted mean diameter [mm]'
    matched['dm_sol']['data'] = dm_sol_matched
    matched['dm_sol']['info'] = 'Retrieved solid/ice phase mass-weighted mean diameter [mm]'
    matched['nw']['data'] = np.ma.log10(nw_matched)
    matched['nw']['info'] = 'Retrieved liquid equivalent normalized intercept parameter [log10(m**-4)]'
    matched['iwc']['data'] = iwc_matched
    matched['iwc']['info'] = 'Retrieved ice water content [g m**-3]'
    if query_k>1:
        matched['dm_liq_stdev'] = {}; matched['dm_sol_stdev'] = {}; matched['nw_stdev'] = {}; matched['iwc_stdev'] = {}
        matched['dm_liq_stdev']['data'] = dm_liq_stdev
        matched['dm_liq_stdev']['info'] = 'Standard deviation in Dm_liquid for n-closest gates from the Barnes-weighted mean [mm]'
        matched['dm_sol_stdev']['data'] = dm_sol_stdev
        matched['dm_sol_stdev']['info'] = 'Standard deviation in Dm_solid for n-closest gates from the Barnes-weighted mean [mm]'
        matched['nw_stdev']['data'] = nw_stdev
        matched['nw_stdev']['info'] = 'Standard deviation in Nw for n-closest gates from the Barnes-weighted mean [m**-4]'
        matched['iwc_stdev']['data'] = iwc_stdev
        matched['iwc_stdev']['info'] = 'Standard deviation in IWC for n-closest gates from the Barnes-weighted mean [g m**-3]'
    if return_indices:
        matched['prind1d']['data'] = prind1d
        matched['prind1d']['info'] = 'Index in the raveled radar array (after removing masked dbz values) for the matched values'
        
    match_obj['kdtree'] = kdtree
    match_obj['matched'] = matched
        
    return match_obj