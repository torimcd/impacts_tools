import xarray as xr
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from datetime import datetime, timedelta
from scipy.optimize import least_squares
try: # try importing the pytmatrix package
    from impacts_tools.forward import *
except ImportError:
    print('WARNING: The pytmatrix package cannot be installed for the psdread() function.')

def get_ames_header(f, datestr_separated):
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
    hdr['VUNIT'] = ['seconds since ' + datestr_separated]
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

def psdread(twodsfile, hvpsfile, datestr, size_cutoff=1., minD=0.15, maxD=30., qc=False, deadtime_thresh=0.7, verbose=True, start_time=None, end_time=None, tres=5.,\
            compute_bulk=False, compute_fits=False, Z_interp=False, matchedZ_W=None, matchedZ_Ka=None, matchedZ_Ku=None, matchedZ_X=None):
    '''
    Load the 2DS and HVPS PSDs processed by UIOOPS and create n-second combined PSDs with optional bulk properties.
    Inputs:
        twodsfile: Path to the 2DS data
        hvpsfile: Path to the HVPS data
        datestr: YYYYMMDD [str]
        size_cutoff: Size [mm] for the 2DS-HVPS crossover
        minD: Minimum size [mm] to consider in the combined PSD
        maxD: Maximum size [mm] to consider in the combined PSD
        qc: Boolean to optionally ignore 1-Hz data from averaging when probe dead time > deadtime_thresh
        deadtime_thresh: Deadtime hreshold [0–1] to ignore 1-Hz data when qc is True
        verbose: Boolean to optionally print all data-related warnings (e.g., high probe dead time)
        start_time: Start time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        end_time: End time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        tres: Averaging interval [s]; tres=1. skips averaging routine
        compute_bulk: Boolean to optionally compute bulk statistics such as N, IWC, Dmm, rho_e
        compute_fits: Boolean to optionally compute gamma fit parameters N0, mu, lambda
        Z_interp: Boolean to optionally simulate Z for additional degrees of riming from Leinonen & Szyrmer (2015; LS15)
        matchedZ_Ka, ...: None (skips minimization) or masked array of matched Z values to perform LS15 m-D minimization
    '''
    p3psd = {}

    if (twodsfile is None) and (hvpsfile is not None):
        print('Only using the HVPS data for {}'.format(datestr))
        size_cutoff = 0.4 # start HVPS PSD at 0.4 mm
    elif (hvpsfile is None) and (twodsfile is not None):
        print('Only using the 2DS data for {}'.format(datestr))
        size_cutoff = 3.2 # end 2DS PSD at 3.2 mm
    elif (twodsfile is None) and (hvpsfile is None):
        print('No input files given...exiting')
        exit()

    # 2DS information
    if twodsfile is not None:
        ds1 = xr.open_dataset(twodsfile)
        time_hhmmss = ds1['time'].values # HHMMSS from flight start date
        time_dt = [datetime(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:])) + timedelta(
            hours=int(str(int(time_hhmmss[i])).zfill(6)[0:2]), minutes=int(str(int(time_hhmmss[i])).zfill(6)[2:4]),
            seconds=int(str(int(time_hhmmss[i])).zfill(6)[4:])) for i in range(len(time_hhmmss))]
        time_str = [datetime.strftime(time_dt[i], '%Y-%m-%dT%H:%M:%S') for i in range(len(time_dt))]
        time = np.array(time_str, dtype='datetime64[s]')
        bin_min_2ds = ds1['bin_min'].values # mm
        bin_max_2ds = ds1['bin_max'].values
        bin_inds = np.where((bin_min_2ds>=minD) & (bin_max_2ds<=size_cutoff))[0] # find bins within user-specified range
        bin_min_2ds = bin_min_2ds[bin_inds]; bin_max_2ds = bin_max_2ds[bin_inds]
        bin_width_2ds = ds1['bin_dD'].values[bin_inds] / 10. # cm
        bin_mid_2ds = bin_min_2ds + (bin_width_2ds * 10.) / 2.
        count_2ds = ds1['count'].values[:, bin_inds]
        sv_2ds = ds1['sample_vol'].values[:, bin_inds] # cm^3
        count_hab_2ds = ds1['habitsd'].values[:, bin_inds, :] * np.tile(np.reshape(sv_2ds, (sv_2ds.shape[0], sv_2ds.shape[1], 1)), (1, 1, 10)) * np.tile(
            np.reshape(bin_width_2ds, (1, len(bin_width_2ds), 1)), (sv_2ds.shape[0], 1, 10))
        ar_2ds = ds1['mean_area_ratio'].values[:, bin_inds] # mean area ratio (circular fit) per bin
        asr_2ds = ds1['mean_aspect_ratio_ellipse'].values[:, bin_inds] # mean aspect ratio (elliptical fit) per bin
        activetime_2ds = ds1['sum_IntArr'].values # s

        if hvpsfile is None:
            count = count_2ds; count_hab = count_hab_2ds; sv = sv_2ds; ar = ar_2ds; asr = asr_2ds; activetime_hvps = np.ones(count.shape[0])
            bin_min = bin_min_2ds; bin_mid = bin_mid_2ds; bin_max = bin_max_2ds; bin_width = bin_width_2ds

    # HVPS information
    if hvpsfile is not None:
        ds2 = xr.open_dataset(hvpsfile)
        bin_min_hvps = ds2['bin_min'].values # mm
        bin_max_hvps = ds2['bin_max'].values
        bin_inds = np.where((bin_min_hvps>=size_cutoff) & (bin_max_hvps<=maxD))[0] # find bins within user-specified range
        bin_min_hvps = bin_min_hvps[bin_inds]; bin_max_hvps = bin_max_hvps[bin_inds]
        bin_width_hvps = ds2['bin_dD'].values[bin_inds] / 10. # cm
        if size_cutoff==2.:
            bin_min_hvps = np.insert(bin_min_hvps, 0, 2.); bin_max_hvps = np.insert(bin_max_hvps, 0, 2.2); bin_width_hvps = np.insert(bin_width_hvps, 0, 0.02)
            bin_inds = np.insert(bin_inds, 0, bin_inds[0]-1)
        bin_mid_hvps = bin_min_hvps + (bin_width_hvps * 10.) / 2.
        count_hvps = ds2['count'].values[:, bin_inds]
        sv_hvps = ds2['sample_vol'].values[:, bin_inds] # cm^3
        count_hab_hvps = (ds2['habitsd'].values[:, bin_inds, :]) * np.tile(np.reshape(sv_hvps, (sv_hvps.shape[0], sv_hvps.shape[1], 1)), (1, 1, 10)) * np.tile(
            np.reshape(bin_width_hvps, (1, len(bin_width_hvps), 1)), (sv_hvps.shape[0], 1, 10))
        ar_hvps = ds2['mean_area_ratio'].values[:, bin_inds] # mean area ratio (circular fit) per bin
        asr_hvps = ds2['mean_aspect_ratio_ellipse'].values[:, bin_inds] # mean aspect ratio (elliptical fit) per bin
        activetime_hvps = ds2['sum_IntArr'].values # s
        if size_cutoff==2.: # normalize counts in first bin (1.8-2.2 mm, now only for 2-2.2 mm)
            count_hvps[:, 0] = count_hvps[:, 0] / 2.
            count_hab_hvps[:, 0, :] = count_hab_hvps[:, 0, :] / 2.

        if twodsfile is None:
            time_hhmmss = ds2['time'].values # HHMMSS from flight start date
            time_dt = [datetime(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:])) + timedelta(
                hours=int(str(int(time_hhmmss[i])).zfill(6)[0:2]), minutes=int(str(int(time_hhmmss[i])).zfill(6)[2:4]),
                seconds=int(str(int(time_hhmmss[i])).zfill(6)[4:])) for i in range(len(time_hhmmss))]
            time_str = [datetime.strftime(time_dt[i], '%Y-%m-%dT%H:%M:%S') for i in range(len(time_dt))]
            time = np.array(time_str, dtype='datetime64[s]')
            count = count_hvps; count_hab = count_hab_hvps; sv = sv_hvps; ar = ar_hvps; asr = asr_hvps; activetime_2ds = np.ones(count.shape[0])
            bin_min = bin_min_hvps; bin_mid = bin_mid_hvps; bin_max = bin_max_hvps; bin_width = bin_width_hvps

    # Combine the datasets
    if (twodsfile is not None) and (hvpsfile is not None):
        count = np.concatenate((count_2ds, count_hvps), axis=1)
        count_hab = np.concatenate((count_hab_2ds, count_hab_hvps), axis=1)
        sv = np.concatenate((sv_2ds, sv_hvps), axis=1)
        ar = np.concatenate((ar_2ds, ar_hvps), axis=1)
        asr = np.concatenate((asr_2ds, asr_hvps), axis=1)
        bin_min = np.concatenate((bin_min_2ds, bin_min_hvps))
        bin_mid = np.concatenate((bin_mid_2ds, bin_mid_hvps))
        bin_max = np.concatenate((bin_max_2ds, bin_max_hvps))
        bin_width = np.concatenate((bin_width_2ds, bin_width_hvps))

    # Average the data
    if start_time is None:
        start_dt64 = time[0]
    else:
        start_dt64 = np.datetime64(start_time)
    if end_time is None:
        end_dt64 = time[-1] if int(tres)>1 else time[-1]+np.timedelta64(1, 's')
    else:
        end_dt64 = np.datetime64(end_time) if int(tres)>1 else np.datetime64(end_time)+np.timedelta64(1, 's')
    dur = (end_dt64 - start_dt64) / np.timedelta64(1, 's') # dataset duration to consider [s]

    # Allocate arrays
    count_aver = np.zeros((int(dur/tres), len(bin_mid)))
    count_hab_aver = np.zeros((int(dur/tres), len(bin_mid), 8))
    sv_aver = np.zeros((int(dur/tres), len(bin_mid)))
    at_2ds_aver = np.ma.array(np.ones(int(dur/tres)), mask=False)
    at_hvps_aver = np.ma.array(np.ones(int(dur/tres)), mask=False)
    ND = np.zeros((int(dur/tres), len(bin_mid)))
    ar_aver = np.zeros((int(dur/tres), len(bin_mid)))
    asr_aver = np.zeros((int(dur/tres), len(bin_mid)))

    time_subset = start_dt64 # allocate time array of N-sec interval obs
    curr_time = start_dt64
    i = 0

    while curr_time+np.timedelta64(int(tres),'s')<=end_dt64:
        if curr_time>start_dt64:
            time_subset = np.append(time_subset, curr_time)
        time_inds = np.where((time>=curr_time) & (time<curr_time+np.timedelta64(int(tres), 's')))[0]
        if qc is True:
            activetime_thresh = 1. - deadtime_thresh
            time_inds = time_inds[(activetime_2ds[time_inds]>=activetime_thresh) & (activetime_hvps[time_inds]>=activetime_thresh)]
        if len(time_inds)>0:
            count_aver[i, :] = np.nansum(count[time_inds, :], axis=0)
            count_hab_aver[i, :, 0] = np.nansum(count_hab[time_inds, :, 3], axis=0) # tiny
            count_hab_aver[i, :, 1] = np.nansum(count_hab[time_inds, :, 0], axis=0) # spherical
            count_hab_aver[i, :, 2] = np.nansum(count_hab[time_inds, :, 1:3], axis=(0, 2)) # oriented + linear
            count_hab_aver[i, :, 3] = np.nansum(count_hab[time_inds, :, 4], axis=0) # hexagonal
            count_hab_aver[i, :, 4] = np.nansum(count_hab[time_inds, :, 5], axis=0) # irregular
            count_hab_aver[i, :, 5] = np.nansum(count_hab[time_inds, :, 6], axis=0) # graupel
            count_hab_aver[i, :, 6] = np.nansum(count_hab[time_inds, :, 7], axis=0) # dendrite
            count_hab_aver[i, :, 7] = np.nansum(count_hab[time_inds, :, 8], axis=0) # aggregate
            ar_aver[i, :] = np.nanmean(ar[time_inds, :], axis=0) # binned mean of area ratio
            asr_aver[i, :] = np.nanmean(asr[time_inds, :], axis=0) # binned mean of aspect ratio
            sv_aver[i, :] = np.nansum(sv[time_inds, :], axis=0)
            at_2ds_aver[i] = np.nansum(activetime_2ds[time_inds]) / len(time_inds)
            at_hvps_aver[i] = np.nansum(activetime_hvps[time_inds]) / len(time_inds)
            ND[i, :] = np.nanmean(count[time_inds, :]/sv[time_inds, :], axis=0) / bin_width # take N(D) for each sec, then average [cm**-4]
        else: # Mask data for current period if dead (active) time from either probe > 0.8*tres (< 0.2*tres) for all 1-Hz times
            if verbose is True:
                print('All 1-Hz data for the {}-s period beginning {} has high dead time. Masking data.'.format(str(tres), np.datetime_as_string(curr_time)))
            at_2ds_aver[i] = np.nansum(activetime_2ds[np.where((time>=curr_time) & (time<curr_time+np.timedelta64(int(tres), 's')))[0]]) / tres; at_2ds_aver.mask[i] = True
            at_hvps_aver[i] = np.nansum(activetime_hvps[np.where((time>=curr_time) & (time<curr_time+np.timedelta64(int(tres), 's')))[0]]) / tres; at_hvps_aver.mask[i] = True
            count_aver[i, :] = np.nan; count_hab_aver[i, :] = np.nan; sv_aver[i, :] = np.nan; ND[i, :] = np.nan; asr_aver[i, :] = np.nan
        i += 1
        curr_time += np.timedelta64(int(tres), 's')

    #ND = np.ma.masked_invalid(count_aver / sv_aver / np.tile(bin_width[np.newaxis, :], (int(dur/tres), 1))) # cm^-4

    # Mask arrays
    count_aver = np.ma.masked_where(np.isnan(count_aver), count_aver)
    count_hab_aver = np.ma.masked_where(np.isnan(count_hab_aver), count_hab_aver)
    sv_aver = np.ma.masked_where(np.isnan(sv_aver), sv_aver)
    ar_aver = np.ma.masked_invalid(ar_aver)
    asr_aver = np.ma.masked_invalid(asr_aver)
    ND[~np.isfinite(ND)] = 0.; ND = np.ma.masked_where(ND==0., ND)

    # Create dictionary
    p3psd['time'] = time_subset
    p3psd['count'] = count_aver
    p3psd['count_habit'] = count_hab_aver
    p3psd['sv'] = sv_aver
    p3psd['area_ratio'] = ar_aver
    p3psd['aspect_ratio'] = asr_aver
    p3psd['ND'] = ND
    p3psd['bin_min'] = bin_min
    p3psd['bin_mid'] = bin_mid
    p3psd['bin_max'] = bin_max
    p3psd['bin_width'] = bin_width
    p3psd['active_time_2ds'] = at_2ds_aver
    p3psd['active_time_hvps'] = at_hvps_aver

    if compute_bulk is True:
        # Compute Z for various degrees of riming and radar wavelengths
        # Based on work from Leionen and Szyrmer 2015
        # (https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2015EA000102)
        # Follows https://github.com/dopplerchase/Leinonen_Python_Forward_Model
        # and uses forward.py and ess238-sup-0002-supinfo.tex in repo
        Z = forward_Z() #initialize class
        # get the PSD in the format to use in the routine (mks units)
        Z.set_PSD(PSD=ND*10.**8, D=bin_mid/1000., dD=bin_width/100., Z_interp=Z_interp)
        Z.load_split_L15() # Load the leinonen output
        Z.fit_sigmas(Z_interp) # Fit the backscatter cross-sections
        Z.fit_rimefrac(Z_interp) # Fit the riming fractions
        Z.calc_Z() # Calculate Z...outputs are Z.Z_x, Z.Z_ku, Z.Z_ka, Z.Z_w for the four radar wavelengths

        # Compute IWC and Dmm following Brown and Francis (1995), modified for a Dmax definition following Hogan et al.
        [
            N0_bf, N0_hy, mu_bf, mu_hy, lam_bf, lam_hy, iwc_bf, iwc_hy, iwc_hab,
            asr_nw, asr_bf, asr_hy, asr_hab, dmm_bf, dmm_hy, dmm_hab, dm_bf, dm_hy,
            dm_hab, rho_bf, rho_hy, rho_hab, rhoe_bf, rhoe_hy, rhoe_hab] = calc_bulk(
            count_aver, count_hab_aver, sv_aver, asr_aver, bin_mid, bin_width)

        # Add bulk variables to the dictionary
        if Z_interp is True:
            p3psd['riming_mass_array'] = [
                0., 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1., 2.]
        else:
            p3psd['riming_mass_array'] = [0., 0.1, 0.2, 0.5, 1., 2.]
        p3psd['a_coeff_array'] = Z.a_coeff
        p3psd['b_coeff_array'] = Z.b_coeff
        p3psd['dbz_W'] = Z.Z_w
        p3psd['dbz_Ka'] = Z.Z_ka
        p3psd['dbz_Ku'] = Z.Z_ku
        p3psd['dbz_X'] = Z.Z_x
        p3psd['N0_bf'] = N0_bf
        p3psd['N0_hy'] = N0_hy
        p3psd['mu_bf'] = mu_bf
        p3psd['mu_hy'] = mu_hy
        p3psd['lambda_bf'] = lam_bf
        p3psd['lambda_hy'] = lam_hy
        p3psd['iwc_bf'] = iwc_bf
        p3psd['iwc_hy'] = iwc_hy
        p3psd['iwc_hab'] = iwc_hab
        p3psd['mean_aspect_ratio'] = asr_nw
        p3psd['mean_aspect_ratio_bf'] = asr_bf
        p3psd['mean_aspect_ratio_hy'] = asr_hy
        p3psd['mean_aspect_ratio_habit'] = asr_hab
        p3psd['dmm_bf'] = dmm_bf
        p3psd['dmm_hy'] = dmm_hy
        p3psd['dmm_hab'] = dmm_hab
        p3psd['dm_bf'] = dm_bf
        p3psd['dm_hy'] = dm_hy
        p3psd['dm_hab'] = dm_hab
        p3psd['eff_density_bf'] = rhoe_bf
        p3psd['eff_density_hy'] = rhoe_hy
        p3psd['eff_density_hab'] = rhoe_hab
        p3psd['density_bf'] = rho_bf
        p3psd['density_hy'] = rho_hy
        p3psd['density_hab'] = rho_hab

        # Optionally constrain the matched Z at Ku- and Ka-band against PSDS to estimate bulk properties
        if (
                matchedZ_W is not None) or (matchedZ_Ka is not None) or (
                matchedZ_Ku is not None) or (matchedZ_X is not None):
            p3psd = calc_riming(
                p3psd, Z, matchedZ_W, matchedZ_Ka, matchedZ_Ku, matchedZ_X,
                compute_fits=compute_fits)

    return p3psd

def cviread(filename, datestr, start_time=None, end_time=None, tres=1.):
    '''
    Load the WHISPER CWC data provided by Darin Toohey (toohey@colorado.edu) and create n-second means of many of these derived properties.
    Inputs:
        filename: Path to the P-3 IWG navigation data
        datestr: Flight start date in YYYY-MM-DD format [str]
        start_time: Start time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        end_time: End time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        tres: Averaging interval [s]; tres=1. skips averaging routine
    '''
    cvi = {}

    # Get header info following the NASA AMES format
    header = get_ames_header(open(filename, 'r', encoding = 'ISO-8859-1'), datestr)

    junk = np.genfromtxt(filename, delimiter=',', skip_header=int(header['NLHEAD']),
        missing_values=header['VMISS'], usemask=True, filling_values=np.nan, encoding = 'ISO-8859-1')

    # Get list of variable names
    name_map = {}
    for var in header['VNAME']:
        name_map[var] = var

    readfile = {}
    if len(header['VNAME']) != len(header['VSCAL']):
        print("ALL variables must be read in this type of file, "
              "please check name_map to make sure it is the "
              "correct length.")
    for jj, name in enumerate(header['VNAME']):
        readfile[name] = np.array(junk[:, jj] * header['VSCAL'][jj])

    # Populate object metadata
    cvi['Information'] = {}
    for ii, comment in enumerate(header['NCOM'][:-1]): # add global attributes
        parsed_comment = comment.split(':')
        cvi['Information'][parsed_comment[0]] = parsed_comment[1][1:]

    # Populate object with variable data and attributes (i.e., units)
    cvi['time'] = {}
    cvi['time']['data']  = np.array([np.datetime64(datestr) + np.timedelta64(int(readfile['time'][i]), 's') for i in
                                     range(len(readfile['time']))])
    cvi['time']['units'] = 'Aircraft flight time as numpy datetime64 object'
    for jj, name in enumerate(header['VNAME'][1:]):
        cvi[name] = {}
        cvi[name]['data'] = readfile[name]
        cvi[name]['data'] = np.ma.masked_where(cvi[name]['data']==-9999., cvi[name]['data'])
        cvi[name]['units'] = header['VUNIT'][jj+1][:]

    if tres>1.:
        if (start_time is None) and (end_time is None):
            start_dt64 = cvi['time']['data'][0]
            end_dt64 = cvi['time']['data'][-1]
        else:
            start_dt64 = np.datetime64(start_time)
            end_dt64 = np.datetime64(end_time)
        dur = (end_dt64 - start_dt64) / np.timedelta64(1, 's') # dataset duration to consider [s]

        # Allocate arrays
        cwc_aver = np.zeros(int(np.ceil(dur/tres)))

        # While loop
        time = cvi['time']['data']
        time_subset = start_dt64 # allocate time array of N-sec interval obs
        curr_time = start_dt64
        i = 0
        while curr_time<end_dt64:
            if curr_time>start_dt64:
                time_subset = np.append(time_subset, curr_time)
            time_inds = np.where((time>=curr_time) & (time<curr_time+np.timedelta64(int(tres), 's')))[0]
            if cvi['CWC']['data'][time_inds].count()>0:
                cwc_aver[i] = np.nanmean(cvi['CWC']['data'][time_inds])
            else:
                cwc_aver[i] = np.nan

            i += 1
            curr_time += np.timedelta64(int(tres), 's')

        # Overwrite 1 Hz data with data from averaging period
        cvi['time']['data'] = time_subset
        cvi['CWC']['data'] = np.ma.masked_invalid(cwc_aver)

    return cvi

def avapsread(filename, datestr):
    # Get header info following the NASA AMES format
    header = get_ames_header(open(filename, 'r', encoding = 'ISO-8859-1'), datestr)

    junk = np.genfromtxt(filename, delimiter=',', skip_header=int(header['NLHEAD']),\
                         missing_values=header['VMISS'], usemask=True, filling_values=np.nan)

    # Get list of variable names
    name_map = {}
    for var in header['VNAME']:
        name_map[var] = var

    readfile = {}
    if len(header['VNAME']) != len(header['VSCAL']):
        print("ALL variables must be read in this type of file, "
              "please check name_map to make sure it is the "
              "correct length.")
    for jj, name in enumerate(header['VNAME']):
        readfile[name] = np.array(junk[:, jj] * header['VSCAL'][jj])

    # Populate object metadata
    avaps = {}

    avaps['Information'] = {}
    for ii, comment in enumerate(header['NCOM'][:-1]): # add global attributes
        parsed_comment = comment.split(':')
        if len(parsed_comment)>1:
            avaps['Information'][parsed_comment[0]] = parsed_comment[1][1:]

    #Populate object with variable data and attributes (i.e., units)
    for jj, name in enumerate(header['VNAME']):
        name_dict = name#.split('_')[0]
        avaps[name_dict] = {}
        avaps[name_dict]['data'] = readfile[name]
        avaps[name_dict]['data'] = np.ma.masked_where(avaps[name_dict]['data']==-9999., avaps[name_dict]['data'])
        avaps[name_dict]['units'] = header['VUNIT'][jj][:].split(',')[0]

    # Fix time variable to be in numpy datetime format
    time = np.array([np.datetime64(avaps['time']['units'].split(' ')[-1])+\
                     np.timedelta64(int(avaps['time']['data'][i]), 's') for i in range(len(avaps['time']['data']))])
    avaps['time']['data'] = time # overwrite time var
    avaps['time']['units'] = 'UTC time'

    return avaps

def tammsread(filename, datestr, start_time=None, end_time=None, tres=1.):
    '''
    Load the WHISPER CWC data provided by Darin Toohey (toohey@colorado.edu) and create n-second means of many of these derived properties.
    Inputs:
        filename: Path to the P-3 IWG navigation data
        datestr: Flight start date in YYYY-MM-DD format [str]
        start_time: Start time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        end_time: End time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        tres: Averaging interval [s]; tres=1. skips averaging routine
    '''
    tamms = {}

    # Get header info following the NASA AMES format
    header = get_ames_header(open(filename, 'r', encoding = 'ISO-8859-1'), datestr)

    junk = np.genfromtxt(filename, delimiter=',', skip_header=int(header['NLHEAD']),
                         missing_values=header['VMISS'], usemask=True, filling_values=np.nan)

    # Get list of variable names
    name_map = {}
    for var in header['VNAME']:
        name_map[var] = var

    readfile = {}
    if len(header['VNAME']) != len(header['VSCAL']):
        print("ALL variables must be read in this type of file, "
              "please check name_map to make sure it is the "
              "correct length.")
    for jj, name in enumerate(header['VNAME']):
        readfile[name] = np.array(junk[:, jj] * header['VSCAL'][jj])

    # Populate object metadata
    tamms['Information'] = {}
    for ii, comment in enumerate(header['NCOM'][:-1]): # add global attributes
        parsed_comment = comment.split(':')
        if len(parsed_comment)>1:
            tamms['Information'][parsed_comment[0]] = parsed_comment[1][1:]

    # Populate object with variable data and attributes (i.e., units)
    tamms['time'] = {}
    tamms['time']['data_raw']  = np.array([np.datetime64(datestr) + np.timedelta64(int(readfile['time'][i]), 's') for i in
                                     range(len(readfile['time']))])
    tamms['time']['units'] = 'Aircraft flight time as numpy datetime64 object'
    for jj, name in enumerate(header['VNAME'][1:]):
        name_dict = name.split('_')[0]
        tamms[name_dict] = {}
        tamms[name_dict]['data_raw'] = readfile[name]
        tamms[name_dict]['data_raw'] = np.ma.masked_where(tamms[name_dict]['data_raw']==-9999., tamms[name_dict]['data_raw'])
        tamms[name_dict]['units'] = header['VUNIT'][jj+1][:].split(',')[0]

    if (start_time is None) and (end_time is None):
        start_dt64 = tamms['time']['data_raw'][0]
        end_dt64 = tamms['time']['data_raw'][-1]
    else:
        start_dt64 = np.datetime64(start_time)
        end_dt64 = np.datetime64(end_time)
    dur = (end_dt64 - start_dt64) / np.timedelta64(1, 's') # dataset duration to consider [s]
    if tres==1.:
        dur += 1 # add an ob to include last second of period

    # Allocate arrays
    for var in list(tamms.keys()):
        if (var!='Information') and (var!='time'):
            tamms[var]['data'] = np.zeros(int(np.ceil(dur/tres)))
            if var=='w':
                tamms['w_std'] = {}
                tamms['w_std']['data'] = np.zeros(int(np.ceil(dur/tres)))
                tamms['w_std']['units'] = 'ms-1'

    # While loop
    time = tamms['time']['data_raw'] # 20 Hz resolution
    time_subset = start_dt64 # allocate time array of N-sec interval obs
    curr_time = start_dt64
    i = 0
    while curr_time<end_dt64:
        if curr_time>start_dt64:
            time_subset = np.append(time_subset, curr_time)
        time_inds = np.where((time>=curr_time) & (time<curr_time+np.timedelta64(int(tres), 's')))[0]
        for jj, var in enumerate(list(tamms.keys())):
            if (var!='Information') and (var!='time') and (var!='w_std'):
                if tamms[var]['data_raw'][time_inds].count()>0:
                    tamms[var]['data'][i] = np.mean(tamms[var]['data_raw'][time_inds].compressed())
                    if var=='w':
                        tamms['w_std']['data'][i] = np.std(tamms['w']['data_raw'][time_inds].compressed())
                else:
                    tamms[var]['data'][i] = np.nan
                    if var=='w':
                        tamms['w_std']['data'][i] = np.nan
        i += 1
        curr_time += np.timedelta64(int(tres), 's')
    if tres==1.:
        time_subset = np.append(time_subset, end_dt64)

    # Mask invalid data
    tamms['time']['data'] = time_subset
    for var in list(tamms.keys()):
        if (var!='Information') and (var!='time'):
            tamms[var]['data'] = np.ma.masked_invalid(tamms[var]['data'])

    return tamms

def fcdpread(filename, datestr, start_time=None, end_time=None, tres=1.):
    '''
    Load the NCAR/UND Fast-CDP data processed by Aaron Bansemer (bansemer@ucar.edu) and create n-second means\
    of the DSD and other bulk properties.
    Inputs:
        filename: Path to the P-3 IWG navigation data
        datestr: Flight start date in YYYY-MM-DD format [str]
        start_time: Start time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        end_time: End time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        tres: Averaging interval [s]; tres=1. skips averaging routine
    '''
    ds = xr.open_dataset(filename)

    time  = np.array([np.datetime64(datestr) + np.timedelta64(int(ds['time'].values[i]), 's') for i in
                      range(len(ds['time']))])
    N_D = ds['CONCENTRATION'].values.T / (10.**8) # cm^-4
    lwc = ds['LWC'].values # g m^-3
    dmm = ds['MMD'].values / 1000. # mm
    n = ds['NT'].values / (10.**6) # cm^-3
    dm = ds['MND'].values / 1000. # mean diameter [mm]
    qc = ds['PROBE_QC'].values # 0: good; 1: medium; 2: bad
    bool_mask = ds['PROBE_QC'].values==2
    bool_mask = np.tile(np.reshape(bool_mask, (len(bool_mask), 1)), (1, 20))

    # Mask bad data
    N_D = np.ma.masked_where(bool_mask==True, N_D)
    lwc = np.ma.masked_where(qc==2, lwc)
    dmm = np.ma.masked_where(qc==2, dmm)
    dmm = np.ma.masked_where(dmm==0., dmm)
    n = np.ma.masked_where(qc==2, n)
    dm = np.ma.masked_where(qc==2, dm)
    dm = np.ma.masked_where(dm==0., dm)

    if tres==1.:
        N_D = np.ma.masked_where(N_D==0., N_D)

        # Assign to dictionary object
        fcdp = {}
        fcdp['time'] = {}; fcdp['time']['data'] = time; fcdp['time']['units'] = 'Aircraft flight time as numpy datetime64 object'
        fcdp['bin_min'] = {}; fcdp['bin_min']['data'] = ds['CONCENTRATION'].attrs['bin_endpoints'][:-1] / 1000.; fcdp['bin_min']['units'] = 'Bin left endpoint [mm]'
        fcdp['bin_mid'] = {}; fcdp['bin_mid']['data'] = ds['CONCENTRATION'].attrs['bin_midpoints'] / 1000.; fcdp['bin_mid']['units'] = 'Bin midpoint [mm]'
        fcdp['bin_max'] = {}; fcdp['bin_max']['data'] = ds['CONCENTRATION'].attrs['bin_endpoints'][1:] / 1000.; fcdp['bin_max']['units'] = 'Bin right endpoint [mm]'
        fcdp['bin_width'] = {}; fcdp['bin_width']['data'] = fcdp['bin_max']['data']-fcdp['bin_min']['data']; fcdp['bin_width']['units'] = 'Binwidth [mm]'
        fcdp['ND'] = {}; fcdp['ND']['data'] = N_D; fcdp['ND']['units'] = 'Number distribution function [cm^-4]'
        fcdp['n'] = {}; fcdp['n']['data'] = n; fcdp['n']['units'] = 'Droplet concentration [L^-1]'
        fcdp['lwc'] = {}; fcdp['lwc']['data'] = lwc; fcdp['lwc']['units'] = 'Liquid water content [g m^-3]'
        fcdp['dm'] = {}; fcdp['dm']['data'] = dm; fcdp['dm']['units'] = 'Mean droplet diameter [mm]'
        fcdp['dmm'] = {}; fcdp['dmm']['data'] = dmm; fcdp['dmm']['units'] = 'Median mass diameter [mm]'
    else:
        if (start_time is None) and (end_time is None):
            start_dt64 = time[0]
            end_dt64 = time[-1]
        else:
            start_dt64 = np.datetime64(start_time)
            end_dt64 = np.datetime64(end_time)
        dur = (end_dt64 - start_dt64) / np.timedelta64(1, 's') # dataset duration to consider [s]

        # Allocate arrays
        ND_aver = np.zeros((int(np.ceil(dur/tres)), N_D.shape[1]))
        lwc_aver = np.zeros(int(np.ceil(dur/tres)))
        n_aver = np.zeros(int(np.ceil(dur/tres)))
        dm_aver = np.zeros(int(np.ceil(dur/tres)))
        dmm_aver = np.zeros(int(np.ceil(dur/tres)))

        # While loop
        time_subset = start_dt64 # allocate time array of N-sec interval obs
        curr_time = start_dt64
        i = 0
        while curr_time<end_dt64:
            if curr_time>start_dt64:
                time_subset = np.append(time_subset, curr_time)
            time_inds = np.where((time>=curr_time) & (time<curr_time+np.timedelta64(int(tres), 's')))[0]
            if dmm[time_inds].count()>0:
                ND_aver[i, :] = np.nanmean(N_D[time_inds, :], axis=0)
                n_aver[i] = np.nanmean(n[time_inds])
                lwc_aver[i] = np.nanmean(lwc[time_inds])
                dm_aver[i] = np.nanmean(dm[time_inds])
                dmm_aver[i] = np.nanmean(dmm[time_inds])
            else:
                ND_aver[i, :] = np.nan; dm_aver[i] = np.nan; dmm_aver[i] = np.nan

            i += 1
            curr_time += np.timedelta64(int(tres), 's')

        # Mask data
        ND_aver = np.ma.masked_invalid(ND_aver)
        n_aver = np.ma.masked_invalid(n_aver)
        lwc_aver = np.ma.masked_invalid(lwc_aver)
        dm_aver = np.ma.masked_invalid(dm_aver)
        dmm_aver = np.ma.masked_invalid(dmm_aver)

        # Assign to dictionary object
        fcdp = {}
        fcdp['time'] = {}; fcdp['time']['data'] = time_subset; fcdp['time']['units'] = 'Aircraft flight time as numpy datetime64 object'
        fcdp['bin_min'] = {}; fcdp['bin_min']['data'] = ds['CONCENTRATION'].attrs['bin_endpoints'][:-1] / 1000.; fcdp['bin_min']['units'] = 'Bin left endpoint [mm]'
        fcdp['bin_mid'] = {}; fcdp['bin_mid']['data'] = ds['CONCENTRATION'].attrs['bin_midpoints'] / 1000.; fcdp['bin_mid']['units'] = 'Bin midpoint [mm]'
        fcdp['bin_max'] = {}; fcdp['bin_max']['data'] = ds['CONCENTRATION'].attrs['bin_endpoints'][1:] / 1000.; fcdp['bin_max']['units'] = 'Bin right endpoint [mm]'
        fcdp['bin_width'] = {}; fcdp['bin_width']['data'] = fcdp['bin_max']['data']-fcdp['bin_min']['data']; fcdp['bin_width']['units'] = 'Binwidth [mm]'
        fcdp['ND'] = {}; fcdp['ND']['data'] = ND_aver; fcdp['ND']['units'] = 'Number distribution function [cm^-4]'
        fcdp['n'] = {}; fcdp['n']['data'] = n_aver; fcdp['n']['units'] = 'Droplet concentration [L^-1]'
        fcdp['lwc'] = {}; fcdp['lwc']['data'] = lwc_aver; fcdp['lwc']['units'] = 'Liquid water content [g m^-3]'
        fcdp['dm'] = {}; fcdp['dm']['data'] = dm_aver; fcdp['dm']['units'] = 'Mean droplet diameter [mm]'
        fcdp['dmm'] = {}; fcdp['dmm']['data'] = dmm_aver; fcdp['dmm']['units'] = 'Median mass diameter [mm]'

    return fcdp

def iwgread(filename, datestr):
    '''
    Load the aircraft IWG navigation data processed by NASA NSERC and create n-second means of many of these derived properties.
    Inputs:
        filename: Path to the P-3 IWG navigation data
        datestr: Flight start date in YYYY-MM-DD format [str]
    '''
    iwg = {}

    # Get header info following the NASA AMES format
    header = get_ames_header(open(filename, 'r'), datestr)

    junk = np.genfromtxt(filename, delimiter=',', skip_header=int(header['NLHEAD']),
                         missing_values=header['VMISS'], usemask=True,
                         filling_values=np.nan)

    # Get list of variable names
    name_map = {}
    for var in header['VNAME']:
        name_map[var] = var

    readfile = {}
    if len(header['VNAME']) != len(header['VSCAL']):
        print("ALL variables must be read in this type of file, "
              "please check name_map to make sure it is the "
              "correct length.")
    for jj, unit in enumerate(header['VUNIT']):
        header['VUNIT'][jj] = unit.split(',')[0]

    for jj, name in enumerate(header['VNAME']):
        if name=='True_Air_Speed' or name=='Indicated_Air_Speed' or name=='Mach_Number':
            header['VMISS'][jj] = -8888.
        if name=='True_Air_Speed' and header['VUNIT'][jj]=='kts': # change TAS to m/s
            header['VSCAL'][jj] = 0.51
            header['VUNIT'][jj] = 'm/s'
        readfile[name] = np.array(junk[:, jj] * header['VSCAL'][jj])
        # Turn missing values to nan
        readfile[name][readfile[name]==header['VMISS'][jj]] = np.nan

    # Populate object metadata
    iwg['Information'] = {}
    instrum_info_counter = 1
    for ii, comment in enumerate(header['NCOM'][:-1]): # add global attributes
        parsed_comment = comment.split(':')
        if len(parsed_comment) > 1:
            iwg['Information'][parsed_comment[0]] = parsed_comment[1][1:]
        else: # handles multiple instrument info lines in *_R0.ict files
            instrum_info_counter += 1
            iwg['Information'][
                'INSTRUMENT_INFO_'+str(instrum_info_counter)] = parsed_comment[0][1:]

    # Populate object with variable data and attributes (i.e., units)
    iwg['time'] = {}
    iwg['time']['data']  = np.array([np.datetime64(datestr) + np.timedelta64(int(readfile['time'][i]), 's') for i in
                                     range(len(readfile['time']))])
    iwg['time']['units'] = 'Aircraft flight time as numpy datetime64 object'#header['VUNIT'][0]
    for jj, name in enumerate(header['VNAME'][2:]):
        iwg[name] = {}
        iwg[name]['data'] = readfile[name]
        iwg[name]['units'] = header['VUNIT'][jj+2][:]

    return iwg

def iwg_average(iwgdata, start_time=None, end_time=None, tres=5.):
    '''
    Average aircraft state parameters according to tres parameter.
    Inputs:
        iwgdata: Aircraft data object obtained from iwgread() function
        start_time: Start time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        end_time: End time [YYYY-MM-DDTHH:MM:SS as str] to consider for the PSDs (optional)
        tres: Averaging interval [s]; tres=1. skips averaging routine
    '''
    if start_time is None:
        start_dt64 = iwgdata['time']['data'][0]
    else:
        start_dt64 = np.datetime64(start_time)

    if end_time is None:
        end_dt64 = iwgdata['time']['data'][-1]
    else:
        end_dt64 = np.datetime64(end_time)

    if tres>1:
        iwg_avg = {}
        iwg_avg['Information'] = iwgdata['Information']

        dur = (end_dt64 - start_dt64) / np.timedelta64(1, 's') # dataset duration to consider [s]

        # Allocate arrays
        for var in iwgdata.keys():
            if (var!='Information') and (var!='time'):
                iwg_avg[var] = {}
                iwg_avg[var]['data'] = np.zeros(int(dur/tres))
                iwg_avg[var]['units'] = iwgdata[var]['units']

        time_subset_begin = start_dt64 # allocate time array of N-sec interval obs
        # get midpoint of N-second interval
        time_subset_mid = start_dt64 + np.timedelta64(int(tres/2.), 's') + np.timedelta64(int(np.mod(tres/2., 1)*1000.), 'ms')
        curr_time = start_dt64
        i = 0

        while curr_time<end_dt64:
            if curr_time>start_dt64:
                time_subset_begin = np.append(time_subset_begin, curr_time)
                time_subset_mid = np.append(time_subset_mid, curr_time + np.timedelta64(int(tres/2.), 's') +
                                            np.timedelta64(int(np.mod(tres/2., 1)*1000.), 'ms'))
            time_inds = np.where((iwgdata['time']['data']>=curr_time) &
                                 (iwgdata['time']['data']<curr_time+np.timedelta64(int(tres), 's')))[0]

            for var in iwg_avg.keys(): # loop through variables
                if (var=='Latitude') or (var=='Longitude') or (var=='GPS_Altitude') or (var=='Pressure_Altitude'): # these require special interpolation about the time midpoint
                    times = np.arange(int(tres))
                    coord_array = iwgdata[var]['data'][time_inds]
                    iwg_avg[var]['data'][i] = np.interp(tres/2., times, coord_array)
                elif (var!='Information') and (var!='time'): # do a simple mean for all other variables
                    var_array = iwgdata[var]['data'][time_inds]
                    iwg_avg[var]['data'][i] = np.mean(var_array)
            i += 1
            curr_time += np.timedelta64(int(tres), 's')

        iwg_avg['time'] = {}; iwg_avg['time_midpoint'] = {}
        iwg_avg['time']['data'] = time_subset_begin
        iwg_avg['time']['units'] = 'Start of {}-second period as numpy datetime64 object'.format(str(int(tres)))
        iwg_avg['time_midpoint']['data'] = time_subset_mid
        iwg_avg['time_midpoint']['units'] = 'Middle of {}-second period as numpy datetime64 object'.format(str(int(tres)))

        return iwg_avg

def calc_bulk(particle_count, habit_count, sample_vol, aspect_ratio, bin_mid, bin_width):
    x0 = [1.e-1, -1., 5.] # initial guess for N0 [cm**-4], mu, lambda [cm**-1]

    # allocate arrays
    N0_bf = np.zeros(particle_count.shape[0])
    N0_hy = np.zeros(particle_count.shape[0])
    mu_bf = np.zeros(particle_count.shape[0])
    mu_hy = np.zeros(particle_count.shape[0])
    lam_bf = np.zeros(particle_count.shape[0])
    lam_hy = np.zeros(particle_count.shape[0])
    iwc_bf = np.zeros(particle_count.shape[0])
    iwc_hy = np.zeros(particle_count.shape[0])
    iwc_hab = np.zeros(particle_count.shape[0])
    asr_nw = np.zeros(particle_count.shape[0])
    asr_bf = np.zeros(particle_count.shape[0])
    asr_hy = np.zeros(particle_count.shape[0])
    asr_hab = np.zeros(particle_count.shape[0])
    dmm_bf = np.zeros(particle_count.shape[0])
    dmm_hy = np.zeros(particle_count.shape[0])
    dmm_hab = np.zeros(particle_count.shape[0])
    dm_bf = np.zeros(particle_count.shape[0])
    dm_hy = np.zeros(particle_count.shape[0])
    dm_hab = np.zeros(particle_count.shape[0])
    rhoe_bf = np.zeros(particle_count.shape[0])
    rhoe_hy = np.zeros(particle_count.shape[0])
    rhoe_hab = np.zeros(particle_count.shape[0])
    rho_bf = np.zeros((particle_count.shape[0], particle_count.shape[1]))
    rho_hy = np.zeros((particle_count.shape[0], particle_count.shape[1]))
    rho_hab = np.zeros((particle_count.shape[0], particle_count.shape[1]))

    # compute particle habit mass outside loop for speed
    a_coeff = np.array([1.96e-3, 1.96e-3, 1.666e-3, 7.39e-3, 1.96e-3, 4.9e-2, 5.16e-4, 1.96e-3])
    a_tile = np.tile(np.reshape(a_coeff, (1, len(a_coeff))), (habit_count.shape[1], 1))
    b_coeff = np.array([1.9, 1.9, 1.91, 2.45, 1.9, 2.8, 1.8, 1.9])
    b_tile = np.tile(np.reshape(b_coeff, (1, len(b_coeff))), (habit_count.shape[1], 1))
    D_tile = np.tile(np.reshape(bin_mid, (len(bin_mid), 1)), (1, habit_count.shape[2]))
    mass_tile = a_tile * (D_tile/10.) ** b_tile

    for time_ind in range(particle_count.shape[0]):
        if particle_count[time_ind, :].count()==particle_count.shape[1]: # time period is not masked...continue on
            Nt = 1000.*np.nansum(particle_count[time_ind, :]/sample_vol[time_ind, :]) # number concentratino [L**-1]

            # spherical volume from Chase et al. (2018) [cm**3 / cm**3]
            vol = (np.pi / 6.) * np.sum(0.6 * ((bin_mid/10.)**3.) * particle_count[time_ind, :] / sample_vol[time_ind, :])

            # number-weighted mean aspect rato
            asr_nw[time_ind] = np.nansum(aspect_ratio[time_ind, :] * particle_count[time_ind, :]) / np.nansum(particle_count[time_ind, :])

            # Brown & Francis products
            mass_particle = (0.00294/1.5) * (bin_mid/10.)**1.9 # particle mass [g]
            mass_bf = mass_particle * particle_count[time_ind, :] # g (binned)
            cumMass_bf = np.nancumsum(mass_bf)
            if cumMass_bf[-1]>0.:
                iwc_bf[time_ind] = 10.**6 * np.nansum(mass_bf / sample_vol[time_ind, :]) # g m^-3
                z_bf = 1.e12 * (0.174/0.93) * (6./np.pi/0.934)**2 * np.nansum(mass_particle**2*particle_count[time_ind, :]/sample_vol[time_ind, :]) # mm^6 m^-3
                sol = least_squares(calc_chisquare, x0, method='lm',ftol=1e-9,xtol=1e-9, max_nfev=int(1e6),\
                                    args=(Nt,iwc_bf[time_ind],z_bf,bin_mid,bin_width,0.00294/1.5,1.9)) # sove the gamma params using least squares minimziation
                N0_bf[time_ind] = sol.x[0]; mu_bf[time_ind] = sol.x[1]; lam_bf[time_ind] = sol.x[2]
                asr_bf[time_ind] = np.sum(aspect_ratio[time_ind, :] * mass_bf / sample_vol[time_ind, :]) / np.sum(mass_bf / sample_vol[time_ind, :]) # mass-weighted aspect ratio
                rhoe_bf[time_ind] = (iwc_bf[time_ind] / 10.**6) / vol # effective density from Chase et al. (2018) [g cm**-3]
                rho_bf[time_ind, :] = (mass_bf / particle_count[time_ind, :]) / (np.pi / 6.) / (bin_mid/10.)**3. # rho(D) following Heymsfield et al. (2003) [g cm**-3]
                dm_bf[time_ind] = 10. * np.sum((bin_mid/10.) * mass_bf / sample_vol[time_ind, :]) / np.sum(mass_bf / sample_vol[time_ind, :]) # mass-weighted mean D from Chase et al. (2020) [mm]
                if cumMass_bf[0]>=0.5*cumMass_bf[-1]:
                    dmm_bf[time_ind] = bin_mid[0]
                else:
                    dmm_bf[time_ind] = bin_mid[np.where(cumMass_bf>0.5*cumMass_bf[-1])[0][0]-1]

            # Heymsfield (2010) products [https://doi.org/10.1175/2010JAS3507.1]
            #mass_hy = (0.0061*(bin_mid/10.)**2.05) * particle_count[time_ind, :] # g (binned) H04 definition used in GPM NCAR files
            mass_particle = 0.00528 * (bin_mid/10.)**2.1 # particle mass [g]
            mass_hy = mass_particle * particle_count[time_ind, :] # g (binned)
            cumMass_hy = np.nancumsum(mass_hy)
            if cumMass_hy[-1]>0.:
                iwc_hy[time_ind] = 10.**6 * np.nansum(mass_hy / sample_vol[time_ind, :]) # g m^-3
                z_hy = 1.e12 * (0.174/0.93) * (6./np.pi/0.934)**2 * np.nansum(mass_particle**2*particle_count[time_ind, :]/sample_vol[time_ind, :]) # mm^6 m^-3
                sol = least_squares(calc_chisquare, x0, method='lm',ftol=1e-9,xtol=1e-9, max_nfev=int(1e6),\
                                    args=(Nt,iwc_hy[time_ind],z_hy,bin_mid,bin_width,0.00528,2.1)) # sove the gamma params using least squares minimziation
                N0_hy[time_ind] = sol.x[0]; mu_hy[time_ind] = sol.x[1]; lam_hy[time_ind] = sol.x[2]
                asr_hy[time_ind] = np.sum(aspect_ratio[time_ind, :] * mass_hy / sample_vol[time_ind, :]) / np.sum(mass_hy / sample_vol[time_ind, :]) # mass-weighted aspect ratio
                rhoe_hy[time_ind] = (iwc_hy[time_ind] / 10.**6) / vol # effective density from Chase et al. (2018) [g cm**-3]
                rho_hy[time_ind, :] = (mass_hy / particle_count[time_ind, :]) / (np.pi / 6.) / (bin_mid/10.)**3. # rho(D) following Heymsfield et al. (2003) [g cm**-3]
                dm_hy[time_ind] = 10. * np.sum((bin_mid/10.) * mass_hy / sample_vol[time_ind, :]) / np.sum(mass_hy / sample_vol[time_ind, :]) # mass-weighted mean D from Chase et al. (2020) [mm]
                if cumMass_hy[0]>=0.5*cumMass_hy[-1]:
                    dmm_hy[time_ind] = bin_mid[0]
                else:
                    dmm_hy[time_ind] = bin_mid[np.where(cumMass_hy>0.5*cumMass_hy[-1])[0][0]-1]


            # Habit-specific products
            mass_hab = np.sum(mass_tile * habit_count[time_ind, :, :], axis=1) # g (binned)
            cumMass_hab = np.nancumsum(mass_hab)
            if cumMass_hab[-1]>0.:
                if cumMass_hab[0]>=0.5*cumMass_hab[-1]:
                    dmm_hab[time_ind] = bin_mid[0]
                else:
                    dmm_hab[time_ind] = bin_mid[np.where(cumMass_hab>0.5*cumMass_hab[-1])[0][0]-1]
            iwc_hab[time_ind] = 10.**6 * np.nansum(mass_hab / sample_vol[time_ind, :]) # g m^-3
            asr_hab[time_ind] = np.sum(aspect_ratio[time_ind, :] * mass_hab / sample_vol[time_ind, :]) / np.sum(mass_hab / sample_vol[time_ind, :]) # mass-weighted aspect ratio
            rhoe_hab[time_ind] = (iwc_hab[time_ind] / 10.**6) / vol # effective density from Chase et al. (2018) [g cm**-3]
            rho_hab[time_ind, :] = (mass_hab / particle_count[time_ind, :]) / (np.pi / 6.) / (bin_mid/10.)**3. # rho(D) following Heymsfield et al. (2003) [g cm**-3]
            dm_hab[time_ind] = 10. * np.sum((bin_mid/10.) * mass_hab / sample_vol[time_ind, :]) / np.sum(mass_hab / sample_vol[time_ind, :]) # mass-weighted mean D from Chase et al. (2020) [mm]

    mu_bf = np.ma.masked_where(N0_bf==0., mu_bf)
    mu_hy = np.ma.masked_where(N0_hy==0., mu_hy)
    lam_bf = np.ma.masked_where(N0_bf==0., lam_bf)
    lam_hy = np.ma.masked_where(N0_hy==0., lam_hy)
    N0_bf = np.ma.masked_where(N0_bf==0., N0_bf)
    N0_hy = np.ma.masked_where(N0_hy==0., N0_hy)
    dmm_bf = np.ma.masked_where(dmm_bf==0., dmm_bf)
    dmm_hy = np.ma.masked_where(dmm_hy==0., dmm_hy)
    dmm_hab = np.ma.masked_where(dmm_hab==0., dmm_hab)
    dm_bf = np.ma.masked_where(dm_bf==0., dm_bf)
    dm_hy = np.ma.masked_where(dm_hy==0., dm_hy)
    dm_hab = np.ma.masked_where(dm_hab==0., dm_hab)
    asr_nw = np.ma.masked_where(np.ma.getmask(dmm_bf), asr_nw)
    asr_bf = np.ma.masked_where(np.ma.getmask(dmm_bf), asr_bf)
    asr_hy = np.ma.masked_where(np.ma.getmask(dmm_hy), asr_hy)
    asr_hab = np.ma.masked_where(np.ma.getmask(asr_hab), iwc_hab)
    rhoe_bf = np.ma.masked_where(np.ma.getmask(dmm_bf), rhoe_bf)
    rhoe_hy = np.ma.masked_where(np.ma.getmask(dmm_hy), rhoe_hy)
    rhoe_hab = np.ma.masked_where(np.ma.getmask(dmm_hab), rhoe_hab)
    iwc_bf = np.ma.masked_where(np.ma.getmask(dmm_bf), iwc_bf)
    iwc_hy = np.ma.masked_where(np.ma.getmask(dmm_hy), iwc_hy)
    iwc_hab = np.ma.masked_where(np.ma.getmask(dmm_hab), iwc_hab)
    rho_bf = np.ma.masked_where(rho_bf==0., rho_bf)
    rho_hy = np.ma.masked_where(rho_hy==0., rho_hy)
    rho_hab = np.ma.masked_where(rho_hab==0., rho_hab)

    return (N0_bf, N0_hy, mu_bf, mu_hy, lam_bf, lam_hy, iwc_bf, iwc_hy, iwc_hab, asr_nw, asr_bf, asr_hy, asr_hab, dmm_bf, dmm_hy, dmm_hab,\
            dm_bf, dm_hy, dm_hab, rho_bf, rho_hy, rho_hab, rhoe_bf, rhoe_hy, rhoe_hab)

def calc_riming(p3psd, Z, matchedZ_W, matchedZ_Ka, matchedZ_Ku, matchedZ_X, compute_fits=False):
    x0 = [1.e-1, -1., 5.] # initial guess for N0 [cm**-4], mu, lambda [cm**-1]

    rmass = np.zeros(len(p3psd['time']))
    rfrac = np.zeros(len(p3psd['time']))
    a_coeff = np.zeros(len(p3psd['time']))
    b_coeff = np.zeros(len(p3psd['time']))
    Nw = np.zeros(len(p3psd['time']))
    N0 = np.zeros(len(p3psd['time']))
    mu = np.zeros(len(p3psd['time']))
    lam = np.zeros(len(p3psd['time']))
    iwc = np.zeros(len(p3psd['time']))
    asr = np.zeros(len(p3psd['time']))
    dm = np.zeros(len(p3psd['time']))
    dmm = np.zeros(len(p3psd['time']))
    rho_eff = np.zeros(len(p3psd['time']))
    dfr_KuKa = np.zeros(len(p3psd['time']))
    error = np.zeros((len(p3psd['time']), len(p3psd['riming_mass_array'])))

    for i in range(len(p3psd['time'])):
        # loop through the different possible riming masses
        for j in range(len(p3psd['riming_mass_array'])):
            if (matchedZ_W is not None) and (np.ma.is_masked(matchedZ_W[i]) is False) and (np.ma.is_masked(p3psd['dbz_W'][i, :]) is False):
                error[i, j] = error[i, j] + np.abs(matchedZ_W[i] - p3psd['dbz_W'][i, j])
            if (matchedZ_Ka is not None) and (np.ma.is_masked(matchedZ_Ka[i]) is False) and (np.ma.is_masked(p3psd['dbz_Ka'][i, :]) is False):
                error[i, j] = error[i, j] + np.abs(matchedZ_Ka[i] - p3psd['dbz_Ka'][i, j])
            if (matchedZ_Ku is not None) and (np.ma.is_masked(matchedZ_Ku[i]) is False) and (np.ma.is_masked(p3psd['dbz_Ku'][i, :]) is False):
                error[i, j] = error[i, j] + np.abs(matchedZ_Ku[i] - p3psd['dbz_Ku'][i, j])
            if (matchedZ_X is not None) and (np.ma.is_masked(matchedZ_X[i]) is False) and (np.ma.is_masked(p3psd['dbz_X'][i, :]) is False):
                error[i, j] = error[i, j] + np.abs(matchedZ_X[i] - p3psd['dbz_X'][i, j])

        if np.sum(error[i, :])>0.:
            rmass[i] = p3psd['riming_mass_array'][np.argmin(error[i, :])]
            a_coeff[i] = p3psd['a_coeff_array'][np.argmin(error[i, :])]
            b_coeff[i] = p3psd['b_coeff_array'][np.argmin(error[i, :])]

            if p3psd['count'][i, :].count()==p3psd['count'].shape[1]: # time period is not masked...continue on
                Nt = 1000.*np.nansum(p3psd['count'][i, :]/p3psd['sv'][i, :]) # concentration [L**-1]
                mass_particle = a_coeff[i] * (p3psd['bin_mid']/10.)**b_coeff[i] # particle mass [g]
                mass = mass_particle * p3psd['count'][i, :] # g (binned)
                cumMass = np.nancumsum(mass)
                if cumMass[-1]>0.:
                    # Nw (follows Chase et al. 2021)
                    # [log10(m**-3 mm**-1)]
                    D_melt = ((6. * mass_particle) / (np.pi * 0.997))**(1./3.)
                    Nw[i] = np.log10((1e5) * (4.**4 / 6) * np.nansum(
                        D_melt**3 * p3psd['ND'][i, :] * p3psd['bin_width'])**5 / np.nansum(
                        D_melt**4 * p3psd['ND'][i, :] * p3psd['bin_width'])**4)

                    # IWC
                    iwc[i] = 10.**6 * np.nansum(mass / p3psd['sv'][i, :]) # g m^-3

                    # DFR
                    dfr_KuKa[i] = p3psd[
                        'dbz_Ku'][i, np.argmin(error[i, :])] - p3psd[
                        'dbz_Ka'][i, np.argmin(error[i, :])] # dB

                    # Optionally compute N0, mu, lambda
                    if compute_fits:
                        z = 10.**(p3psd['dbz_X'][i,np.argmin(error[i, :])]/10.) # mm^6 m^-3

                        # solve gamma params using least squares minimziation
                        sol = least_squares(
                            calc_chisquare, x0, method='lm', ftol=1e-9, xtol=1e-9,
                            max_nfev=int(1e6), args=(
                                Nt, iwc[i], z, p3psd['bin_mid'], p3psd['bin_width'],
                                a_coeff[i], b_coeff[i], np.argmin(error[i, :])))
                        N0[i] = sol.x[0]; mu[i] = sol.x[1]; lam[i] = sol.x[2]

                    # Mass-weighted mean aspect ratio
                    asr[i] = np.sum(
                        p3psd['aspect_ratio'][i, :] * mass / p3psd['sv'][i, :]) / np.sum(
                        mass / p3psd['sv'][i, :])

                    # Bulk riming fraction (see Eqn 1 of Morrison and Grabowski
                    # [2010, https://doi.org/10.1175/2010JAS3250.1] for binned version)
                    rfrac[i] = np.sum(
                        np.squeeze(Z.rimefrac[0, :, np.argmin(error[i, :])])
                        * mass / p3psd['sv'][i, :]) / np.nansum(
                        mass / p3psd['sv'][i, :]) # SUM(rimed mass conc)/iwc

                    # Effective density (follows Chase et al. 2018)
                    vol = (np.pi / 6.) * np.sum(
                        0.6 * ((p3psd['bin_mid']/10.)**3.) * p3psd['count'][i, :]
                        / p3psd['sv'][i, :]) # [cm**3 / cm**3]
                    rho_eff[i] = (iwc[i] / 10.**6) / vol # [g cm**-3]

                    # Mass-weighted mean diameter (follows Chase et al. 2020)
                    # M3/M2 if b==2, more generally M(b+1)/Mb
                    dm[i] = 10. * np.sum(
                        (p3psd['bin_mid']/10.) * mass / p3psd['sv'][i, :]) / np.sum(
                        mass / p3psd['sv'][i, :]) # [mm]

                    # Mass-weighted median diameter [mm]
                    if cumMass[0]>=0.5*cumMass[-1]:
                        dmm[i] = p3psd['bin_mid'][0]
                    else:
                        dmm[i] = p3psd[
                            'bin_mid'][np.where(cumMass>0.5*cumMass[-1])[0][0]-1]

    p3psd['sclwp'] = np.ma.masked_where(np.sum(error, axis=1)==0., rmass)
    p3psd['riming_frac'] = np.ma.masked_where(np.sum(error, axis=1)==0., rfrac)
    p3psd['a_coeff'] = np.ma.masked_where(np.sum(error, axis=1)==0., a_coeff)
    p3psd['b_coeff'] = np.ma.masked_where(np.sum(error, axis=1)==0., b_coeff)
    if compute_fits:
        p3psd['mu_ls'] = np.ma.masked_where(N0==0., mu)
        p3psd['lambda_ls'] = np.ma.masked_where(N0==0., lam)
        p3psd['N0_ls'] = np.ma.masked_where(N0==0., N0)
    p3psd['Nw_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., Nw)
    p3psd['iwc_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., iwc)
    p3psd['mean_aspect_ratio_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., asr)
    p3psd['dm_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., dm)
    p3psd['dmm_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., dmm)
    p3psd['eff_density_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., rho_eff)
    p3psd['dfr_KuKa_ls'] = np.ma.masked_where(np.sum(error, axis=1)==0., dfr_KuKa)

    return p3psd

def calc_chisquare(
    x, Nt_obs, iwc_obs, z_obs, bin_mid, bin_width, a_coefficient, b_coefficient,
    rime_ind=None, exponential=False):
    '''
    Compute gamma fit parameters for the PSD.

    Follows McFarquhar et al. (2015) by finding N0-mu-lambda minimizing first
    (Nt), third (mass), sixth (reflectivity) moments.

    Inputs:
        x: N0, mu, lambda to test on the minimization procedure
        Nt_obs: Observed number concentration [L^-1]
        iwc_obs: Observed IWC using an assumed m-D relation [g m**-3]
        z_obs: Observed Z (following Hogan et al. 2012 definition) using assumed m-D relation [mm**6 m**-3]
        bin_mid: Midpoints for the binned particle size [mm]
        bin_width: Bin width for the binned particle size [cm]
        a_coefficient: Prefactor component to the assumed m-D reltation [cm**-b]
        b_coefficient: Exponent component to the assumed m-D reltation
        rime_ind (optional, for LS products only): Riming category index to use for the reflectivity moment
        exponential: Boolean, True if setting mu=0 for the fit (exponential form)
    Outputs:
        chi_square: Chi-square value for the provided N0-mu-lambda configuration
    '''
    Dmax = bin_mid / 10. # midpoint in cm
    dD = bin_width # bin width in cm
    mass_particle = a_coefficient * Dmax**b_coefficient # binned particle mass [g]

    if exponential: # exponential form with mu=0
        ND_fit = x[0] * np.exp(-x[2]*Dmax)
    else: # traditional gamma function with variable mu
        ND_fit = x[0] * Dmax**x[1] * np.exp(-x[2]*Dmax)
        
    Nt_fit = 1000.*np.nansum(ND_fit*dD) # L**-1
    iwc_fit = 10.**6  * np.nansum(mass_particle*ND_fit*dD) # g m**-3
    if rime_ind is not None:
        Z_fit = forward_Z() #initialize class
        Z_fit.set_PSD(PSD=ND_fit[np.newaxis,:]*10.**8, D=Dmax/100., dD=dD/100., Z_interp=True) # get the PSD in the format to use in the routine (mks units)
        Z_fit.load_split_L15() # Load the leinonen output
        Z_fit.fit_sigmas(Z_interp=True) # Fit the backscatter cross-sections
        Z_fit.calc_Z() # Calculate Z...outputs are Z.Z_x, Z.Z_ku, Z.Z_ka, Z.Z_w for the four radar wavelengths
        z_fit = 10.**(Z_fit.Z_x[0, rime_ind] / 10.) # mm**6 m**-3
    else:
        z_fit = 1.e12 * (0.174/0.93) * (6./np.pi/0.934)**2 * np.nansum(mass_particle**2*ND_fit*dD) # mm**6 m**-3

    csq_Nt = ((Nt_obs-Nt_fit) / np.sqrt(Nt_obs*Nt_fit))**2
    csq_iwc = ((iwc_obs-iwc_fit) / np.sqrt(iwc_obs*iwc_fit))**2
    csq_z = ((z_obs-z_fit) / np.sqrt(z_obs*z_fit))**2
    chi_square = [csq_Nt, csq_iwc, csq_z]

    return chi_square

def load_scat_curves():
    '''
    Returns a dictionary object of DFR_Ku-Ka and DFR_Ka-W values for unrimed and rimed (effective LWP = 0.1, 0.2, 0.5, 1 kg m**-2) particles.
    Follows Leinonen & Szyrmer (2015; https://doi.org/10.1002/2015EA000102) with modifications outlined in notebooks/exploratory/jf-scatdb-tripfreq_curves.ipynb.

    Returns
    ----------
    scat_curves: Dict with keys 'dfr_ku_ka_lwp00', 'dfr_ku_ka_lwp01', 'dfr_ku_ka_lwp02', 'dfr_ku_ka_lwp05', 'dfr_ku_ka_lwp10',
        'dfr_ka_w_lwp00', 'dfr_ka_w_lwp01', 'dfr_ka_w_lwp02', 'dfr_ka_w_lwp05', 'dfr_ka_w_lwp10'
    '''
    scat_curves = {}
    scat_curves['dfr_ku_ka_lwp00'] = np.array([-0.06384744,0.10719788,0.28875311,0.53266324,0.82943129,1.14343318,1.45959242,\
                                               1.77734675,2.09975012,2.4287072,2.7639136,3.10327,3.44368493,3.78180182,4.11449736,\
                                               4.43914913,4.75372913,5.05678849,5.3473865,5.62499912,5.88942779,6.14071847,6.37909447,\
                                               6.60490328,6.81857567,7.02059511,7.2114752,7.39174341,7.5619294,7.72255691,7.87413806,\
                                               8.01716958,8.15213029,8.27947955,8.39965638,8.51307911,8.62014531,8.72123203,8.8166962,\
                                               8.90687517,8.99208731,9.07263274,9.14879403,9.22083699,9.28901144,9.35355198,9.4146788,\
                                               9.47259839,9.52750436,9.57957812,9.62898959,9.67589791,9.72045206,9.76279152,9.80304683,\
                                               9.84134016,9.87778586,9.91249097,9.94555567,9.97707375,10.00713302,10.03581571,10.06319887,\
                                               10.08935465,10.11435068,10.13825037,10.16111317,10.18299485,10.20394774,10.22402095,10.24326063,\
                                               10.26171009,10.27941006,10.29639884,10.31271242,10.32838469,10.34344755,10.35793104,10.37186346,\
                                               10.38527148,10.39818026,10.41061352,10.42259368,10.43414188,10.44527809,10.4560212,10.46638907,\
                                               10.47639858,10.48606572,10.49540561,10.50443259,10.51316023,10.52160143,10.52976837,10.53767265,\
                                               10.54532528,10.55273668,10.55991678,10.566875,10.57362033,10.58016127,10.58650595,10.59266209,\
                                               10.59863705,10.60443783,10.61007112,10.61554328,10.62086038,10.62602822,10.63105234])
    scat_curves['dfr_ku_ka_lwp01'] = np.array([-0.0712766,0.02178642,0.20864484,0.51518598,0.88117726,1.26741244,1.66233205,2.06273367,2.46543034,\
                                               2.86633835,3.26138783,3.64722115,4.02141193,4.38239217,4.72927279,5.06165804,5.37949202,5.68294402,\
                                               5.97232745,6.24804428,6.51054789,6.76031861,6.99784794,7.22362865,7.43814872,7.64188774,7.83531498,\
                                               8.01888832,8.19305375,8.35824514,8.51488409,8.66337985,8.80412915,8.93751605,9.06391181,9.18367462,\
                                               9.2971495,9.40466813,9.50654875,9.60309614,9.69460165,9.78134327,9.86358576,9.94158094,10.0155679,\
                                               10.08577331,10.15241183,10.21568647,10.27578902,10.33290051,10.38719164,10.43882324,10.48794679,\
                                               10.53470481,10.5792314,10.6216526,10.66208691,10.70064568,10.73743352,10.77254868,10.80608346,\
                                               10.83812456,10.86875341,10.89804648,10.92607563,10.95290837,10.97860814,11.00323456,11.02684369,\
                                               11.04948823,11.07121777,11.09207895,11.11211567,11.13136928,11.14987869,11.16768058,11.18480952,\
                                               11.20129809,11.21717705,11.23247537,11.24722046,11.26143815,11.27515287,11.28838771,11.30116448,\
                                               11.31350383,11.32542529,11.33694736,11.34808754,11.35886243,11.36928776,11.37937845,11.38914863,\
                                               11.39861174,11.40778052,11.41666709,11.42528295,11.43363904,11.44174576,11.44961301,11.4572502,\
                                               11.46466631,11.47186987,11.47886903,11.48567155,11.49228482,11.49871591,11.50497155,11.51105817,11.51698191])
    scat_curves['dfr_ku_ka_lwp02'] = np.array([-0.05680912,0.12046722,0.33178736,0.61793811,0.94889302,
         1.29250341,1.63922676,1.98887445,2.34178828,2.6968072,
         3.05162638,3.40350216,3.74976302,4.0880855,4.4166039,
         4.73392152,5.03907118,5.33145386,5.61077217,5.87696735,
         6.13016361,6.37062145,6.5986995,6.81482428,7.01946659,
         7.21312341,7.39630427,7.56952109,7.73328075,7.88807974,
         8.0344004,8.17270837,8.30345091,8.42705592,8.54393147,
         8.65446565,8.75902675,8.85796361,8.9516061,9.04026574,
         9.12423639,9.20379493,9.279202,9.35070282,9.4185279,
         9.4828938,9.5440039,9.60204909,9.65720851,9.70965019,
         9.75953175,9.80700098,9.85219644,9.89524809,9.93627772,
         9.97539957,10.01272072,10.04834161,10.08235645,10.11485358,
        10.14591592,10.17562126,10.20404264,10.23124862,10.2573036,
        10.2822681,10.30619896,10.32914964,10.35117041,10.37230855,
        10.39260859,10.41211241,10.43085948,10.44888698,10.46622996,
        10.48292147,10.49899269,10.51447303,10.52939028,10.54377068,
        10.55763901,10.57101871,10.58393196,10.59639971,10.60844183,
        10.6200771,10.63132333,10.6421974,10.6527153,10.66289222,
        10.67274255,10.68227996,10.69151744,10.70046731,10.70914132,
        10.71755059,10.72570573,10.73361684,10.74129352,10.74874491,
        10.75597974,10.7630063,10.76983252,10.77646595,10.7829138,
        10.78918294,10.79527995,10.80121109,10.80698237,10.81259951])
    scat_curves['dfr_ku_ka_lwp05'] = np.array([-0.10311124,0.29175946,0.68235314,1.0358194,1.38696029,
         1.73513864,2.0818466,2.43052578,2.78317815,3.13970401,
         3.49842498,3.85682235,4.21215102,4.56184745,4.90374863,
         5.23617164,5.55790574,5.86815802,6.16648094,6.45269836,
         6.72683875,6.98907922,7.23970056,7.47905274,7.70752888,
         7.92554638,8.13353338,8.33191931,8.52112858,8.70157632,
         8.8736658,9.03778683,9.19431495,9.34361107,9.48602146,
         9.62187793,9.75149809,9.8751857,9.99323113,10.10591168,
        10.21349205,10.31622476,10.41435053,10.50809873,10.59768772,
        10.68332535,10.76520925,10.84352732,10.91845804,10.99017093,
        11.05882686,11.12457845,11.18757048,11.24794017,11.30581759,
        11.36132599,11.41458211,11.46569652,11.51477393,11.56191349,
        11.60720905,11.65074949,11.69261895,11.73289707,11.77165926,
        11.80897693,11.84491768,11.87954555,11.91292118,11.94510202,
        11.97614251,12.00609421,12.03500604,12.06292432,12.08989303,
        12.11595384,12.14114632,12.16550799,12.18907448,12.21187964,
        12.2339556,12.25533288,12.2760405,12.29610605,12.31555576,
        12.33441458,12.35270626,12.37045341,12.38767754,12.40439915,
        12.42063778,12.43641204,12.4517397,12.46663766,12.4811221,
        12.49520842,12.50891134,12.52224493,12.53522259,12.54785718,
        12.56016095,12.57214564,12.58382247,12.59520218,12.60629504,
        12.61711089,12.62765917,12.6379489,12.64798875,12.657787  ])
    scat_curves['dfr_ku_ka_lwp10'] = np.array([-0.27516392,0.03782814,0.49692132,1.23542261,1.98955997,
         2.64680795,3.21460106,3.72043101,4.18521284,4.62155705,
         5.03633937,5.43303372,5.81325351,6.17767524,6.52656331,
         6.86005336,7.17829394,7.48150781,7.77000943,8.04420004,
         8.30455277,8.55159462,8.7858888,9.00801923,9.21857765,
         9.41815346,9.60732602,9.78665901,9.95669649,10.11796035,
        10.27094876,10.41613541,10.55396939,10.68487542,10.80925444,
        10.92748436,11.03992097,11.14689892,11.24873276,11.34571799,
        11.43813205,11.52623541,11.61027248,11.69047261,11.767051,
        11.84020955,11.91013768,11.97701314,12.04100273,12.10226299,
        12.16094087,12.21717433,12.27109295,12.32281842,12.37246509,
        12.42014044,12.46594552,12.50997534,12.5523193,12.59306155,
        12.63228129,12.67005311,12.70644732,12.74153017,12.77536412,
        12.8080081,12.83951774,12.86994554,12.8993411,12.92775129,
        12.95522041,12.98179037,13.00750081,13.03238927,13.05649129,
        13.07984055,13.10246896,13.1244068,13.14568278,13.16632416,
        13.18635681,13.20580532,13.22469306,13.24304222,13.26087395,
        13.27820835,13.29506455,13.31146078,13.32741441,13.34294199,
        13.35805931,13.37278141,13.38712267,13.40109681,13.41471691,
        13.42799551,13.44094456,13.4535755,13.46589928,13.47792635,
        13.48966675,13.50113006,13.51232549,13.52326183,13.53394753,
        13.54439069,13.55459908,13.56458014,13.57434103,13.58388862])
    scat_curves['dfr_ka_w_lwp00'] = np.array([-0.72546926,0.07166054,1.02401374,1.98890886,
          2.90093642,3.72912595,4.46460879,5.11154038,
          5.67865803,6.17509182,6.60896078,6.9871694,
          7.31559214,7.59932061,7.84286855,8.05031519,
          8.22539587,8.37155514,8.49197604,8.58959585,
          8.66711548,8.72700655,8.77151873,8.80268837,
          8.8223486,8.83214088,8.83352763,8.82780541,
          8.81611843,8.79947195,8.77874533,8.75470449,
          8.72801373,8.69924674,8.6688968,8.63738609,
          8.60507422,8.57226593,8.53921797,8.50614535,
          8.4732268,8.44060971,8.40841441,8.37673799,
          8.34565764,8.31523355,8.28551148,8.25652493,
          8.22829709,8.20084251,8.17416846,8.14827624,
          8.12316219,8.0988186,8.0752345,8.05239628,
          8.03028829,8.00889328,7.98819281,7.9681676,
          7.94879779,7.93006318,7.9119434,7.89441814,
          7.8774672,7.86107064,7.84520886,7.82986266,
          7.81501328,7.80064247,7.78673249,7.77326617,
          7.76022685,7.74759847,7.73536551,7.72351303,
          7.71202661,7.70089241,7.6900971,7.67962788,
          7.66947246,7.65961904,7.6500563,7.64077336,
          7.63175982,7.62300567,7.61450136,7.60623768,
          7.59820585,7.59039744,7.58280434,7.57541883,
          7.56823347,7.56124114,7.55443502,7.54780857,
          7.54135552,7.53506985,7.52894579,7.52297781,
          7.51716061,7.51148908,7.50595834,7.50056371,
          7.49530067,7.49016491,7.48515227,7.48025876,
          7.47548054,7.47081395])
    scat_curves['dfr_ka_w_lwp01'] = np.array([-0.68742979,0.00294939,0.85800718,1.76549561,
          2.63128598,3.41933406,4.12023142,4.73600522,
          5.27298002,5.73878624,6.14120161,6.48769289,
          6.78520859,7.04007176,7.25792874,7.44373911,
          7.60179815,7.73578285,7.84881305,7.94352001,
          8.02211652,8.0864643,8.13813588,8.17846935,
          8.20861539,8.2295764,8.24223823,8.24739529,
          8.24576961,8.23802512,8.22477779,8.20660262,
          8.18403822,8.15758969,8.12773027,8.09490231,
          8.05951783,8.02195893,7.9825782,7.94169932,
          7.89961777,7.85660175,7.8128933,7.76870956,
          7.72424419,7.67966884,7.63513471,7.59077407,
          7.54670186,7.5030172,7.45980481,7.41713648,
          7.37507233,7.33366208,7.29294616,7.25295682,
          7.21371903,7.1752514,7.13756699,7.10067399,
          7.06457638,7.02927453,6.9947657,6.96104449,
          6.92810324,6.89593245,6.864521,6.8338565,
          6.80392552,6.77471379,6.74620637,6.71838787,
          6.6912425,6.66475429,6.63890711,6.61368481,
          6.58907126,6.56505045,6.54160651,6.51872378,
          6.49638682,6.47458047,6.45328986,6.43250044,
          6.41219795,6.39236851,6.37299856,6.3540749,
          6.33558466,6.31751537,6.29985485,6.28259133,
          6.26571335,6.2492098,6.23306991,6.21728323,
          6.20183964,6.18672934,6.17194283,6.15747091,
          6.14330468,6.12943552,6.11585507,6.10255527,
          6.0895283,6.07676658,6.06426281,6.0520099,
          6.04000098,6.02822943])
    scat_curves['dfr_ka_w_lwp02'] = np.array([-0.62959995,0.76015151,1.93432572,3.03701542,
          4.03197713,4.89486081,5.6269162,6.24228676,
          6.7583211,7.19131439,7.55521498,7.86154001,
          8.11968521,8.33730883,8.52068079,8.67496979,
          8.80446972,8.91277526,9.00291729,9.07746772,
          9.13862099,9.18825813,9.22799764,9.25923625,
          9.28318216,9.30088223,9.31324458,9.32105755,
          9.32500565,9.32568325,9.32360616,9.31922177,
          9.31291768,9.30502932,9.29584653,9.28561928,
          9.27456278,9.26286187,9.25067488,9.23813705,
          9.22536355,9.21245204,9.199485,9.18653174,
          9.17365018,9.16088836,9.14828581,9.13587475,
          9.12368107,9.11172526,9.10002318,9.08858677,
          9.07742456,9.06654226,9.05594318,9.0456286,
          9.03559809,9.02584984,9.01638087,9.00718725,
          8.99826427,8.98960662,8.98120851,8.97306378,
          8.965166,8.95750855,8.95008469,8.94288761,
          8.93591051,8.92914658,8.92258911,8.91623145,
          8.91006706,8.90408953,8.89829261,8.89267018,
          8.88721628,8.88192513,8.87679111,8.87180877,
          8.86697284,8.86227822,8.85771997,8.85329333,
          8.84899371,8.84481668,8.84075796,8.83681343,
          8.83297913,8.82925123,8.82562606,8.82210009,
          8.81866989,8.8153322,8.81208387,8.80892186,
          8.80584325,8.80284524,8.79992513,8.79708033,
          8.79430833,8.79160674,8.78897324,8.78640562,
          8.78390174,8.78145953,8.77907702,8.77675231,
          8.77448356,8.772269  ])
    scat_curves['dfr_ka_w_lwp05'] = np.array([-0.68228683,0.56251633,2.11601112,3.56211519,
          4.81607629,5.87430047,6.75581259,7.48560663,
          8.0877343,8.58314768,8.98947428,9.32139063,
          9.59109882,9.80875826,9.98284472,10.12044286,
         10.22748466,10.30894475,10.36900114,10.41116807,
         10.4384058,10.45321169,10.45769547,10.45364182,
         10.44256209,10.42573735,10.40425418,10.37903439,
         10.35085997,10.32039389,10.28819763,10.25474593,
         10.22043931,10.18561468,10.15055437,10.11549389,
         10.08062856,10.0461192,10.01209706,9.97866807,
          9.94591645,9.91390799,9.88269266,9.85230711,
          9.82277664,9.794117,9.76633597,9.73943461,
          9.71340846,9.68824851,9.66394204,9.64047332,
          9.61782429,9.59597504,9.57490427,9.55458966,
          9.53500822,9.51613654,9.49795103,9.48042806,
          9.46354418,9.44727622,9.43160136,9.41649727,
          9.40194212,9.38791468,9.37439429,9.36136097,
          9.34879536,9.33667878,9.32499319,9.31372125,
          9.30284625,9.29235212,9.28222346,9.27244547,
          9.26300394,9.25388527,9.24507641,9.23656489,
          9.22833875,9.22038653,9.21269729,9.20526055,
          9.19806628,9.19110489,9.18436722,9.17784448,
          9.17152829,9.16541063,9.15948382,9.15374053,
          9.14817372,9.1427767,9.13754302,9.13246655,
          9.12754141,9.12276195,9.11812281,9.11361881,
          9.10924502,9.10499671,9.10086935,9.09685861,
          9.09296031,9.08917049,9.08548532,9.08190114,
          9.07841444,9.07502184])
    scat_curves['dfr_ka_w_lwp10'] = np.array([-12.40985649,-5.54001381,-1.1417156,2.09968621,
          4.54291378,6.40442132,7.84189555,8.9661311,
          9.85417004,10.5602022,11.12315861,11.57172051,
         11.92757601,12.20753628,12.42491936,12.59046641,
         12.71296039,12.79965524,12.85658361,12.88878554,
         12.90048429,12.89522554,12.87598994,12.84528547,
         12.80522399,12.7575849,12.70386824,12.64533898,
         12.58306404,12.51794317,12.45073482,12.38207768,
         12.31250886,12.24247907,12.17236548,12.10248251,
         12.03309105,11.96440624,11.89660413,11.82982737,
         11.76419005,11.69978192,11.63667193,11.57491139,
         11.51453657,11.45557102,11.39802757,11.34191001,
         11.28721459,11.23393124,11.18204475,11.1315356,
         11.08238087,11.03455486,10.98802971,10.9427759,
         10.89876272,10.85595855,10.81433125,10.77384839,
         10.73447748,10.69618613,10.65894222,10.62271402,
         10.5874703,10.55318042,10.51981435,10.48734278,
         10.45573711,10.42496951,10.39501292,10.36584108,
         10.33742851,10.30975052,10.28278322,10.2565035,
         10.23088901,10.20591816,10.1815701,10.15782469,
         10.13466251,10.11206482,10.09001355,10.06849125,
         10.04748113,10.02696699,10.0069332,9.98736471,
          9.96824703,9.94956615,9.93130861,9.91346141,
          9.89601204,9.87894844,9.86225895,9.84593238,
          9.8299579,9.8143251,9.79902391,9.78404465,
          9.76937795,9.75501479,9.74094648,9.7271646,
          9.71366104,9.70042799,9.68745788,9.67474341,
          9.66227755,9.65005348])

    return scat_curves
