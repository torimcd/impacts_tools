import pandas as pd
import numpy as np
import xarray as xr
import scipy.interpolate
import os
import pytmatrix
import sys

class forward_Z():
    """ 
    Class to forward model reflectivity using Chase et al (2021) DDA database.
    doi: 10.1175/JAMC-D-20-0177.1
    
    Ku- and Ka-band only are supported for now. Database has all 4 frequencies.    
    """
    
    def set_PSD(self, PSD=None, D=None, dD=None):
        """
        Set the PSD objects in the class. 
        
        PSD: Matrix, (n_samples, n_bins); units: m**-4
        D: Array, (n_bins,); units m
        dD: Array, (n_bins,); units m
        """
        self.PSD = PSD.values
        self.dD = dD.values
        self.D = D.values
        
        #rescale to match shape of PSD. This allows fast computations through vectorization
        psd_shape = self.PSD.shape
        self.dD = np.reshape(self.dD, [1, psd_shape[1]])
        self.dD = np.tile(self.dD, (psd_shape[0], 1))
        
        #rescale to match shape of PSD. This allows fast computations through vectorization 
        self.D = np.reshape(self.D, [1, psd_shape[1]])
        self.D = np.tile(self.D, (psd_shape[0], 1))
        
        # set a placeholder for the backscatter cross-section 
        self.sigma_ku = xr.DataArray(
            data = np.zeros(psd_shape),
            dims = PSD.dims,
            coords = PSD.coords
        )
        self.sigma_ka = xr.DataArray(
            data = np.zeros(psd_shape),
            dims = PSD.dims,
            coords = PSD.coords
        )
        self.sigma_w = xr.DataArray(
            data = np.zeros(psd_shape),
            dims = PSD.dims,
            coords = PSD.coords
        )
    
    def load_db(self, pressure=None, temperature=None):
        """ 
        This method loads the results from Chase et al (2021).
        
        Optional pressure [hPa] to include particle terminal fall speed.
        """
        #load database
        cwd = os.path.dirname(os.path.abspath(__file__))
        fname_scatdb = f'{cwd}/base_df_DDA.csv'
        scatdb = pd.read_csv(
            fname_scatdb, index_col=0, header=0
        ) # Dmax [m], Mass [kg], backscatter at Ku, Ka, W [m^2]
        scatdb = scatdb.rename(columns={'D': 'Dmax'})
        
        if (pressure is not None) and (temperature is not None): # DOI: 10.1175/2010JAS3379.1
            # NOTE: Dmax and M in SI units, the database already uses that
            T_K = temperature + 273.15 # degC to K
            p_Pa = 100. * pressure # hPa to Pa
            rho_a = p_Pa / (287.15 * T_K)
            eta = 18.27 * (291.15 + 120.) / (T_K + 120.) * (
                T_K / 291.15
            )**1.5 / 1e6  # Sutherland's formula for dynamic viscosity
            nu = eta / rho_a # kinetic viscosity
            
            # Ar-D relationship from https://doi.org/10.1175/2009JAS3004.1 (Eqn 1)
            Dmax_cm = 100. * scatdb['Dmax'] # m to cm
            Ar = np.zeros(len(Dmax_cm))
            Ar[Dmax_cm <= 0.02] = np.exp(-38. * Dmax_cm[Dmax_cm <= 0.02])
            Ar[Dmax_cm > 0.02] = 0.16 * (Dmax_cm[Dmax_cm > 0.02]) ** -0.27
            
            # modified Best number
            X = rho_a / eta**2 * 8 * scatdb['M'] * 9.81 / (np.pi * np.sqrt(Ar))
            
            # Reynolds number
            Re = 16. * (np.sqrt(1 + 4. * np.sqrt(X) / 64. / np.sqrt(0.35)) - 1.)**2
            
            # fall speed
            V = nu / scatdb['Dmax'] * Re
            scatdb.insert(2, 'V', V) # m/s
        elif pressure is not None: # DOI: 10.1175/JAS-D-12-0124.1
            V = np.zeros(len(scatdb['M']))

            bool_dmax = scatdb['Dmax'] < 41.e-6
            V[bool_dmax] = 0.0028 * (1.e6 * scatdb['Dmax'][bool_dmax]) ** 2

            bool_dmax = (scatdb['Dmax'] >= 41.e-6) & (scatdb['Dmax'] < 839.e-6)
            V[bool_dmax] = 0.0791 * (1.e6 * scatdb['Dmax'][bool_dmax]) ** 1.101

            bool_dmax = scatdb['Dmax'] >= 839.e-6
            V[bool_dmax] = 62.29 * (1.e6 * scatdb['Dmax'][bool_dmax]) ** 0.1098

            # simple pressure correction (Eqn. 10a)
            c = (1000. / pressure) ** 0.4
            
            # complex pressure correction (Eqn. 14-16)
            # c0 = -1.04 + 0.298 * np.log(pressure)
            # c1 = 0.67 - 0.097 * np.log(pressure)
            # c = c0 + c1 * np.log(1.e6 * scatdb['Dmax'])

            Vsurf = V
            V = c * V
            scatdb.insert(2, 'Vsurf', Vsurf / 100.) # m/s at 1000 hPa
            scatdb.insert(3, 'V', V / 100.) # m/s
        
        self.database = scatdb.drop(scatdb[scatdb.Dmax < 1.e-5].index)
        
    def fit_sigmas(self):
        """ 
        This method is to fit a flexible function to the Chase et al (2021) data.
        It interpolates the backscatter cross-section to whatever values of D
        are inputed to the class. D should be in m.
        
        If Vt is in scattering database that too is interpolated to whatever
        values of D are inputted to the class.
        """
        bins = np.array([
            0., 20., 40., 60., 80., 100., 125., 150., 200., 250., 300., 350., 400.,
            475., 550., 625., 700., 800., 900., 1000., 1200., 1400., 1600., 1800.,
            2200., 2600., 3000., 3400., 3800., 4200., 4600., 5000., 6000., 7000.,
            8000., 9000., 10000., 12000., 14000., 16000., 18000., 20000., 25000.,
            30000., 40000., 50000.
        ]) / 10**6. # [m]
        self.database['bin_i'] = np.digitize(self.database.Dmax, bins=bins)
        df = self.database.groupby('bin_i').median()
        df = df.reindex(np.arange(0, len(bins)))
        df = df.interpolate()
        df = df.dropna(how='all')

        f_ku = scipy.interpolate.interp1d(
            np.log10(df.Dmax.values[:-1]), np.log10(df.sigma_ku.values[:-1]),
            fill_value='extrapolate', kind='linear', bounds_error=False
        )
        sigma_ku = 10. ** f_ku(np.log10(self.D[0, :]))

        f_ka = scipy.interpolate.interp1d(
            np.log10(df.Dmax.values[:-1]), np.log10(df.sigma_ka.values[:-1]),
            fill_value='extrapolate', kind='linear', bounds_error=False
        )
        sigma_ka = 10. ** f_ka(np.log10(self.D[0, :]))
        
        f_w = scipy.interpolate.interp1d(
            np.log10(df.Dmax.values[:-1]), np.log10(df.sigma_w.values[:-1]),
            fill_value='extrapolate', kind='linear', bounds_error=False
        )
        sigma_w = 10. ** f_w(np.log10(self.D[0, :]))

        # reshape things to have vectorized calculations
        psd_shape = self.PSD.shape
        sigma_ku = np.reshape(sigma_ku,[1, psd_shape[1]])
        sigma_ku = np.tile(sigma_ku, (psd_shape[0], 1))
        sigma_ka = np.reshape(sigma_ka,[1, psd_shape[1]])
        sigma_ka = np.tile(sigma_ka, (psd_shape[0], 1))
        sigma_w = np.reshape(sigma_w,[1, psd_shape[1]])
        sigma_w = np.tile(sigma_w, (psd_shape[0], 1))
        
        # if V in scatd, then interpolate to get V(D)
        if 'V' in df:
            f_V = scipy.interpolate.interp1d(
                df.Dmax.values[:-1], df.V.values[:-1],
                fill_value='extrapolate', kind='linear', bounds_error=False
            )
            VD = f_V(self.D[0, :])
            self.VD = xr.DataArray(
                data = VD,
                dims = self.sigma_ka.dims[1], # particle size dimension
                coords = dict(size=self.sigma_ka.coords['size'])
            )
        # if V in scatd, then interpolate to get V(D)
        if 'Vsurf' in df:
            f_Vsurf = scipy.interpolate.interp1d(
                df.Dmax.values[:-1], df.Vsurf.values[:-1],
                fill_value='extrapolate', kind='linear', bounds_error=False
            )
            VD_surf = f_Vsurf(self.D[0, :])
            self.VD_surf = xr.DataArray(
                data = VD_surf,
                dims = self.sigma_ka.dims[1], # particle size dimension
                coords = dict(size=self.sigma_ka.coords['size'])
            )
        
        # store it into the class
        self.sigma_ku.values[:] = np.copy(sigma_ku) * 1e6 # convert to mm^2 
        self.sigma_ka.values[:] = np.copy(sigma_ka) * 1e6 # convert to mm^2 
        self.sigma_w.values[:] = np.copy(sigma_w) * 1e6 # convert to mm^2

    def calc_Z(self):
        """
        This method actualy calculates Z [dBZ] 
        """
        
        #create the coeficients in equation
        from pytmatrix import tmatrix_aux
        
        # Ku-band
        lamb = tmatrix_aux.wl_Ku # [mm]
        K = tmatrix_aux.K_w_sqr[lamb]
        coef2 = (lamb ** 4) / (np.pi ** 5 * K) # [mm^4]
        
        # Ka-band
        lamb = tmatrix_aux.wl_Ka # [mm]
        K = tmatrix_aux.K_w_sqr[lamb]
        coef3 = (lamb ** 4) / (np.pi ** 5 * K) # [mm^4]
        
        # W-band
        lamb = tmatrix_aux.wl_W # [mm]
        K = tmatrix_aux.K_w_sqr[lamb]
        coef4 = (lamb ** 4) / (np.pi ** 5 * K) # [mm^4]
        
        # binned Z, linear units
        z_ku_bin = xr.DataArray(
            data = coef2 * self.sigma_ku * self.PSD * self.dD,
            dims = self.sigma_ku.dims
        )
        z_ka_bin = xr.DataArray(
            data = coef3 * self.sigma_ka * self.PSD * self.dD,
            dims = self.sigma_ka.dims
        )
        z_w_bin = xr.DataArray(
            data = coef4 * self.sigma_w * self.PSD * self.dD,
            dims = self.sigma_w.dims
        )
        
        # calculate, output is in dBZ
        Z_ku = xr.DataArray(
            data = 10. * np.log10(np.nansum(z_ku_bin, axis=1)),
            dims = self.sigma_ku.dims[0]
        )
        Z_ka = xr.DataArray(
            data = 10. * np.log10(np.nansum(z_ka_bin, axis=1)),
            dims = self.sigma_ka.dims[0]
        )
        Z_w = xr.DataArray(
            data = 10. * np.log10(np.nansum(z_w_bin, axis=1)),
            dims = self.sigma_w.dims[0]
        )
        
        # eliminate any missing values
        z_ku_bin = z_ku_bin.where(~np.isinf(z_ku_bin))
        z_ka_bin = z_ka_bin.where(~np.isinf(z_ka_bin))
        z_w_bin = z_w_bin.where(~np.isinf(z_w_bin))
        Z_ku = Z_ku.where(~np.isinf(Z_ku))
        Z_ka = Z_ka.where(~np.isinf(Z_ka))
        Z_w = Z_w.where(~np.isinf(Z_w))

        self.binZ_ku = z_ku_bin
        self.binZ_ka = z_ka_bin
        self.binZ_w = z_w_bin
        self.Z_ku = Z_ku
        self.Z_ka = Z_ka
        self.Z_w = Z_w