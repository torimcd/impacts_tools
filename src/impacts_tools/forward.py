import pandas as pd 
import numpy as np 
import scipy.interpolate
import os
import pytmatrix

class forward_Z():
    """ 
    Class to forward model reflectivity using Leinonen and Szrymer 2015. 
    
    X-, Ku-, Ka-, and W-band are supported. 
    
    """    
    
    def set_PSD(self, PSD=None, D=None, dD=None):
        """
        This sets the PSD objects in the class. It expects the following: 
        
        PSD: Matrix, (n_samples, n_bins); units: m**-4
        D: Array, (n_bins,); units m
        dD: Array, (n_bins,); units m
        """
        #set the number of simulations
        n_sims = 11
        #reshape PSD to have a third dimension. 1 for each L15 sim
        PSD = np.reshape(PSD,[PSD.shape[0],PSD.shape[1],1])
        PSD = np.tile(PSD,(1,1,n_sims))
        
        self.PSD = PSD
        self.dD = dD
        self.D = D
        
        
        #time to reshape things
        psd_shape = self.PSD.shape
        
        #rescale to match shape of PSD. This allows fast computations through vectorization 
        self.dD = np.reshape(self.dD,[1,psd_shape[1]])
        self.dD = np.tile(self.dD,(psd_shape[0],1))
        self.dD = np.reshape(self.dD,[psd_shape[0],psd_shape[1],1])
        self.dD = np.tile(self.dD,(1,1,psd_shape[2]))
        
        #rescale to match shape of PSD. This allows fast computations through vectorization 
        self.D = np.reshape(self.D,[1,psd_shape[1]])
        self.D = np.tile(self.D,(psd_shape[0],1)) 
        self.D = np.reshape(self.D,[psd_shape[0],psd_shape[1],1])
        self.D = np.tile(self.D,(1,1,psd_shape[2]))
        
        #set a placeholder for the backscatter cross-section 
        self.sigma_x = np.zeros(psd_shape)
        self.sigma_ku = np.zeros(psd_shape)
        self.sigma_ka = np.zeros(psd_shape) 
        self.sigma_w = np.zeros(psd_shape)

        #set a placeholder for the riming fraction 
        self.rimefrac = np.zeros(psd_shape)
    
    def load_split_L15(self):
        """ 
        This method loads the results from Leinonen and Szyrmer 2015 and then splits the particles into each rimed category.
        There are 6 categories. Each category of partilces were exposed to a larger amount of supercooled liquid water path. 
        The order of less rimed to heavily rimes is: No riming; 0.1 kg/m^2; 0.2 kg/m^2; 0.5 kg/m^2; 1.0 kg/m^2; 2.0 kg/m^2.
        """
        #load text file
        cwd = os.path.dirname(os.path.abspath(__file__))
        header = [
            'rimemodel', 'lwp', 'mass', 'Dmax', 'rad_gy', 'axis_ratio',
            'rimed_fraction', 'Xchh', 'Xvv', 'Kuchh', 'Kucvv', 'Kachh', 'Kacvv',
            'Wchh','Wcvv'
        ]
        leinonen = pd.read_csv(
            cwd + '/database.tex', delim_whitespace=True, names=header,
            header=None, index_col=None
        )
        
        #split methods 
        leinonen_A = leinonen.where(leinonen.rimemodel == 'A')
        leinonen_B = leinonen.where(leinonen.rimemodel == 'B')
        
        #grab all the rimed instances
        bins = np.arange(-0.05, 2.05, 0.1)
        bin_i = np.digitize(leinonen_B.lwp,bins=bins)
        leinonen_B['bin_i'] = bin_i
        grouped = leinonen_B.groupby('bin_i')
        groups = grouped.groups
        list_of_keys = list(groups.keys())
        list_of_subsetted_data = []
        for i in list_of_keys:
                g_i = np.asarray(groups[i].values,dtype=int)
                d = leinonen_B.iloc[g_i]
                list_of_subsetted_data.append(d)

        L01 = list_of_subsetted_data[0]
        L02 = list_of_subsetted_data[1]
        L05 = list_of_subsetted_data[2]
        L10 = list_of_subsetted_data[3]
        L20 =list_of_subsetted_data[4].dropna()
        
        #grab the non-rimed situation 
        bin_i = np.digitize(leinonen_A.lwp,bins=bins)
        leinonen_A['bin_i'] = bin_i
        grouped = leinonen_A.groupby('bin_i')
        groups = grouped.groups
        list_of_keys = list(groups.keys())
        list_of_subsetted_data = []
        for i in list_of_keys:
                g_i = np.asarray(groups[i].values,dtype=int)
                d = leinonen_A.iloc[g_i]
                list_of_subsetted_data.append(d)

        L00 = list_of_subsetted_data[0]
        
        #store them in the class. 
        self.L00 = L00
        self.L01 = L01
        self.L02 = L02
        self.L05 = L05
        self.L10 = L10
        self.L20 = L20
        
    def fit_sigmas(self):
        """ 
        This method is to fit a flexible function to the Leinonen and Szyrmer (2015) data. Essentially, it interpolates
        the backscatter cross-section to whatever values of D are inputed to the class. Please make sure you have the correct units.
        D should be in m.
        """
        
        #loop over the various degrees of riming 
        list_o_objects = [self.L00,self.L01,self.L02,self.L05,self.L10,self.L20]
        for i,ii in enumerate(list_o_objects):
            bins = np.append(np.linspace(1e-4,3e-3,5),np.linspace(3e-3,2.20e-2,7))
            whichbin = np.digitize(ii.Dmax,bins=bins)
            ii['bin_i'] = whichbin
            df = ii.groupby('bin_i').median()
            df = df.reindex(np.arange(0,len(bins)))
            df = df.interpolate()
            df = df.dropna(how='all')
            
            #fit the functions for each frequency 
            f_x = scipy.interpolate.interp1d(
                np.log10(df.Dmax.values[:-1]), np.log10(df.Xchh.values[:-1]),
                fill_value='extrapolate', kind='linear', bounds_error=False
            )
            sigma_x = 10**f_x(np.log10(self.D[0,:,0]))

            f_ku = scipy.interpolate.interp1d(
                np.log10(df.Dmax.values[:-1]), np.log10(df.Kuchh.values[:-1]),
                fill_value='extrapolate', kind='linear', bounds_error=False
            )
            sigma_ku = 10**f_ku(np.log10(self.D[0,:,0]))

            f_ka = scipy.interpolate.interp1d(
                np.log10(df.Dmax.values[:-1]), np.log10(df.Kachh.values[:-1]),
                fill_value='extrapolate', kind='linear', bounds_error=False
            )
            sigma_ka = 10**f_ka(np.log10(self.D[0,:,0]))

            f_w = scipy.interpolate.interp1d(
                np.log10(df.Dmax.values[:-1]), np.log10(df.Wchh.values[:-1]),
                fill_value='extrapolate', kind='linear', bounds_error=False
            )
            sigma_w = 10**f_w(np.log10(self.D[0,:,0]))

            #time to reshape things again so we can have vectorized calculations 
            psd_shape = self.PSD.shape

            sigma_x = np.reshape(sigma_x,[1,psd_shape[1]])
            sigma_x = np.tile(sigma_x,(psd_shape[0],1))

            sigma_ku = np.reshape(sigma_ku,[1,psd_shape[1]])
            sigma_ku = np.tile(sigma_ku,(psd_shape[0],1))

            sigma_ka = np.reshape(sigma_ka,[1,psd_shape[1]])
            sigma_ka = np.tile(sigma_ka,(psd_shape[0],1))

            sigma_w = np.reshape(sigma_w,[1,psd_shape[1]])
            sigma_w = np.tile(sigma_w,(psd_shape[0],1))

            #store it into the class, the 3rd dimension is now the various degrees of riming. 
            self.sigma_x[:,:,i] = np.copy(sigma_x)*1e6 #convert to mm^2 
            self.sigma_ku[:,:,i] = np.copy(sigma_ku)*1e6 #convert to mm^2 
            self.sigma_ka[:,:,i] = np.copy(sigma_ka)*1e6 #convert to mm^2 
            self.sigma_w[:,:,i] = np.copy(sigma_w)*1e6 #convert to mm^2 

        #initialize the riming categories
        rcat_orig = np.array([0., 0.1, 0.2, 0.5, 1., 2.])
        rcat_new = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1., 2.])

        #interpolate m-D coefficients
        a_orig = np.array([0.015, 0.0354, 0.0545, 0.126, 0.369, 1.7])
        b_orig = np.array([2.08, 2.06, 2.05, 2.03, 2.11, 2.3])
        a_new = np.interp(rcat_new, rcat_orig, a_orig)
        b_new = np.interp(rcat_new, rcat_orig, b_orig)
        a_new = (10.**(3-2*b_new)) * a_new # change a coefficient to cgs units


        #initialize the backscatter arrays with the added riming categories
        sigma_x = np.zeros((psd_shape[1], len(rcat_new)))
        sigma_ku = np.zeros((psd_shape[1], len(rcat_new)))
        sigma_ka = np.zeros((psd_shape[1], len(rcat_new)))
        sigma_w = np.zeros((psd_shape[1], len(rcat_new)))

        #loop through the bins and interpolate
        for i in range(psd_shape[1]):
            sigma_x[i,:] = 10.**np.interp(rcat_new, rcat_orig, np.log10(self.sigma_x[0,i,:6]))
            sigma_ku[i,:] = 10.**np.interp(rcat_new, rcat_orig, np.log10(self.sigma_ku[0,i,:6]))
            sigma_ka[i,:] = 10.**np.interp(rcat_new, rcat_orig, np.log10(self.sigma_ka[0,i,:6]))
            sigma_w[i,:] = 10.**np.interp(rcat_new, rcat_orig, np.log10(self.sigma_w[0,i,:6]))

        #reshape and tile the arrays
        sigma_x = np.reshape(sigma_x, (1, psd_shape[1], len(rcat_new)))
        sigma_x = np.tile(sigma_x, (psd_shape[0], 1, 1))

        sigma_ku = np.reshape(sigma_ku, (1, psd_shape[1], len(rcat_new)))
        sigma_ku = np.tile(sigma_ku, (psd_shape[0], 1, 1))

        sigma_ka = np.reshape(sigma_ka, (1, psd_shape[1], len(rcat_new)))
        sigma_ka = np.tile(sigma_ka, (psd_shape[0], 1, 1))

        sigma_w = np.reshape(sigma_w, (1, psd_shape[1], len(rcat_new)))
        sigma_w = np.tile(sigma_w, (psd_shape[0], 1, 1))

        #store it into the class again
        self.sigma_x = np.copy(sigma_x)
        self.sigma_ku = np.copy(sigma_ku)
        self.sigma_ka = np.copy(sigma_ka)
        self.sigma_w = np.copy(sigma_w)
        self.a_coeff = a_new
        self.b_coeff = b_new

    def fit_rimefrac(self):
        """ 
        This method is to fit a flexible function to the Leinonen and Szyrmer (2015) data. Essentially, it interpolates
        the riming fraction to whatever values of D are inputed to the class. Please make sure you have the correct units.
        D should be in m.
        """
        
        #loop over the various degrees of riming 
        list_o_objects = [self.L00,self.L01,self.L02,self.L05,self.L10,self.L20]
        for i,ii in enumerate(list_o_objects):
            bins = np.append(np.linspace(1e-4,3e-3,5),np.linspace(3e-3,2.20e-2,7))
            whichbin = np.digitize(ii.Dmax,bins=bins)
            ii['bin_i'] = whichbin
            df = ii.groupby('bin_i').median()
            df = df.reindex(np.arange(0,len(bins)))
            df = df.interpolate()
            df = df.dropna(how='all')
            
            #fit the functions for each frequency 
            f = scipy.interpolate.interp1d(np.log10(df.Dmax.values[:-1]),df.rimed_fraction.values[:-1],fill_value='extrapolate',kind='linear',bounds_error=False)
            rimefrac = f(self.D[0,:,0])

            #time to reshape things again so we can have vectorized calculations 
            psd_shape = self.PSD.shape

            rimefrac = np.reshape(rimefrac,[1,psd_shape[1]])
            rimefrac = np.tile(rimefrac,(psd_shape[0],1))

            #store it into the class, the 3rd dimension is now the various degrees of riming. 
            self.rimefrac[:,:,i] = np.copy(rimefrac)

        #initialize the riming categories
        rcat_orig = np.array([0., 0.1, 0.2, 0.5, 1., 2.])
        rcat_new = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1., 2.])

        #initialize the riming fraction array with the added riming categories
        rimefrac = np.zeros((psd_shape[1], len(rcat_new)))

        #loop through the bins and interpolate
        for i in range(psd_shape[1]):
            rimefrac[i,:] = np.interp(rcat_new, rcat_orig, self.rimefrac[0,i,:6])

        #reshape and tile the array
        rimefrac = np.reshape(rimefrac, (1, psd_shape[1], len(rcat_new)))
        rimefrac = np.tile(rimefrac, (psd_shape[0], 1, 1))

        #store it into the class again
        self.rimefrac = np.copy(rimefrac)

    def calc_Z(self):
        """
        Here is the method that actualy calculates Z. Output is in dBZ. 
        
        The resulting shape is 2d. Axis 1 is still 
        """
        
        #create the coeficients in equation
        from pytmatrix import tmatrix_aux
        
        #X-band
        lamb = tmatrix_aux.wl_X #is in mm
        K = tmatrix_aux.K_w_sqr[lamb]
        coef = (lamb**4)/(np.pi**5*K) #mm^4
        
        #Ku-band
        lamb = tmatrix_aux.wl_Ku #is in mm
        K = tmatrix_aux.K_w_sqr[lamb]
        coef2 = (lamb**4)/(np.pi**5*K) #mm^4
        
        #Ka-band
        lamb = tmatrix_aux.wl_Ka #is in mm
        K = tmatrix_aux.K_w_sqr[lamb]
        coef3 = (lamb**4)/(np.pi**5*K) #mm^4
        
        #W-band
        lamb = tmatrix_aux.wl_W #is in mm
        K = tmatrix_aux.K_w_sqr[lamb]
        coef4 = (lamb**4)/(np.pi**5*K) #mm^4
        
        # binned Z, linear units
        z_x_bin = coef * self.sigma_x * self.PSD * self.dD
        z_ku_bin = coef2 * self.sigma_ku * self.PSD * self.dD
        z_ka_bin = coef3 * self.sigma_ka * self.PSD * self.dD
        z_w_bin = coef4 * self.sigma_w * self.PSD * self.dD
        
        #calculate, output is in dBZ
        Z_x = 10*np.log10(coef*np.nansum(self.sigma_x*self.PSD*self.dD,axis=1))
        Z_ku = 10*np.log10(coef2*np.nansum(self.sigma_ku*self.PSD*self.dD,axis=1))
        Z_ka = 10*np.log10(coef3*np.nansum(self.sigma_ka*self.PSD*self.dD,axis=1))
        Z_w = 10*np.log10(coef4*np.nansum(self.sigma_w*self.PSD*self.dD,axis=1))
        
        # eliminate any missing values
        Z_x[np.isinf(Z_x)] = np.nan
        Z_ku[np.isinf(Z_ku)] = np.nan
        Z_ka[np.isinf(Z_ka)] = np.nan
        Z_w[np.isinf(Z_w)] = np.nan

        self.Z_x = Z_x.T
        self.Z_ku = Z_ku.T
        self.Z_ka = Z_ka.T
        self.Z_w = Z_w.T
        self.binZ_x = z_x_bin.T # sims x bins x times
        self.binZ_ku = z_ku_bin.T
        self.binZ_ka = z_ka_bin.T
        self.binZ_w = z_w_bin.T