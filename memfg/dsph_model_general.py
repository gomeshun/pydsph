import pandas as pd
import numpy as np
import multiprocessing as multi

from .dequad_ver2 import dequad
from .dequad import dequad as dequad_ver1

from numpy import array,pi,sqrt,exp,power,log,log10,log1p,cos,tan,sin, sort,argsort, inf
from scipy.stats import norm
from scipy.special import k0, betainc, beta, hyp2f1, erf, gamma, gammainc
from scipy import integrate
from scipy.constants import parsec, degree # parsec in meter, degree in radian
from scipy.integrate import quad
from scipy.interpolate import interp1d,Akima1DInterpolator

from multiprocessing import Pool

import warnings

from .dsph_model import model, stellar_model, DM_model

GMsun_m3s2 = 1.32712440018e20


class Anisotropy_Model(model):
    name = "anisotropy model"

    
class Constant_Anisotropy_Model(Anisotropy_Model):
    name = "Constant_Anisotropy_Model"
    required_params_name = ['beta_ani']
    
    def kernel(self,u,R,**kwargs):
        """
        kernel function K(u). LOSVD is given by
        
            sigmalos2(R) = 2 * \int_1^\infty du \nu_\ast(uR)/\Sigma_\ast(R) * GM(uR) * K(u)/u.
        """
        b = self.params.beta_ani
        u2 = u**2
        kernel = sqrt(1-1/u2)*((1.5-b)*u2*hyp2f1(1.0,1.5-b,1.5,1-u2)-0.5)
        return kernel
    
class Osipkov_Merritt_Model(Anisotropy_Model):
    name = "Osipkov_Merritt_Model"
    required_params_name = ["r_a"]
    
    def kernel(self,u,R,**kwargs):
        """
        u, R: 1d array
        """
        R = R[:,np.newaxis]  # axis = 0
        u = u[np.newaxis, :] # axis = 1
        u_a = self.params.r_a / R
        u2_a = u_a**2
        u2 = u**2
        return (u2+u2_a)*(u2_a+0.5)/(u*(u2_a+1)**1.5) * np.arctan(np.sqrt((u2-1)/(u2_a+1))) - np.sqrt(1-1/u2)/2/(u2_a+1)

    
class Baes_Anisotopy_Model(Anisotropy_Model):
    name = "Baes_Anisotropy_Model"
    required_params_name = ["beta_0", "beta_inf","r_a","eta"]
    
    def beta(self,r):
        b0,binf = self.params.beta_0, self.params.beta_inf
        r_a, eta = self.params.r_a, self.params.eta
        x = power(r/r_a,eta)
        return (b0+binf*x)/(1+x)
    
    def f(self,r):
        b0,binf = self.params.beta_0, self.params.beta_inf
        r_a, eta = self.params.r_a, self.params.eta
        x = power(r/r_a,eta)
        return power(r,2*b0)*power(1+x,2*(binf-b0)/eta)
        
    
    def integrand_kernel(self,u_integ,R):
        """
        u = r/R,
        us = r_a/R
        """
        u2_integ = u_integ**2
        r_integ = R*u_integ
        return u_integ/sqrt(u2_integ-1)*(1-self.beta(r_integ)/u2_integ)/self.f(r_integ)
    
    
    def kernel(self,u,R,**kwargs):
        """
        kernel function K(u). LOSVD is given by
        
            sigmalos2(R) = 2 * \int_1^\infty du \nu_\ast(uR)/\Sigma_\ast(R) * GM(uR) * K(u)/u.
            
        u: ndarray, shape = (n_u,)
        R: ndarray, shape = (n_R,)
        
        return: ndarray, shape = (n_R,n_u)
        """
        n = 128 if ("n" not in kwargs) else kwargs["n"]
        
        u_expanded = u[np.newaxis,:,np.newaxis]  # axis = 1
        R_expanded = R[:,np.newaxis,np.newaxis]  # axis = 0
        #print("u_shape:{} R.shape:{}".format(u.shape,R.shape))
        
        def integrand(_u):     
            #_u_array = np.array(_u)[np.newaxis,np.newaxis,:]  # axis = 2
            return self.integrand_kernel(_u,R_expanded)
            
        integration = dequad(integrand,1,u_expanded,n,axis=2,replace_inf_to_zero=True,replace_nan_to_zero=True)  # shape = (n_R, n_u)
            
        return integration * self.f(R_expanded[...,0]*u_expanded[...,0])/u_expanded[...,0]

    
    
class dSph_model(model):
    name = 'dSph_model'
    required_params_name = []
    required_models = [stellar_model,DM_model,Anisotropy_Model]
    ncpu = multi.cpu_count()
#    def __init__(self,stellar_model,DM_model,**params_dSph_model):
#        """
#        params_dSph_model: pandas.Series, index = (params_stellar_model,params_DM_model,center_of_dSph)
#        """
#        # NOTE IT IS NOT COM{PATIBLE TO THE CODE BELOW!!!
#        super().__init__(**params_dSph_model)
#        self.submodels = (stellar_model,DM_model)
#        self.name = ' and '.join((model.name for model in self.submodels))


    def sigmar2(self,r_pc):
        RELERROR_INTEG = 1e-6
        density_3d = self.submodels["stellar_model"].density_3d
        enclosure_mass = self.submodels["DM_model"].enclosure_mass
        f = self.submodels["Anisotropy_Model"].f
        integrand = lambda r: density_3d(r)*f(r)*GMsun_m3s2*enclosure_mass(r)/r**2/f(r_pc)/density_3d(r_pc)*1e-6/parsec
        integ, abserr = integrate.quad(integrand,r_pc,np.inf)
        return integ

    
    def integrand_sigmalos2(self,u,R_pc,n_kernel=128):
        '''
        integrand of sigmalos2 at R = R_pc.
        u is a variable of integration, u=r/R.
        Domain: 1 < u < oo.
        
        u: ndarray: shape = (n_u,)
        R_pc: ndarray: shape = (n_R,)
        '''
        
        R_pc = R_pc[:,np.newaxis] # axis = 0
        u = u[np.newaxis,:]  # axis = 1
        
        density_3d = self.submodels["stellar_model"].density_3d
        density_2d = self.submodels["stellar_model"].density_2d
        enclosure_mass = self.submodels["DM_model"].enclosure_mass
        kernel = self.submodels["Anisotropy_Model"].kernel
        r = R_pc*u
        # Note that parsec = parsec/m.
        # If you convert m -> pc,      ... var[m] * [1 pc/ parsec m] = var/parsec[pc].
        #                pc^1 -> m^pc, ... var[pc^1] * parsec(=[pc/m]) = var[m^-1]
        # Here var[m^3 pc^-1 s^-2] /parsec[m/pc] * 1e-6[km^2/m^2] = var[km^2/s^2]
        return 2.0 * kernel(u[0,:],R_pc[:,0],n=n_kernel)/u *  density_3d(r)/density_2d(R_pc)*GMsun_m3s2 * enclosure_mass(r) / parsec * 1e-6

    
    def sigmalos2_dequad(self,R_pc,n=1024,n_kernel=128,ignore_RuntimeWarning=True):
        
        
        def func(u):
            '''
            shape: (n_u,) -> (n_R,n_u)
            Note that the shape of kernel return. 
            '''
            return self.integrand_sigmalos2(u,R_pc,n_kernel)
        with warnings.catch_warnings():
            if ignore_RuntimeWarning:
                warnings.simplefilter('ignore',RuntimeWarning)
            integ = dequad(func,1,np.inf,axis=1,n=n,replace_inf_to_zero=True,replace_nan_to_zero=True)
            return integ 
    
    
    def sigmalos_dequad(self,R_pc,n=1024,n_kernel=128,ignore_RuntimeWarning=True):
        return np.sqrt(self.sigmalos2_dequad(R_pc,n,n_kernel,ignore_RuntimeWarning))

    
    