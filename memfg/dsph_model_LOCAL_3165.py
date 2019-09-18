import pandas as pd
import numpy as np
import multiprocessing as multi

from .dequad import dequad_hinf, dequad

from numpy import array,pi,sqrt,exp,power,log,log10,log1p,cos,tan,sin, sort,argsort, inf
from scipy.stats import norm
from scipy.special import k0, betainc, beta, hyp2f1, erf, gamma, gammainc
from scipy import integrate
from scipy.constants import parsec, degree # parsec in meter, degree in radian
from scipy.integrate import quad
from scipy.interpolate import interp1d,Akima1DInterpolator

from multiprocessing import Pool

GMsun_m3s2 = 1.32712440018e20
R_trunc_pc = 1866.

class model:
    #params, required_params_name = pd.Series(), ['',]
    '''
    params, required_prams_name is undefined; must be defined in child class
    '''
    def __init__(self,show_init=False,submodels_dict={},**params):
        self.params = pd.Series(params)
        self.required_params_name = list(self.required_params_name)
        self.submodels = submodels_dict
        self.required_models = {}
        if 'submodels' in self.params.index:
            raise TypeError('submodels -> submodels_dict?')
        if set(self.params.index) != set(self.required_params_name):
            raise TypeError(self.name+' has the paramsters: '+str(self.required_params_name)+" but in put is "+str(self.params.index))
        for required_model in self.required_models:
            isinstance_ =  [isinstance(model,required_model) for model in self.submodels.values]
            if not any(isinstance_):
                raise TypeError("no "+str(required_model))
            
        if self.submodels != {}:
            self.name = ': ' + ' and '.join((model.name for model in self.submodels.values()))
        if show_init:
            print("initialized:")
            print(self.show_params("all"))

    def __repr__(self):
        ret = self.name + ":\n" + self.show_params().__repr__()
        #if len(self.submodels) > 0:
        #    ret += '\n'
        #    for model in self.submodels.values():
        #        ret += model.__repr__() + '\n'
        return ret

    def show_params(self,models='all'):
        ret = self.params
        if self.submodels != {} and models=='all':
            ret = pd.concat([model.show_params('all') for model in self.submodels.values()])
        return ret

    def show_required_params_name(self,models='all'):
        ret = self.required_params_name[:] # need copy because we must keep self.required_params_name
        if self.submodels != {} and models=='all':
            [ ret.extend(model.show_required_params_name('all')) for model in self.submodels.values() ]
        return ret

    def is_required_params_name(self,params_name_candidates):
        return [ (p in self.required_params_name) for p in params_name_candidates ]

    def update(self,new_params_dict=None,target='all',**kwargs):
        new_params = pd.Series(new_params_dict) if type(new_params_dict)==dict else None
        if new_params is None:
            new_params = pd.Series(kwargs)
        #print(np.isin(new_params.index, self.show_required_params_name('all')+['this','all']))
        if not np.any(np.isin(new_params.index, self.show_required_params_name('all')+['this','all'])):
            raise TypeError("new params has no required parameters.\nrequired parameters:{}".format(self.required_params_name))
        if target in ('this','all'):
            self.params.update(new_params)
            [model.params.update(new_params) for model in self.submodels.values()] if target in ('all',) else None
        else:
            [(model.params.update(new_params) if target==model.name else None) for model in self.submodels.values()]


class stellar_model(model):
    name = "stellar model"
    def density(self,distance_from_center,dimension):
        if dimension == "2d":
            return self.density_2d(distance_from_center)
        elif dimension == "3d":
            return self.density_3d(distance_from_center)
    def density_2d_truncated(self,R_pc,R_trunc_pc):
        """
        Truncated 2D density. Note that
            \int_0^{R_trunc} 2\pi R density_2d_truncated(R,R_trunc) = 1 .
        """
        return self.density_2d(R_pc)/self.cdf_R(R_trunc_pc)


class plummer_model(stellar_model):
    name = "Plummer model"
    required_params_name = ['re_pc',]
    def density_2d(self,R_pc):
        re_pc= self.params.re_pc
        return 1/(1+(R_pc/re_pc)**2)**2 /np.pi/re_pc**2
    def logdensity_2d(self,R_pc):
        re_pc= self.params.re_pc
        return -np.log1p((R_pc/re_pc)**2)*2 -log(np.pi) -log(re_pc)*2
    def density_2d_normalized_re(self,R_pc):
        re_pc= self.params.re_pc
        return 4/(1+(R_pc/re_pc)**2)**2
      
    def density_3d(self,r_pc):
        re_pc= self.params.re_pc
        return (3/4/np.pi/re_pc**3)/np.sqrt(1+(r_pc/re_pc)**2)**5
    def cdf_R(self,R_pc):
        '''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc= self.params.re_pc
        return 1/(1+(re_pc/R_pc)**2)
    def mean_density_2d(self,R_pc):
        '''
        return the mean density_2d in R < R_pc with the weight 2*pi*R
        mean_density_2d = \frac{\int_\RoIR \dd{R} 2\pi R \Sigma(R)}{\int_\RoIR \dd{R} 2\pi R}
            = \frac{cdf_R(R)}{\pi R^2}
        '''
        re_pc= self.params.re_pc
        return 1/pi/(R_pc**2+re_pc**2)
    
    def _half_light_radius(self,re_pc):
        '''
        Half-light-raduis means that the radius in which the half of all stars are include
        '''
        return re_pc
      
    def half_light_radius(self):
        '''
        Half-light-raduis means that the radius in which the half of all stars are include
        '''
        return self._half_light_radius(self.params.re_pc)

class sersic_model(stellar_model):
    name = "Sersic model"
    required_params_name = ['re_pc','n']
    def b(self,n):
        return 2*n - 0.324
    def norm(self):
        n = self.params.n
        return pi*self.params.re_pc**2 *power(self.b(n),-2*n) * gamma(2*n+1)
    def density_2d(self,R_pc):
        re_pc= self.params.re_pc
        n = self.params.n
        return exp(-self.b(n)*power(R_pc/self.params.re_pc,1/n))/self.norm()
    def cdf_R(self,R_pc):
        '''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc= self.params.re_pc
        n = self.params.n
        return gammainc(2*n,self.b(n)*power(R_pc/re_pc,1/n)) - gammainc(2*n,0)
    def mean_density_2d(self,R_pc):
        '''
        return the mean density_2d in R < R_pc with the weight 2*pi*R
        mean_density_2d = \frac{\int_\RoIR \dd{R} 2\pi R \Sigma(R)}{\int_\RoIR \dd{R} 2\pi R}
            = \frac{cdf_R(R)}{\pi R^2}
        '''
        return self.cdf_R(R_pc)/pi/R_pc**2
    
    def half_light_radius(self):
        return self.params.re_pc
    
class exp2d_model(stellar_model):
    name = "exp2d model"
    required_params_name = ['re_pc',]
    def density_2d(self,R_pc):
        re_pc = self.params.re_pc
        return (1./2/pi/re_pc**2)*exp(-R_pc/re_pc) 
    def logdensity_2d(self,R_pc):
        re_pc = self.params.re_pc
        return log(1./2/pi) -log(re_pc)*2 +(-R_pc/re_pc) 
    def density_3d(self,r_pc):
        re_pc = self.params.re_pc
        return (1./2/pi**2/re_pc**3)*k0(r_pc/re_pc)
    def cdf_R(self,R_pc):
        '''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc = self.params.re_pc
        return 1. - exp(-R_pc/re_pc)*(1+R_pc/re_pc)
    def mean_density_2d(self,R_pc):
        re_pc = self.params.re_pc
        return self.cdf_R(R_pc)/pi/R_pc**2
    
    def _half_light_radius(self,re_pc):
        return 1.67834699001666*re_pc
    
    def half_light_radius(self):
        return self._half_light_radius(self.params.re_pc)
    
class exp3d_model(stellar_model):
    name = "exp3d model"
    required_params_name = ['re_pc',]
    def density_2d(self,R_pc):
        re_pc = self.params.re_pc
        return (1./2/pi/re_pc**2)*exp(-R_pc/re_pc) 
    def density_3d(self,r_pc):
        re_pc = self.params.re_pc
        return (1./2/pi**2/re_pc**3)*k0(r_pc/re_pc)
    def cdf_R(self,R_pc):
        '''
        cdf_R(R) = \int_0^R \dd{R'} 2\pi R' \Sigma(R')
        '''
        re_pc = self.params.re_pc
        return 1. - exp(-R_pc/re_pc)*(1+R_pc/re_pc)
    def mean_density_2d(self,R_pc):
        re_pc = self.params.re_pc
        return self.cdf_R(R_pc)/pi/R_pc**2
    def half_light_radius(self):
        return 1.67834699001666*self.params.re_pc
        
class uniform2d_model(stellar_model):
    name = "uniform model"
    required_params_name = ['Rmax_pc',]
    def density_2d(self,R_pc):
        return 1./(pi*self.params.Rmax_pc**2)*np.ones_like(R_pc)
    def cdf_R(self,R_pc):
        return (R_pc/self.params.Rmax_pc)**2    

class DM_model(model):
    name = "DM model"

class NFW_model(DM_model):
    name = "NFW model"
    required_params_name = ['rs_pc','rhos_Msunpc3','a','b','g','R_trunc_pc']
    
    #def __init__(self,params):
    #    super().__init__(params)
    #    if set(self.params.index) != set(NFW_params_name):
    #        raise TypeError('NFW_model has the paramsters: '+str(NFW_params_name))

    def mass_density_3d(self,r_pc):
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        x = r_pc/rs_pc
        return rhos_Msunpc3*power(x,-g)*power(1+power(x,a),-(b-g)/a)
        
    def enclosure_mass(self,r_pc):
        #ret = (array(r_pc.shape) if len(r_pc)>1 else 0)
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        
        x = power(r_pc/rs_pc,a)
        
        argbeta0 = (3-g)/a
        argbeta1 = (b-3)/a
        
        return (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x/(1+x)) 
    
    def enclosure_mass_truncated(self,r_pc):
        #ret = (array(r_pc.shape) if len(r_pc)>1 else 0)
        rs_pc, rhos_Msunpc3,a,b,g = self.params.rs_pc, self.params.rhos_Msunpc3, self.params.a, self.params.b,self.params.g
        
        is_in_Rtrunc = r_pc<self.params.R_trunc_pc
        is_outof_Rtrunc = np.logical_not(is_in_Rtrunc)
        
        x = power(r_pc/rs_pc,a)*is_in_Rtrunc
        x_truncd = power(self.params.R_trunc_pc/rs_pc,a)
        argbeta0 = (3-g)/a
        argbeta1 = (b-3)/a
        
        #ret[is_in_Rtrunc] = (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x/(1+x))
        #ret[is_outof_Rtrunc] = (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x_truncd/(1+x_truncd))
        return is_in_Rtrunc * (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x/(1+x)) + is_outof_Rtrunc * (4.*pi*rs_pc**3*rhos_Msunpc3/a)*beta(argbeta0,argbeta1)*betainc(argbeta0,argbeta1,x_truncd/(1+x_truncd))
        
class dSph_model(model):
    name = 'dSph_model'
    required_params_name = ['anib']
    required_models = [stellar_model,DM_model]
    ncpu = multi.cpu_count()
#    def __init__(self,stellar_model,DM_model,**params_dSph_model):
#        """
#        params_dSph_model: pandas.Series, index = (params_stellar_model,params_DM_model,center_of_dSph)
#        """
#        # NOTE IT IS NOT COM{PATIBLE TO THE CODE BELOW!!!
#        super().__init__(**params_dSph_model)
#        self.submodels = (stellar_model,DM_model)
#        self.name = ' and '.join((model.name for model in self.submodels))
    def mykernel_over_u(self,u,out="linear"):
        '''
        My kernel over u. Using 2F1.Using this kernel over u K(u)/u, sigmalos2 is given by
            sigmalos2(R) = 2 * \int_1^oo du \Sigma_\ast(uR)/\nu_\ast(R) * GM(uR) * K(u)/u.
            
        Descriptions:
            args: u = r_integrated/R, 1<u<oo
        '''
        anib = self.params.anib
        u2 = u**2
        if out=="linear":
            return 1/u * sqrt(1-1/u2)*((1.5-anib)*u2*hyp2f1(1.0,1.5-anib,1.5,1-u2)-0.5)
        elif out=="log":
            return -log(u) + log(1-1/u2)/2 + log((1.5-anib)*u2*hyp2f1(1.0,1.5-anib,1.5,1-u2)-0.5)

    def sigmar2(self,r_pc):
        RELERROR_INTEG = 1e-6
        anib = self.params.anib
        integrand = lambda r,r1: self.submodels["stellar_model"].density_3d(r)*np.power(r/r1,-2*anib)*GMsun_m3s2*self.submodels["DM_model"].enclosure_mass(r)/r**2/self.submodels["stellar_model"].density_3d(r_pc)*1e-6/parsec
        integ, abserr = integrate.quad(integrand,r_pc,np.inf,args=(r_pc,))
        return integ
    
    def naive_sigmalos2(self,R_pc):
        RELERROR_INTEG = 1e-6
        anib = self.params.anib
        integrand = lambda r: (1-anib*(R_pc/r)**2)*self.submodels["stellar_model"].density_3d(r)*self.sigmar2(r)/np.sqrt(1-(R_pc/r)**2)
        rs_interp = np.logspace(-2,6,51)
        integrand_interp = interp1d(rs_interp,[integrand(r) for r in rs_interp],kind="quadratic") 
        integ, abserr = integrate.quad(integrand_interp,R_pc,np.inf)
        return 2*integ/self.submodels["stellar_model"].density_2d(R_pc)
    
    def integrand_sigmalos2(self,c,arg_R_pc):
        '''
        integrand of sigmalos2 at R = R_pc.
        c is a variable of integration, related to u (=r/R) as u = 1/c.
        domain: 0 < c < 1, OR 1 < u < oo.
        The variable c has the mean of cos(\theta), where \theta is the angle of the popsition of integration
        on the LINE-of-sight. (integration on los)
        '''
        params_dSph,params_stellar,params_DM = self.params,self.submodels["stellar_model"].params,self.submodels["DM_model"].params
        anib = params_dSph.anib
        #print(params_dSph,params_stellar,params_DM)
        rhos_Msunpc3,rs_pc,a,b,g = [params_DM[key] for key in ('rhos_Msunpc3','rs_pc','a','b','g')]
        re_pc = params_stellar.re_pc
        u2 = 1./c/c
        #####weight = sqrt(z2-1)*((1.5-anib)*power(z2,anib)*hyp2f1(0.5,anib,1.5,1-z2)-0.5)
        #weight = sqrt(z2-1)*((1.5-anib)*z2*hyp2f1(1.0,1.5-anib,1.5,1-z2)-0.5) # looks like faster
        ####weight = sqrt(z2-1)*((1.5-anib)*hyp2f1(1,anib,1.5,1-1./z2)-0.5) #NOTE: It looks simple but slow because of the calculation of 2f1
        return sqrt(u2-1)*((1.5-anib)*u2*hyp2f1(1.0,1.5-anib,1.5,1-u2)-0.5) * self.submodels["stellar_model"].density_3d(arg_R_pc/c)*GMsun_m3s2*self.submodels["DM_model"].enclosure_mass(arg_R_pc/c)/parsec # NOTE: without u^-2 !! 1/parsec because we divide it by Sigmaast [pc^-2] so convert one of [m] -> [pc]  
    
    def integrand_sigmalos2_using_mykernel(self,u,arg_R_pc):
        '''
        integrand of sigmalos2 at R = R_pc.
        u is a variable of integration, u=r/R.
        Domain: 1 < u < oo.
        '''
        stellar_model,DM_model = self.submodels["stellar_model"],self.submodels["DM_model"]
        u2,r = u**2,arg_R_pc*u
        # Note that parsec = parsec/m.
        # If you convert m -> pc,      ... var[m] * [1 pc/ parsec m] = var/parsec[pc].
        #                pc^1 -> m^pc, ... var[pc^1] * parsec(=[pc/m]) = var[m^-1]
        # Here var[m^3 pc^-1 s^-2] /parsec[m/pc] * 1e-6[km^2/m^2] = var[km^2/s^2]
        return 2 * self.mykernel_over_u(u) *  stellar_model.density_3d(r)/stellar_model.density_2d(arg_R_pc)*GMsun_m3s2 * DM_model.enclosure_mass(r) / parsec * 1e-6

    def sigmalos2_scaler_using_mykernel(self,R_pc):
        '''
        sigmalos2[km^2 s^-2].
        '''
        u_trunc = self.submodels["DM_model"].params.R_trunc_pc/R_pc
        u_max = 10 * u_trunc
        u_re = self.submodels["stellar_model"].params.re_pc/R_pc

        # outer region is almost 0, so divede the region for good point -> u_max
        
        RELERR_INTEG = 1e-6
        integ, abserr =  integrate.quad(self.integrand_sigmalos2_using_mykernel, 1,u_max, args=(R_pc,),points=(1.001,max(u_re,1.08),2.71,u_trunc))
        return integ
    
    def sigmalos2_dequad(self,R_pc,dtype=np.float64):
        def func(u):
            u_ = np.array(u)[np.newaxis,:]
            R_pc_ = np.array(R_pc)[:,np.newaxis]
            return self.integrand_sigmalos2_using_mykernel(u_,R_pc_)
        ret = dequad_hinf(func,1,axis=1,width=5e-3,pN=1000,mN=1000,dtype=np.dtype,show_fig=show_fig)
        if np.any(isnan(ret)):
            raise TypeError("nan! {}".format(ret))
        return ret
    
    def sigmalos_dequad(self,R_pc,show_fig=False,dtype=np.float64,ignore_nan=False):
        def func(u):
            u_ = np.array(u)[np.newaxis,:]
            R_pc_ = np.array(R_pc)[:,np.newaxis]
            return self.integrand_sigmalos2_using_mykernel(u_,R_pc_)
        return np.sqrt(dequad(func,1,inf,axis=1,width=5e-3,pN=1000,mN=1000,dtype=dtype,show_fig=show_fig,ignore_nan=ignore_nan))
    
    def downsampling(self,array,downsampling_rate=0.5):
        '''
        downsampling too dense ones.
        '''
        ordered_array = sort(array)
        arg_ordered_array = argsort(array)
        diff = ordered_array[1:] - ordered_array[:-1]
        arg_arg_ordered_array = argsort(diff)[int(downsampling_rate*len(array)):] # ordering by diff and extract not too dense ones
        #display(pd.DataFrame({"arg_ordered_array":arg_ordered_array,"array":array}))
        #display(pd.DataFrame({"arg_arg_ordered_array":arg_arg_ordered_array}))
        return array[arg_ordered_array[arg_arg_ordered_array]]
    
    def downsamplings(self,array,downsampling_rate=0.5,iteration=1):
        ret = array
        func = self.downsampling
        #print("print array")
        #display(ret)
        for i in range(iteration):
            #print("print return i:{}".format(i))
            ret = func(ret,downsampling_rate)
            #display(ret)
        return ret
    
    def sigmalos_dequad_interp1d_downsampled(self,R_pc,downsampling_rate=0.6,iteration=5,kind="cubic",offset=5,sep_offset=1,points=[25.,50.,100.,191.,400.,1950.,2000.,2050.],show_fig=False,dtype=np.float64,ignore_nan=True):
        R_pc_sorted = sort(R_pc)
        R_pc_downsampled = np.unique(np.concatenate([self.downsamplings(R_pc,iteration=iteration),R_pc_sorted[:offset:sep_offset],R_pc_sorted[-offset::sep_offset],points]))
        sigmalos_ = self.sigmalos_dequad(R_pc_downsampled,show_fig=show_fig,dtype=dtype,ignore_nan=ignore_nan)
        interpd_func = interp1d(R_pc_downsampled, sigmalos_,kind=kind)
        #interpd_func = Akima1DInterpolator(R_pc_downsampled, sigmalos_)
        return interpd_func(R_pc) # here R_pc is original, so return unsorted results
        
    
    def sigmalos2_scaler(self,R_pc,using_mykernel=False): # sigmalos2[km^2 s^-2] 
        params_dSph,params_stellar,params_DM = self.params,self.submodels["stellar_model"].params,self.submodels["DM_model"].params
        anib = params_dSph.anib
        #print(params_dSph,params_stellar,params_DM)
        rhos_Msunpc3,rs_pc,a,b,g = [params_DM[key] for key in ('rhos_Msunpc3','rs_pc','a','b','g')]
        re_pc = params_stellar.re_pc

        RELERR_INTEG = 1e-6
        #print("R_pc:",R_pc)
        integ, abserr =  integrate.quad(self.integrand_sigmalos2, 0,1, args=(R_pc,),points=(0.5,0.9,0.99,0.999,0.9999,0.99999))
        return 2*integ/self.submodels["stellar_model"].density_2d(R_pc)*1e-6
    
    def sigmalos2_vector(self,Rs_pc):
        sigmalos2_scaler = self.sigmalos2_scaler
        return [sigmalos2_scaler(R_pc) for R_pc in Rs_pc]
    
    def sigmalos2_vector_using_mykernel(self,Rs_pc):
        sigmalos2_scaler = self.sigmalos2_scaler_using_mykernel
        return [sigmalos2_scaler(R_pc) for R_pc in Rs_pc]
    
    def sigmalos2_multi_using_mykernel(self,R_pc):
        p = Pool(self.ncpu)
        R_pc_splited = np.array_split(R_pc,self.ncpu)
        #args = [(Rs,anib,rhos_Msunpc3,rs_pc,a,b,g,re_pc) for Rs in R_pc_v_splited]
        ret = p.map(self.sigmalos2_vector_using_mykernel,R_pc_splited)
        p.close()
        return np.concatenate(ret)

    def sigmalos2(self,R_pc):
        return (self.sigmalos2_scaler(R_pc) if len(R_pc)==1 else self.sigmalos2_vector(R_pc))
    
    def sigmalos2_using_mykernel(self,R_pc):
        return (self.sigmalos2_scaler_uging_mykernel(R_pc) if len(R_pc)==1 else self.sigmalos2_vector_using_mykernel(R_pc))
    
    def sigmalos_interp1d(self,R_pc,dR=1,dRtrunc=10,step_around_Rtrunc=10,step_outer=8,step_center=4,kind="cubic"):
        '''
        inner part has large error, so more refine interp
        '''
        re = self.submodels["stellar_model"].params.re_pc
        Rtrunc = self.submodels["DM_model"].params.R_trunc_pc
        n = len(R_pc)
        R_pc_sorted_ = np.sort(R_pc)
        R_pc_around_Rtrunc = np.sort(R_pc_sorted_[np.argsort(np.abs(R_pc_sorted_-Rtrunc))[:step_around_Rtrunc]])
        Rmin,Rmax = R_pc_sorted_[0],R_pc_sorted_[-1]
        #R_pc_lo_ = np.linspace(Rmin*0.5,Rmin*1.5,int((n*0.2)/step))
        #R_pc_lo_ = R_pc_sorted_[:int(n*0.05)]
        #R_pc_hi_ = np.linspace(R_pc_lo_[-1]+0.1,Rmax*1.1,int(n/step))
        #R_pc_ = np.hstack((R_pc_lo_,R_pc_hi_))
        R_pc_zero = R_pc_sorted_[:step_center]
        R_pc_hi_ = np.logspace(np.log10(R_pc_sorted_[step_center]),np.log10(Rmax)+1e-8,step_outer)
        R_pc_ = np.sort(np.unique(np.concatenate((R_pc_zero,R_pc_around_Rtrunc,R_pc_hi_),axis=None)))
        #R_pc_ = np.linspace(Rmin,Rmax,int(n/step))
        sigmalos2_ = self.sigmalos2_multi_using_mykernel(R_pc_)
        interpd_func = interp1d(R_pc_, sqrt(sigmalos2_),kind=kind)
        return interpd_func(R_pc) # here R_pc is original, so return unsorted results
    
    def sigmalos_interp1d_dequad(self,R_pc,dR=1,dRtrunc=10,step_around_Rtrunc=10,step_outer=20,step_center=4,kind="cubic"):
        '''
        inner part has large error, so more refine interp
        '''
        re = self.submodels["stellar_model"].params.re_pc
        Rtrunc = self.submodels["DM_model"].params.R_trunc_pc
        n = len(R_pc)
        R_pc_sorted_ = np.sort(R_pc)
        R_pc_around_Rtrunc = np.sort(R_pc_sorted_[np.argsort(np.abs(R_pc_sorted_-Rtrunc))[:step_around_Rtrunc]])
        Rmin,Rmax = R_pc_sorted_[0],R_pc_sorted_[-1]
        #R_pc_lo_ = np.linspace(Rmin*0.5,Rmin*1.5,int((n*0.2)/step))
        #R_pc_lo_ = R_pc_sorted_[:int(n*0.05)]
        #R_pc_hi_ = np.linspace(R_pc_lo_[-1]+0.1,Rmax*1.1,int(n/step))
        #R_pc_ = np.hstack((R_pc_lo_,R_pc_hi_))
        R_pc_zero = R_pc_sorted_[:step_center]
        R_pc_hi_ = np.logspace(np.log10(R_pc_sorted_[step_center]),np.log10(Rmax)+1e-8,step_outer)
        R_pc_ = np.sort(np.unique(np.concatenate((R_pc_zero,R_pc_around_Rtrunc,R_pc_hi_),axis=None)))
        #R_pc_ = np.linspace(Rmin,Rmax,int(n/step))
        sigmalos2_ = self.sigmalos2_dequad(R_pc_)
        interpd_func = interp1d(R_pc_, sqrt(sigmalos2_),kind=kind)
        return interpd_func(R_pc) # here R_pc is original, so return unsorted results
    
    def sigmalos2_interp1d(self,R_pc,step_center=10,dR=1,dRtrunc=10,step_around_Rtrunc=10,step_before_Rtrunc=8,step_outer=8,step_inner=16,kind="cubic"):
        '''
        inner part has large error, so more refine interp
        '''
        re = self.submodels["stellar_model"].params.re_pc
        Rtrunc = self.submodels["DM_model"].params.R_trunc_pc
        n = len(R_pc)
        R_pc_sorted_ = np.sort(R_pc)
        R_pc_around_Rtrunc = np.sort(R_pc_sorted_[np.argsort(np.abs(R_pc_sorted_-Rtrunc))[:step_around_Rtrunc]])
        Rmin,Rmax = R_pc_sorted_[0],R_pc_sorted_[-1]
        #R_pc_lo_ = np.linspace(Rmin*0.5,Rmin*1.5,int((n*0.2)/step))
        #R_pc_lo_ = R_pc_sorted_[:int(n*0.05)]
        #R_pc_hi_ = np.linspace(R_pc_lo_[-1]+0.1,Rmax*1.1,int(n/step))
        #R_pc_ = np.hstack((R_pc_lo_,R_pc_hi_))
        R_pc_zero = R_pc_sorted_[:step_center]
        R_pc_lo_ = np.linspace(R_pc_sorted_[step_center],R_pc_sorted_[R_pc_sorted_<re][-1],step_inner)
        R_pc_hi_ = np.linspace(re,Rmax,step_outer)
        R_pc_ = np.sort(np.unique(np.concatenate((R_pc_zero,R_pc_lo_,R_pc_around_Rtrunc,R_pc_hi_),axis=None)))
        #R_pc_ = np.linspace(Rmin,Rmax,int(n/step))
        sigmalos2_ = self.sigmalos2_multi_using_mykernel(R_pc_)
        interpd_func = interp1d(R_pc_, sigmalos2_,kind=kind)
        return interpd_func(R_pc) # here R_pc is original, so return unsorted results
        

class KI17_model:
    def __init__(self,params_KI17_model):
        """
        params_KI17_model: pandas.Series, index = (params_dSph_model,params_FG_model,s)
        """
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dm_model = NFW_model(a=2.78,b=7.78,g=0.675,rhos_Msunpc3=np.power(10,-2.05),rs_pc=np.power(10,3.96),R_trunc_pc=2000)
    mystellar_model = plummer_model(re_pc=221)
    draco_model = dSph_model(submodels_dict={"DM_model":dm_model,"stellar_model":mystellar_model},anib=1-np.power(10,0.13),dist_center_pc=76e3,ra_center_deg=0,de_center_deg=0)
    
    Rs = np.logspace(-1,9,100)
    ss = draco_model.sigmalos2(Rs)
    ss2 = [draco_model.sigmar2(R) for R in Rs]
    ss3 = [draco_model.naive_sigmalos2(R) for R in Rs]
    print(draco_model)
    print(draco_model.integrand_sigmalos2(1,1))
    plt.plot(Rs,np.sqrt(ss))
    plt.plot(Rs,np.sqrt(ss2))
    plt.plot(Rs,np.sqrt(ss3))
    plt.xscale("log")
    plt.yscale("log")
    #plt.ylim(0,40)
    plt.show()
    input("press any key")







