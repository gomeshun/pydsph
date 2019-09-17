# version 1.0.1
# 
# update of version 1.0.1
# - Use SkyCoord.radial_velocity_err as the err of the input array.
from . import mcgenerator, coord
from .dsph_model import dSph_model, plummer_model, exp2d_model, NFW_model, uniform2d_model
from numpy import array,power,sqrt,log,exp,sin
from scipy.special import logsumexp
from scipy.stats import norm
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

#update likelihood
DEBUG = False

#dSph_property = pd.read_csv("dSph_property.csv",index_col=0)
#prior_lim = []
#draco_prop = dSph_property.loc["Draco"]
#sculptor_prop = dSph_property.loc["Sculptor"]
#RA0 = draco_prop.RAdeg
#DE0 = draco_prop.DEdeg
#DIST = draco_prop.DIST
#err_DIST = draco_prop.err_DIST
#dSph_property

def is_positive(*args):
    return array(args)>0

class modKI17:
    
    ######## initialization ########
    def __init__(self,sc_obsdata,sc_center0,model,paramlims_fname,prior_norm_fname):
        """
        parameters:
            - sc_obsdata:
                SkyCoord of observed data
            - sc_center0: 
                SkyCoord of ad hoc center of dSph
            - paramlims_fname
                Filename of prior configuration. File format: 
                    $     ,prms_min,prms_max
                    $ (p0),(p0_min),(p0_max)
                    $ (p1),(p1_loc),(p1_max)
                    $ ...
            - prior_norm_fname
                Filename of prior configuration. File format: 
                    $   ,loc   ,scale
                    $ p0,p0_loc,p0_scale
                    $ p1,p1_loc,p1_scale
                    $ ...
        """
        self.init_prior(paramlims_fname,prior_norm_fname)
        
        # index for the likelihood, prior and posterior
        #self.param_names = ["re_pc","odds","dra0","dde0",
        #    "log10_rs_pc","log10_rhos_Msunpc3","a","b","g",
        #    "mlog10_1manib",
        #    "vmem","vfg0","vfg1","dvfg0","dvfg1", "sfg0","dist"]
        self.param_names = self.prior_lim.index
        
        #if not np.any(self.param_names == self.prior_lim.index):
        #    raise TypeError("parameter order is not matched!")
        
        self.init_data(sc_obsdata)
        
        self.init_center0(sc_center0)
        
        self.R_RoI = np.max(self.separation_pc(0,0,sc_center0.distance)) # use Rs.max as the RoI
        
        self.beta = 1/np.log(len(self.sc_obsdata))
        
        #print(Rs.describe())
        #print("beta: {}".format(self.beta)) if beta != 1 else None
        
        self.init_model(model)
    
    def init_prior(self,paramlims_fname,prior_norm_fname):
        # set the support of parameter
        self.prior_lim = pd.read_csv(paramlims_fname,index_col=0)
        self.prior_lim_param_names = self.prior_lim.index
        
        # set prior_norm of parameter
        _df_prior_norm = pd.read_csv(prior_norm_fname,index_col=0)
        self.prior_norm = norm(loc=_df_prior_norm["loc"],scale=_df_prior_norm["scale"])
        self.prior_norm_param_names = _df_prior_norm.index
    
    def init_data(self,sc_obsdata):
        self.sc_obsdata = sc_obsdata
        if hasattr(self.sc_obsdata,"radial_velocity_err"):
            print("sc_obsdata has radial_velocity_err. Likelihood function is defined with velocity error:\n{}".format(self.sc_obsdata.radial_velocity_err))
        #self.sc_center0 = sc_center0
    
    def init_model(self,model):
        if model == "Plummer":
            mem = plummer_model(re_pc=200)
        elif model == "exp2d":
            mem = exp2d_model(re_pc=200)
        else: 
            raise TypeError("Undefined stellar model!")
        dm = NFW_model(
            a=2.78,b=7.78,g=0.675,
            rhos_Msunpc3=np.power(10,-2.05),rs_pc=np.power(10,3.96),
            R_trunc_pc=2000
        )
        self.dsph = dSph_model(anib=-0.5,submodels_dict={"stellar_model":mem,"DM_model":dm},show_init=True)
        self.fg = uniform2d_model(Rmax_pc=self.R_RoI,show_init=True)
    
    def init_center0(self,center0):
        self.sc_center0 = center0
        try:
            getattr(self.sc_center0,"distance_err")
        except AttrbuteError as e: 
            print(e)
            raise e
        

    ######## utils ########
    @property
    def n_components(self):
        if "sfg1" in self.prior_lim_param_names:
            return 3
        elif "sfg0" in self.prior_lim_param_names:
            return 2
        else:
            return 1
    
    @staticmethod
    def params_to_series(index,**kwargs):
        """
        return the series whoes elements are "kwargs" ordered by "index".
        it is useful to avoid to mistake the order of the parameters
        """
        sr = pd.Series(index=index)
        for key in kwargs:
            sr[key] = kwargs[key]
            #display(sr)
        display("params_to_ser",sr) if DEBUG else None
        return sr
    
    @staticmethod
    def params_to_array(index,**kwargs):
        """
        return the series whoes elements are "kwargs" ordered by "index".
        it is useful to avoid to mistake the order of the parameters
        """
        ret = modKI17.params_to_series(index=index,**kwargs).values
        display("params_to_arr",ret) if DEBUG else None
        return ret
    
    @staticmethod
    def array_to_series(index,params):
        ret = pd.Series(params,index=index)
        display("array_to_ser",ret) if DEBUG else None
        return ret

    ######## mathematics ########
    def sc_center(self,dra0,dde0,dist):
        return SkyCoord(
            ra=self.sc_center0.ra+dra0*u.deg,
            dec=self.sc_center0.dec+dde0*u.deg,
            distance=dist*u.pc)

    def separation_pc(self,dra0,dde0,dist):
        c = self.sc_center(dra0,dde0,dist)
        return c.distance.value*np.sin(self.sc_obsdata.separation(c).rad)

    def lnprior(self,params):
        if not self.is_parameters_in_domain(params):
            display("out ouf dom:",self.is_parameters_in_domain(params)) if DEBUG else None
            return -np.inf
        else:
            #note that the order of the 
            # _prior_param_names = ["re_pc","odds","dra0","dde0","dist"]
            sr_params = self.array_to_series(index=self.param_names,params=params) # unpack parameters
            #args_prior = self.params_to_array(index=self.prior_norm_param_names,
            #                re_pc=sr_params.re_pc, odds=sr_params.odds,
            #                dra0=sr_params.dra0, dde0=sr_params.dde0,
            #                dist=sr_params.dist
            #               )
            args_prior = array([sr_params[p_name] for p_name in self.prior_norm_param_names]) # extract prior parameters
            logGs = self.prior_norm.logpdf(args_prior)
            #logGs = []
            #logGs.append(norm.logpdf(re,  loc=191,      scale=5.7)     )
            #logGs.append(norm.logpdf(odds,loc=8.794,    scale=0.5107)  )
            #logGs.append(norm.logpdf(dra0,loc=4.212e-3, scale=7.052e-3))
            #logGs.append(norm.logpdf(dde0,loc=-1.991e-3,scale=3.302e-3))
            #logGs.append(norm.logpdf(dist,loc=self.sc_center0.distance,scale=self.sc_center0.distance_err))
            return np.sum(logGs)

    def __call__(self,**args):
        params = self.params_to_array(index=self.param_names,**args)
        ret = self.lnprior(params)
        if ret == -np.inf:
            return ret
        else:
            return ret + np.sum(self.lnlikeli(params)) 
        
    def lnpost(self,params):
        sr_params = self.array_to_series(index=self.param_names,params=params)
        return self.__call__(**sr_params)
    
    def lnposterior_general(self,params):
        lnp = self.lnprior(params)
        if lnp > -np.inf:
            lnl = self.lnlikeli(params) 
            return (self.beta*lnl+lnp, lnl)
        else:
            return (-np.inf, np.nan)
    
    def is_parameters_in_domain(self,params):
        """
        check the parameters are in valid domain.
        note that the params is ordered.
        """
        sr_params = self.array_to_series(index=self.param_names,params=params)
        prms_df = pd.DataFrame({**self.prior_lim,"inprms":sr_params})
        display(prms_df) if DEBUG else None
        is_in_minmax = np.all((prms_df.prms_min < prms_df.inprms) & (prms_df.inprms < prms_df.prms_max))
        #is_ordered = prms_df.inprms.vfg0 < prms_df.inprms.vfg1
        if self.n_components == 3:
            sfg0,sfg1 = prms_df.inprms.sfg0,prms_df.inprms.sfg1
            sfg2 = 1-sfg0-sfg1
            is_ordered = (sfg0 > sfg1) & (sfg1 > sfg2) & (sfg2 > 0)
        elif self.n_components == 2:
            sfg0 = prms_df.inprms.sfg0
            sfg1 = 1-sfg0
            is_ordered = (sfg0 > sfg1) & (sfg1 > 0)
        else: # self.n_components == 1:
            is_ordered = True
        display("is_in_minmax",(prms_df.prms_min < prms_df.inprms) & (prms_df.inprms < prms_df.prms_max),"is_ordered",is_ordered) if DEBUG else None
        return is_in_minmax & is_ordered
    
    def memfg_ratio_at_R(self,mem,odds,R):
        return 1/(1+ 1/(odds * mem.density_2d_normalized_re(R)))
    
    def lnfmems(self,re_pc,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem,dist):

        if DEBUG:
            prms = pd.Series(locals()).drop("self")
            display("args of loglikeli:",prms)
        
        vs=self.sc_obsdata.radial_velocity.value
        mem,dm= self.dsph.submodels["stellar_model"],self.dsph.submodels["DM_model"]
        
        #update parameters
        mem.update({"re_pc":re_pc*(dist/self.sc_center0.distance.value)}) # Note that re_pc given by the stelar fit is just the angle (re_rad), not re_pc !!!
        dm.update({"rs_pc":power(10,log10_rs_pc),"rhos_Msunpc3":power(10,log10_rhos_Msunpc3),"a":a,"b":b,"g":g})
        self.dsph.update({"anib":1-power(10,-mlog10_1manib)})
        ref_R = mem.half_light_radius() # 1.67834699001666*re
        
        Rs = self.separation_pc(dra0,dde0,dist) # here 
        
        sigmalos = self.dsph.sigmalos_dequad_interp1d_downsampled(Rs)
        #sigmalos = self.dsph.sigmalos_dequad(Rs)
        
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if hasattr(self.sc_obsdata,"radial_velocity_err") else 0)
        ret = norm.logpdf(vs,loc=vmem,scale=sqrt(sigmalos**2+vobs_err**2))
        return ret 
    
    def lnlikeli(self,params):
        return self.loglikelihood(*params)
    
    def weighted_distribution_functions(
        self,re_pc,odds,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem,vfg0,vfg1,dvfg0,dvfg1, sfg0, dist
    ):
        prms = pd.Series(locals()).drop("self")
        display("args of loglikeli:",prms) if DEBUG else None
        
        vs=self.sc_obsdata.radial_velocity.value
        mem,dm,fg = self.dsph.submodels["stellar_model"],self.dsph.submodels["DM_model"],self.fg
        
        #update parameters
        mem.update({"re_pc":re_pc*(dist/self.sc_center0.distance.value)}) # Note that re_pc given by the stelar fit is just the angle (re_rad), not re_pc !!!
        dm.update({"rs_pc":power(10,log10_rs_pc),"rhos_Msunpc3":power(10,log10_rhos_Msunpc3),"a":a,"b":b,"g":g})
        self.dsph.update({"anib":1-power(10,-mlog10_1manib)})
        ref_R = mem.half_light_radius() # 1.67834699001666*re
        
        Rs = self.separation_pc(dra0,dde0,dist) # here 
        
        s = 1/(1+ 1/(odds * mem.density_2d_normalized_re(Rs))) # Not s but s(R)
        sigmalos = self.dsph.sigmalos_dequad_interp1d_downsampled(Rs)
        #sigmalos = self.dsph.sigmalos_dequad(Rs)
        sfg1 = 1-sfg0
        
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if hasattr(self.sc_obsdata,"radial_velocity_err") else 0)
        logfmem = norm.logpdf(vs,loc=vmem,scale=sqrt(sigmalos**2+vobs_err**2))
        
        logffg0 = norm.logpdf(vs,loc=vfg0,scale=sqrt(dvfg0**2+vobs_err**2))
        logffg1 = norm.logpdf(vs,loc=vfg1,scale=sqrt(dvfg1**2+vobs_err**2))
        
        logfs = [logfmem,logffg0,logffg1]
        ss = [s,(1-s)*sfg0,(1-s)*sfg1]
                         
        display("sigmalos:{}".format(sigmalos)) if DEBUG else None
        print("fmem:{}".format(fmem)) if DEBUG else None
        print("s*fmem+(1-s)*ffg:{}".format(s*fmem+(1-s)*ffg)) if DEBUG else None
        
        ret = np.array(ss)*np.exp(logfs)
        
        return ret
    
    def membership_prob(self,re_pc,odds,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem,vfg0,vfg1,dvfg0,dvfg1, sfg0, dist):
        params = pd.Series(locals()).drop("self")
        weighted_dist_funcs = self.weighted_distribution_functions(**params)  # (n_distfunc, n_star)
        normalizations = weighted_dist_funcs.sum(axis=0)
        memfg_ratios = (weighted_dist_funcs/normalizations)[0]  # (n_distfunc, n_star)
        return memfg_ratios
    
    def loglikelihood(
        self,re_pc,odds,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem,vfg0,vfg1,dvfg0,dvfg1, sfg0, dist
    ): # R_trunc_pc is fixed 2000 pc but its not affect to the result (truncation radius is not used in the calculation of sigmalos in "dSph_Model")

        prms = pd.Series(locals()).drop("self")
        display("args of loglikeli:",prms) if DEBUG else None
        
        vs=self.sc_obsdata.radial_velocity.value
        mem,dm,fg = self.dsph.submodels["stellar_model"],self.dsph.submodels["DM_model"],self.fg
        
        #update parameters
        mem.update({"re_pc":re_pc*(dist/self.sc_center0.distance.value)}) # Note that re_pc given by the stelar fit is just the angle (re_rad), not re_pc !!!
        dm.update({"rs_pc":power(10,log10_rs_pc),"rhos_Msunpc3":power(10,log10_rhos_Msunpc3),"a":a,"b":b,"g":g})
        self.dsph.update({"anib":1-power(10,-mlog10_1manib)})
        ref_R = mem.half_light_radius() # 1.67834699001666*re
        
        Rs = self.separation_pc(dra0,dde0,dist) # here 
        
        s = 1/(1+ 1/(odds * mem.density_2d_normalized_re(Rs))) # Not s but s(R)
        sigmalos = self.dsph.sigmalos_dequad_interp1d_downsampled(Rs)
        #sigmalos = self.dsph.sigmalos_dequad(Rs)
        sfg1 = 1-sfg0
        
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if hasattr(self.sc_obsdata,"radial_velocity_err") else 0)
        logfmem = norm.logpdf(vs,loc=vmem,scale=sqrt(sigmalos**2+vobs_err**2))
        
        logffg0 = norm.logpdf(vs,loc=vfg0,scale=sqrt(dvfg0**2+vobs_err**2))
        logffg1 = norm.logpdf(vs,loc=vfg1,scale=sqrt(dvfg1**2+vobs_err**2))
        
        logfs = [logfmem,logffg0,logffg1]
        ss = [s,(1-s)*sfg0,(1-s)*sfg1]
                         
        display("sigmalos:{}".format(sigmalos)) if DEBUG else None
        print("fmem:{}".format(fmem)) if DEBUG else None
        print("s*fmem+(1-s)*ffg:{}".format(s*fmem+(1-s)*ffg)) if DEBUG else None
        
        ret = np.sum(logsumexp(a=logfs,b=ss,axis=0)) # note that logsumexp must be used to aviod over/underflow of the likelihood
        
        return ret

    
    
class modKI17_1gauss(modKI17):
    def loglikelihood(self,re_pc,odds,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem,vfg0,dvfg0,dist
    ):
        mem = self.dsph.submodels["stellar_model"]
        mem.update({"re_pc":re_pc*(dist/self.sc_center0.distance.value)}) # TODO: "update" is required. This is very confusing and may cause bugs.
        
        vs = self.sc_obsdata.radial_velocity.value
        vobs_err = (self.sc_obsdata.radial_velocity_err.value if hasattr(self.sc_obsdata,"radial_velocity_err") else 0)
        s = 1/(1+ 1/(odds * mem.density_2d_normalized_re(self.separation_pc(dra0,dde0,dist)))) # In this line, we use "mem.foo", so we should update "mem" in advance.
        
        lnfmems = self.lnfmems(re_pc,dra0,dde0,log10_rs_pc,log10_rhos_Msunpc3,a,b,g,mlog10_1manib,vmem,dist)
        lnffg0s = norm.logpdf(vs,loc=vfg0,scale=sqrt(dvfg0**2+vobs_err**2))
    
        logfs = [lnfmems,lnffg0s]
        ss = [s,(1-s)]
        
        ret = np.sum(logsumexp(a=logfs,b=ss,axis=0)) # note that logsumexp must be used to aviod over/underflow of the likelihood
        
        return ret
    
    
class modKI17_memonly:
    def __init__(self,vs,dRAs,dDEs,dsph_name,paramlims_fname,beta=1):
        """
        vs: velosity data
        dRAs, dDEs: position data
        RA0,DE0: adhoc central position of dSph 
        paramlims_fname: filename of prior configuration
        """
        self.params_name = ["re","dra0","dde0","log10_rs_pc","log10_rhos_Msunpc3","a","b","g","mlog10_1manib","vmem","dist"]
        self.prior_lim = pd.read_csv(paramlims_fname,index_col=0).loc[self.params_name]
        
        #for val in vs,dRAs,dDEs:
        #    if isinstance(val,pd.DataFrame):
        #        val = val.values
        self.vs = vs
        self.dRAs,self.dDEs = dRAs,dDEs
        
        # read property
        self.RA0,self.DE0 = dSph_property.loc[dsph_name][["RAdeg","DEdeg"]]
        self.DIST,self.err_DIST = dSph_property.loc[dsph_name][["DIST","err_DIST"]]

        self.R_RoI = np.max(self.separation_pc(0,0,self.DIST)) # use Rs.max as the RoI
        self.beta = beta
        #print(Rs.describe())
        print("beta: {}".format(self.beta)) if beta != 1 else None
        mem = plummer_model(re_pc=200)
        dm = NFW_model(
            a=2.78,b=7.78,g=0.675,
            rhos_Msunpc3=np.power(10,-2.05),rs_pc=np.power(10,3.96),
            R_trunc_pc=2000
        )
        self.dsph = dsph_model(anib=-0.5,submodels_dict={"stellar_model":mem,"DM_model":dm},show_init=True)
        
        #self.fg = dsph_model.uniform2d_model(Rmax_pc=self.R_RoI,show_init=True)

    def separation_pc(self,dra0,dde0,dist):
        return coord.projected_distance(
            dist=dist,
            ra_center = self.RA0+dra0,
            de_center = self.DE0+dde0,
            ra = self.RA0+self.dRAs,
            de = self.DE0+self.dDEs,
            dtype="deg")

    def lnprior(self,**prms):
        if not self.is_parameters_in_domain(**prms):
            display("out ouf dom:",self.is_parameters_in_domain(**prms)) if DEBUG else None
            return -np.inf
        else:
            #G_re = norm.pdf(re,loc=191,scale=5.7)
            #G_odds = norm.pdf(odds,loc=8.794,scale=0.5107)
            #G_dra0 = norm.pdf(dra0,loc=4.212e-3,scale=7.052e-3)
            #G_dde0 = norm.pdf(dde0,loc=-1.991e-3,scale=3.302e-3)
            G_dist = norm.pdf(prms["dist"],loc=self.DIST,scale=self.err_DIST)
        
            return np.sum(np.log([G_dist,1]))

    def __call__(self,**args):
        ret = self.lnprior(**args)
        if ret == -np.inf:
            return ret
        else:
            return ret + np.sum(self.lnlikelis(**args)) 
        
    def lnprob(self,params):
        prms = {key:val for key,val in zip(self.params_name,params)}
        return self.__call__(**prms)
    
    def is_parameters_in_domain(self,**prms):
        """
        check the parameters are in valid domain.
        """
        prms_df = pd.DataFrame({**self.prior_lim,"inprms":prms})
        display(prms_df) if DEBUG else None
        is_in_minmax = np.all((prms_df.prms_min < prms_df.inprms) & (prms_df.inprms < prms_df.prms_max))
        #is_ordered = prms_df.inprms.vfg0 < prms_df.inprms.vfg1
        display("is_in_minmax",(prms_df.prms_min < prms_df.inprms) & (prms_df.inprms < prms_df.prms_max)) if DEBUG else None
        return is_in_minmax
    """
    def is_parameters_in_domain(self,re,odds,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem,vfg0,vfg1,dvfg0,dvfg1, sfg0):
        is_positive_ = np.all(is_positive(re,odds,dvfg0,dvfg1))
        is_vfg_ordered_ = vfg0 < vfg1
        is_ffg_normalized_ = 0<sfg0<1
        is_in_domain_ = -1<mlog10_1manib<1 and -4<log10_rhos_Msunpc3<4 and 0<log10_rs_pc<5 and 0.5<a<3 and 3<b<10 and 0<g<1.2
        return (is_positive_ and is_vfg_ordered_ and is_ffg_normalized_ and is_in_domain_)
    """
    
    def lnlikelis(
        self,re,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem, dist
    ): # R_trunc_pc is fixed 2000 pc
        prms = pd.Series(locals()).drop("self")
        display("args of loglikeli:",prms) if DEBUG else None
        
        vs=self.vs
        mem,dm = self.dsph.submodels["stellar_model"],self.dsph.submodels["DM_model"]
        
        #update parameters
        mem.update({"re_pc":re*(dist/self.DIST)}) # Note that re_pc given by the stelar fit is just the angle (re_rad), not re_pc !!!
        dm.update({"rs_pc":power(10,log10_rs_pc),"rhos_Msunpc3":power(10,log10_rhos_Msunpc3),"a":a,"b":b,"g":g})
        self.dsph.update({"anib":1-power(10,-mlog10_1manib)})
        ref_R = mem.half_light_radius() # 1.67834699001666*re
        
        Rs = self.separation_pc(dra0,dde0,dist) # here 
        
        #s = 1/(1+ 1/(odds * mem.density_2d_normalized_re(Rs)))
        sigmalos = self.dsph.sigmalos_dequad(Rs)
        if np.any(np.isnan(sigmalos)):
            raise TypeError("sigmalos is nan! {}".format(sigmalos))
        #sigmalos = self.dsph.sigmalos_dequad(Rs)
        #sfg1 = 1-sfg0
        
        fmem = norm.pdf(vs,loc=vmem,scale=sigmalos)
        
        #ffg0 = sfg0 * norm.pdf(vs,loc=vfg0,scale=dvfg0)
        #ffg1 = sfg1 * norm.pdf(vs,loc=vfg1,scale=dvfg1)
        #ffg = ffg0+ffg1
        
        display("sigmalos:{}".format(sigmalos)) if DEBUG else None
        print("fmem:{}".format(fmem)) if DEBUG else None
        #print("s*fmem+(1-s)*ffg:{}".format(s*fmem+(1-s)*ffg)) if DEBUG else None
        #ret = np.log(s*fmem+(1-s)*ffg)
        ret = np.log(fmem)
        
        return self.beta * ret


class modKI17_photometry:
    def __init__(self,data,center0,
                 #area_fname,area_config_fname,
                 #nrow=10000,
                 model="Plummer"):
        '''
        modKI17 likelihood for the photometric observation.
        
        Parameters:
            data       : SkyCoord of observed stars
            center0    : SkyCoord of default center position
            # area_fname : filename of random sampled area
            # area_config_fname: filename of random sampled area
            # n_row      : number of the row you want to use in "area_fname".
        '''
        self.obs_data = data
        self.center0 = center0
        self.R_RoI = self.center0.distance.pc * sin(center0.separation(data).rad.max())
        
        if model=="Plummer":
            self.mem = plummer_model(re_pc=200)
        elif model=="exp2d":
            self.mem = exp2d_model(re_pc=200)
        else: 
            raise TypeError("Undefined stellar model!")
        self.fg = uniform2d_model(Rmax_pc=self.R_RoI)
        
        #df_sample = pd.read_csv(area_fname)[:nrow]
        #self.sc_sample = SkyCoord(ra=df_sample.ra*u.deg,dec=df_sample.dec*u.deg)
        #self.df_sample_config  = pd.read_csv(area_config_fname,squeeze=True,index_col=0,header=None)
        # For the general case, R_RoI is ambiguous
    
    @property
    def beta(self):
        return 1/np.log(len(self.obs_data))
        
    def _separation_pc(self,sc,dra0,dde0):
        center = SkyCoord(ra=self.center0.ra+dra0*u.deg,dec=self.center0.dec+dde0*u.deg,distance=self.center0.distance)
        return self.center0.distance.pc*np.sin(sc.separation(center).rad)
    
    def Rs(self,dra0,dde0):
        return self._separation_pc(sc=self.obs_data,dra0=dra0,dde0=dde0)
    
    def __call__(self,prms):
        if self.lnprior(*prms) == -np.inf:
            return -np.inf
        else:
            return np.sum(self.loglikelis(*prms))
        
    def lnpost(self,prms):
        if self.lnprior(*prms) == -np.inf:
            return -np.inf
        else:
            return self.loglikeli(*prms)
    
    def loglikeli(self,re,odds,dra0,dde0):
        #return np.sum(self.loglikelis(re,odds,dra0,dde0))
        logfmem,logffg,logftot,s = self.logfs(re,odds,dra0,dde0)
        return np.sum(logftot)
    
    def lnprior(self,re,odds,dra0,dde0):
        if re<0 or odds<0 or np.abs(dra0)>1 or np.abs(dde0)>1:
            return -np.inf
        else:
            return 0
    
    def logfs(self,re,odds,dra0,dde0,Rs=None):
        '''
        Return f_mem and f_fg
        '''
        mem,fg = self.mem,self.fg
        mem.update({"re_pc":re})
        ref_R = self.mem.half_light_radius() # 1.67834699001666*re for sersic model
        
        # C1 and C0 are the normalization factor of the member and foreground distribution.
        # For general area case, which is given by
        # \int_Omega d\Omega ~ Omega * (mean of p(R_i), where R_i denotes each samplinig points) 
        #C0 = mem.density_2d_trunc()
        #logC0 = log(C0)
        #C1_0 = 1 / np.mean(mem.density_2d(Rs_rand)) # C1/C0
        #logC1_0 = -(logsumexp(mem.logdensity_2d(Rs_rand)) - log(len(Rs_rand)))
        #C1_0 = exp(logC1_0)
        #C1_0 = 1 / self.mem.cdf_R(self.R_RoI) / Omega
        # C1 = 1 / Omega / self.center0.distance.pc**2 / (np.mean(mem.density_2d(_Rs)))
        #C1 = C1_0 * C0
        #C1 = 1 / self.mem.cdf_R(self.R_RoI)
        #logC1 = logC1_0 + logC0
        #s = C0/(sigmafg*mem.density_2d(self.Rs)*C1 + C0)
        #s = C0/(sigmafg*mem.density_2d(self.Rs)*C1)
        #s = 1/(1+ 1/odds * exp(logC1_0+mem.logdensity_2d(ref_R)))
        s = 1/(1+ 1/odds * mem.density_2d_truncated(ref_R,self.R_RoI)/fg.density_2d_truncated(ref_R,self.R_RoI))
        #print("mem.density2d:",mem.density_2d(ref_R))
        #print("mem.mean_density2d:",mem.mean_density_2d(self.R_RoI))
        
        _Rs = (self.Rs(dra0,dde0) if Rs is None else Rs)
        #ret = np.log(C0) + np.log(s*C1_0*mem.density_2d(Rs)+(1-s))
        logfmem = np.log(mem.density_2d_truncated(_Rs,self.R_RoI))#logC1+mem.logdensity_2d(_Rs)
        logffg  = np.log( fg.density_2d_truncated(_Rs,self.R_RoI))#logC0*np.ones(shape=_Rs.shape)
        logfs = array([logfmem,logffg])
        ss = array([s,(1-s)])
        #print(logfmem.shape,logffg.shape,logfs.shape)
        logftot = logsumexp(a=np.transpose(logfs),b=ss,axis=1) # Note: (stars,components)*(components,), summed along with components axis 
        
        return (logfmem,logffg,logftot,s)
    
    def mem_cdf(R,dra0,dde0):
        R_photo = 0
        pass
    
    def __loglikelis(self,re,odds,dra0,dde0,Rs=None): # old fashion. do not use!
        mem = self.mem
        mem.update({"re_pc":re})
        ref_R = self.mem.half_light_radius() # 1.67834699001666*re for sersic model
        
        # C1 and C0 are the normalization factor of the member and foreground distribution.
        # For general area case, which is given by
        # \int_Omega d\Omega ~ Omega * (mean of p(R_i), where R_i denotes each samplinig points) 
        Omega = self.df_sample_config.loc["area"]
        Rs_rand = self._Rs(sc=self.sc_sample,dra0=dra0,dde0=dde0)
        C0 = 1 / Omega / self.center0.distance.pc**2
        C1_0 = 1 / np.mean(mem.density_2d(Rs_rand)) # C1/C0
        # C1 = 1 / Omega / self.center0.distance.pc**2 / (np.mean(mem.density_2d(_Rs)))
        C1 = C1_0 * C0
        #### C1 = 1 / self.center0.distance.pc**2 / (mem.cdf_R())
        #s = C0/(sigmafg*mem.density_2d(self.Rs)*C1 + C0)
        #s = C0/(sigmafg*mem.density_2d(self.Rs)*C1)
        s = 1/(1+ 1/odds * C1_0*mem.density_2d(ref_R))
        #print("mem.density2d:",mem.density_2d(ref_R))
        #print("mem.mean_density2d:",mem.mean_density_2d(self.R_RoI))
        
        _Rs = (self.Rs(dra0,dde0) if Rs==None else Rs)
        ret = np.log(C0) + np.log(s*C1_0*mem.density_2d(_Rs)+(1-s))

        return ret

    def lnposterior_general(self,params):
        lnp = self.lnprior(*params)
        if lnp > -np.inf:
            lnl = self.loglikeli(*params) 
            #print((self.beta*lnl+lnp)/lnl)
            return (self.beta*lnl+lnp, lnl)
        else:
            return (-np.inf, np.nan)
