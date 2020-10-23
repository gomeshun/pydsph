import MCgenerator, dSph_model, coord
from numpy import power,sqrt
from scipy.stats import norm
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

#update likelihood
DEBUG = False

dSph_property = pd.read_csv("dSph_property.csv",index_col=0)
prior_lim = []
draco_prop = dSph_property.loc["Draco"]
sculptor_prop = dSph_property.loc["Sculptor"]
#RA0 = draco_prop.RAdeg
#DE0 = draco_prop.DEdeg
#DIST = draco_prop.DIST
#err_DIST = draco_prop.err_DIST
#dSph_property

def is_positive(*args):
    return np.array(args)>0

class modKI17:
    def __init__(self,sc_obsdata,dsph_name,prior_fname,beta=1):
        """
        sc_obsdata: SkyCoord of observed data
        sc_center: SkyCoord of ad hoc center of dSph
        prior_fname: filename of prior configuration
        """
        self.prior_lim = pd.read_csv(prior_fname,index_col=0)
        
        #for val in vs,dRAs,dDEs:
        #    if isinstance(val,pd.DataFrame):
        #        val = val.values
        self.sc_obsdata = sc_obsdata
        #self.sc_center0 = sc_center0
        
        # read property
        RA0,DE0 = dSph_property.loc[dsph_name][["RAdeg","DEdeg"]]
        DIST,err_DIST = dSph_property.loc[dsph_name][["DIST","err_DIST"]]
        self.sc_center0 = SkyCoord(ra=RA0*u.deg,dec=DE0*u.deg,distance=DIST)
        self.sc_center0.distance_err = err_DIST

        self.RoI_R = np.max(self.Rs(0,0)) # use Rs.max as the RoI
        self.beta = beta
        #print(Rs.describe())
        print("beta: {}".format(self.beta)) if beta != 1 else None
        mem = dSph_model.plummer_model(re_pc=200)
        dm = dSph_model.NFW_model(
            a=2.78,b=7.78,g=0.675,
            rhos_Msunpc3=np.power(10,-2.05),rs_pc=np.power(10,3.96),
            R_trunc_pc=2000
        )
        self.dsph = dSph_model.dSph_model(anib=-0.5,submodels_dict={"stellar_model":mem,"DM_model":dm},show_init=True)
        
        self.fg = dSph_model.uniform2d_model(Rmax_pc=self.RoI_R,show_init=True)
    
    def sc_center(self,dra0,dde0):
        return SkyCoord(
            ra=self.sc_center0.ra+dra0*u.deg,
            dec=self.sc_center0.dec+dde0*u.deg)
        
    def Rs(self,dra0,dde0):
        return self.sc_center0.distance.value*np.sin(self.sc_obsdata.separation(self.sc_center(dra0,dde0)).rad)

    def lnprior(self,re,odds,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem,vfg0,vfg1,dvfg0,dvfg1, sfg0,dist):
        if not self.is_parameters_in_domain(re=re, odds=odds,
            dra0=dra0,dde0=dde0,
            log10_rs_pc=log10_rs_pc,log10_rhos_Msunpc3=log10_rhos_Msunpc3,
            a=a,b=b,g=g,mlog10_1manib=mlog10_1manib,
            vmem=vmem,vfg0=vfg0,vfg1=vfg1,dvfg0=dvfg0,dvfg1=dvfg1, sfg0=sfg0, 
            dist=dist):
            display("out ouf dom:",self.is_parameters_in_domain(re=re, odds=odds,
                dra0=dra0,dde0=dde0,
                log10_rs_pc=log10_rs_pc,log10_rhos_Msunpc3=log10_rhos_Msunpc3,
                a=a,b=b,g=g,mlog10_1manib=mlog10_1manib,
                vmem=vmem,vfg0=vfg0,vfg1=vfg1,dvfg0=dvfg0,dvfg1=dvfg1, sfg0=sfg0, 
                dist=dist)) if DEBUG else None
            return -np.inf
        else:
            logGs = []
            logGs.append(norm.logpdf(re,  loc=191,      scale=5.7)     )
            logGs.append(norm.logpdf(odds,loc=8.794,    scale=0.5107)  )
            logGs.append(norm.logpdf(dra0,loc=4.212e-3, scale=7.052e-3))
            logGs.append(norm.logpdf(dde0,loc=-1.991e-3,scale=3.302e-3))
            logGs.append(norm.logpdf(dist,loc=self.sc_center0.distance,scale=self.sc_center0.distance_err)
        
            return np.sum(np.logGs)

    def __call__(self,**args):
        ret = self.lnprior(**args)
        if ret == -np.inf:
            return ret
        else:
            return ret + np.sum(self.lnlikelis(**args)) 
        
    def lnprob(self,params):
        re,odds,dra0,dde0,log10_rs_pc,log10_rhos_Msunpc3,a,b,g,mlog10_1manib,vmem,vfg0,vfg1,dvfg0,dvfg1, sfg0, dist = params
        return self.__call__(
            re=re, odds=odds,
            dra0=dra0,dde0=dde0,
            log10_rs_pc=log10_rs_pc,log10_rhos_Msunpc3=log10_rhos_Msunpc3,
            a=a,b=b,g=g,mlog10_1manib=mlog10_1manib,
            vmem=vmem,vfg0=vfg0,vfg1=vfg1,dvfg0=dvfg0,dvfg1=dvfg1, sfg0=sfg0, 
            dist=dist)
    
    def is_parameters_in_domain(self,**prms):
        """
        check the parameters are in valid domain.
        """
        prms_df = pd.DataFrame({**self.prior_lim,"inprms":prms})
        display(prms_df) if DEBUG else None
        is_in_minmax = np.all((prms_df.prms_min < prms_df.inprms) & (prms_df.inprms < prms_df.prms_max))
        #is_ordered = prms_df.inprms.vfg0 < prms_df.inprms.vfg1
        is_ordered = prms_df.inprms.sfg0 > 0.5
        display("is_in_minmax",(prms_df.prms_min < prms_df.inprms) & (prms_df.inprms < prms_df.prms_max),"is_ordered",is_ordered) if DEBUG else None
        return is_in_minmax & is_ordered
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
        self,re,odds,dra0,dde0,
            log10_rs_pc,log10_rhos_Msunpc3,a,b,g,
            mlog10_1manib,
            vmem,vfg0,vfg1,dvfg0,dvfg1, sfg0, dist
    ): # R_trunc_pc is fixed 2000 pc
        raise TypeError("This function is not use logsumexp")
        prms = pd.Series(locals()).drop("self")
        display("args of loglikeli:",prms) if DEBUG else None
        
        vs=self.sc_obsdata.radial_velocity.value
        mem,dm,fg = self.dsph.submodels["stellar_model"],self.dsph.submodels["DM_model"],self.fg
        
        #update parameters
        mem.update({"re_pc":re*(dist/self.sc_center0.distance.value)}) # Note that re_pc given by the stelar fit is just the angle (re_rad), not re_pc !!!
        dm.update({"rs_pc":power(10,log10_rs_pc),"rhos_Msunpc3":power(10,log10_rhos_Msunpc3),"a":a,"b":b,"g":g})
        self.dsph.update({"anib":1-power(10,-mlog10_1manib)})
        ref_R = mem.half_light_radius() # 1.67834699001666*re
        
        Rs = self.Rs(dra0,dde0) # here 
        
        s = 1/(1+ 1/(odds * mem.density_2d_normalized_re(Rs)))
        sigmalos = self.dsph.sigmalos_dequad_interp1d_downsampled(Rs)
        #sigmalos = self.dsph.sigmalos_dequad(Rs)
        sfg1 = 1-sfg0
        
        fmem = norm.pdf(vs,loc=vmem,scale=sigmalos)
        
        ffg0 = sfg0 * norm.pdf(vs,loc=vfg0,scale=dvfg0)
        ffg1 = sfg1 * norm.pdf(vs,loc=vfg1,scale=dvfg1)
        ffg = ffg0+ffg1
        
        display("sigmalos:{}".format(sigmalos)) if DEBUG else None
        print("fmem:{}".format(fmem)) if DEBUG else None
        print("s*fmem+(1-s)*ffg:{}".format(s*fmem+(1-s)*ffg)) if DEBUG else None
        ret = np.log(s*fmem+(1-s)*ffg)
        
        return self.beta * ret

class modKI17_memonly:
    def __init__(self,vs,dRAs,dDEs,dsph_name,prior_fname,beta=1):
        """
        vs: velosity data
        dRAs, dDEs: position data
        RA0,DE0: adhoc central position of dSph 
        prior_fname: filename of prior configuration
        """
        self.params_name = ["re","dra0","dde0","log10_rs_pc","log10_rhos_Msunpc3","a","b","g","mlog10_1manib","vmem","dist"]
        self.prior_lim = pd.read_csv(prior_fname,index_col=0).loc[self.params_name]
        
        #for val in vs,dRAs,dDEs:
        #    if isinstance(val,pd.DataFrame):
        #        val = val.values
        self.vs = vs
        self.dRAs,self.dDEs = dRAs,dDEs
        
        # read property
        self.RA0,self.DE0 = dSph_property.loc[dsph_name][["RAdeg","DEdeg"]]
        self.DIST,self.err_DIST = dSph_property.loc[dsph_name][["DIST","err_DIST"]]

        self.RoI_R = np.max(self.Rs(0,0,self.DIST)) # use Rs.max as the RoI
        self.beta = beta
        #print(Rs.describe())
        print("beta: {}".format(self.beta)) if beta != 1 else None
        mem = dSph_model.plummer_model(re_pc=200)
        dm = dSph_model.NFW_model(
            a=2.78,b=7.78,g=0.675,
            rhos_Msunpc3=np.power(10,-2.05),rs_pc=np.power(10,3.96),
            R_trunc_pc=2000
        )
        self.dsph = dSph_model.dSph_model(anib=-0.5,submodels_dict={"stellar_model":mem,"DM_model":dm},show_init=True)
        
        #self.fg = dSph_model.uniform2d_model(Rmax_pc=self.RoI_R,show_init=True)

    def Rs(self,dra0,dde0,dist):
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
        
        Rs = self.Rs(dra0,dde0,dist) # here 
        
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
 