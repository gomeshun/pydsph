import pandas as pd
import numpy as np
import emcee 
from astropy.io import ascii
from astropy.coordinates import SkyCoord,Distance
import astropy.units as u
from . import modki17
from tqdm import tqdm as tqdm

class dSphData:
    '''
    data class of the observed data of dSph.
    
    attributes:
        - 
    
    '''
    idx_photo_default = dict(index_col=0,cut="in_cmdcut",ra_idx="raMean",dec_idx="decMean")
    idx_spec_default  = dict(index_col=0,cut="in_cmdcut",ra_idx="ra",    dec_idx="dec"    )
    
    def __init__(self,dsph_name,
                 fname_photo=None,fname_spec=None,
                 idx_photo=None,idx_spec=None):
        '''
        initialize a sampler instance.
        
        input:
            - dsph_name(str)   : dwarf spheroidal name listed in "Nearbygalaxies.dat".
            - fname_photo(str) : file name of photometry data
            - fname_spec(str)  : file name of spectroscopy data
            - idx_photo,idx_spec(dict) : dictionary of argument passed to "self.load_file".
        '''
        idx_photo = dSphData.idx_photo_default if idx_photo is None else idx_photo
        idx_spec = dSphData.idx_spec_default if idx_spec is None else idx_spec
        
        self.load_dsph_property(dsph_name)
        if not fname_photo is None:
            self.load_csv("photo",fname_photo,**idx_photo)
        if not  fname_spec is None:
            self.load_csv("spec", fname_spec, **idx_spec)
        
    def load_dsph_property(self,dsph_name,ra=None,dec=None,distance=None):
        dsph_prop = ascii.read("../data/NearbyGalaxies.dat").to_pandas().set_index("GalaxyName").loc[dsph_name]
        self.dsph_prop = dsph_prop
        _ra  = "{}h{}m{}s".format(dsph_prop.RAh,dsph_prop.RAm,dsph_prop.RAs) if (ra  is None) else ra
        _dec = "{}d{}m{}s".format(dsph_prop.DEd,dsph_prop.DEm,dsph_prop.DEs) if (dec is None) else dec
        _distance = Distance(distmod=dsph_prop["(m-M)o"]).pc if (distance is None) else distance
        self.dsph_prop_sc = SkyCoord(ra=_ra,dec=_dec,distance=_distance*u.pc)
    
    def load_csv(self,name,fname,index_col=None,cut=None,ra_idx="ra",dec_idx="dec"):
        df_tot = pd.read_csv(fname,index_col=index_col)
        _cut = [True]*len(df_tot) if (cut is None) else cut
        df = df_tot[df_tot[cut]].reset_index(drop=True)
        sc = SkyCoord(ra = df[ra_idx].values*u.deg, dec = df[dec_idx].values*u.deg)
        setattr(self,"df_"+name,df)
        setattr(self,"sc_"+name,sc)
        setattr(self,"sep_"+name,self.dsph_prop_sc.separation(sc))
        
class Sampler:
    def __init__(self,dsphdata,model,param_names,init_low,init_high):
        self.dsphdata = dsphdata
        self.photometry_model = modki17.modKI17_photometry(dsphdata.sc_photo,dsphdata.dsph_prop_sc)
        
        if not (len(param_names)==len(init_low) and len(param_names)==len(init_high)):
            raise TypeError("invalid param_names,low or high")
        
        self.param_names = param_names
        self.init_low = init_low
        self.init_high = init_high
        self.n_param = len(param_names)
        
    def init_sampler(self,nwalkers=16):
        self.pos0 = np.random.uniform(low=self.init_low,high=self.init_high,size=(nwalkers,self.n_param))
        self.sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers, dim=self.n_param,
            lnpostfn=self.photometry_model.lnpost, threads=16)
        return self.pos0
    
    def prepare_csv(self,fname):
        df = pd.DataFrame(columns=[*(self.param_names),"lnpost"])
        df.to_csv(fname,index=None,header=True)
        
    def to_csv(self,fname,mode="w",header=True):
        with open(fname,mode) as f:
            df = pd.DataFrame(
                np.concatenate([self.sampler.flatchain,self.sampler.flatlnprobability[:,np.newaxis]],axis=1),
                columns=[*(self.param_names),"lnpost"] )
            df.to_csv(f,index=None,header=header)
    
    def sample(self,p0,n_run = 5000,fname=None,mode="w",header=True,reset=False,desc=None):
        for pos, prob, state in tqdm(self.sampler.sample(p0,iterations=n_run),total=n_run,desc=desc):
            pass
        else:
            if not (fname is None):
                self.to_csv(fname,mode,header)
            if reset:
                self.sampler.reset()
            return pos
    
    def run_mcmc(self,output_fname,n_burnin,n_run=5000,to_csv=True,mode="w",header=True):
        self.init_sampler()
        self.prepare_csv(output_fname)
        pos = self.sample(self.pos0,n_burnin,desc="Burn in: ")
        self.sampler.reset()
        
        pos = self.sample(self.pos0,n_run,output_fname,desc="Sampling: ")
        if to_csv:
            self.to_csv(output_fname,mode,header)
        return pos
    
    def run_mcmc_epoch(self,output_fname,n_burnin=1000,n_run=500,n_epoch=12):
        self.init_sampler()
        self.prepare_csv(output_fname)
        pos = self.sample(self.pos0,n_burnin,desc="Burn in: ")
        self.sampler.reset()

        # create file
        self.prepare_csv(output_fname)
        # run
        for i in range(n_epoch):
            pos = self.sample(pos,n_run,output_fname,mode="a",header=False,reset=True,desc="{} th Sampling: ".format(i))