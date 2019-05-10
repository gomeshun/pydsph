import pandas as pd
import numpy as np
import emcee 
from astropy.io import ascii
from astropy.coordinates import SkyCoord,Distance
import astropy.units as u
from . import modki17
from tqdm import tqdm as tqdm

class MyEnsembleSampler(emcee.EnsembleSampler):
    @property
    def flatblobs(self):
        return np.array(self.blobs).T.flatten()

class Sampler:
    def __init__(self,mode,dsphdata,dsphprop,model,param_names,init_low,init_high,paramlims_fname=None,prior_norm_fname=None):
        
        if mode == "photo":
            self.model = modki17.modKI17_photometry(
                dsphdata.to_sc("photo"),dsphprop.to_sc(),
                model=model
            )
        elif mode == "spec":
            self.model = modki17.modKI17(
                dsphdata.to_sc("spec"),dsphprop.to_sc(),
                paramlims_fname,prior_norm_fname
            )
        
        if not (len(param_names)==len(init_low) and len(param_names)==len(init_high)):
            raise TypeError("invalid param_names,low or high")
        
        self.param_names = param_names
        self.init_low = init_low
        self.init_high = init_high
        self.n_param = len(param_names)
        
    def init_sampler(self,nwalkers=16):
        '''
        Initialize the sampler for the parameter estimation.
        '''
        self.pos0 = np.random.uniform(low=self.init_low,high=self.init_high,size=(nwalkers,self.n_param))
        self.sampler = MyEnsembleSampler(
            nwalkers=nwalkers, dim=self.n_param,
            lnpostfn=self.model.lnpost, threads=16)
        return self.pos0
    
    def wbic_init_sampler(self,nwalkers=16):
        '''
        Initialize the sampler for the wbic estimation.
        '''
        self.wbic_pos0 = np.random.uniform(low=self.init_low,high=self.init_high,size=(nwalkers,self.n_param))
        self.wbic_sampler = MyEnsembleSampler(
            nwalkers=nwalkers, dim=self.n_param,
            lnpostfn=self.model.lnposterior_general, threads=16)
        return self.wbic_pos0
    
    def prepare_csv(self,fname):
        df = pd.DataFrame(columns=[*(self.param_names),"lnpost"])
        df.to_csv(fname,index=None,header=True)
        
    def to_csv(self,fname,mode="w",header=True):
        with open(fname,mode) as f:
            df = pd.DataFrame(
                np.concatenate([self.sampler.flatchain,self.sampler.flatlnprobability[:,np.newaxis]],axis=1),
                columns=[*(self.param_names),"lnpost"] )
            df.to_csv(f,index=None,header=header)
            
    def wbic_prepare_csv(self,fname):
        df = pd.DataFrame(columns=[*(self.param_names),"lngenpost","lnlike"])
        df.to_csv(fname,index=None,header=True)
        
    def wbic_to_csv(self,fname,mode="w",header=True):
        with open(fname,mode) as f:
            df = pd.DataFrame(
                np.concatenate([self.wbic_sampler.flatchain,self.wbic_sampler.flatlnprobability[:,np.newaxis],self.wbic_sampler.flatblobs[:,np.newaxis]],axis=1),
                columns=[*(self.param_names),"lngenpost","lnlike"] )
            df.to_csv(f,index=None,header=header)
    
    ######## parameter sampler ########
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
        
        pos = self.sample(pos,n_run,output_fname,desc="Sampling: ")
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
            
    ######## WBIC sampler ########
    def wbic_sample(self,p0,n_run = 5000,fname=None,mode="w",header=True,reset=False,desc=None):
        for pos, prob, state, blobs in tqdm(self.wbic_sampler.sample(p0,iterations=n_run),total=n_run,desc=desc):
            pass
        else:
            if not (fname is None):
                self.wbic_to_csv(fname,mode,header)
            if reset:
                self.wbic_sampler.reset()
            return pos
    
    def wbic_run_mcmc(self,output_fname,n_burnin,n_run=5000,to_csv=True,mode="w",header=True):
        self.wbic_init_sampler()
        self.wbic_prepare_csv(output_fname)
        pos = self.wbic_sample(self.wbic_pos0,n_burnin,desc="WBIC Burn in: ")
        self.wbic_sampler.reset()
        
        pos = self.wbic_sample(pos,n_run,output_fname,desc="WBIC Sampling: ")
        if to_csv:
            self.wbic_to_csv(output_fname,mode,header)
        return pos
    
    def wbic_run_mcmc_epoch(self,output_fname,n_burnin=1000,n_run=500,n_epoch=12):
        self.wbic_init_sampler()
        self.wbic_prepare_csv(output_fname)
        pos = self.wbic_sample(self.wbic_pos0,n_burnin,desc="WBIC Burn in: ")
        self.wbic_sampler.reset()

        # create file
        self.wbic_prepare_csv(output_fname)
        # run
        for i in range(n_epoch):
            pos = self.wbic_sample(pos,n_run,output_fname,mode="a",header=False,reset=True,desc="{} th WBIC Sampling: ".format(i))
