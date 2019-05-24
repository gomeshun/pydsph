from numpy import sinh,cosh,exp,log,pi,arange,isnan,isinf,float64,float128
from functools import lru_cache
import numpy as np
#import cupy as cp
import matplotlib.pyplot as plt

DEBUG = False
BUF_DEBUG = None

@lru_cache(maxsize = None)
def generate_x_w(a,b,width,mN,pN,xp=np):
    '''
    x = phi(t)
    w = phi'(t)
    '''
    #print("generate_x_w in",a,b,width,mN,pN)
    pi2 = xp.pi/2
    ts = width*xp.arange(-mN,pN)#.astype(dtype)
    if np.all(not np.isinf(a)) and np.all(np.isinf(b)):
        ps = pi2*xp.sinh(ts)
        xs = xp.exp(ps) + a
        ws = width *pi2 *xp.cosh(ts)*xp.exp(ps)    
        
    elif np.all(np.isinf(a)) and np.all(np.isinf(b)):
        ps = pi2*xp.sinh(ts)
        xs = xp.sinh(ps)
        ws = width *pi2 *xp.cosh(ts)*xp.cosh(ps)  
        
    elif np.all(not np.isinf(a)) and np.all(not np.isinf(b)):
        ps = pi2*xp.sinh(ts)
        xs = (b-a)/2*xp.tanh(ps) + (a+b)/2
        ws = width * (b-a)/2 *pi2 *xp.cosh(ts)/(xp.cosh(ps)**2)
        
    return xs,ws

def dequad(func,a,b,width=5e-3,pN=1000,mN=1000,axis=None,kind="linear",dtype=float64,show_fig=False,show_integrand_array=False,ignore_nan=False,xp=np):
    '''
    func: func(ndarray_in) = ndarray_out
    axis: define the axis of ndarray_out to use integrate.
    
    Note: If "func" is like func: array(n,) -> array(m,n),
              f(x) * w ~ (m,n) * (n,) ~ (m,n) * (1,n)
          then works well (axis should set as axis=1). 
    '''
    if kind == "linear":
        #print(a,b,width,mN,pN)
        #print(generate_x_w(a,b,width,mN,pN,xp=np))
        xs,ws = generate_x_w(a,b,width,mN,pN,xp=np)
        wsfs = ws*func(xs)
        isnan = xp.isnan(wsfs)
        isinf = xp.isinf(wsfs)
        wsfs[isnan | isinf] = 0
        
        print(wsfs[wsfs!=0]) if DEBUG else None
        
        if np.any(isnan) and not ignore_nan:
            raise TypeError("dequad_hinf: wsfs is nan! {}".format(wsfs))
        if show_fig:
            if len(wsfs.shape)>1:
                np.set_printoptions(threshold=20)
                display(wsfs) if show_integrand_array else None
                plt.plot(width*xp.arange(-mN,pN),*wsfs,label=(wsfs).sum(axis=axis))
                plt.legend()
            else:
                plt.plot(ts,wsfs)
        return (wsfs).sum(axis=axis)
    elif kind == "log":
        """
        R = \sum_{i=mN}^{pN} w_i * f(x_i)
        """
        ts = width*arange(-mN,pN)
        xs = exp(sinh(ts)*pi/2) + a
        logwsp = log(width *pi/2) + ts + pi/2*sinh(ts)
        logwsm = log(width *pi/2) - ts + pi/2*sinh(ts)
        logfs = func(xs)
        wsfs = (exp(logwsp*logfs)+exp(logwsm*logfs))/2
        wsfs[isnan(wsfs) | isinf(wsfs)] = 0
        return (wsfs).sum(axis=axis)

def dequad_hinf(func,a,width=5e-3,pN=1000,mN=1000,axis=None,kind="linear",dtype=float64,show_fig=False,show_integrand_array=True):
    '''
    func: func(ndarray_in) = ndarray_out
    axis: define the axis of ndarray_out to use integrate.
    
    Note: If "func" is like func: array(n,) -> array(m,n),
              f(x) * w ~ (m,n) * (n,) ~ (m,n) * (1,n)
          then works well (axis should set as axis=1). 
    '''
    if kind == "linear":
        xs,ws = generate_x_w(a,np.inf,width,mN,pN,xp=np)
        wsfs = ws*func(xs)
        isnan = np.isnan(wsfs)
        isinf = np.isinf(wsfs)
        wsfs[isnan | isinf] = 0
        
        print(wsfs[wsfs!=0]) if DEBUG else None
        
        if np.any(isnan):
            raise TypeError("dequad_hinf: wsfs is nan! {}".format(wsfs))
        if show_fig:
            if len(wsfs.shape)>1:
                np.set_printoptions(threshold=20)
                display(wsfs) if show_integrand_array else None
                plt.plot(width*np.arange(-mN,pN),*wsfs,label=(wsfs).sum(axis=axis))
                plt.legend()
            else:
                plt.plot(ts,wsfs)
        return (wsfs).sum(axis=axis)
    elif kind == "log":
        """
        R = \sum_{i=mN}^{pN} w_i * f(x_i)
        """
        ts = width*arange(-mN,pN)
        xs = exp(sinh(ts)*pi/2) + a
        logwsp = log(width *pi/2) + ts + pi/2*sinh(ts)
        logwsm = log(width *pi/2) - ts + pi/2*sinh(ts)
        logfs = func(xs)
        wsfs = (exp(logwsp*logfs)+exp(logwsm*logfs))/2
        wsfs[isnan(wsfs) | isinf(wsfs)] = 0
        return (wsfs).sum(axis=axis)

"""
def dequad_hinf_cupysum(func,a,width=5e-3,pN=1000,mN=1000,axis=None,kind="linear",dtype=float64,show_fig=False,show_integrand_array=True,xp=np):
    '''
    func: func(ndarray_in) = ndarray_out
    axis: define the axis of ndarray_out to use integrate.
    
    Note: If "func" is like func: array(n,) -> array(m,n),
              f(x) * w ~ (m,n) * (n,) ~ (m,n) * (1,n)
          then works well (axis should set as axis=1). 
    '''
    
    stream = cupy.cuda.Stream.null
    start = stream.record()
    
    if kind == "linear":
        xs,ws = generate_x_w(a,width,mN,pN,xp)
        wsfs = cp.asarray(ws)*cp.asarray(func(xs))
        isnan = xp.isnan(wsfs)
        isinf = xp.isinf(wsfs)
        wsfs[isnan | isinf] = 0
        
        print(wsfs[wsfs!=0]) if DEBUG else None
        
        if xp.any(isnan):
            raise TypeError("dequad_hinf: wsfs is nan! {}".format(wsfs))
        if show_fig:
            if len(wsfs.shape)>1:
                np.set_printoptions(threshold=20)
                display(wsfs) if show_integrand_array else None
                plt.plot(ts,*wsfs,label=(wsfs).sum(axis=axis))
                plt.legend()
            else:
                plt.plot(ts,wsfs)

        end = stream.record()
        end.synchronize()
        elapsed = cupy.cuda.get_elapsed_time(start, end)
        #print(elapsed)
                
        return (wsfs).sum(axis=axis)
    elif kind == "log":
        '''
        R = \sum_{i=mN}^{pN} w_i * f(x_i)
        '''
        ts = width*arange(-mN,pN)
        xs = exp(sinh(ts)*pi/2) + a
        logwsp = log(width *pi/2) + ts + pi/2*sinh(ts)
        logwsm = log(width *pi/2) - ts + pi/2*sinh(ts)
        logfs = func(xs)
        wsfs = (exp(logwsp*logfs)+exp(logwsm*logfs))/2
        wsfs[isnan(wsfs) | isinf(wsfs)] = 0
        return (wsfs).sum(axis=axis)
"""
