from numpy import sinh,cosh,exp,log,pi,arange,isnan,isinf,float64,float128
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False
BUF_DEBUG = None

def dequad_hinf(func,a,width=5e-3,pN=1000,mN=1000,axis=None,kind="linear",dtype=float64,show_fig=False,show_integrand_array=True):
    '''
    func: func(ndarray_in) = ndarray_out
    axis: define the axis of ndarray_out to use integrate.
    
    Note: If "func" is like func: array(n,) -> array(m,n),
              f(x) * w ~ (m,n) * (n,) ~ (m,n) * (1,n)
          then works well (axis should set as axis=1). 
    '''
    if kind == "linear":
        ts = width*arange(-mN,pN)#.astype(dtype)
        xs = exp(sinh(ts)*pi/2) + a
        ws = width *pi/2 *cosh(ts)*exp(pi/2*sinh(ts))
        fs = func(xs)
        wsfs = ws*fs
        wsfs[isnan(wsfs) | isinf(wsfs)] = 0
        
        print(wsfs[wsfs!=0]) if DEBUG else None
        
        if np.any(isnan(wsfs)):
            raise TypeError("dequad_hinf: wsfs is nan! {}".format(wsfs))
        if show_fig:
            if len(wsfs.shape)>1:
                np.set_printoptions(threshold=20)
                display(wsfs) if show_integrand_array else None
                plt.plot(ts,*wsfs,label=(wsfs).sum(axis=axis))
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
