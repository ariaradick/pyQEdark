"""
Set of relevant velocity distribution functions for dm-electron scattering
All of these return a normalized function ready to be integrated, and only
a function of velocity

author: Aria
date: 2/8/20
"""

import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pyQEdark.constants import ckms, ccms, c_light

def f_SHM(*args, vp=2):
    """
    Standard Halo Model, velocity follows a Maxwell-Boltzmann distribution
    _params = [v0, vesc]
    """
    if len(args) == 2:
        V, _params = args
        return_func = False
    elif len(args) == 1:
        _params = args[0]
        return_func = True
    else:
        raise TypeError('Wrong number of arguments!')

    v0 = _params[0]
    vesc = _params[1]

    KK_ = v0**3 * np.pi * ( np.sqrt(np.pi)*erf(vesc/v0) - 2 * (vesc/v0) * \
            np.exp(-vesc**2/v0**2) )

    def shm_(v):
        return v**vp*1/KK_*np.exp(-v**2/v0**2)*np.heaviside(vesc-v, .5)

    if return_func:
        return shm_
    else:
        return shm_(V)

def f_DPL(_params, vp=2):
    """
    Double Power Law Profile, 1.5 <= k <= 3.5 found to give best fit to N-body
    simulations. Ref [7] in Aria's notes
    _params = [v0, vesc, k]
    """
    v0 = _params[0]
    vesc = _params[1]
    k = _params[2]

    def dpl_int(v):
        if v < vesc:
            return v**2*( np.exp( (vesc**2 - v**2) / (k * v0**2) ) - 1 )**k
        else:
            return 0

    KK_ = 4*np.pi*quad(dpl_int, 0, vesc)[0]

    def dpl_(v):
        if v < vesc:
            return v**vp*1/KK_*(np.exp((vesc**2-v**2)/(k*v0**2))-1)**k
        else:
            return 0

    return np.vectorize(dpl_)

def f_Tsa(*args, vp=2):
    """
    Tsallis Model, q = .773, v0 = 267.2 km/s, and vesc = 560.8 km/s
    give best fits from arXiv:0909.2028
    """
    if len(args) == 2:
        V, _params = args
        return_func = False
    elif len(args) == 1:
        _params = args[0]
        return_func = True
    else:
        raise TypeError('Wrong number of arguments!')

    v0   = _params[0]
    vesc = _params[1]
    q = 1 - v0**2 / vesc**2

    # KK_ = 4/3*np.pi*vesc**3*hyp2f1(3/2, 1/(q-1), 5/2, (1-q)*vesc**2/v0**2)

    def tsa_inttest(vx):
        if q == 1:
            if vx <= vesc:
                return vx**2*np.exp(-vx**2/v0**2)
            else:
                return 0
        else:
            if vx <= vesc:
                return vx**2*(1-(1-q)*vx**2/v0**2)**(1/(1-q))
            else:
                return 0

    KK_ = 4 * np.pi * quad(tsa_inttest, 0, vesc)[0]

    def tsa_(v):
        return v**vp * 1/KK_ * ( 1 - (1-q)*v**2/v0**2 )**(1/(1-q))

    tsa_fn = lambda v: np.piecewise(v, [v<=vesc, v>vesc], [tsa_, 0])

    if return_func:
        return tsa_fn
    else:
        return tsa_fn(V)

def f_MSW(*args, vp=2):
    """
    Strigari-Weschler Empirical model from arXiv:1210.2721
    _params = [v0, vesc, p]
    """
    if len(args) == 2:
        V, _params = args
        return_func = False
    elif len(args) == 1:
        _params = args[0]
        return_func = True
    else:
        raise TypeError('Wrong number of arguments!')

    v0 = _params[0]
    vesc = _params[1]
    p = _params[2]

    # prefactor = np.pi**(3/2) * 2**(3/2+p) * vesc**(2+2*p) * gamma(1+p)
    # t = np.zeros(5)
    # t[0] = -vesc**2/((2+3*p+p**2)*v0)/(2**(3/2+p) * np.sqrt(np.pi) * gamma(1+p))
    # t[1] = v0**(3/2+p) / vesc**(1/2+p) * iv(3/2+p,vesc/v0)
    # t[2] = v0**(1/2+p) / vesc**(-1/2+p) * iv(5/2+p, vesc/v0)
    # t[3] = - v0**(3/2+p) / vesc**(1/2+p) * modstruve(3/2+p, vesc/v0)
    # t[4] = - v0**(1/2+p) / vesc**(-1/2+p) * modstruve(5/2+p, vesc/v0)
    # KK_ = prefactor * np.sum(t)

    def msw_inttest(vx):
        if vx <= vesc:
            return vx**2*np.exp(-vx/v0)*(vesc**2-vx**2)**p
        else:
            return 0

    KK_ = 4 * np.pi * quad(msw_inttest, 0, vesc)[0]

    def swm_(v):
        return v**vp*1/KK_*np.exp(-v/v0)*(vesc**2-v**2)**p * \
               np.heaviside(vesc-v, .5)

    if return_func:
        return swm_
    else:
        return swm_(V)

def f_FromFile( *args, vp=2, in_unit='kms', out_unit='kms', delimiter=',',
                interp1d_args={} ):
    """
    Returns either an eta function eta(Vmin) or an array of eta values eta(Vmin)
    depending on args.
    args = ( Vmin, path_to_data )
    with Vmin being optional. If Vmin is not passed, this will return a fn.
    """

    corr_dict = { 'kms' : ckms,
                  'cms' : ccms,
                  'ms'  : c_light,
                  'nat' : 1 }

    if len(args) == 1:
        return_func = True
        path_to_data = args[0]

    elif len(args) == 2:
        return_func = False
        V, path_to_data = args

    else:
        raise TypeError('Wrong number of arguments!')

    data = np.loadtxt(path_to_data, delimiter=delimiter)

    # change the data into the correct units:
    data[:,0] *= corr_dict[out_unit] / corr_dict[in_unit]
    # data[:,1] *= corr_dict[in_unit] / corr_dict[out_unit]

    v2data = np.zeros_like(data)
    v2data[:,:] = data[:,:]
    v2data[:,1] *= data[:,0]**2

    arb_fn = interp1d( data[:,0], data[:,1], bounds_error=False,
                       fill_value=0., **interp1d_args )

    def int_fn(vx):
        return vx**2 * arb_fn(vx)
    KK_ = 4*np.pi*trapezoid(v2data[:,1], x=v2data[:,0])

    def norm_fn(v_):
        return v_**vp * arb_fn(v_) / KK_

    if return_func:
        return norm_fn
    else:
        return norm_fn(V)
