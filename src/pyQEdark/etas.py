"""
Integrated velocity distributions eta (<1/v> weighted with given vdfs), taking
into account the velocity of earth

Integration method from Tien-Tien Yu

author: Aria Radick
date created: 5/16/20
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.integrate import nquad, quad, trapezoid
from pyQEdark.constants import ckms, ccms, c_light

def etaSHM(*args):

    """
    Standard Halo Model with sharp cutoff.
    Fiducial values are v0=220 km/s, vE=232 km/s, vesc= 544 km/s
    params = [v0, vE, vesc], input parameters must be scalars
    """

    if len(args) == 1:
        return_func = True
        _params = args[0]

    elif len(args) == 2:
        return_func = False
        vmin, _params = args

    else:
        raise TypeError('Wrong number of arguments!')

    v0 = _params[0]
    vE = _params[1]
    vesc = _params[2]

    KK=v0**3*(-2.0*np.exp(-vesc**2/v0**2)*np.pi*vesc/v0+np.pi**1.5*erf(vesc/v0))

    def eta_a(_vmin):
        a = v0**2 * np.pi / (2 * vE * KK)
        nn1_ = -4*np.exp(-vesc**2/v0**2)*vE
        nn2_ = np.sqrt(np.pi)*v0*(erf((_vmin+vE)/v0) - erf((_vmin-vE)/v0))
        return a * (nn1_ + nn2_)
    def eta_b(_vmin):
        a = v0**2 * np.pi / (2 * vE * KK)
        nn1_ = -2*np.exp(-vesc**2/v0**2)*(vE+vesc - _vmin)
        nn2_ = np.sqrt(np.pi)*v0*(erf(vesc/v0)-erf((_vmin-vE)/v0))
        return a * (nn1_ + nn2_)

    eta = lambda vmin: np.piecewise( vmin,
                                     [ vmin <= (vesc-vE),
                                       np.logical_and(vmin > (vesc-vE),
                                                      vmin <= (vesc+vE)),
                                       vmin > (vesc+vE) ],
                                     [ eta_a, eta_b, 0 ] )

    if return_func:
        return eta
    else:
        return eta(vmin)

def etaTsa(*args):

    """
    Tsallis Model, q = .773, v0 = 267.2 km/s, and vesc = 560.8 km/s
    give best fits from arXiv:0909.2028.
    params = [v0, vE, q], input parameters must be scalars
    """

    if len(args) == 1:
        return_func = True
        _params = args[0]

    elif len(args) == 2:
        return_func = False
        vmin, _params = args

    else:
        raise TypeError('Wrong number of arguments!')

    v0   = _params[0]
    vE   = _params[1]
    vesc = _params[2]
    q = 1 - v0**2 / vesc**2

    if q == 1:
        tsa_ = lambda vx: vx**2*np.exp(-vx**2/v0**2)
        func = lambda vx2: np.exp(-vx2/v0**2)

    else:
        tsa_ = lambda vx: vx**2*(1-(1-q)*vx**2/v0**2)**(1/(1-q))
        func = lambda vx2: (1-(1-q)*vx2/v0**2)**(1/(1-q))

    tsa_inttest = lambda vx: np.piecewise(vx, [vx <= vesc, vx > vesc],
                                          [tsa_, 0])

    K_=4*np.pi*quad(tsa_inttest, 0, vesc)[0]

    def eta_a(_vmin):
        def bounds_cosq():
            return [-1,1]
        def bounds_vX(cosq):
            return [_vmin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]
        def intfunc(vx,cosq):
            return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
        return nquad(intfunc, [bounds_vX,bounds_cosq])[0]

    def eta_b(_vmin):
        def bounds_cosq(vx):
            return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)]
        def bounds_vX():
            return [_vmin, vE+vesc]
        def intfunc(cosq,vx):
            return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
        return nquad(intfunc, [bounds_cosq,bounds_vX])[0]

    eta = lambda vmin: np.piecewise( vmin,
                                     [ vmin <= (vesc-vE),
                                       np.logical_and(vmin > (vesc-vE),
                                                      vmin <= (vesc+vE)),
                                       vmin > (vesc+vE) ],
                                     [ np.vectorize(eta_a),
                                       np.vectorize(eta_b), 0 ] )

    if return_func:
        return eta
    else:
        return eta(vmin)

def etaTsa_q(*args):

    """
    Tsallis Model, q = .773, v0 = 267.2 km/sveldist_plot, and vesc = 560.8 km/s
    give best fits from arXiv:0909.2028.
    params = [v0, vE, q], input parameters must be scalars
    """

    if len(args) == 1:
        return_func = True
        _params = args[0]

    elif len(args) == 2:
        return_func = False
        vmin, _params = args

    else:
        raise TypeError('Wrong number of arguments!')

    v0 = _params[0]
    vE = _params[1]
    q = _params[2]

    if q < 1:
        vesc = v0/np.sqrt(1-q)
    else:
        vesc = 544/ckms # km/s, standard fiducial value

    if q == 1:
        tsa_ = lambda vx: vx**2*np.exp(-vx**2/v0**2)
        func = lambda vx2: np.exp(-vx2/v0**2)

    else:
        tsa_ = lambda vx: vx**2*(1-(1-q)*vx**2/v0**2)**(1/(1-q))
        func = lambda vx2: (1-(1-q)*vx2/v0**2)**(1/(1-q))

    tsa_inttest = lambda vx: np.piecewise(vx, [vx <= vesc, vx > vesc],
                                          [tsa_, 0])

    K_=4*np.pi*quad(tsa_inttest, 0, vesc)[0]

    def eta_a(_vmin):
        def bounds_cosq():
            return [-1,1]
        def bounds_vX(cosq):
            return [_vmin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]
        def intfunc(vx,cosq):
            return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
        return nquad(intfunc, [bounds_vX,bounds_cosq])[0]

    def eta_b(_vmin):
        def bounds_cosq(vx):
            return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)]
        def bounds_vX():
            return [_vmin, vE+vesc]
        def intfunc(cosq,vx):
            return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
        return nquad(intfunc, [bounds_cosq,bounds_vX])[0]

    eta = lambda vmin: np.piecewise( vmin,
                                     [ vmin <= (vesc-vE),
                                       np.logical_and(vmin > (vesc-vE),
                                                      vmin <= (vesc+vE)),
                                       vmin > (vesc+vE) ],
                                     [ np.vectorize(eta_a),
                                       np.vectorize(eta_b), 0 ] )

    if return_func:
        return eta
    else:
        return eta(vmin)

def etaMSW(*args):

    """
    empirical model by Mao, Strigari, Weschler arXiv:1210.2721
    params = [v0, vE, vesc, p], input parameters must be scalars
    """

    if len(args) == 1:
        return_func = True
        _params = args[0]

    elif len(args) == 2:
        return_func = False
        vmin, _params = args

    else:
        raise TypeError('Wrong number of arguments!')

    v0 = _params[0]
    vE = _params[1]
    vesc = _params[2]
    p = _params[3]

    def msw(vx):
        return vx**2*np.exp(-vx/v0)*(vesc**2-vx**2)**p

    msw_inttest = lambda vx: np.piecewise( vx,
                                           [vx <= vesc, vx > vesc],
                                           [msw, 0] )

    K_=4*np.pi*quad(msw_inttest, 0, vesc)[0]

    def func(vx2):
        return np.exp(-np.sqrt(vx2)/v0)*(vesc**2-vx2)**p

    def eta_a(_vmin):
        def bounds_cosq():
            return [-1,1]
        def bounds_vX(cosq):
            return [_vmin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]
        def intfunc(vx,cosq):
            return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
        return nquad(intfunc, [bounds_vX,bounds_cosq])[0]

    def eta_b(_vmin):
        def bounds_cosq(vx):
            return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)]
        def bounds_vX():
            return [_vmin, vE+vesc]
        def intfunc(cosq,vx):
            return (2*np.pi/K_)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
        return nquad(intfunc, [bounds_cosq,bounds_vX])[0]

    eta = lambda vmin: np.piecewise( vmin,
                                     [ vmin <= (vesc-vE),
                                       np.logical_and(vmin > (vesc-vE),
                                                      vmin <= (vesc+vE)),
                                       vmin > (vesc+vE) ],
                                     [ np.vectorize(eta_a),
                                       np.vectorize(eta_b), 0 ] )

    if return_func:
        return eta
    else:
        return eta(vmin)

def etaDebris(vmin, _params):

    vflow = _params[0]
    vE = _params[1]
    x = vmin

    eta_ = np.piecewise(x, [x < vflow-vE, (vflow-vE <= x) & (x < vflow+vE)],
                        [1/vflow, lambda x: (vflow+vE-x)/(2*vflow*vE), 0])

    return eta_

def etaFromVDF(*args, vesc=544, vE=232, method='quad', **kwargs):
    """
    Calculates eta from an arbitrary velocity distribution entered by either a
    data file or a function.
    path_or_fn must either be the path to your data file or a function.
    For a data file, assumes csv file unless delimiter is set in kwargs.
    """

    if len(args) == 1:
        return_func = True
        path_or_fn = args[0]

    elif len(args) == 2:
        return_func = False
        Vmin, path_or_fn = args

    else:
        raise TypeError('Wrong number of arguments!')

    if isinstance(path_or_fn, str):
        if 'delimiter' in kwargs:
            data = np.loadtxt(path_or_fn, delimiter=kwargs.get('delimiter'))
        else:
            data = np.loadtxt(path_or_fn, delimiter=',')

        arb_fn = interp1d(data[:,0], data[:,1], bounds_error=False,
                          fill_value=0.)
        v2data = data[:,:]
        v2data[:,1] *= data[:,0]**2
        KK_ = 4*np.pi*trapezoid(v2data[:,1], x=v2data[:,0])

    elif callable(path_or_fn):
        arb_fn = path_or_fn
        def v2f_fn(v):
            return v**2*arb_fn(v)
        KK_ = 4*np.pi*quad(v2f_fn, 0, vesc)[0]

    else:
        print('etaFromVDF failed because you did not enter a valid string '+\
              'or function.')
        return

    def norm_fn(v_):
        return arb_fn(v_)/KK_

    def eta_a(_vmin):
        def bounds_cosq():
            return [-1,1]
        def bounds_vX(cosq):
            return [_vmin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]
        def eta(vx, cosq):
            return (2*np.pi)*vx*norm_fn(vx**2+vE**2+2*vx*vE*cosq)
        return nquad(eta, [bounds_vX, bounds_cosq])[0]

    def eta_b(_vmin):
        def bounds_cosq(vx):
            return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)]
        def bounds_vX():
            return [_vmin, vE+vesc]
        def eta(cosq,vx):
            return (2*np.pi)*vx*norm_fn(vx**2+vE**2+2*vx*vE*cosq)
        return nquad(eta, [bounds_cosq,bounds_vX])[0]

    eta = lambda vmin: np.piecewise( vmin,
                                     [ vmin <= (vesc-vE),
                                       np.logical_and(vmin > (vesc-vE),
                                                      vmin <= (vesc+vE)),
                                       vmin > (vesc+vE) ],
                                     [ np.vectorize(eta_a),
                                       np.vectorize(eta_b), 0 ] )

    if return_func:
        return eta
    else:
        return eta(vmin)

def etaFromFile(*args, in_unit='kms', out_unit='kms', delimiter=',',
                kind='slinear'):
    """
    Returns either an eta function eta(Vmin) or an array of eta values eta(Vmin)
    depending on args.
    args = ( (Vmin,) path_to_data )
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
        Vmin, path_to_data = args

    else:
        raise TypeError('Wrong number of arguments!')

    data = np.loadtxt(path_to_data, delimiter=delimiter)

    # change the data into the correct units:
    data[:,0] *= corr_dict[out_unit] / corr_dict[in_unit]
    data[:,1] *= corr_dict[in_unit] / corr_dict[out_unit]

    eta_fn = interp1d( data[:,0], data[:,1], bounds_error=False,
                       fill_value='extrapolate', kind=kind )

    eta = lambda x: np.piecewise( x, [x <= data[-1,0], x > data[-1,0]],
                                     [eta_fn, 0] )

    if return_func:
        return eta
    else:
        return eta(Vmin)
