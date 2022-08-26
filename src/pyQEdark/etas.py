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
from scipy.integrate import nquad, quad, trapezoid, quad_vec, odeint

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

def detaSHM_dvmin(*args):
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

    def deta_dvmin_a(_vmin):
        pref = np.exp(-(vE+_vmin)**2/v0**2)*np.pi*v0**2 / (2*KK*vE**2*_vmin)
        alpha = v0**2 + 2 * vE * _vmin
        beta = np.exp(4*vE*_vmin/v0**2) * (v0**2-2*vE*_vmin)
        return pref*(beta-alpha)

    def deta_dvmin_b(_vmin):
        pref = np.exp(-(vE+_vmin)**2/v0**2)*np.pi*v0**2 / (2*KK*vE**2*_vmin)
        gamma = np.exp((vE-vesc+_vmin)*(vE+vesc-_vmin)/v0**2) * \
               (v0**2-vE**2+vesc**2-_vmin**2)
        beta = np.exp(4*vE*_vmin/v0**2) * (v0**2-2*vE*_vmin)
        return pref*(beta-gamma)

    deta_dvmin = lambda vmin: np.piecewise( vmin,
                                     [ vmin <= (vesc-vE),
                                       np.logical_and(vmin > (vesc-vE),
                                                      vmin <= (vesc+vE)),
                                       vmin > (vesc+vE) ],
                                     [ deta_dvmin_a, deta_dvmin_b, 0 ] )

    if return_func:
        return deta_dvmin
    else:
        return deta_dvmin(vmin)

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

def detaTsa_dvmin(*args):

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

    def deta_dvmin_a(_vmin):
        f_to_int = lambda cth: _vmin * cth / K_ * \
                               func( _vmin**2+vE**2+2*_vmin*vE*cth )
        return 2*np.pi*quad_vec(f_to_int, -1, 1)[0]

    def deta_dvmin_b(_vmin):
        f_to_int = lambda cth: _vmin * cth / K_ * \
                               func( _vmin**2+vE**2+2*_vmin*vE*cth )
        bound_up = (-vE**2 + vesc**2 - _vmin**2) / (2 * vE * _vmin)
        return 2*np.pi*quad(f_to_int, -1, bound_up)[0]

    deta_dvmin_b_vec = np.vectorize(deta_dvmin_b)

    deta_dvmin = lambda vmin: np.piecewise( vmin,
                                     [ vmin <= (vesc-vE),
                                       np.logical_and(vmin > (vesc-vE),
                                                      vmin <= (vesc+vE)),
                                       vmin > (vesc+vE) ],
                                     [ deta_dvmin_a,
                                       deta_dvmin_b_vec, 0 ] )

    if return_func:
        return deta_dvmin
    else:
        return deta_dvmin(vmin)

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

def etaMSW(*args, method='fast', N_MC=100000):

    """
    empirical model by Mao, Strigari, Weschler arXiv:1210.2721
    params = [v0, vE, vesc, p], input parameters must be scalars

    method = 'fast' : uses monte carlo, for ~.5% error compared to nquad
           = 'slow' : uses scipy.integrate.nquad, which is more accurate, but
                      ~300 times slower
    """

    if len(args) == 1:
        return_func = True
        _params = args[0]

    elif len(args) == 2:
        return_func = False
        vmin, _params = args

    else:
        raise TypeError('Wrong number of arguments!')

    v0, vE, vesc, p = _params

    def msw_to_norm(v):
        if v <= vesc:
            return v**2*np.exp(-v/v0)*(vesc**2-v**2)**p
        else:
            return 0

    KK = 4*np.pi*quad( msw_to_norm, 0, vesc )[0]

    if method == 'fast':

        def msw_un(vx):
            return np.exp(-vx/v0)*(vesc**2-vx**2)**p

        msw_fn = lambda vx: np.piecewise( vx,
                                          [ vx <= vesc, vx > vesc],
                                          [ msw_un, 0] )

        def eta(vmin):
            vmin = np.atleast_1d(vmin)

            v1 = vmin.min()
            v2 = vE+vesc

            c1,c2 = (-1,1)

            A = (v2-v1)*(c2-c1)

            rando_v = (v2-v1) * np.random.random_sample(size=N_MC) + v1
            rando_c = (c2-c1) * np.random.random_sample(size=N_MC) + c1

            vv = np.sqrt(rando_v**2 + vE**2 + 2*rando_v*vE*rando_c)
            f_vals = rando_v*msw_fn(vv)

            I = np.zeros_like(vmin)
            for i in range(len(vmin)):
                f_tmp = np.zeros(N_MC)
                is_greater_than_vmin = rando_v >= vmin[i]
                f_tmp[:] = f_vals[:] * is_greater_than_vmin[:]
                I[i] = np.sum(f_tmp[:])

            # I = np.zeros_like(vmin)
            # for i in range(N_MC):
            #     less_than_v = vmin <= rando_v[i]
            #     I += f_vals[i] * less_than_v

            I *= A*2*np.pi / (N_MC * KK)
            return I

    elif method == 'slow':
        def func(vx2):
            return np.exp(-np.sqrt(vx2)/v0)*(vesc**2-vx2)**p

        def eta_a(_vmin):
            def bounds_cosq():
                return [-1,1]

            def bounds_vX(cosq):
                return [_vmin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]

            def intfunc(vx, cosq):
                vv = vx**2 + vE**2 + 2 * vx * vE * cosq
                ff = np.exp(-np.sqrt(vv)/v0)*(vesc**2-vv)**p
                return (2*np.pi/KK)*vx*ff

            return nquad(intfunc, [bounds_vX,bounds_cosq])[0]

        def eta_b(_vmin):
            def bounds_cosq(vx):
                return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)]

            def bounds_vX():
                return [_vmin, vE+vesc]

            def intfunc(cosq, vx):
                vv = vx**2 + vE**2 + 2 * vx * vE * cosq
                ff = np.exp(-np.sqrt(vv)/v0)*(vesc**2-vv)**p
                return (2*np.pi/KK)*vx*ff

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

def detaMSW_dvmin(*args):

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

    def deta_dvmin_a(_vmin):
        f_to_int = lambda cth: _vmin * cth / K_ * \
                               func( _vmin**2+vE**2+2*_vmin*vE*cth )
        return 2*np.pi*quad_vec(f_to_int, -1, 1)[0]

    def deta_dvmin_b(_vmin):
        f_to_int = lambda cth: _vmin * cth / K_ * \
                               func( _vmin**2+vE**2+2*_vmin*vE*cth )
        bound_up = (-vE**2 + vesc**2 - _vmin**2) / (2 * vE * _vmin)
        return 2*np.pi*quad(f_to_int, -1, bound_up)[0]

    deta_dvmin_b_vec = np.vectorize(deta_dvmin_b)

    deta_dvmin = lambda vmin: np.piecewise( vmin,
                                     [ vmin <= (vesc-vE),
                                       np.logical_and(vmin > (vesc-vE),
                                                      vmin <= (vesc+vE)),
                                       vmin > (vesc+vE) ],
                                     [ deta_dvmin_a,
                                       deta_dvmin_b_vec, 0 ] )

    if return_func:
        return deta_dvmin
    else:
        return deta_dvmin(vmin)

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
