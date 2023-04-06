import numpy as np
from numpy.linalg import norm
from scipy.special import erf
from scipy.integrate import quad, nquad
from scipy.optimize import approx_fprime
from pyQEdark import ckms
from pyQEdark.Streams import _calc_lam, _calc_tc, _solve_for_Mu
from pyQEdark.etas import etaSHMi

v_sun = np.array([11.1, 247.24, 7.25])/ckms # velocity of sun in galactic frame

vE_solar = 29.79/ckms # magnitude of earth velocity in solar frame

# epsilon is used to determine the direction of earth's velocity in solar frame
epsilon_1 = np.array([0.9940, 0.1095, 0.0031])
epsilon_2 = np.array([-0.0517, 0.4945, -0.8677])
epsilon_3 = np.cross(epsilon_1, epsilon_2)

# epsilon_hat is used for determining useful quantities b_i
eps_hat = np.array([epsilon_1/norm(epsilon_1),
                    epsilon_2/norm(epsilon_2),
                    epsilon_3/norm(epsilon_3)])

omega = 2*np.pi/365.25 # days^-1
t_Mar21 = 79.26 # days

mean_stream_disp = 55.937 / ckms

def piece_it(
    arr,
    thresholds=(0, 1/6, 1/2),
    funcs=(lambda x: 0, lambda x: 6 * x, lambda x: 3 / 2 - 3 * x, lambda x: 0)
    ):
    """
    From https://stackoverflow.com/questions/73889832/
    """
    masks = np.array([arr < threshold for threshold in thresholds])
    masks = [masks[0]] + [x for x in masks[1:] & ~masks[:-1]] + [~masks[-1]]
    result = np.empty_like(arr)
    for mask, func in zip(masks, funcs):
        result[mask] = func(arr[mask])
    return result

def vE_fn_solar(t):
    """
    Returns the velocity of the earth in the solar frame at a given time
    t : units of days (t=0 corresponds to Jan 1 at 00:00)
    """
    return vE_solar * ( epsilon_1*np.cos(omega*(t-t_Mar21)) + \
                        epsilon_2*np.sin(omega*(t-t_Mar21)) )

def v_lab(t):
    """
    Returns the velocity of the lab in the galactic frame at a given time
    t : units of days (t=0 corresponds to Jan 1 at 00:00)
    """
    return v_sun + vE_fn_solar(t)

def _calc_tc(Mu_gal):
    """
    Calculates the critical time t_c that maximizes mu_lab
    Mu_gal : mean stream velocity vector in the galactic frame (nat. units)
    """
    Mu_sol_hat = (Mu_gal - v_sun) / norm(Mu_gal - v_sun)

    b = np.array([np.dot(Mu_sol_hat, eps_hat[i]) for i in range(3)])

    tc1 = (1/omega)*np.arctan2(-b[1],-b[0]) + t_Mar21
    tc2 = tc1 + 365.25/2

    mulab1 = norm(Mu_gal - v_lab(tc1))
    mulab2 = norm(Mu_gal - v_lab(tc2))

    if mulab1 > mulab2:
        tc = tc1
    else:
        tc = tc2

    return tc % 365.25

def mugal_to_musol(Mu_gal):
    Mu_sol = Mu_gal - v_sun
    musol = norm(Mu_sol)
    Mu_sol_hat = Mu_sol / musol

    b3 = np.dot(Mu_sol_hat, epsilon_3)
    b = np.sqrt(1 - b3**2)

    tc = _calc_tc(Mu_gal)

    return (musol, tc, b)

class VelocityDistribution:
    """
    Creates an object which contains the velocity distribution function (vdf),
    eta (<1/v> wrt vdf in lab frame), and deta_dvmin. All units are assumed to
    be natural.

    vdf_gal(v, params) : functional form of the vdf in the galactic frame,
    assumed to be isotropic (only a function of |\vec{v}|) and pre-normalized

    params : parameters of vdf_gal

    vesc : velocity for which vdf_gal(v > vesc) = 0

    eta(vmin, t, params) : (optional) analytic form of eta, if not provided
    then eta will be calculated by integrating d^3v vdf_gal(v+vlab) θ(v-vmin)

    deta_dvmin(vmin, t, params) : (optional) analytic form of deta_dmvin, if not
    provided then this will be calculated via finite difference
    """
    def __init__(self, vdf_gal, params, vesc=None, eta=None, deta_dvmin=None):

        self.params = params
        self.vesc = vesc
        self.vdf = lambda v: vdf_gal(v, self.params)

        if eta is not None:
            self.eta = lambda vmin, t: eta(vmin, t, self.params)
        elif vesc is not None:
            self.eta = self._gen_eta_vesc(vesc)
        else:
            self.eta = self._gen_eta()
        
        if deta_dvmin is not None:
            self.deta_dvmin = lambda vmin, t: deta_dvmin(vmin, t, self.params)
        else:
            self.deta_dvmin = self._gen_deta_dvmin()

    def _gen_eta_vesc(self, vesc):

        func = lambda vx2: self.vdf(np.sqrt(vx2))

        def eta_a(_vmin, t):
            vE = norm(v_lab(t))
            def bounds_cosq():
                return [-1,1]
            def bounds_vX(cosq):
                return [_vmin, -cosq*vE+np.sqrt((cosq**2-1)*vE**2+vesc**2)]
            def intfunc(vx,cosq):
                return (2*np.pi)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(intfunc, [bounds_vX,bounds_cosq])[0]

        def eta_b(_vmin, t):
            vE = norm(v_lab(t))
            def bounds_cosq(vx):
                return [-1, (vesc**2-vE**2-vx**2)/(2*vx*vE)]
            def bounds_vX():
                return [_vmin, vE+vesc]
            def intfunc(cosq,vx):
                return (2*np.pi)*vx*func(vx**2+vE**2+2*vx*vE*cosq)
            return nquad(intfunc, [bounds_cosq,bounds_vX])[0]
        
        def eta_fn(vmin, t):
            vE = norm(v_lab(t))
            return np.piecewise(vmin,
                                [ vmin <= (vesc-vE),
                                np.logical_and(vmin > (vesc-vE),
                                                vmin <= (vesc+vE)),
                                vmin > (vesc+vE) ],
                                [ np.vectorize(eta_a, excluded=['t']),
                                  np.vectorize(eta_b, excluded=['t']),
                                  0 ], t)
        
        return eta_fn
    
    def _gen_eta(self):

        func = lambda vx2: self.vdf(np.sqrt(vx2))

        def eta_a(vmin, t):
            intfunc = lambda v, x: 2*np.pi*v*func(v**2+vE**2+2*v*vE*x)
            return nquad(intfunc, [[vmin, np.inf],[-1,1]])
        
        return np.vectorize(eta_a, excluded=['t'])

    def _gen_deta_dvmin(self, dv=1/(1e3*ckms)):
        func_prime = lambda vmin,t: (self.eta(vmin+dv, t) - self.eta(vmin,t))/dv
        return func_prime

class VelDistSHM(VelocityDistribution):
    def __init__(self, v0=228.6/ckms, vesc=528/ckms):
        self.v0 = v0
        self.vesc = vesc
        self.KK = v0**3 * np.pi * (np.sqrt(np.pi)*erf(vesc/v0) - 2 * (vesc/v0)*\
                  np.exp(-vesc**2/v0**2))
    
    def _mb(self, v):
        return 1/self.KK*np.exp(-v**2/self.v0**2)

    def vdf(self, v):
        return np.piecewise( v, [v < self.vesc, v >= self.vesc], [self._mb, 0] )
    
    def _eta_a(self, vmin, t):
        vE = norm(v_lab(t))
        a = self.v0**2 * np.pi / (2 * vE * self.KK)
        nn1 = -4*np.exp(-self.vesc**2/self.v0**2)*vE
        nn2 = np.sqrt(np.pi)*self.v0*( erf((vmin+vE)/self.v0) - \
                                       erf((vmin-vE)/self.v0) )
        return a * (nn1 + nn2)

    def _eta_b(self, vmin, t):
        vE = norm(v_lab(t))
        a = self.v0**2 * np.pi / (2 * vE * self.KK)
        nn1 = -2*np.exp(-self.vesc**2/self.v0**2)*(vE+self.vesc - vmin)
        nn2 = np.sqrt(np.pi)*self.v0*(erf(self.vesc/self.v0) - \
                                      erf((vmin-vE)/self.v0))
        return a * (nn1 + nn2)
    
    def etap(self, vmin, t):
        vE = norm(v_lab(t))
        return np.piecewise( vmin, [ vmin <= (self.vesc-vE),
                                     np.logical_and(vmin > (self.vesc-vE),
                                                    vmin <= (self.vesc+vE)),
                                     vmin > (self.vesc+vE) ],
                                     [ self._eta_a, self._eta_b, 0 ], t )
    
    def eta(self, vmin, t):
        vE = norm(v_lab(t))
        thresholds = (0, self.vesc-vE, self.vesc+vE)
        funcs=(lambda x: 0, lambda x: self._eta_a(x,t), 
               lambda x: self._eta_b(x,t), lambda x: 0)
        return piece_it(vmin, thresholds=thresholds, funcs=funcs)
    
    def _de_dv_a(self, vmin, t):
        vE = norm(v_lab(t))
        pref = np.pi * self.v0**2 / (self.KK*vE)
        alpha = np.exp(-(vmin-vE)**2/self.v0**2)
        beta = np.exp(-(vmin+vE)**2/self.v0**2)
        return pref*(beta-alpha)

    def _de_dv_b(self, vmin, t):
        vE = norm(v_lab(t))
        pref = np.pi * self.v0**2 / (self.KK*vE)
        beta = np.exp(-self.vesc**2/self.v0**2)
        alpha = np.exp(-(vmin-vE)**2/self.v0**2)
        return pref*(beta-alpha)

    def deta_dvmin(self, vmin, t):
        vE = norm(v_lab(t))
        thresholds = (0, self.vesc-vE, self.vesc+vE)
        funcs=(lambda x: 0, lambda x: self._de_dv_a(x,t), 
               lambda x: self._de_dv_b(x,t), lambda x: 0)
        return piece_it(vmin, thresholds=thresholds, funcs=funcs)

    def deta_dvmin_p(self, vmin, t):
        vE = norm(v_lab(t))
        return np.piecewise( vmin,
                             [ vmin <= (self.vesc-vE),
                               np.logical_and(vmin > (self.vesc-vE),
                                              vmin <= (self.vesc+vE)),
                               vmin > (self.vesc+vE) ],
                             [ self._de_dv_a, self._de_dv_b, 0 ], t )

class VelDistStream(VelocityDistribution):

    def __init__(self, *args):
        if len(args) == 2:
            Mu_gal = args[0]
            sigma = args[1]

            self.tc = _calc_tc(Mu_gal)
            self.lam = _calc_lam(Mu_gal)
            self.b = np.sin(self.lam)

        elif len(args) == 4:
            mu_norm = args[0]
            tc = args[1]
            lam = args[2]
            sigma = args[3]
            Mu_gal = _solve_for_Mu(mu_norm, tc, lam)

            self.tc = tc
            self.lam = lam
            self.b = np.sin(self.lam)

        else:
            raise TypeError('Wrong number of arguments!')

        self.Mu_gal = Mu_gal
        self.sigma = sigma

        self.mu_sol = norm(Mu_gal-v_sun)

        self.KK = np.sqrt((2*np.pi)**3)*sigma**3

        # self._Mu_gal_calc = _solve_for_Mu(norm(self.Mu_gal),self.tc,self.lam)

    def f_gal(self, v):
        v_ = v[:]
        vexp = norm(v_ - self.Mu_gal)
        len_v = np.sqrt( np.sum(v_**2) )
        ee = (1/self.KK) * np.exp( -.5*vexp**2/sigma**2 )
        return ee

    def f_lab(self, v, t):
        v_ = v[:]
        vlab = v_lab(t)
        vexp = norm(v_ + vlab - self.Mu_gal)
        ee = (1/self.KK) * np.exp( -.5*vexp**2/sigma**2 )
        return ee

    def F_lab(self, v, t):
        vlab = v_lab(t)
        mu_lab = norm(self.Mu_gal - vlab)
        pref = 2*np.pi*v*self.sigma**2/mu_lab
        ee1 = np.exp(-(v+mu_lab)**2/(2*self.sigma**2))
        ee2 = np.exp(2*v*mu_lab / self.sigma**2) - 1
        return pref*ee1*ee2/self.KK

    def eta(self, vmin, t):
        vlab = v_lab(t)
        mu_lab = norm(self.Mu_gal - vlab)
        pref = np.sqrt(2*np.pi**3)*self.sigma**3 / mu_lab
        ee = erf((vmin+mu_lab)/(np.sqrt(2)*self.sigma)) - \
             erf((vmin-mu_lab)/(np.sqrt(2)*self.sigma))
        return pref*ee/self.KK
    
    def deta_dmugal(self, vmin, t, idx):
        Mu_lab = self.Mu_gal - v_lab(t)
        mu_lab = norm(Mu_lab)
        pref1 = - Mu_lab[idx] / (2*mu_lab**3)
        erf_term = erf((vmin+mu_lab)/(np.sqrt(2)*self.sigma)) - \
                   erf((vmin-mu_lab)/(np.sqrt(2)*self.sigma))
        pref2 = Mu_lab[idx] / (np.sqrt(2*np.pi)*mu_lab**2*self.sigma)
        ee1 = np.exp(-(vmin+mu_lab)**2 / (2*self.sigma**2))
        ee2 = np.exp(-(vmin-mu_lab)**2 / (2*self.sigma**2))
        return pref1*erf_term + pref2*(ee1 + ee2)
    
    def deta_dvmin(self, vmin, t):
        vlab = v_lab(t)
        mu_lab = norm(self.Mu_gal - vlab)
        pref = 1 / (np.sqrt(2*np.pi)*self.sigma*mu_lab)
        ee1 = np.exp(-(vmin+mu_lab)**2 / (2*self.sigma**2))
        ee2 = np.exp(-(vmin-mu_lab)**2 / (2*self.sigma**2))
        return pref*(ee1-ee2)

class VelDistStreamSol(VelocityDistribution):
    def __init__(self, mu_sol, tc, b, sigma):
        self.mu_sol = mu_sol
        self.tc = tc
        self.b = b
        self.sigma = sigma
        self.KK = np.sqrt((2*np.pi)**3)*sigma**3
    
    def mu_lab(self, t):
        xi = np.cos(omega*(t-self.tc))
        return np.sqrt(self.mu_sol**2 + vE_solar**2 + \
               2 * self.mu_sol * vE_solar * self.b * xi)
    
    def eta(self, vmin, t):
        mu_lab = self.mu_lab(t)
        pref = np.sqrt(2*np.pi**3)*self.sigma**3 / mu_lab
        ee = erf( (vmin+mu_lab)/(np.sqrt(2)*self.sigma) ) - \
             erf( (vmin-mu_lab)/(np.sqrt(2)*self.sigma) )
        return pref*ee/self.KK

    def deta_dvmin(self, vmin, t):
        mu_lab = self.mu_lab(t)
        pref = 1 / (np.sqrt(2*np.pi)*self.sigma*mu_lab)
        ee1 = np.exp(-(vmin+mu_lab)**2 / (2*self.sigma**2))
        ee2 = np.exp(-(vmin-mu_lab)**2 / (2*self.sigma**2))
        return pref*(ee1-ee2)
    
    def deta_dmulab(self, vmin, t):
        mu_lab = self.mu_lab(t)
        pref = np.pi*self.sigma**2 / mu_lab**2
        ee1 = 2*mu_lab*(np.exp(-(vmin+mu_lab)**2 / (2*self.sigma**2)) + \
                        np.exp(-(vmin-mu_lab)**2 / (2*self.sigma**2)))
        ee2 = -np.sqrt(2*np.pi)*self.sigma*\
              (erf( (vmin+mu_lab)/(np.sqrt(2)*self.sigma) ) - \
              erf( (vmin-mu_lab)/(np.sqrt(2)*self.sigma) ))
        return pref * (ee1 + ee2) / self.KK
    
    def deta_dmusol(self, vmin, t):
        deta_dmulab = self.deta_dmulab(vmin, t)

        mu_lab = self.mu_lab(t)
        dmulab_dmusol = (self.mu_sol + self.b * vE_solar * \
                         np.cos(omega*(t-self.tc))) / mu_lab
        
        return deta_dmulab * dmulab_dmusol
    
    def deta_dtc(self, vmin, t):
        deta_dmulab = self.deta_dmulab(vmin, t)

        mu_lab = self.mu_lab(t)
        dmulab_dtc = (self.b*vE_solar*self.mu_sol*omega*\
                     np.sin(omega*(t-self.tc))) / mu_lab
        
        return deta_dmulab * dmulab_dtc
    
    def deta_db(self, vmin, t):
        deta_dmulab = self.deta_dmulab(vmin, t)

        mu_lab = self.mu_lab(t)
        dmulab_db = (vE_solar*self.mu_sol*np.cos(omega*(t-self.tc)))/mu_lab

        return deta_dmulab * dmulab_db
    

