"""
A set of parameters, functions, and classes used to describe DM streams in our
galaxy. All velocities are in natural units (c = hbar = 1), BUT, time t is in
units of days (to easily track the passing of a year).

Author: Aria Radick
Date Created: 3/15/2022
"""

import numpy as np
from numpy.linalg import norm, inv, det
from scipy.integrate import nquad
from scipy.interpolate import interp1d, interp2d
from scipy.stats import poisson
from scipy.special import erf
from pyQEdark.constants import ckms

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

# Parameters for some known stellar streams:
Mu_gal_Sausage = np.array([0, 36.94, -2.92])/ckms
Sigma_Sausage = np.diag([108.33**2, 62.60**2, 57.99**2])/ckms**2

Mu_gal_Nyx = np.array([133.9, 130.17, 53.65])/ckms
Sigma_Nyx = np.diag([67.13**2, 45.80**2, 65.82**2])/ckms**2

Mu_gal_S1 = np.array([-34.2, -306.3, -64.4])/ckms
Sigma_S1 = np.diag([81.9**2, 46.3**2, 62.9**2])/ckms**2

Mu_gal_S2a = np.array([5.8, 163.6, -250.4])/ckms
Sigma_S2a = np.diag([45.9**2, 13.8**2, 26.8**2])/ckms**2

Mu_gal_S2b = np.array([-50.6, 138.5, 183.1])/ckms
Sigma_S2b = np.diag([90.8**2, 25.0**2, 43.8**2])/ckms**2

_Sig_mean = np.zeros(3)
for i in range(3):
    _Sig_mean[i] = np.mean([Sigma_Nyx[i,i], Sigma_S1[i,i], Sigma_S2a[i,i],
                           Sigma_S2b[i,i]])
sig_mean = norm(np.sqrt(_Sig_mean)) / np.sqrt(3)

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

def Mu_lab(t, Mu_gal):
    """
    Determines the mean stream velocity in the lab frame at a time t
    t : units of days (t=0 corresponds to Jan 1 at 00:00)
    Mu_gal : mean stream velocity vector in the galactic frame (nat. units)
    """
    return norm(Mu_gal - v_sun - vE_fn_solar(t))

def _calc_tc(Mu_gal):
    """
    Calculates the critical time t_c that maximizes mu_lab
    Mu_gal : mean stream velocity vector in the galactic frame (nat. units)
    """
    Mu_sol_hat = (Mu_gal - v_sun) / norm(Mu_gal - v_sun)

    b = np.array([np.dot(Mu_sol_hat, eps_hat[i]) for i in range(3)])

    tc1 = (1/omega)*np.arctan2(b[1],b[0]) + t_Mar21
    tc2 = tc1 + 365.25/2

    if Mu_lab(tc1,Mu_gal) > Mu_lab(tc2,Mu_gal):
        tc = tc1
    else:
        tc = tc2

    while tc < 0:
        tc += 365.25
    while tc > 365.25:
        tc -= 365.25

    return tc

def _calc_lam(Mu_gal):
    """
    Calculates the angle λ between mu in the solar frame and the normal to
    Earth's rotational plane.
    Mu_gal : mean stream velocity vector in the galactic frame (nat. units)
    """
    Mu_sol_hat = (Mu_gal - v_sun) / norm(Mu_gal - v_sun)
    b3 = np.dot(Mu_sol_hat, eps_hat[2])
    return np.arccos(b3)

def _solve_for_Mu(Mu_norm, tc, lam, diagnostics=False):
    """
    Transforms from (Mu_norm, t_c, λ) to Mu_gal
    """
    b = np.sin(lam)
    te_c = omega*(tc-t_Mar21)
    theta = te_c - np.pi

    vs_eps = np.dot(eps_hat, v_sun)
    vsa,vsb,vsc = vs_eps

    b1 = b*np.cos(theta)
    b2 = b*np.sin(theta)
    b3 = np.cos(lam)

    Z = np.sqrt(-vsb**2 + b2**2*(-vsa**2 + vsb**2) + 2*b1*b3*vsa*vsc - \
                vsc**2 + 2*b2*vsb*(b1*vsa + b3*vsc) + \
                b3**2*(-vsa**2 + vsc**2) + Mu_norm**2)

    mu_sol_p = Z - b1*vsa - b2*vsb - b3*vsc
    mu_sol_m = -Z - b1*vsa - b2*vsb - b3*vsc

    if mu_sol_p < mu_sol_m:
        Z = -Z

    mu1 = (1 - b1**2)*vsa - b1*b2*vsb - b1*b3*vsc + b1*Z
    mu2 = (1 - b2**2)*vsb - b1*b2*vsa - b2*b3*vsc + b2*Z
    mu3 = (1 - b3**2)*vsc - b1*b3*vsa - b2*b3*vsb + b3*Z

    Mu_gal_eps = np.array([mu1,mu2,mu3])
    Mu_gal = np.dot(np.transpose(eps_hat), Mu_gal_eps)

    if diagnostics:
        tc_calc = _calc_tc(Mu_gal)
        lam_calc = _calc_lam(Mu_gal)
        return Mu_gal, tc_calc, lam_calc
    else:
        return Mu_gal

def _mu_solver_min(tc, lam):
    """
    Related to the '_solve_for_Mu' function. Determines the minimum allowed
    norm(mu_gal) for a given t_c and λ pair.
    """
    b = np.sin(lam)
    te_c = omega*(tc-t_Mar21)
    theta = te_c - np.pi

    vs_eps = np.dot(eps_hat, v_sun)

    b1 = b*np.cos(theta)
    b2 = b*np.sin(theta)
    b3 = np.cos(lam)

    vsa,vsb,vsc = vs_eps

    under_sqrt = -(-vsb**2 + b2**2*(-vsa**2 + vsb**2) + 2*b1*b3*vsa*vsc - \
                 vsc**2 + 2*b2*vsb*(b1*vsa + b3*vsc) + b3**2*(-vsa**2 + vsc**2))

    return np.sqrt(under_sqrt)

class Stream_Analytic():

    def __init__(self, Mu_gal, sigma):
        self.Mu_gal = Mu_gal
        self.sigma = sigma

        self.KK = np.sqrt((2*np.pi)**3)*sigma**3

        self.tc = _calc_tc(self.Mu_gal)
        self.lam = _calc_lam(self.Mu_gal)
        self.b = np.sin(self.lam)

        # self._Mu_gal_calc = _solve_for_Mu(norm(self.Mu_gal), self.tc, self.lam)

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
    
    def _deta_dmugal(self, vmin, t, idx):
        Mu_lab = self.Mu_gal - v_lab(t)
        mu_lab = norm(Mu_lab)
        pref1 = - Mu_lab[idx] / (2*mu_lab**3)
        erf_term = erf((vmin+mu_lab)/(np.sqrt(2)*self.sigma)) - \
                   erf((vmin-mu_lab)/(np.sqrt(2)*self.sigma))
        pref2 = Mu_lab[idx] / (np.sqrt(2*np.pi)*mu_lab**2*self.sigma)
        ee1 = np.exp(-(vmin+mu_lab)**2 / (2*self.sigma**2))
        ee2 = np.exp(-(vmin-mu_lab)**2 / (2*self.sigma**2))
        return pref1*erf_term + pref2*(ee1 + ee2)
    
    def _deta_dvmin(self, vmin, t):
        vlab = v_lab(t)
        mu_lab = norm(self.Mu_gal - vlab)
        pref = 1 / (np.sqrt(2*np.pi)*self.sigma*mu_lab)
        ee1 = np.exp(-(vmin+mu_lab)**2 / (2*self.sigma**2))
        ee2 = np.exp(-(vmin-mu_lab)**2 / (2*self.sigma**2))
        return pref*(ee1-ee2)


class Stream_From_B_Analytic(Stream_Analytic):

    def __init__(self, Mu_norm, tc, lam, sigma):
        self.Mu_gal = _solve_for_Mu(Mu_norm, tc, lam)
        self.sigma = sigma

        self.KK = np.sqrt((2*np.pi)**3)*sigma**3

        self.tc = tc
        self._tc_calc = _calc_tc(self.Mu_gal)
        self.lam = lam
        self._lam_calc = _calc_lam(self.Mu_gal)
        self.b = np.sin(self.lam)

class _Stream_Functions:
    def _f_gal(self, v):
        v_ = v[:]
        vexp = v_ - self.Mu_gal
        len_v = np.sqrt( np.sum(v_**2) )
        ee = (1/self.KK) * np.exp( -.5*np.dot(np.transpose(vexp), \
                                              np.dot(inv(self.Sigma), vexp)) )*\
             np.heaviside(self.vesc - len_v, 0)
        return ee

    def _f_lab(self, v, t):
        v_ = v[:]
        vlab = v_lab(t)
        vexp = v_ + vlab - self.Mu_gal
        len_v = np.sqrt( np.sum((v_+vlab)**2) )
        ee = (1/self.KK) * np.exp( -.5*np.dot(np.transpose(vexp), \
                                              np.dot(inv(self.Sigma), vexp)) )*\
             np.heaviside(self.vesc - len_v, 0)
        return ee

    def _f_gal_novesc(self, v):
        v_ = v[:]
        vexp = v_ - self.Mu_gal
        len_v = np.sqrt( np.sum(v_**2) )
        ee = (1/self.KK) * np.exp( -.5*np.dot( np.transpose(vexp), \
                                               np.dot(inv(self.Sigma), vexp) ) )
        return ee

    def _f_lab_novesc(self, v, t):
        v_ = v[:]
        vlab = v_lab(t)
        vexp = v_ + vlab - self.Mu_gal
        len_v = np.sqrt( np.sum((v_+vlab)**2) )
        ee = (1/self.KK) * np.exp( -.5*np.dot( np.transpose(vexp), \
                                               np.dot(inv(self.Sigma), vexp) ) )
        return ee

    def F_lab(self, v, t):
        """
        Calculates integral(f_lab dOmega)

        Note that v is NOT a 3-vector. It is the magnitude of the velocity here.
        """
        v_ = np.atleast_1d(v)
        I = np.zeros_like(v_)

        for j in range(len(v_)):
            vv = np.array([v_[j]*np.cos(self._Ph)*np.sin(self._Th),
                           v_[j]*np.sin(self._Ph)*np.sin(self._Th),
                           v_[j]*np.cos(self._Th)])

            f_vals = np.zeros(self.N_MC)
            for i in range(self.N_MC):
                f_vals[i] = v_[j]**2 * np.sin(self._Th[i]) * \
                            self.f_lab(vv[:,i], t)

            I[j] = np.sum(f_vals) * 2 * np.pi**2 / self.N_MC
        return I

    def eta(self, vmin, t):
        vmin = np.atleast_1d(vmin)

        f_vals = np.zeros(self.N_MC)
        for i in range(self.N_MC):
            f_vals[i] = self._V[i] * np.sin(self._Th[i]) * \
                        self.f_lab(self._vv[:,i], t)

        I = np.zeros_like(vmin)
        for i in range(len(vmin)):
            f_tmp = np.zeros(self.N_MC)
            grtr_than_vmin = self._V >= vmin[i]
            f_tmp[:] = f_vals[:] * grtr_than_vmin[:]
            I[i] = np.sum(f_tmp[:])

        I *= self._Volume_MC / self.N_MC
        return I

class Stream(_Stream_Functions):
    """
    Creates a class to easily organize information about a given stream from
    its mean velocity in the galactic frame (Mu_gal), its velocity dispersion
    matrix (Sigma), and the assumed escape velocity (vesc). All parameters are
    assumed to be in natural units.
    """
    def __init__(self, Mu_gal, Sigma, vesc=None, eta_path=None, N_MC=1000000):
        self.Mu_gal = Mu_gal
        self.Sigma = Sigma
        self.vesc = vesc
        self._dpath = eta_path
        self.N_MC = N_MC

        vlabmax = norm(v_lab(t_Mar21))

        if vesc is not None:
            v2 = vesc + vlabmax
        else:
            v2 = norm(self.Mu_gal) + 5*norm(self.Sigma) + vlabmax

        self._v2 = v2

        ph2 = 2*np.pi
        th2 = np.pi

        self._Volume_MC = v2*ph2*th2

        V = v2 * np.random.random_sample(size=N_MC)
        Ph = ph2 * np.random.random_sample(size=N_MC)
        Th = th2 * np.random.random_sample(size=N_MC)

        self._V = V
        self._Ph = Ph
        self._Th = Th

        self._vv = np.array([V*np.cos(Ph)*np.sin(Th),
                             V*np.sin(Ph)*np.sin(Th),
                             V*np.cos(Th)])

        def f_to_norm(v,phi,theta):
            v_vec = np.array([v*np.cos(phi)*np.sin(theta),
                              v*np.sin(phi)*np.sin(theta),
                              v*np.cos(theta)])

            vexp = v_vec - self.Mu_gal

            ee = v**2 * np.sin(theta) * \
                 np.exp( -.5*np.dot( np.transpose(vexp), \
                                     np.dot(inv(Sigma), vexp) ) )
            return ee

        if vesc is not None:
            self.KK = nquad(f_to_norm, [(0,vesc), (0,2*np.pi), (0,np.pi)])[0]
            self.f_galactic = self._f_gal
            self.f_lab = self._f_lab
        else:
            self.KK = np.sqrt((2*np.pi)**3*det(Sigma))
            self.f_galactic = self._f_gal_novesc
            self.f_lab = self._f_lab_novesc

        self.tc = _calc_tc(self.Mu_gal)
        self.lam = _calc_lam(self.Mu_gal)
        self.b = np.sin(self.lam)

        self._Mu_gal_calc = _solve_for_Mu(norm(self.Mu_gal), self.tc, self.lam)

class Stream_From_B(_Stream_Functions):
    def __init__(self, Mu_norm, tc, lam, Sigma, vesc=None, eta_path=None,
                 N_MC=1000000):
        self.Mu_gal = _solve_for_Mu(Mu_norm, tc, lam)
        self.Sigma = Sigma
        self.vesc = vesc
        self.N_MC = N_MC

        vlabmax = norm(v_lab(t_Mar21))

        if vesc is not None:
            v2 = vesc + vlabmax
        else:
            v2 = norm(self.Mu_gal) + 5*norm(self.Sigma) + vlabmax

        ph2 = 2*np.pi
        th2 = np.pi

        self._Volume_MC = v2*ph2*th2

        V = v2 * np.random.random_sample(size=N_MC)
        Ph = ph2 * np.random.random_sample(size=N_MC)
        Th = th2 * np.random.random_sample(size=N_MC)

        self._V = V
        self._Ph = Ph
        self._Th = Th

        self._vv = np.array([V*np.cos(Ph)*np.sin(Th),
                             V*np.sin(Ph)*np.sin(Th),
                             V*np.cos(Th)])

        def f_to_norm(v,phi,theta):
            v_vec = np.array([v*np.cos(phi)*np.sin(theta),
                              v*np.sin(phi)*np.sin(theta),
                              v*np.cos(theta)])

            vexp = v_vec - self.Mu_gal

            ee = v**2 * np.sin(theta) * \
                 np.exp( -.5*np.dot( np.transpose(vexp), \
                                     np.dot(inv(Sigma), vexp) ) )
            return ee

        if vesc is not None:
            self.KK = nquad(f_to_norm, [(0,vesc), (0,2*np.pi), (0,np.pi)])[0]
            self.f_galactic = self._f_gal
            self.f_lab = self._f_lab
        else:
            self.KK = np.sqrt((2*np.pi)**3*det(Sigma))
            self.f_galactic = self._f_gal_novesc
            self.f_lab = self._f_lab_novesc

        self.tc = tc
        self._tc_calc = _calc_tc(self.Mu_gal)
        self.lam = lam
        self._lam_calc = _calc_lam(self.Mu_gal)
        self.b = np.sin(self.lam)

def mu_sig_min(mX, exposure, Mus, TCs, Lams, Sigma, stream_frac,
               test_xsec=1e-37, analytic=True):
    """
    Finds the |μ| value that minimizes the significance of a chi^2 test between
    the SHM alone and the SHM with a stream at (tc, λ).

    mX : mass of dark matter [eV]
    exposure : exposure [kg y]
    Mus : array of mu values to scan over [dimensionless]
    TCs : array of critical times [days]
    Lams : array of λ, the angle between the normal to earth's rotational plane
           and μ_solar [rads]
    Sigma : matrix of velocity dispersion squared [dimensionless]
    stream_frac : fraction of DM contained in the stream
    test_xsec : assumed DM-e cross-section value [cm^2]
    """
    from time import process_time
    start = process_time()
    from pyQEdark import Crystal_DMe, ccms, hbar
    from pyQEdark.vdfparams.nat import v0new, vescnew
    from pyQEdark.stats import chisq_test

    if analytic:
        from pyQEdark.Streams import Stream_From_B_Analytic as Stream_From_B
    else:
        from pyQEdark.Streams import Stream_From_B

    TCs = np.atleast_1d(TCs)
    Lams = np.atleast_1d(Lams)

    test_xsec *= 1 / (hbar**2 * ccms**2)

    Mu_Sig_Min = np.zeros(( len(TCs), len(Lams) ))

    Stream_Si = Crystal_DMe('Si', sig_test=test_xsec, interp=False)
    SHM_Si = Crystal_DMe('Si', sig_test=test_xsec)

    for n in range(len(TCs)):
        tc = TCs[n]

        vE = norm(v_lab(tc))
        SHM_Si.set_params(vparams=[v0new, vE, vescnew])
        SHM_Rates = SHM_Si.Rate(mX, Ne_max=4)*exposure

        for m in range(len(Lams)):
            lam = Lams[m]
            Probs = np.zeros_like(Mus)

            for i in range(len(Mus)):

                if Mus[i] <= _mu_solver_min(tc, lam):
                    Probs[i] = np.nan

                else:
                    stream = Stream_From_B(Mus[i], tc, lam, Sigma)

                    Stream_Si.set_params(eta = lambda vmin: \
                                         stream.eta(vmin, stream.tc))

                    Stream_Rates_th = Stream_Si.Rate(mX, Ne_max=4)*exposure
                    Stream_Rates = (1-stream_frac)*SHM_Rates + \
                                   stream_frac*Stream_Rates_th
                    Probs[i] = chisq_test(Stream_Rates, SHM_Rates)

            Mu_Sig_Min[n,m] = Mus[np.nanargmax(Probs)]

    print(process_time()-start, 'seconds passed.')

    return np.transpose(Mu_Sig_Min)

def mu_sig_min_am(mX, exposure, Mus, TCs, Lams, Sigma, stream_frac, times,
                  test_xsec=1e-37, analytic=True):
    """
    Finds the |μ| value that minimizes the significance of a chi^2 test between
    the SHM alone and the SHM with a stream at (tc, λ). Includes effects of
    annual modulation.

    mX : mass of dark matter [eV]
    exposure : exposure [kg y]
    Mus : array of mu values to scan over [dimensionless]
    TCs : array of critical times [days]
    Lams : array of λ, the angle between the normal to earth's rotational plane
           and μ_solar [rads]
    Sigma : matrix of velocity dispersion squared [dimensionless]
    stream_frac : fraction of DM contained in the stream
    times : list of edges of time bins for a modulation analysis [days]
    test_xsec : assumed DM-e cross-section value [cm^2]
    """
    from time import process_time
    start = process_time()
    from pyQEdark import Crystal_DMe, ccms, hbar
    from pyQEdark.vdfparams.nat import v0new, vescnew
    from pyQEdark.stats import chisq_test

    if analytic:
        from pyQEdark.Streams import Stream_From_B_Analytic as Stream_From_B
    else:
        from pyQEdark.Streams import Stream_From_B

    TCs = np.atleast_1d(TCs)
    ntc = len(TCs)
    Lams = np.atleast_1d(Lams)
    nlam = len(Lams)

    test_xsec *= 1 / (hbar**2 * ccms**2)

    Ts = (times[1:] + times[:-1])/2
    nT = len(Ts)
    exposure = exposure / nT

    Mu_Sig_Min = np.zeros(( ntc, nlam ))

    Si_crys = Crystal_DMe('Si', sig_test=test_xsec, interp=False)

    SHM_Rates = np.zeros((4, nT))
    for i in range(nT):
        vE = norm(v_lab(Ts[i]))
        Si_crys.set_params(vparams=[v0new, vE, vescnew])
        SHM_Rates[:,i] = Si_crys.Rate(mX, Ne_max=4)*exposure

    for n in range(len(TCs)):
        tc = TCs[n]

        for m in range(len(Lams)):
            lam = Lams[m]
            Probs = np.zeros_like(Mus)

            for i in range(len(Mus)):

                if Mus[i] <= _mu_solver_min(tc, lam):
                    Probs[i] = np.nan

                else:
                    stream = Stream_From_B(Mus[i], tc, lam, Sigma)

                    Stream_Rates_th = np.zeros((4, nT))
                    for t_idx in range(nT):
                        Si_crys.set_params(eta = lambda vmin: \
                                           stream.eta(vmin, Ts[t_idx]))
                        Stream_Rates_th[:,t_idx] = Si_crys.Rate(mX, Ne_max=4)\
                                                   *exposure

                    Stream_Rates = (1-stream_frac)*SHM_Rates + \
                                   stream_frac*Stream_Rates_th

                    Probs[i] = chisq_test(Stream_Rates.flatten(),
                                          SHM_Rates.flatten())

            Mu_Sig_Min[n,m] = Mus[np.nanargmax(Probs)]

    print(process_time()-start, 'seconds passed.')

    return np.transpose(Mu_Sig_Min)

def stream_vs_shm_chisq(mX, exposure, Mus, TCs, Lams, Sigma, stream_fraction,
                        test_xsec=1e-37, N_test=100000, analytic=True):
    from time import process_time
    start = process_time()

    from pyQEdark import Crystal_DMe, ccms, hbar
    from pyQEdark.vdfparams.nat import v0new, vescnew
    from pyQEdark.stats import chisq_test

    if analytic:
        from pyQEdark.Streams import Stream_From_B_Analytic as Stream_From_B
    else:
        from pyQEdark.Streams import Stream_From_B

    TCs = np.atleast_1d(TCs)
    Lams = np.atleast_1d(Lams)

    test_xsec *= 1 / (hbar**2 * ccms**2)

    Stream_Si = Crystal_DMe('Si', sig_test=test_xsec, interp=False)
    SHM_Si = Crystal_DMe('Si', sig_test=test_xsec)

    Probs = np.zeros( (len(Lams),len(TCs)) )

    for n in range(len(TCs)):
        tc = TCs[n]

        vE = norm(v_lab(tc))
        SHM_Si.set_params(vparams=[v0new, vE, vescnew])
        SHM_Rates = SHM_Si.Rate(mX, Ne_max=4)*exposure

        for m in range(len(Lams)):
            lam = Lams[m]

            stream = Stream_From_B(Mus[m,n], tc, lam, Sigma)

            Stream_Si.set_params(eta = lambda vmin: \
                                 stream.eta(vmin, stream.tc))

            Stream_Rates_th = Stream_Si.Rate(mX, Ne_max=4)*exposure

            probs_temp = np.zeros(N_test)
            Stream_Rates = np.zeros((N_test,4))
            for k in range(4):
                Stream_Rates[:,k] = poisson.rvs((1-stream_fraction)*\
                                           SHM_Rates[k] + stream_fraction*\
                                           Stream_Rates_th[k],
                                           size=N_test)
            
            for k in range(N_test):
                probs_temp[k] = chisq_test(Stream_Rates[k], SHM_Rates)

            Probs[m,n] = np.mean(probs_temp)

    print(process_time()-start, 'seconds passed.')
    return Probs

def stream_vs_shm_chisq_am(mX, exposure, Mus, TCs, Lams, Sigma, stream_fraction,
                        times, test_xsec=1e-37, N_test=100000, analytic=True):
    from time import process_time
    start = process_time()

    from pyQEdark import Crystal_DMe, ccms, hbar
    from pyQEdark.vdfparams.nat import v0new, vescnew
    from pyQEdark.stats import chisq_test

    if analytic:
        from pyQEdark.Streams import Stream_From_B_Analytic as Stream_From_B
    else:
        from pyQEdark.Streams import Stream_From_B

    TCs = np.atleast_1d(TCs)
    Lams = np.atleast_1d(Lams)

    Ts = (times[1:] + times[:-1])/2
    nT = len(Ts)
    exp = exposure / nT

    xsec = test_xsec / (hbar**2 * ccms**2)

    Si_crys = Crystal_DMe('Si', sig_test=xsec, interp=False)

    SHM_Rates = np.zeros((4, nT))
    for i in range(nT):
        vE = norm(v_lab(Ts[i]))
        Si_crys.set_params(vparams=[v0new, vE, vescnew])
        SHM_Rates[:,i] = Si_crys.Rate(mX, Ne_max=4)*exp
    flatSHM = SHM_Rates.flatten()

    Probs = np.zeros( (len(Lams),len(TCs)) )

    for n in range(len(TCs)):
        tc = TCs[n]

        for m in range(len(Lams)):
            lam = Lams[m]

            stream = Stream_From_B(Mus[m,n], tc, lam, Sigma)

            Stream_Rates_th = np.zeros((4, nT))
            for t_idx in range(nT):
                Si_crys.set_params(eta = lambda vmin: \
                                     stream.eta(vmin, Ts[t_idx]))
                Stream_Rates_th[:,i] = Si_crys.Rate(mX, Ne_max=4)*exp
            flatStream_th = Stream_Rates_th.flatten()

            flatStream = np.zeros((nT*4, N_test))
            for k in range(nT*4):
                flatStream[k] = poisson.rvs((1-stream_fraction)*\
                                           flatSHM[k] + stream_fraction*\
                                           flatStream_th[k], size=N_test)

            probs_temp = np.zeros(N_test)
            for mc_idx in range(N_test):
                probs_temp[mc_idx] = chisq_test(flatStream[:,mc_idx],
                                                flatSHM[:])

            Probs[m,n] = np.mean(probs_temp)

    print(process_time()-start, 'seconds for chisq with modulation.')
    return Probs

def stream_vs_shm_ll_am(mX, exposure, Mus, TCs, Lams, Sigma, stream_fraction,
                        times, test_xsec=1e-37, N_test=100000, analytic=True):
    from time import process_time
    start = process_time()

    from pyQEdark import Crystal_DMe, ccms, hbar
    from pyQEdark.vdfparams.nat import v0new, vescnew
    from pyQEdark.stats import avg_LL_discovery

    if analytic:
        from pyQEdark.Streams import Stream_From_B_Analytic as Stream_From_B
    else:
        from pyQEdark.Streams import Stream_From_B

    TCs = np.atleast_1d(TCs)
    Lams = np.atleast_1d(Lams)

    Ts = (times[1:] + times[:-1])/2
    nT = len(Ts)
    exp = exposure / nT

    xsec = test_xsec / (hbar**2 * ccms**2)

    Si_crys = Crystal_DMe('Si', sig_test=xsec, interp=False)

    SHM_Rates = np.zeros((4, nT))
    for i in range(nT):
        vE = norm(v_lab(Ts[i]))
        Si_crys.set_params(vparams=[v0new, vE, vescnew])
        SHM_Rates[:,i] = Si_crys.Rate(mX, Ne_max=4)*exp
    flatSHM = SHM_Rates.flatten()

    Probs = np.zeros( (len(Lams), len(TCs)) )
    for i in range(len(Lams)):
        lam = Lams[i]
        for j in range(len(TCs)):
            tc = TCs[j]

            test_Stream = Stream_From_B_Analytic(Mus[i,j], tc, lam, Sigma)

            Stream_Rates = np.zeros((4, nT))
            for t_idx in range(nT):
                Si_crys.set_params(eta = lambda vmin: \
                                test_Stream.eta(vmin, Ts[t_idx]))
                Stream_Rates[:,t_idx] = Si_crys.Rate(mX, Ne_max=4)*exp
            flatStream = Stream_Rates.flatten()

            Probs[i,j] = avg_LL_discovery(stream_fraction, flatStream, 
                                          flatSHM, N_mc=N_test)

    print(process_time()-start, 'seconds for LL with modulation.')
    return Probs

Gaia_Sausage = Stream(Mu_gal_Sausage, Sigma_Sausage)
# print(Gaia_Sausage.tc, Gaia_Sausage.b, Gaia_Sausage.lam)

Nyx = Stream(Mu_gal_Nyx, Sigma_Nyx)
# print(Nyx.tc, Nyx.b, Nyx.lam)

S1 = Stream(Mu_gal_S1, Sigma_S1)
# print(S1.tc, S1.b, S1.lam)

S2a = Stream(Mu_gal_S2a, Sigma_S2a)
# print(S2a.tc, S2a.b, S2a.lam)

S2b = Stream(Mu_gal_S2b, Sigma_S2b)
# print(S2b.tc, S2b.b, S2b.lam)
