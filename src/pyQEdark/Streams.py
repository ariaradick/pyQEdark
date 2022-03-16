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
    mu_gal : mean stream velocity in the galactic frame (nat. units)
    """
    return norm(Mu_gal - v_sun - vE_fn_solar(t))

def _calc_tc(Mu_gal):
    """
    Calculates the critical time t_c that maximizes mu_lab
    mu_gal : mean stream velocity in the galactic frame (nat. units)
    """
    Mu_sol_hat = (Mu_gal - v_sun) / norm(Mu_gal - v_sun)

    b = np.array([np.dot(Mu_sol_hat, eps_hat[i]) for i in range(3)])

    tc1 = (1/omega)*np.arctan2(b[1],b[0]) + t_Mar21
    tc2 = tc1 + 365.25/2

    if Mu_lab(tc1,Mu_gal) > Mu_lab(tc2,Mu_gal):
        return tc1
    else:
        return tc2

def _calc_lam(Mu_gal):
    """
    Calculates the angle λ between mu in the solar frame and the normal to
    Earth's rotational plane.
    mu_gal : mean stream velocity in the galactic frame (nat. units)
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
                b3**2*(-vsa**2 + vsc**2) + Mu_norm**2) * np.abs(b1)

    mu_sol_p = Z/b1 - b1*vsa - b2*vsb - b3*vsc
    mu_sol_m = -Z/b1 - b1*vsa - b2*vsb - b3*vsc

    if mu_sol_p < mu_sol_m:
        Z = -Z

    mu1 = (1 - b1**2)*vsa - b1*b2*vsb - b1*b3*vsc + Z
    mu2 = (b2**3*vsa+b1*vsb-b1*b2**2*vsb+b2*((-1+b3**2)*vsa-b1*b3*vsc+Z))/b1
    mu3 = (b3**3*vsa+b1*vsc-b1*b3**2*vsc+b3*((-1+b2**2)*vsa-b1*b2*vsb+Z))/b1

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

    epsilon_3 = np.cross(epsilon_1, epsilon_2)
    eps_hat = np.array([epsilon_1/norm(epsilon_1), epsilon_2/norm(epsilon_2),
                        epsilon_3/norm(epsilon_3)])

    vs_eps = np.dot(eps_hat, v_sun)

    b1 = b*np.cos(theta)
    b2 = b*np.sin(theta)
    b3 = np.cos(lam)

    vsa,vsb,vsc = vs_eps

    under_sqrt = -(-vsb**2 + b2**2*(-vsa**2 + vsb**2) + 2*b1*b3*vsa*vsc - \
                 vsc**2 + 2*b2*vsb*(b1*vsa + b3*vsc) + b3**2*(-vsa**2 + vsc**2))

    return np.sqrt(under_sqrt)

class _Stream_Functions:
    def _f_gal(self, v):
        v_ = v[:]
        vexp = v_ - self.Mu_gal
        len_v = np.sqrt( np.sum(v_**2) )
        ee = (1/self.KK) * np.exp( -.5*np.dot( np.transpose(vexp), \
                                               np.dot(inv(Sigma), vexp) ) )*\
             np.heaviside(self.vesc - len_v, 0)
        return ee

    def _f_lab(self, v, t):
        v_ = v[:]
        vlab = v_lab(t)
        vexp = v_ + vlab - self.Mu_gal
        len_v = np.sqrt( np.sum((v_+vlab)**2) )
        ee = (1/self.KK) * np.exp( -.5*np.dot( np.transpose(vexp), \
                                               np.dot(inv(Sigma), vexp) ) )*\
             np.heaviside(self.vesc - len_v, 0)
        return ee

    def _f_gal_novesc(self, v):
        v_ = v[:]
        vexp = v_ - self.Mu_gal
        len_v = np.sqrt( np.sum(v_**2) )
        ee = (1/self.KK) * np.exp( -.5*np.dot( np.transpose(vexp), \
                                               np.dot(inv(Sigma), vexp) ) )
        return ee

    def _f_lab_novesc(self, v, t):
        v_ = v[:]
        vlab = v_lab(t)
        vexp = v_ + vlab - self.Mu_gal
        len_v = np.sqrt( np.sum((v_+vlab)**2) )
        ee = (1/self.KK) * np.exp( -.5*np.dot( np.transpose(vexp), \
                                               np.dot(inv(Sigma), vexp) ) )
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
    def __init__(self, Mu_gal, Sigma, vesc=None, N_MC=1000000):
        self.Mu_gal = Mu_gal
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

        self.tc = _calc_tc(self.Mu_gal)
        self.lam = _calc_lam(self.Mu_gal)
        self.b = np.sin(self.lam)

class Stream_From_B(_Stream_Functions):
    def __init__(self, Mu_norm, tc, lam, Sigma, vesc=None, N_MC=1000000):
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
        self.lam = lam
        self.b = np.sin(self.lam)
