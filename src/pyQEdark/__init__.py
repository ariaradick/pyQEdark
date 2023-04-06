__all__ = ['constants', 'crystaldme', 'dmvdf', 'etas', 'plotting',
           'stats', 'vdfs', 'vdfparams']

from dataclasses import dataclass

import numpy as np
from numpy import loadtxt

from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad, cumulative_trapezoid

import io
import pkgutil

from pyQEdark import *
from pyQEdark.constants import *
from pyQEdark.veldist import VelDistSHM
from pyQEdark.etas import etaSHM
from pyQEdark.vdfparams.nat import v0new, vEfid, vescnew

def FDM(q, n):
        """
        Dark matter - electron scattering form factor
        """
        return (alpha * m_e / q)**n

def vmin(q, Ee, mx):
    """
    Threshold velocity for detection.
    """
    return q/(2*mx) + Ee/q

def dvmin_dmx(q, mx):
        return -q/(2*mx**2)

def mu_xe(mx):
    """
    reduced mass of dark matter and electron system
    """
    return (m_e*mx)/(m_e + mx)

@dataclass
class Crystal:
    def __init__(self, material):

        if material == 'Si':
            self.dq = .02*alpha*m_e
            self.dE = .1
            self.mcell = 52.33e9
            self.density = 2328*hbar**3*c_light**5/evtoj
            self.Egap = 1.2
            self.dE_bin = 3.8
            self.datapath = 'QEdark_data/Si_f2.txt'

        elif material == 'Ge':
            self.dq = .02*alpha*m_e
            self.dE = .1
            self.mcell = 135.33e9
            self.density = 5323*hbar**3*c_light**5/evtoj
            self.Egap = 0.7
            self.dE_bin = 2.9
            self.datapath = 'QEdark_data/Ge_f2.txt'
        
        _vcell = self.mcell/self.density
        _Epref = 2*np.pi**2/(alpha*m_e**2*_vcell)
        _wk = 2/137
        tmpdata = io.BytesIO(pkgutil.get_data(__name__, self.datapath))
        _data = np.transpose(np.resize(loadtxt(tmpdata),(500,900)))
        self.data = _Epref / (self.dE * self.dq) * _wk / 4 * _data

        self.nE = len(self.data[0,:])
        self.nq = len(self.data[:,0])

        self.q_list = np.arange(1,self.nq+1)*self.dq
        self.E_list = np.arange(1,self.nE+1)*self.dE

        # self.f2crys = RectBivariateSpline(self.q_list, self.E_list, self.data)

@dataclass
class ExptSetup:
    def __init__(self, 
                 Material = Crystal('Si'),
                 vparams = [v0new, vEfid, vescnew],
                 eta_fn = etaSHM,
                 rho_x = .4,
                 sig_e = 1e-37,
                 FDMn = 0
                 ):

        self.mat = Material
        self.rho_x = rho_x * 1e15 * c_light**3 * hbar**3
        self.sig_e = sig_e / (hbar**2 * ccms**2)
        self.FDMn = FDMn

def _dR_dEi_dqi(mx, t, Ei, qi, setup=ExptSetup(), veldist=VelDistSHM()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    q = setup.mat.dq*(qi+1)
    Ee = setup.mat.dE*(Ei+1)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2

    y = 1/q * veldist.eta(vmin(q,Ee,mx), t) * \
        FDM(q, setup.FDMn)**2 * \
        setup.mat.data[qi, Ei]

    return prefactor * y * corr

def _dR_dEi(mx, t, Ei, setup=ExptSetup(), veldist=VelDistSHM()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    Ee = setup.mat.dE*(Ei+1)
    qs = setup.mat.q_list
    q_is = np.arange(setup.mat.nq)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2

    y = 1/qs * veldist.eta(vmin(qs,Ee,mx), t) * \
        FDM(qs, setup.FDMn)**2 * \
        setup.mat.data[q_is, Ei]

    return np.trapz(prefactor * y * corr, dx=setup.mat.dq)

def RateNe(mx, t, ne, setup=ExptSetup(), veldist=VelDistSHM()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    binsize = setup.mat.dE_bin
    a = np.rint( ( (ne-1)*binsize + setup.mat.Egap ) / setup.mat.dE - 1 \
                ).astype(int)
    b = np.rint( ( ne*binsize + setup.mat.Egap ) / setup.mat.dE ).astype(int)
    E_idx = np.arange(a,b)

    qs = setup.mat.q_list
    Es = setup.mat.E_list[a:b]
    Ees, Qs = np.meshgrid(Es, qs)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2
    
    y = 1/Qs * veldist.eta(vmin(Qs,Ees,mx), t) * \
        FDM(Qs, setup.FDMn)**2 * \
        setup.mat.data[:, a:b]

    drde = np.trapz(y, dx=setup.mat.dq, axis=0)
    ratene = np.trapz(drde, dx=setup.mat.dE)
    return ratene*prefactor*corr

def _dRNe_dmx(mx, t, ne, setup=ExptSetup(), veldist=VelDistSHM()):
    term1 = -(3*m_e+mx)/((m_e+mx)*mx) * \
            RateNe(mx, t, ne, setup=setup, veldist=veldist)
    
    corr = c_light**2*sec2year / (hbar*evtoj)

    binsize = setup.mat.dE_bin
    a = np.rint( ( (ne-1)*binsize + setup.mat.Egap ) / setup.mat.dE - 1 \
                ).astype(int)
    b = np.rint( ( ne*binsize + setup.mat.Egap ) / setup.mat.dE ).astype(int)
    E_idx = np.arange(a,b)

    qs = setup.mat.q_list
    Es = setup.mat.E_list[a:b]
    Ees, Qs = np.meshgrid(Es, qs)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2
    
    y = 1/Qs * veldist.deta_dvmin(vmin(Qs,Ees,mx), t) * dvmin_dmx(Qs,mx) * \
        FDM(Qs, setup.FDMn)**2 * \
        setup.mat.data[:, a:b]

    drde = np.trapz(y, dx=setup.mat.dq, axis=0)
    term2 = np.trapz(drde, dx=setup.mat.dE)*prefactor*corr

    return term1+term2

def _dRNe_dmugal(mx, t, ne, muidx, setup=ExptSetup(), veldist=VelDistSHM()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    binsize = setup.mat.dE_bin
    a = np.rint( ( (ne-1)*binsize + setup.mat.Egap ) / setup.mat.dE - 1 \
                ).astype(int)
    b = np.rint( ( ne*binsize + setup.mat.Egap ) / setup.mat.dE ).astype(int)
    E_idx = np.arange(a,b)

    qs = setup.mat.q_list
    Es = setup.mat.E_list[a:b]
    Ees, Qs = np.meshgrid(Es, qs)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2
    
    y = 1/Qs * veldist.deta_dmugal(vmin(Qs,Ees,mx), t, muidx) * \
        FDM(Qs, setup.FDMn)**2 * \
        setup.mat.data[:, a:b]

    drde = np.trapz(y, dx=setup.mat.dq, axis=0)
    ratene = np.trapz(drde, dx=setup.mat.dE)
    return ratene*prefactor*corr

def _dRNe_dmusol(mx, t, ne, setup=ExptSetup(), veldist=VelDistSHM()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    binsize = setup.mat.dE_bin
    a = np.rint( ( (ne-1)*binsize + setup.mat.Egap ) / setup.mat.dE - 1 \
                ).astype(int)
    b = np.rint( ( ne*binsize + setup.mat.Egap ) / setup.mat.dE ).astype(int)
    E_idx = np.arange(a,b)

    qs = setup.mat.q_list
    Es = setup.mat.E_list[a:b]
    Ees, Qs = np.meshgrid(Es, qs)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2
    
    y = 1/Qs * veldist.deta_dmusol(vmin(Qs,Ees,mx), t) * \
        FDM(Qs, setup.FDMn)**2 * \
        setup.mat.data[:, a:b]

    drde = np.trapz(y, dx=setup.mat.dq, axis=0)
    ratene = np.trapz(drde, dx=setup.mat.dE)
    return ratene*prefactor*corr

def _dRNe_dtc(mx, t, ne, setup=ExptSetup(), veldist=VelDistSHM()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    binsize = setup.mat.dE_bin
    a = np.rint( ( (ne-1)*binsize + setup.mat.Egap ) / setup.mat.dE - 1 \
                ).astype(int)
    b = np.rint( ( ne*binsize + setup.mat.Egap ) / setup.mat.dE ).astype(int)
    E_idx = np.arange(a,b)

    qs = setup.mat.q_list
    Es = setup.mat.E_list[a:b]
    Ees, Qs = np.meshgrid(Es, qs)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2
    
    y = 1/Qs * veldist.deta_dtc(vmin(Qs,Ees,mx), t) * \
        FDM(Qs, setup.FDMn)**2 * \
        setup.mat.data[:, a:b]

    drde = np.trapz(y, dx=setup.mat.dq, axis=0)
    ratene = np.trapz(drde, dx=setup.mat.dE)
    return ratene*prefactor*corr

def _dRNe_db(mx, t, ne, setup=ExptSetup(), veldist=VelDistSHM()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    binsize = setup.mat.dE_bin
    a = np.rint( ( (ne-1)*binsize + setup.mat.Egap ) / setup.mat.dE - 1 \
                ).astype(int)
    b = np.rint( ( ne*binsize + setup.mat.Egap ) / setup.mat.dE ).astype(int)
    E_idx = np.arange(a,b)

    qs = setup.mat.q_list
    Es = setup.mat.E_list[a:b]
    Ees, Qs = np.meshgrid(Es, qs)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2
    
    y = 1/Qs * veldist.deta_db(vmin(Qs,Ees,mx), t) * \
        FDM(Qs, setup.FDMn)**2 * \
        setup.mat.data[:, a:b]

    drde = np.trapz(y, dx=setup.mat.dq, axis=0)
    ratene = np.trapz(drde, dx=setup.mat.dE)
    return ratene*prefactor*corr

def Rate_ct(mx, t, Ne=[1,2,3,4], setup=ExptSetup(), veldist=VelDistSHM()):
    """
    Calculates the rate, binned in number of electrons, for a given experimental
    setup
    mx : mass of dark matter (eV)
    Ne : range of electron bins of interest (np.arange, range, ...) should
         be ordered and with step sizes 1 (e.g., [1,2,3,4])
    setup : instance of ExptSetup
    """
    corr = c_light**2*sec2year / (hbar*evtoj)

    nearr = np.insert(np.array(Ne), 0, 0)

    ne_binsize = setup.mat.dE_bin
    E_idx_oi = np.rint( (nearr*ne_binsize + \
                         setup.mat.Egap) / setup.mat.dE - 1 ).astype(int)

    a = E_idx_oi[0]
    b = E_idx_oi[-1] + 1
    E_idx = np.arange(a,b)

    qs = setup.mat.q_list
    Es = setup.mat.E_list[a:b]
    Ees, Qs = np.meshgrid(Es, qs)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2
    
    y = 1/Qs * veldist.eta(vmin(Qs,Ees,mx), t) * \
        FDM(Qs, setup.FDMn)**2 * \
        setup.mat.data[:, a:b]

    drde = np.trapz(y, dx=setup.mat.dq, axis=0)
    cumulative_rate = cumulative_trapezoid(drde, dx=setup.mat.dE, initial=0.)

    ratene = cumulative_rate[E_idx_oi[1:]-E_idx_oi[0]] - \
             cumulative_rate[E_idx_oi[:-1]-E_idx_oi[0]]
    return ratene*prefactor*corr


def Rate(mx, t, Ne=[1,2,3,4], setup=ExptSetup(), veldist=VelDistSHM()):
    return np.array([RateNe(mx, t, n, setup=setup, veldist=veldist) for n in Ne])

def dR_dsige(mx, t, Ne=[1,2,3,4], setup=ExptSetup(), veldist=VelDistSHM()):
    return Rate(mx, t, Ne=Ne, setup=setup, veldist=veldist) / setup.sig_e

def dR_dmx(mx, t, Ne=[1,2,3,4], setup=ExptSetup(), veldist=VelDistSHM()):
    return np.array([_dRNe_dmx(mx, t, n, setup=setup, veldist=veldist) for n in Ne])

def dR_dmugal(mx, t, muidx, Ne=[1,2,3,4], setup=ExptSetup(), 
              veldist=VelDistSHM()):
    return np.array([_dRNe_dmugal(mx, t, n, muidx, setup=setup, 
                                  veldist=veldist) for n in Ne])

def dR_dmusol(mx, t, Ne=[1,2,3,4], setup=ExptSetup(), veldist=VelDistSHM()):
    return np.array([_dRNe_dmusol(mx, t, n, setup=setup, 
            veldist=veldist) for n in Ne])

def dR_dtc(mx, t, Ne=[1,2,3,4], setup=ExptSetup(), veldist=VelDistSHM()):
    return np.array([_dRNe_dtc(mx, t, n, setup=setup, 
            veldist=veldist) for n in Ne])

def dR_db(mx, t, Ne=[1,2,3,4], setup=ExptSetup(), veldist=VelDistSHM()):
    return np.array([_dRNe_db(mx, t, n, setup=setup, 
            veldist=veldist) for n in Ne])