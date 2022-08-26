__all__ = ['constants', 'crystaldme', 'dmvdf', 'etas', 'plotting',
           'stats', 'vdfs', 'vdfparams']

import numpy as np

import io
import pkgutil

from pyQEdark import *
from pyQEdark.constants import *
from pyQEdark.dmvdf import DM_Halo
from pyQEdark.crystaldme import Crystal_DMe

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

def mu_xe(mx):
    """
    reduced mass of dark matter and electron system
    """
    return (m_e*mx)/(m_e + mx)

class Crystal():
    
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
        _data = np.transpose(np.resize(np.loadtxt(tmpdata),(500,900)))
        self.data = _Epref / (self.dE * self.dq) * _wk / 4 * _data

        self.nE = len(self.data[0,:])
        self.nq = len(self.data[:,0])

def dR_dEe(Ee, mX, sigma_e, CrysObj, rho_X, eta_fn, FDMn):
    q_ = np.arange(1, CrysObj.nq+1)*CrysObj.dq
    Ncell = 1 / CrysObj.mcell
    _rho_X = rho_X * 1e15 * c_light**3 * hbar**3
    _sigma_e = sigma_e / (hbar**2 * ccms**2)
    _etas = eta_fn(vmin(q_, Ee, mX))
    prefactor = _rho_X / mX * Ncell * _sigma_e * alpha *\
                            m_e**2 / mu_xe(mX)**2
    _y = 1/q_**2 * _etas * FDM(q_, FDMn)**2 * CrysObj.data[:,Ei]
    return prefactor*np.trapz(_y, x=q_)

def rate_Ne(Ne, mX, sigma_e, CrysObj, rho_X, eta_fn, FDMn):
    binsize = CrysObj.dE_bin
    a = int( ( (Ne-1)*binsize + CrysObj.Egap ) / CrysObj.dE )
    b = int( (Ne*binsize + CrysObj.Egap) / CrysObj.dE )
    I_ = np.arange(a,b)

