__all__ = ['constants', 'crystaldme', 'dmvdf', 'etas', 'plotting',
           'stats', 'vdfs', 'vdfparams']

from dataclasses import dataclass

import numpy as np

from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad, cumulative_trapezoid

import io
import pkgutil

from pyQEdark import *
from pyQEdark.constants import *
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
        _data = np.transpose(np.resize(np.loadtxt(tmpdata),(500,900)))
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
        self.vparams = vparams
        self.eta = lambda vmin: eta_fn(vmin, self.vparams)

def _dR_dEdq_interp(mx, setup=ExptSetup()):
    q_list = np.arange(1,self.nq+1)*self.dq
    E_list = np.arange(1,self.nE+1)*self.dE

    E, Q = np.meshgrid(E_list, q_list)
    Vmin = vmin(Q, E, mx)

    EtaGrid = setup.eta(Vmin)
    
    corr = c_light**2*sec2year / (hbar*evtoj)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2

    y = 1/q * setup.eta(vmin(q,Ee,mx)) * \
        FDM(q, setup.FDMn)**2 * \
        setup.mat.f2crys(q,Ee)

    return prefactor * y * corr

def _dR_dEi_dqi(mx, Ei, qi, setup=ExptSetup()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    q = setup.mat.dq*(qi+1)
    Ee = setup.mat.dE*(Ei+1)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2

    y = 1/q * setup.eta(vmin(q,Ee,mx)) * \
        FDM(q, setup.FDMn)**2 * \
        setup.mat.data[qi, Ei]

    return prefactor * y * corr

def _dR_dEi(mx, Ei, setup=ExptSetup()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    Ee = setup.mat.dE*(Ei+1)
    qs = setup.mat.q_list
    q_is = np.arange(setup.mat.nq)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2

    y = 1/qs * setup.eta(vmin(qs,Ee,mx)) * \
        FDM(qs, setup.FDMn)**2 * \
        setup.mat.data[q_is, Ei]

    return np.trapz(prefactor * y * corr, dx=setup.mat.dq)

def RateNe(mx, ne, setup=ExptSetup()):
    corr = c_light**2*sec2year / (hbar*evtoj)

    binsize = setup.mat.dE_bin
    a = np.rint( ( (ne-1)*binsize + setup.mat.Egap ) / setup.mat.dE - 1 \
                ).astype(int)
    b = np.rint( ( ne*binsize + setup.mat.Egap ) / setup.mat.dE - 1 \
                ).astype(int)
    E_idx = np.arange(a,b)

    qs = setup.mat.q_list
    Es = setup.mat.E_list[a:b]
    Ees, Qs = np.meshgrid(Es, qs)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2
    
    y = 1/Qs * setup.eta(vmin(Qs,Ees,mx)) * \
        FDM(Qs, setup.FDMn)**2 * \
        setup.mat.data[:, a:b]

    drde = np.trapz(y, dx=setup.mat.dq, axis=0)
    ratene = np.trapz(drde, dx=setup.mat.dE)
    return ratene*prefactor*corr

def Rate_cumulative_trapezoid(mx, Ne=[1,2,3,4], setup=ExptSetup()):
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
    b = E_idx_oi[-1]
    E_idx = np.arange(a,b)

    qs = setup.mat.q_list
    Es = setup.mat.E_list[a:b]
    Ees, Qs = np.meshgrid(Es, qs)

    Ncell = 1 / setup.mat.mcell
    prefactor = setup.rho_x / mx * Ncell * setup.sig_e * alpha *\
                m_e**2 / mu_xe(mx)**2
    
    y = 1/Qs * setup.eta(vmin(Qs,Ees,mx)) * \
        FDM(Qs, setup.FDMn)**2 * \
        setup.mat.data[:, a:b]

    drde = np.trapz(y, dx=setup.mat.dq, axis=0)
    cumulative_rate = cumulative_trapezoid(drde, dx=setup.mat.dE, initial=0.)

    ratene = cumulative_rate[E_idx_oi[1:]-E_idx_oi[0]-1] - \
             cumulative_rate[E_idx_oi[:-1]-E_idx_oi[0]]
    return ratene*prefactor*corr


def Rate(mx, Ne=[1,2,3,4], setup=ExptSetup()):
    return np.array([RateNe(mx, n, setup) for n in Ne])

class Crystal_DMe:

    """
    material : an instance of "Crystal"

    rho_x : the local dark matter density [GeV / cm^3]

    sig_test : the test total cross-section for dRdE plots [cm^2]

    FDMn : the power of momentum scaling in the dark matter - electron
           scattering form factor [dimensionless]

    eta_fn(vmin, vparams) : <1/v> wrt vdf [dimensionless]

    vparams : list of parameters corresponding to the chosen vdf [dim'less]
    """

    def __init__(self, 
                 MaterialObj,
                 vparams = [v0new, vEfid, vescnew],
                 eta_fn = etaSHM,
                 rho_x = .46,
                 sig_e = 1e-37,
                 FDMn = 0
                 ):

        self.mat = MaterialObj
        self.rho_x = rho_x * 1e15 * c_light**3 * hbar**3
        self.sig_e = sig_e / (hbar**2 * ccms**2)
        self.FDMn = FDMn
        self.vparams = vparams
        self.eta = lambda vmin: eta_fn(vmin, self.vparams)

    def get_evals(self, Ne_min=1, Ne_max=12):
        Ne_list = np.arange(Ne_min-1, Ne_max+1)
        binsize = self.mat.dE_bin
        evals = self.mat.Egap + binsize*Ne_list
        return evals

    def EtaGrid(self, _mx):
        """
        Calculates eta at each E and q value in our grid.
        """
        q_list = np.arange(1,self.mat.nq+1)*self.mat.dq
        E_list = np.arange(1,self.mat.nE+1)*self.mat.dE

        E, Q = np.meshgrid(E_list, q_list)
        Vmin = vmin(Q, E, _mx)

        return self.eta(Vmin)

    def _rateNe(self, mx, ne, etas_):
        binsize = self.mat.dE_bin
        a = int( ( (ne-1)*binsize + self.mat.Egap ) / self.mat.dE )
        b = int( (ne*binsize + self.mat.Egap) / self.mat.dE )
        I_ = np.arange(a,b)

        qunit = self.mat.dq
        q_ = np.arange(1, self.mat.nq+1)*qunit
        Q = np.ones( (len(q_), len(I_)) )
        for i in range(len(I_)):
            Q[:,i] = q_[:]

        _Ncell = 1 / self.mat.mcell
        prefactor = self.rho_x / mx * _Ncell * self.sig_e * alpha *\
                    m_e**2 / mu_xe(mx)**2

        dRdE_ = np.sum(qunit / Q * etas_[:,a:b] * \
                       FDM(Q, self.FDMn)**2 * self.mat.data[:,a:b])*\
                       prefactor * self.mat.dE

        return dRdE_

    def Rate(self, mX, binned=True, **kwargs):

        corr = c_light**2*sec2year / (hbar*evtoj)

        mX = np.atleast_1d(mX)
        N_mX = len(mX)

        if binned:
            if 'Ne' in kwargs.keys():
                Ne_list = np.atleast_1d(kwargs['Ne'])
            else:
                if 'Ne_min' in kwargs.keys():
                    Ne_min = kwargs['Ne_min']
                else:
                    Ne_min = 1

                if 'Ne_max' in kwargs.keys():
                    Ne_max = kwargs['Ne_max']
                else:
                    Ne_max = 12
                Ne_list = np.arange(Ne_min, Ne_max+1)

        else:
            Ne_list = np.arange(1,13)

        N_Ne = len(Ne_list)

        rates = np.zeros( (N_mX, N_Ne) )

        for i in range(N_mX):
            etas = self.EtaGrid(mX[i])
            for j in range(N_Ne):
                rates[i,j] = self._rateNe(mX[i], Ne_list[j], etas)

        if binned:
            if N_mX == 1:
                return rates[0]*corr
            else:
                return rates*corr

        else:
            if N_mX == 1:
                return np.sum(rates[0])*corr
            else:
                return np.sum(rates, axis=1)*corr

    def sig_min(self, *args, N_event=3):
        corr = c_light**2*sec2year / (hbar*evtoj)

        if len(args)==3:
            mX, exposure, Ne = args
            Ne = np.atleast_1d(Ne)
            N_Ne = len(Ne)
        elif len(args)==2:
            mX, exposure = args
        else:Ne_max=Ne_max
        N_mX = len(mX)
        sigtest_m = self.sig_e * c_light**2 * hbar**2

        if len(args)==3:
            output_ = N_event * sigtest_m / ( self.Rate(mX, Ne=Ne) * \
                      exposure )
            output_ = np.swapaxes(output_, 0, 1)
            if N_Ne==1:
                return output_[0]*1e4
            else:
                return output_*1e4

        elif len(args)==2:
            output_ = N_event * sigtest_m / ( self.Rate(mX, binned=False) * \
                      exposure )
            return output_*1e4