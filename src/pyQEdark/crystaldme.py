"""
Analyzes the output of QEDark, giving rates for different materials

author: Aria Radick
date created: 11/25/19
"""

import numpy as np
import io
import pkgutil

from pyQEdark.constants import *
from pyQEdark.dmvdf import DM_VDF

class Crystal_DMe(DM_VDF):

    """
    material is a string that represents the given material, presently only
    'Si' and 'Ge' are incorporated.

    keyword arguments can be:

    rho_x : the local dark matter density [eV^4]

    sig_test : the test total cross-section for dRdE plots [eV^-2]

    FDMn : the power of momentum scaling in the dark matter - electron
           scattering form factor (dimensionless)

    vdf : a keyword string that sets the velocity distribution function and
          determines what your vparams should be:
          'shm' : [v0, vE, vesc]
          'tsa' : [v0, vE, vesc]
          'msw' : [v0, vE, vesc, p]

    vparams : list of parameters corresponding to the chosen vdf [nat. units]
    """

    def __init__(self, material, **kwargs):

        self._setup_mat(material)

        self.allowed_keys = {'rho_x', 'sig_test', 'FDMn', 'interp'}

        self.in_vcorr = 1
        self.out_vcorr = 1
        self.vcorr = 1

        defaults = { 'rho_x' : .4e15 * c_light**3 * hbar**3,
                     'sig_test' : 1e-41 / (hbar**2 * c_light**2),
                     'FDMn' : 0,
                     'vdf' : 'shm',
                     'vparams' : np.array([220, 232, 544]) / ckms,
                     'interp' : True }

        self.set_params(**defaults)
        self.set_params(**kwargs)

    def _setup_mat(self, material):

        SiDict = { 'dq'       : .02*alpha*m_e,
                   'dE'       : .1,
                   'mcell'    : 52.33e9,
                   'density'  : 2328*hbar**3*c_light**5/evtoj,
                   'Egap'     : 1.2,
                   'dE_bin'   : 3.8,
                   'datapath' : 'QEdark_data/Si_f2.txt' }

        GeDict = { 'dq'       : .02*alpha*m_e,
                   'dE'       : .1,
                   'mcell'    : 135.33e9,
                   'density'  : 5323*hbar**3*c_light**5/evtoj,
                   'Egap'     : 0.7,
                   'dE_bin'   : 2.9,
                   'datapath' : 'QEdark_data/Ge_f2.txt' }

        self.material = material

        materials = { 'Si': SiDict,
                      'Ge': GeDict }
        matdict = materials[material]
        self.__dict__.update( (k,v) for k,v in matdict.items() )

        _vcell = self.mcell/self.density
        _Epref = 2*np.pi**2/(alpha*m_e**2*_vcell)
        _wk = 2/137
        tmpdata = io.BytesIO(pkgutil.get_data(__name__, self.datapath))
        _data = np.transpose(np.resize(np.loadtxt(tmpdata),(500,900)))
        self.data = _Epref / (self.dE * self.dq) * _wk / 4 * _data

        self.nE = len(self.data[0,:])
        self.nq = len(self.data[:,0])

    def vmin(self, _q, _Ee, _mx):
        """
        Threshold velocity for detection.
        """
        return _q/(2*_mx) + _Ee/_q

    def FDM(self, _q):
        """
        Dark matter - electron scattering form factor
        """
        return (alpha * m_e / _q)**self.FDMn

    def mu_xe(self, _mx):
        """
        reduced mass of dark matter and electron system
        """
        return (m_e*_mx)/(m_e + _mx)

    def get_evals(self, Ne_min=1, Ne_max=12):
        binsize = self.dE_bin
        firstE = self.Egap + (Ne_min-1)*binsize
        finalE = int( (Ne_max*binsize + self.Egap) )
        evals = np.arange(firstE, finalE+binsize, binsize)
        return evals

    def EtaGrid(self, _mx):
        """
        Calculates eta at each E and q value in our grid.
        """
        q_list = np.arange(1,self.nq+1)*self.dq
        E_list = np.arange(1,self.nE+1)*self.dE

        E, Q = np.meshgrid(E_list, q_list)
        Vmin = self.vmin(Q, E, _mx)

        return self.eta(Vmin)

    def _rateNe(self, mx, ne, etas_):
        binsize = self.dE_bin
        a = int( ( (ne-1)*binsize + self.Egap ) / self.dE )
        b = int( (ne*binsize + self.Egap) / self.dE )
        I_ = np.arange(a,b)

        qunit = self.dq
        q_ = np.arange(1, self.nq+1)*qunit
        Q = np.ones( (len(q_), len(I_)) )
        for i in range(len(I_)):
            Q[:,i] = q_[:]

        _Ncell = 1 / self.mcell
        prefactor = self.rho_x / mx * _Ncell * self.sig_test * alpha *\
                    m_e**2 / self.mu_xe(mx)**2

        dRdE_ = np.sum(qunit / Q * etas_[:,a:b] * \
                       self.FDM(Q)**2 * self.data[:,a:b])*prefactor*self.dE

        return dRdE_

    def Rate(self, mX, xsec=None, binned=True, out_unit='kgy', **kwargs):

        self.set_params(**kwargs)

        if xsec is None:
            xsec_corr = 1
        else:
            xsec_corr = xsec / self.sig_test

        if out_unit == 'nat':
            corr = 1
        elif out_unit == 'kgy':
            corr = c_light**2*sec2year / (hbar*evtoj)

        mX = np.atleast_1d(mX)
        mX_MeV = np.around(mX*1e-6, decimals=6)
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
                return rates[0]*corr*xsec_corr
            else:
                return rates*corr*xsec_corr

        else:
            if N_mX == 1:
                return np.sum(rates[0])*corr*xsec_corr
            else:
                return np.sum(rates, axis=1)*corr*xsec_corr

    def sig_min(self, *args, N_event=3, out_unit='cm', **kwargs):
        self.set_params(**kwargs)
        corr = c_light**2*sec2year / (hbar*evtoj)
        corr_dict = {'m' : 1, 'cm' : 1e4}

        if len(args)==3:
            mX, exposure, Ne = args
            Ne = np.atleast_1d(Ne)
            N_Ne = len(Ne)
        elif len(args)==2:
            mX, exposure = args
        else:
            raise TypeError('Wrong number of arguments.')

        mX = np.atleast_1d(mX)
        N_mX = len(mX)
        sigtest_m = self.sig_test * c_light**2 * hbar**2

        if len(args)==3:
            output_ = N_event * sigtest_m / ( self.Rate(mX, Ne=Ne) * \
                      exposure )
            output_ = np.swapaxes(output_, 0, 1)
            if N_Ne==1:
                return output_[0]*corr_dict[out_unit]
            else:
                return output_*corr_dict[out_unit]

        elif len(args)==2:
            output_ = N_event * sigtest_m / ( self.Rate(mX, binned=False) * \
                      exposure )
            return output_*corr_dict[out_unit]
