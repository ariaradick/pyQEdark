__all__ = ['constants', 'crystaldme', 'dmvdf', 'etas', 'plotting',
           'stats', 'vdfs', 'vdfparams']

import numpy as np

import io
import pkgutil

from pyQEdark import *
from pyQEdark.constants import *
from pyQEdark.dmvdf import DM_Halo
from pyQEdark.crystaldme import Crystal_DMe

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