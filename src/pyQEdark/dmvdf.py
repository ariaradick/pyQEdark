"""
Basic class for describing a velocity distribution function.

author: Aria Radick
date: 7/19/21
"""

import numpy as np
from scipy.interpolate import interp1d
from pyQEdark.constants import ckms, ccms, c_light

class DM_VDF:

    def __init__(self, in_unit='kms', out_unit='kms', **kwargs):

        self.allowed_keys = {'interp'}

        corr_dict = { 'kms' : ckms,
                      'cms' : ccms,
                      'ms'  : c_light,
                      'nat' : 1 }

        self.in_vcorr = corr_dict[in_unit]
        self.out_vcorr = corr_dict[out_unit]
        self.vcorr = corr_dict[out_unit] / corr_dict[in_unit]

        self.interp = True

        self.set_params(**kwargs)

    def _setup(self, **kwargs):
        from pyQEdark.vdfs import f_SHM, f_Tsa, f_MSW
        from pyQEdark.etas import etaSHM, etaTsa, etaMSW, etaFromVDF

        name_dict = { 'shm' : 'Standard Halo Model',
                      'tsa' : 'Tsallis Model',
                      'msw' : 'Empirical Model' }

        f_dict = { 'shm' : f_SHM,
                   'tsa' : f_Tsa,
                   'msw' : f_MSW }

        eta_dict = { 'shm' : etaSHM,
                     'tsa' : etaTsa,
                     'msw' : etaMSW }

        if 'vdf' in kwargs.keys():

            if isinstance(kwargs['vdf'], str):
                vdf = kwargs['vdf']
                self.vdf = vdf
                if 'vparams' in kwargs.keys():
                    self.vparams = kwargs['vparams']

                self.vname = name_dict[vdf]

                if vdf == 'msw':
                    fparams = [self.vparams[0], self.vparams[2],
                               self.vparams[3]]
                else:
                    fparams = [self.vparams[0], self.vparams[2]]

                f_VDF = f_dict[vdf](fparams, vp=0)
                self.f_VDF = lambda v: f_VDF(v) / self.vcorr**3

                v2f = f_dict[vdf](fparams, vp=2)
                self.v2f = lambda v: v2f(v) / self.vcorr

                self.v0 = self.vparams[0] / self.in_vcorr * ckms
                self.vE = self.vparams[1] / self.in_vcorr * ckms
                self.vesc = self.vparams[2] / self.in_vcorr * ckms
                if vdf == 'msw':
                    self.p = self.vparams[3]
                else:
                    self.p = None

                eta_fn = eta_dict[vdf](self.vparams)
                self.eta_fn = lambda x: eta_fn(x) / self.vcorr

                if vdf == 'shm' or not self.interp:
                    self.etaInterp = None
                    self.eta = self.eta_fn
                else:
                    self.etaInterp = self.make_etaInterp()
                    self.eta = self.etaInterp

            elif callable(kwargs['vdf']):
                self.vname = 'Custom'
                self.vdf = 'custom'
                self.vsavename = 'custom'
                self.f_VDF = lambda x: kwargs['vdf'](x) / self.vcorr**3
                self.v2f = lambda x: x**2 * self.f_VDF(x) / self.vcorr

                self.v0 = None
                self.vE = 232/ckms * self.in_vcorr
                self.vesc = 544/ckms * self.in_vcorr
                self.p = None

                self.set_custom_params(**kwargs)

                vesctmp = self.vesc
                vEtmp = self.vE

                self.vE *= ckms / self.in_vcorr
                self.vesc *= ckms / self.in_vcorr

                eta_fn = etaFromVDF(kwargs['vdf'], vesc=vesctmp, vE=vEtmp)
                self.eta_fn = lambda x: eta_fn(x) / self.vcorr

                if self.interp:
                    self.etaInterp = self.make_etaInterp()
                    self.eta = self.etaInterp
                else:
                    self.etaInterp = None
                    self.eta = self.eta_fn

            else:
                raise TypeError("Keyword argument 'vdf' must be a string " +\
                                "or function.")

        if 'eta' in kwargs.keys():

            if callable(kwargs['eta']):
                self.eta_fn = lambda x: kwargs['eta'](x) / self.vcorr
                self.vname = 'Custom'
                self.vdf = 'custom'
                self.vsavename = 'custom'

                self.v0 = None
                self.vE = 232/ckms * self.in_vcorr
                self.vesc = 544/ckms * self.in_vcorr
                self.p = None

                self.set_custom_params(**kwargs)

                self.vE *= ckms / self.in_vcorr
                self.vesc *= ckms / self.in_vcorr

                if self.interp:
                    self.etaInterp = self.make_etaInterp()
                    self.eta = self.etaInterp
                else:
                    self.etaInterp = None
                    self.eta = self.eta_fn

            else:
                raise TypeError("Keyword argument 'eta' must be a function.")

    def set_params(self, **kwargs):
        self.__dict__.update((k,v) for k,v in kwargs.items() \
                             if k in self.allowed_keys)

        if 'vdf' in kwargs.keys() or 'eta' in kwargs.keys():
            self._setup(**kwargs)

        elif 'vparams' in kwargs.keys():
            from utils.velocity_dists import f_SHM, f_Tsa, f_MSW
            from utils.etas import etaSHM, etaTsa, etaMSW

            name_dict = { 'shm' : 'Standard Halo Model',
                          'tsa' : 'Tsallis Model',
                          'msw' : 'Empirical Model' }

            f_dict = { 'shm' : f_SHM,
                       'tsa' : f_Tsa,
                       'msw' : f_MSW }

            eta_dict = { 'shm' : etaSHM,
                         'tsa' : etaTsa,
                         'msw' : etaMSW }

            self.vparams = kwargs['vparams']
            vdf = self.vdf

            if vdf == 'msw':
                fparams = [self.vparams[0], self.vparams[2],
                           self.vparams[3]]
            else:
                fparams = [self.vparams[0], self.vparams[2]]

            f_VDF = f_dict[vdf](fparams, vp=0)
            self.f_VDF = lambda v: f_VDF(v) / self.vcorr**3

            v2f = f_dict[vdf](fparams, vp=2)
            self.v2f = lambda v: v2f(v) / self.vcorr

            self.v0 = self.vparams[0] / self.in_vcorr * ckms
            self.vE = self.vparams[1] / self.in_vcorr * ckms
            self.vesc = self.vparams[2] / self.in_vcorr * ckms
            if vdf == 'msw':
                self.p = self.vparams[3]
            else:
                self.p = None

            eta_fn = eta_dict[vdf](self.vparams)
            self.eta_fn = lambda x: eta_fn(x) / self.vcorr

            if vdf == 'shm' or not self.interp:
                self.etaInterp = None
                self.eta = self.eta_fn
            else:
                self.etaInterp = self.make_etaInterp()
                self.eta = self.etaInterp

        elif 'interp' in kwargs.keys():
            if self.vdf == 'shm' or not self.interp:
                self.etaInterp = None
                self.eta = self.eta_fn
            else:
                self.etaInterp = self.make_etaInterp()
                self.eta = self.etaInterp

    def set_custom_params(self, **kwargs):
        custom_keys = {'vname', 'vsavename', 'vE', 'vesc'}
        self.__dict__.update((k,v) for k,v in kwargs.items() \
                             if k in custom_keys)

    def make_etaInterp(self):
        vmin = np.linspace(0, (self.vE+self.vesc+1)/ckms, N_vmin)
        eta = self.eta_fn(Vmin*self.in_vcorr)*self.out_vcorr

        return interp1d(vmin*self.in_vcorr, eta/self.out_vcorr,
                        bounds_error=False, fill_value=0.)
