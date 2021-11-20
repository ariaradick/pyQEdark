"""
Basic class for describing a velocity distribution function.

author: Aria Radick
date: 7/19/21
"""

import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad_vec, nquad, quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import erf
from pyQEdark.constants import ckms, ccms, c_light

class DM_Halo:

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
                self.f_VDF = lambda x: kwargs['vdf'](x) / self.vcorr**3
                self.v2f = lambda x: x**2 * self.f_VDF(x) / self.vcorr

                self.v0 = None
                self.vE = 232/ckms * self.in_vcorr
                self.vesc = 544/ckms * self.in_vcorr
                self.p = None

                self._set_custom_params(**kwargs)

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

                self.v0 = None
                self.vE = 232/ckms * self.in_vcorr
                self.vesc = 544/ckms * self.in_vcorr
                self.p = None

                self._set_custom_params(**kwargs)

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

    def _set_custom_params(self, **kwargs):
        custom_keys = {'vE', 'vesc'}
        self.__dict__.update((k,v) for k,v in kwargs.items() \
                             if k in custom_keys)

    def make_etaInterp(self):
        N_vmin = 1000
        vmin = np.linspace(0, (self.vE+self.vesc+1)/ckms, N_vmin)
        eta = self.eta_fn(vmin*self.in_vcorr)*self.out_vcorr

        return interp1d(vmin*self.in_vcorr, eta/self.out_vcorr,
                        bounds_error=False, fill_value=0.)

class Stream:
    def __init__(self, Mu, Sigma, vesc, in_unit='kms', out_unit='kms',
                 N_MC=1000000):

        from pyQEdark.vdfparams.nat import vE_fn_solar, v_sun, \
                                           epsilon_1, epsilon_2
        self._v_lab = lambda t: (vE_fn_solar(t) + v_sun)

        corr_dict = { 'kms' : ckms,
                      'cms' : ccms,
                      'ms'  : c_light,
                      'nat' : 1 }

        self.in_vcorr = corr_dict[in_unit]
        self.out_vcorr = corr_dict[out_unit]
        self.vcorr = corr_dict[out_unit] / corr_dict[in_unit]

        self.Mu = Mu / self.in_vcorr
        sx2 = Sigma[0,0]
        sy2 = Sigma[1,1]
        sz2 = Sigma[2,2]
        self.s2 = np.array([sx2, sy2, sz2]) / self.in_vcorr**2
        self.vesc = vesc / self.in_vcorr
        self.N_MC = N_MC

        v2 = vesc / self.in_vcorr + np.sqrt(np.sum(self._v_lab(79.26)**2))
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

            ee = v**2 * np.sin(theta) * \
                 np.exp( -.5*np.sum( (v_vec-self.Mu)**2/self.s2 ) )
            return ee

        self.KK = nquad(f_to_norm, [(0,self.vesc), (0,2*np.pi), (0,np.pi)])[0]



        neg_Mu_lab = lambda t: -np.sqrt( np.sum((self.Mu - self._v_lab(t))**2) )
        self.tc = minimize(neg_Mu_lab, 79.26, method='Nelder-Mead').x[0]

        Mu_hat_solar = (self.Mu - v_sun) / np.linalg.norm(self.Mu - v_sun)
        epsilon_1_hat = epsilon_1 / np.linalg.norm(epsilon_1)
        epsilon_2_hat = epsilon_2 / np.linalg.norm(epsilon_2)
        self.b = np.sqrt( np.vdot(Mu_hat_solar, epsilon_1_hat)**2 + \
                          np.vdot(Mu_hat_solar, epsilon_2_hat)**2 )

        # negF_tavg = lambda v: -quad(lambda t: self.F_lab(v,t), 0, 365.25)[0] / \
        #                       365.25
        # self.v_mp = minimize(negF_tavg, vecMag(self._v_lab(79.26)),
        #                      method='Nelder-Mead').x[0]

        self.v_mp = -quad(neg_Mu_lab, 0, 365.25)[0] / 365.25

    def f_galactic(self, v):
        v_ = v[:] / self.in_vcorr
        len_v = np.sqrt( np.sum(v_**2) )
        ee = (1/self.KK) * np.exp( -.5*np.sum( (v_-self.Mu)**2/self.s2 ) ) * \
             np.heaviside(self.vesc - len_v, 0)
        return ee / self.out_vcorr

    def f_lab(self, v, t):
        v_ = v[:] / self.in_vcorr
        v_lab = self._v_lab(t)
        len_v = np.sqrt( np.sum((v_+v_lab)**2) )
        ee = (1/self.KK) * np.exp(-.5*np.sum((v_+v_lab-self.Mu)**2/self.s2)) * \
             np.heaviside(self.vesc - len_v, 0)
        return ee / self.out_vcorr

    def F_lab(self, v, t):
        """
        Calculates integral(f_lab dOmega)

        Note that v is NOT a 3-vector. It is the magnitude of the velocity here.
        """
        v_ = np.atleast_1d(v) / self.in_vcorr
        v_lab = self._v_lab(t)
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
        vmin = np.atleast_1d(vmin) / self.in_vcorr

        f_vals = np.zeros(self.N_MC)
        for i in range(self.N_MC):
            f_vals[i] = self._V[i] * np.sin(self._Th[i]) * \
                        self.f_lab(self._vv[:,i]*self.in_vcorr, t) * \
                        self.out_vcorr

        I = np.zeros_like(vmin)
        for i in range(len(vmin)):
            f_tmp = np.zeros(self.N_MC)
            grtr_than_vmin = self._V >= vmin[i]
            f_tmp[:] = f_vals[:] * grtr_than_vmin[:]
            I[i] = np.sum(f_tmp[:])

        I *= self._Volume_MC / self.N_MC / self.out_vcorr
        return I

class SimpleStream():
    """
    A DM stream with uniform dispersion sigma.
    """
    def __init__(self, Mu_gal, sigma, vesc, in_unit='kms', out_unit='kms'):

        from pyQEdark.vdfparams.nat import vE_fn_solar, v_sun, \
                                           epsilon_1, epsilon_2
        self._v_lab = lambda t: (vE_fn_solar(t) + v_sun)

        corr_dict = { 'kms' : ckms,
                      'cms' : ccms,
                      'ms'  : c_light,
                      'nat' : 1 }

        self.in_vcorr = corr_dict[in_unit]
        self.out_vcorr = corr_dict[out_unit]
        self.vcorr = corr_dict[out_unit] / corr_dict[in_unit]

        self._Mu_gal = Mu_gal / self.in_vcorr
        sx2 = Sigma[0,0]
        sy2 = Sigma[1,1]
        sz2 = Sigma[2,2]
        self._sig2 = sigma**2 / self.in_vcorr**2
        self._vesc = vesc / self.in_vcorr

        mu = norm(self._Mu_gal)

        self.KK = np.pi*self._sig2**(3/2) * (\
                  (-2*np.sqrt(np._sig2)/mu) * \
                  np.exp(-(mu+self._vesc)**2 / (2*np._sig2)) * \
                  (np.exp(2*mu*np._vesc/self._sig2)-1) + \
                  np.sqrt(2*np.pi)*(erf((mu+self._vesc)/np.sqrt(2*self._sig2))-\
                                    erf((mu-self._vesc)/np.sqrt(2*self._sig2))))

        neg_Mu_lab=lambda t: -np.sqrt(np.sum((self._Mu_gal-self._v_lab(t))**2))
        self.tc = minimize(neg_Mu_lab, 79.26, method='Nelder-Mead').x[0]

        Mu_hat_solar = (self._Mu_gal - v_sun) / norm(self._Mu_gal - v_sun)
        epsilon_1_hat = epsilon_1 / norm(epsilon_1)
        epsilon_2_hat = epsilon_2 / norm(epsilon_2)
        self.b = np.sqrt( np.vdot(Mu_hat_solar, epsilon_1_hat)**2 + \
                          np.vdot(Mu_hat_solar, epsilon_2_hat)**2 )

        negF_tavg = lambda v: -quad(lambda t: self.F_lab(v,t), 0, 365.25)[0] / \
                              365.25
        self.v_mp = minimize(negF_tavg, norm(self._v_lab(79.26)),
                             method='Nelder-Mead').x[0]

    def f_galactic(self, v_vec):
        v_ = v_vec[:] / self.in_vcorr
        len_v = norm(v_)
        ee = (1/self.KK) * np.exp(-.5*np.sum((v_-self._Mu_gal)**2/self._sig2))*\
             np.heaviside(self.vesc - len_v, 0)
        return ee / self.out_vcorr

    def f_lab(self, v_vec, t):
        v_lab = self._v_lab(t)*self.in_vcorr
        return self.f_galactic(v_vec+v_lab) / self.out_vcorr

    def F_lab(self, v, t):
        v_ = v / self.in_vcorr
        v_lab = self._v_lab(t)
        mu_lab = norm(self._Mu_gal - v_lab)
        e1 = np.exp(-(mu_lab+v_)**2 / (2*self._sig2))
        e2 = np.exp(2*mu_lab*v_/self._sig2)
        return v_*self._sig2 / (mu_lab*self.KK) * e1 * (e2-1)
