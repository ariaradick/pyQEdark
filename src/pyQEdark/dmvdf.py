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

        # self.v_mp = -quad(neg_Mu_lab, 0, 365.25)[0] / 365.25

    def _calc_v_mp(self):
        negF_tavg = lambda v: -quad(lambda t: self.F_lab(v,t), 0, 365.25)[0] / \
                              365.25
        self.v_mp = minimize(negF_tavg, norm(self.Mu-self._v_lab(79.26))).x[0]
        return

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

class Stream_From_B:
    def __init__(self, Mu_norm, tc, lam, Sigma, vesc, in_unit='kms',
                 out_unit='kms', N_MC=1000000):

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

        self.Mu = self.solve_for_Mu(Mu_norm, tc, lam) / self.in_vcorr
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

        self.tc = tc
        self.b = np.sin(lam)

        # negF_tavg = lambda v: -quad(lambda t: self.F_lab(v,t), 0, 365.25)[0] / \
        #                       365.25
        # self.v_mp = minimize(negF_tavg, vecMag(self._v_lab(79.26)),
        #                      method='Nelder-Mead').x[0]

        # self.v_mp = -quad(neg_Mu_lab, 0, 365.25)[0] / 365.25

    def solve_for_Mu(self, Mu_norm, tc, lam, diagnostics=False):
        from pyQEdark.vdfparams.kms import v_sun, epsilon_1, epsilon_2, \
                                           t_Mar21, omega

        b = np.sin(lam)
        te_c = omega*(tc-t_Mar21)

        epsilon_3 = np.cross(epsilon_1, epsilon_2)
        eps_hat = np.array([epsilon_1/norm(epsilon_1),
                            epsilon_2/norm(epsilon_2),
                            epsilon_3/norm(epsilon_3)])

        vs_eps = np.dot(eps_hat, v_sun)

        b1 = b*np.cos(te_c)
        b2 = b*np.sin(te_c)
        b3 = np.cos(lam)

        vsa,vsb,vsc = vs_eps

        under_sqrt = -vsb**2 + b2**2*(-vsa**2 + vsb**2) + 2*b1*b3*vsa*vsc - \
                     vsc**2 + 2*b2*vsb*(b1*vsa + b3*vsc) + \
                     b3**2*(-vsa**2 + vsc**2) + Mu_norm**2
        mu1 = (b2**2 + b3**2)*vsa - b1*b2*vsb - b1*b3*vsc + \
              b1*np.sqrt(under_sqrt)
        mu2 = (1 - b2**2)*vsb - b1*b2*vsa - b2*b3*vsc + b2*np.sqrt(under_sqrt)
        mu3 = (1 - b3**2)*vsc - b1*b3*vsa - b2*b3*vsb + b3*np.sqrt(under_sqrt)

        Mu_gal_eps = np.array([mu1,mu2,mu3])
        Mu_gal = np.dot(np.transpose(eps_hat), Mu_gal_eps)

        if diagnostics:
            Mu_solar = Mu_gal - v_sun
            b_vec = [np.dot(Mu_solar, eps_hat[i])/norm(Mu_solar) \
                     for i in range(3)]
            lam_calc = np.arccos(b_vec[2])
            tc_calc = np.arctan2(b_vec[1], b_vec[0])/omega + t_Mar21
            return Mu_gal, tc_calc, lam_calc
        else:
            return Mu_gal

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
