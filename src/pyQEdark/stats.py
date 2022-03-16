import numpy as np
from scipy.stats import chisquare, norm, poisson, chi2
from scipy.special import gammaincc, gamma
from scipy.optimize import brentq, minimize, Bounds
from multiprocessing import Pool, cpu_count

from pyQEdark.constants import ckms
from pyQEdark.crystaldme import Crystal_DMe
from pyQEdark.vdfparams.nat import v0fid, vescfid, v0new, vescnew, pfid, vEfid

def p_from_Z(signif):
    return 1-norm.cdf(signif)

def Z_from_p(p):
    return norm.ppf(1-p)

def chisq_test(data, theory, ddof=0):
    data_ = np.zeros_like(data)
    theory_ = np.zeros_like(theory)

    data_[:] = data[:]
    theory_[:] = theory[:]

    i_to_del = []
    for i in range(len(data_)):
        if data_[i] == 0.0 or theory_[i] == 0.0:
            i_to_del.append(i)

    data_ = np.delete(data_, i_to_del)
    theory_ = np.delete(theory_, i_to_del)
    return chisquare(data_, theory_, ddof=ddof)

def _t_mu(data, theory):
    t = np.zeros_like(data)
    nu = len(data)

    data_ = np.zeros_like(data)
    theory_ = np.zeros_like(theory)

    data_[:] = data[:]
    theory_[:] = theory[:]

    for i in range(len(data)):
        if theory_[i] == 0.0:
            nu -= 1
        elif data_[i] == 0.0:
            t[i] = theory_[i]
        else:
            t[i] = theory_[i]-data_[i]+data_[i]*np.log(data_[i]/theory_[i])
    t = 2*np.sum(t)

    return (t,nu)

def t_mu_test(data, theory, ddof=0, method='chi2'):
    tmu, nu = _t_mu(data, theory)

    if method == 'chi2':
        p = chi2.sf(tmu,nu-ddof)
        return (tmu, p)

    elif method == 'mc':
        ts_hist, ts_bins = _generate_mc(theory)
        mcp = _integrate_mc(tmu, ts_bins, ts_hist)
        return (tmu, mcp)

def _generate_mc(theory, N_mc=10000):
    N_Ne = len(theory)
    pois = np.zeros( (N_Ne, N_mc) )
    ts = np.zeros( (N_mc) )
    for i in range(N_Ne):
        pois[i] = poisson.rvs(theory[i], size=N_mc)
    for j in range(N_mc):
        ts[j] = _t_mu(pois[:,j], theory)[0]
    return np.histogram(ts, bins=100, density=True)

def _integrate_mc(ts, bins, theory):
    ts_i = np.digitize(ts,bins)
    if ts_i == len(bins):
        return 0.
    else:
        bin_width = bins[1]-bins[0]
        first_val = (bins[ts_i] - ts) * theory[ts_i-1]
        other_vals = theory[ts_i:]*bin_width
        return first_val + np.sum(other_vals)

def find_exposure(data, theory, signif, ddof=0, method='chi2', bqlims=1e6):
    def zero_func(x):
        data_ = np.zeros_like(data)
        theory_ = np.zeros_like(theory)
        data_[:] = x*data[:]
        theory_[:] = x*theory[:]

        return t_mu_test(data_,theory_,ddof=ddof,method=method)[1] - \
               p_from_Z(signif)

    return brentq(zero_func, 1/bqlims, bqlims)

def find_MLE(material, exposure, vE, data, bkg, vdf='shm', v0=220/ckms,
             vesc=544/ckms, p=1.5, FDMn=0, eta_db_path=None):

    CrysList = []
    for vv in vE:
        CrysList.append(Crystal_DMe(material, save_loc=eta_db_path, vdf=vdf,
                                 vparams=[v0, vv, vesc, p], FDMn=FDMn))

    def dLL_dmx(x):
        N_Ne = len(data[0])
        mx = x[0]
        sig = x[1]

        rates = np.zeros((len(CrysList), N_Ne))
        dR_dmx = np.zeros((len(CrysList), N_Ne))

        for i in range(len(CrysList)):
            rates[i] = exposure[i]*(CrysList[i].Rate(mx, xsec=sig)+bkg[i])
            dR_dmx[i] = exposure[i]*CrysList[i]._dR_dmx(mx, xsec=sig)

        ans = dR_dmx * (1 - data/rates)

        return np.sum(ans)

    def dLL_dsig(x):
        N_Ne = len(data[0])
        mx = x[0]
        sig = x[1]

        rates = np.zeros((len(CrysList), N_Ne))
        dR_dsig = np.zeros((len(CrysList), N_Ne))

        for i in range(len(CrysList)):
            rates[i] = exposure[i]*(CrysList[i].Rate(mx, xsec=sig)+bkg[i])
            dR_dsig[i] = exposure[i]*CrysList[i]._dR_dsig(mx, xsec=sig)

        ans = dR_dsig * (1 - data/rates)

        return np.sum(ans)

    def dLL(x):
        b = np.array([dLL_dmx(x), dLL_dsig(x)])
        print(b)
        return b

    def LL(x):
        N_Ne = len(data[0])
        mx = x[0]
        sig = x[1]
        rates = np.zeros((len(CrysList), N_Ne))
        for i in range(len(CrysList)):
            rates[i] = exposure[i]*(CrysList[i].Rate(mx, xsec=sig)+bkg[i])
        return _t_mu(data.flatten(), rates.flatten())[0]

    bounds = Bounds([6e5,1e-60],[1e15,np.inf])

    init_xsec = CrysList[0].sig_test
    init_mass = 100e6 # eV
    init = [init_mass, init_xsec]

    return minimize( LL, init, method='Nelder-Mead',
                     bounds=bounds ).x

class DMe_MLE:

    def __init__( self, material, mass_detector, time, data, save_loc=None,
                  vdfs=['shm', 'tsa', 'msw'],
                  vdfparams = [{'v0':v0fid, 'vesc':vescfid, 'p':pfid},
                               {'v0':v0new, 'vesc':vescnew, 'p':pfid}],
                  FDMn=[0,2], bkg=None, N_proc='max' ):

        self.data = data

        def vE_fn(t):
            # t in years
            return vEfid + 15*np.sin(2*np.pi*t)/ckms

        N_vdf = len(vdfs)
        N_vp = len(vdfparams)
        N_fdmn = len(FDMn)
        N_Ne = len(data[0])

        time = np.atleast_1d(time)

        if len(time) == 1:
            time = np.array([0, time[0]])

        avg_time = (time[1:] + time[:-1]) / 2
        d_time = time[1:] - time[:-1]

        exposure = mass_detector*d_time
        self.exposure = exposure

        vE_list = vE_fn(avg_time)
        N_vE = len(vE_list)

        if bkg is None:
            Bkg = np.zeros((N_vE, N_Ne))
        else:
            if np.isscalar(bkg):
                Bkg = bkg*np.ones((N_vE, N_Ne))
            elif bkg.shape == (N_Ne,):
                Bkg = np.zeros((N_vE, N_Ne))
                for i in range(N_vE):
                    Bkg[i,:] = bkg[:]
            elif bkg.shape == (N_vE, N_Ne):
                Bkg = bkg
            else:
                raise TypeError('Shape of background must match shape of data.')

        CrysList = [[] for i in range(len(vdfparams))]

        for i in range(len(vdfparams)):
            for vE in vE_list:
                CrysList[i].append(Crystal_DMe(material, save_loc=save_loc,
                                               vparams=[vdfparams[i]['v0'],
                                                        vE,
                                                        vdfparams[i]['vesc'],
                                                        vdfparams[i]['p']]))

        best_fits = np.zeros( (N_fdmn, N_vdf, N_vp, 2) )

        params_to_pass = []

        for n in range(N_fdmn):
            for i in range(N_vdf):
                for j in range(N_vp):
                    params_to_pass.append((material, exposure, vE_list,
                                           data, Bkg, vdfs[i],
                                           vdfparams[j]['v0'],
                                           vdfparams[j]['vesc'],
                                           vdfparams[j]['p'],
                                           FDMn[n],
                                           save_loc))
                    # best_fits[n,i,j] = find_MLE(material, exposure, vE_list,
                    #                             data, Bkg, vdf=vdfs[i],
                    #                             v0=vdfparams[j]['v0'],
                    #                             vesc=vdfparams[j]['vesc'],
                    #                             p=vdfparams[j]['p'],
                    #                             FDMn=FDMn[n],
                    #                             eta_db_path=save_loc)

        if N_proc == 'max':
            pool = Pool(cpu_count())
        elif np.isscalar(N_proc):
            pool = Pool(N_proc)

        results = pool.starmap(find_MLE, params_to_pass)

        bf_idx = 0

        for n in range(N_fdmn):
            for i in range(N_vdf):
                for j in range(N_vp):
                    best_fits[n,i,j] = results[bf_idx]
                    bf_idx += 1

        self.bf_mx_xsec = best_fits

        R_pred = np.zeros( (N_fdmn, N_vdf, N_vp, N_vE, N_Ne) )

        for n in range(N_fdmn):
            for i in range(N_vdf):
                for j in range(N_vp):
                    for k in range(N_vE):
                        CrysList[j][k].set_params(vdf=vdfs[i], FDMn=FDMn[n])
                        R_pred[n,i,j,k] = exposure[k]*CrysList[j][k].Rate(
                                            best_fits[n,i,j,0],
                                            xsec=best_fits[n,i,j,1]
                                            )

        self.preds = R_pred

        t_mu_vec = np.zeros((N_fdmn, N_vdf, N_vp))

        for n in range(N_fdmn):
            for i in range(N_vdf):
                for j in range(N_vp):
                    t_mu_vec[n,i,j] = _t_mu(data.flatten(),
                                            R_pred[n,i,j].flatten())[0]

        self.LL_vals = t_mu_vec

        n_m, v_m, vp_m = np.unravel_index(np.argmin(t_mu_vec, axis=None),
                                          t_mu_vec.shape)

        self.theory = np.zeros((N_vE, N_Ne))
        self.theory[:,:] = R_pred[n_m, v_m, vp_m, :, :]

        self.best_fits = { 'mass' : best_fits[n_m, v_m, vp_m, 0],
                           'xsec' : best_fits[n_m, v_m, vp_m, 1],
                           'vdf'  : vdfs[v_m],
                           'vparams' : vdfparams[vp_m],
                           'FDMn' : FDMn[n_m] }

        self._FDM_str = [ '= 1', '~ 1/q', '~ 1/q^2' ]

    def print_results(self):
        from pyQEdark.constants import ccms, hbar
        xsec_corr = ccms**2 * hbar**2

        print('The best fit mass is '+str(np.around(self.best_fits['mass']*1e-6,
              decimals=1))+' MeV.')
        print('The best fit cross-section is ' + \
              str(self.best_fits['xsec'] * xsec_corr) + ' cm^2.')
        print('The best fit VDF is ' + self.best_fits['vdf'] +'.')
        if self.best_fits['vdf'] == 'msw':
            print('''The best fit VDF parameters are v0 = {} km/s,
                     vesc = {} km/s, and p = {}.'''.format(
                                         self.best_fits['vparams']['v0']*ckms,
                                         self.best_fits['vparams']['vesc']*ckms,
                                         self.best_fits['vparams']['p'] ))
        else:
            print('''The best fit VDF parameters are v0 = {} km/s,  vesc = {}
                     km/s.'''.format( self.best_fits['vparams']['v0']*ckms,
                                     self.best_fits['vparams']['vesc']*ckms ))
        print('The best fit dark matter model is F_DM ' + \
              self._FDM_str[self.best_fits['FDMn']] + '.')

    def get_p_val(self, method='chi2'):
        return t_mu_test(self.data.flatten(), self.theory.flatten(), ddof=2,
                         method=method)[1]
