import numpy as np
from scipy.stats import chisquare, norm, poisson, chi2
from scipy.special import gammaincc, gamma
from scipy.optimize import brentq, minimize

from pyQEdark.constants import ckms
from pyQEdark.crystaldme import Crystal_DMe

def p_from_Z(signif):
    return 1-norm.cdf(signif)

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

def find_MLE(material, exposure, vE, data, vdf='shm', v0=220/ckms,
             vesc=544/ckms, p=1.5, FDMn=0, eta_db_path=None):

    def LL(x, CrysList, exposure, data):
        N_Ne = len(data[0])
        mx = x[0]
        sig = x[1]
        rates = np.zeros((len(CrysList), N_Ne))
        for i in range(len(CrysList)):
            rates[i] = exposure[i]*CrysList[i].Rate(mx, xsec=sig)
        return _t_mu(data.flatten(), rates.flatten())[0]

    CrysL = []
    for vv in vE:
        CrysL.append(Crystal_DMe(material, save_loc=eta_db_path, vdf=vdf,
                                 vparams=[v0, vv, vesc, p], FDMn=FDMn))

    init_xsec = CrysL[0].sig_test
    init_mass = 10e6 # eV
    init = [init_mass, init_xsec]

    return minimize(LL, init, args=(CrysL, exposure, data),
                    method='Nelder-Mead').x
