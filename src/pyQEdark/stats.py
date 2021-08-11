import numpy as np
from scipy.stats import chisquare, norm, poisson, chi2
from scipy.special import gammaincc, gamma
from scipy.optimize import brentq

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
    t = 0.0
    nu = len(data)

    data_ = np.zeros_like(data)
    theory_ = np.zeros_like(theory)

    data_[:] = data[:]
    theory_[:] = theory[:]

    for i in range(len(data)):
        if theory_[i] == 0.0:
            nu -= 1
        elif data_[i] == 0.0:
            t += theory_[i]
        else:
            t += theory_[i]-data_[i]+data_[i]*np.log(data_[i]/theory_[i])
    t *= 2

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
