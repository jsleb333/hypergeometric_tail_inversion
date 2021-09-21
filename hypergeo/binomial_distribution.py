import numpy as np
from scipy.special import betainc
from scipy.optimize import bisect


def binomial_tail(k, m, p):
    return betainc(m-k, k+1, 1-p) # Quite faster than scipy.stats.binom.cdf


def binomial_tail_inverse(k, m, delta):
    # Note that one cannot use the regularized incomplete beta function 'betaincinv' of scipy because of numerical instabilities when k is small.
    func = lambda p: binomial_tail(k, m, p) - delta
    return bisect(func, 0, 1, xtol=1e-100, rtol=1e-15, maxiter=200)