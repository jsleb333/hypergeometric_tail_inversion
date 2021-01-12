import numpy as np
from scipy.special import betainc, betaincinv
from scipy.optimize import bisect


def binomial_tail(k, m, p):
    return betainc(m-k, k+1, 1-p) # Quite faster than scipy.stats.binom.cdf


def binomial_tail_inverse(k, m, delta):
    return 1-betaincinv(m-k, k+1, delta)