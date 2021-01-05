from scipy.special import binom

from source.binomial_distribution import *


def test_binomial_tail():
    k, m, p = 5, 13, .3
    cdf = sum(binom(m, i) * p**i * (1-p)**(m-i) for i in range(k+1))
    assert np.isclose(binomial_tail(k, m, p), cdf)


def test_binomial_tail_inverse_is_inverse():
    k, m, p = 5, 13, .3
    assert np.isclose(binomial_tail_inverse(k, m, binomial_tail(k,m,p)), p)
