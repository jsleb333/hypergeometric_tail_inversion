from scipy.special import binom, comb

from source.hypergeometric_distribution import *


def test_hypergeometric_pmf():
    k, m, K, M = 5, 13, 16, 30
    assert np.isclose(hypergeometric_pmf(k, m, K, M), binom(K, k) * binom(M-K, m-k) / binom(M, m))


def test_hypergeometric_tail():
    k, m, K, M = 5, 13, 16, 30
    assert np.isclose(hypergeometric_tail(k, m, K, M), sum(binom(K, j)*binom(M-K, m-j)/binom(M, m) for j in range(k+1)))


def test_berkopec_single_term():
    k, m, K, M = 5, 13, 16, 30
    assert berkopec_single_term(k,m,K,M) == binom(K, k)*binom(M-K-1, M-K-m+k)/binom(M, m)


def test_berkopec_formula_equals_tail():
    k, m, K, M = 50, 200, 500, 1000
    berkopec = hypergeometric_berkopec_tail(k, m, K, M)
    # assert np.isclose(berkopec, hypergeometric_tail(k, m, K, M), atol=10-35, rtol=10e-20)
    k, m, K, M = 10, 200, 10, 1000
    berkopec = hypergeometric_berkopec_tail(k, m, K, M)
    print(f'{berkopec:e}')
    assert berkopec == 1


def test_hypergeometric_tail_inverse_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    k, m, K, M = 20, 200, 42, 222
    assert hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M), M) == K
    assert hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)-10e-22, M) == K+1
    assert hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)+10e-18, M) == K

    for delta in [0.05, 0.1, 0.25, 10e-20]:
        assert hypergeometric_tail(k, m, hypergeometric_tail_inverse(k,m,delta,M), M) <= delta


def test_hypergeometric_tail_inverse_log_delta_is_same_as_delta():
    delta = 0.05
    k, m, _, M = 20, 200, 42, 222
    assert hypergeometric_tail_inverse(k, m, delta, M) == hypergeometric_tail_inverse(k, m, np.log(delta), M, log_delta=True)


def test_berkopec_hypergeometric_tail_inverse_below_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    assert berkopec_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M), M) == K
    assert berkopec_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)-0.001, M) == K+1
    assert berkopec_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)+0.001, M) == K

    for delta in [0.05, 0.1, 0.25, 10e-20]:
        assert hypergeometric_tail(k, m, berkopec_hypergeometric_tail_inverse(k,m,delta,M), M) <= delta


def test_berkopec_hypergeometric_tail_inverse_above_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    assert berkopec_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M), M, start='above') == K
    assert berkopec_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)-0.001, M, start='above') == K+1
    assert berkopec_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)+0.001, M, start='above') == K

    for delta in [0.05, 0.1, 0.25, 10e-20]:
        assert hypergeometric_tail(k, m, berkopec_hypergeometric_tail_inverse(k,m,delta,M, start='above'), M) <= delta


def test_logberkopec_hypergeometric_tail_inverse_below_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    assert logberkopec_hypergeometric_tail_inverse(k, m, np.log(hypergeometric_tail(k,m,K,M)), M) == K
    assert logberkopec_hypergeometric_tail_inverse(k, m, np.log(hypergeometric_tail(k,m,K,M)-0.001), M) == K+1
    assert logberkopec_hypergeometric_tail_inverse(k, m, np.log(hypergeometric_tail(k,m,K,M)+0.001), M) == K

    for delta in [0.05, 0.1, 0.25]:
        assert np.log(hypergeometric_tail(k, m, logberkopec_hypergeometric_tail_inverse(k,m,np.log(delta),M), M)) <= np.log(delta)


def test_logberkopec_hypergeometric_tail_inverse_below_is_same_as_log_above():
    k, m, K, M = 5, 13, 16, 30
    for delta in [0.05, 0.1, 0.25]:
        assert logberkopec_hypergeometric_tail_inverse(k, m, np.log(delta), M, start='below') == logberkopec_hypergeometric_tail_inverse(k, m, np.log(delta), M, start='above')
    k, m, K, M = 7, 50, 40, 200
    for delta in [0.05, 0.1, 0.25]:
        assert logberkopec_hypergeometric_tail_inverse(k, m, np.log(delta), M, start='below') == logberkopec_hypergeometric_tail_inverse(k, m, np.log(delta), M, start='above')


def test_naive_hypergeometric_tail_inverse_below_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    assert naive_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M), M) == K
    assert naive_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)-0.001, M) == K+1
    assert naive_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)+0.001, M) == K

    for delta in [0.05, 0.1, 0.25]:
        assert hypergeometric_tail(k, m, naive_hypergeometric_tail_inverse(k,m,delta,M), M) <= delta


def test_naive_hypergeometric_tail_inverse_above_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    assert naive_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M), M, start='above') == K
    assert naive_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)-0.001, M, start='above') == K+1
    assert naive_hypergeometric_tail_inverse(k, m, hypergeometric_tail(k,m,K,M)+0.001, M, start='above') == K

    for delta in [0.05, 0.1, 0.25]:
        assert hypergeometric_tail(k, m, naive_hypergeometric_tail_inverse(k,m,delta,M, start='above'), M) <= delta
