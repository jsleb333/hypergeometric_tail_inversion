from scipy.special import binom

from source.hypergeometric_distribution import *


def test_hypergeometric_pmf():
    k, m, K, M = 5, 13, 16, 30
    assert np.isclose(hypergeometric_pmf(k, m, K, M), binom(K, k) * binom(M-K, m-k) / binom(M, m))


def test_hypergeometric_left_tail():
    k, m, K, M = 5, 13, 16, 30
    assert np.isclose(hypergeometric_left_tail(k, m, K, M), sum(binom(K, j)*binom(M-K, m-j)/binom(M, m) for j in range(k+1)))


def test_berkopec_single_term():
    k, m, K, M = 5, 13, 16, 30
    return np.isclose(berkopec_single_term(k,m,K,M), binom(K, k)*binom(M-K-1, M-K-m+k)/binom(M, m))


def test_hypergeometric_left_tail_inverse_above_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    assert hypergeometric_left_tail_inverse(k, m, hypergeometric_left_tail(k,m,K,M), M) == K
    assert hypergeometric_left_tail_inverse(k, m, hypergeometric_left_tail(k,m,K,M)-0.001, M) == K+1
    assert hypergeometric_left_tail_inverse(k, m, hypergeometric_left_tail(k,m,K,M)+0.001, M) == K

    for delta in [0.05, 0.1, 0.25]:
        assert hypergeometric_left_tail(k, m, hypergeometric_left_tail_inverse(k,m,delta,M), M) <= delta


def test_hypergeometric_left_tail_inverse_below_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    assert hypergeometric_left_tail_inverse(k, m, hypergeometric_left_tail(k,m,K,M), M, start='below') == K
    assert hypergeometric_left_tail_inverse(k, m, hypergeometric_left_tail(k,m,K,M)-0.001, M, start='below') == K+1
    assert hypergeometric_left_tail_inverse(k, m, hypergeometric_left_tail(k,m,K,M)+0.001, M, start='below') == K

    delta = 0.1
    assert hypergeometric_left_tail(k, m, hypergeometric_left_tail_inverse(k,m,delta,M, start='below'), M) <= delta
    delta = 0.05
    assert hypergeometric_left_tail(k, m, hypergeometric_left_tail_inverse(k,m,delta,M, start='below'), M) <= delta


def test_hypergeometric_left_tail_inverse_above_is_same_as_below():
    k, m, K, M = 5, 13, 16, 30
    for delta in [0.05, 0.1, 0.25]:
        assert hypergeometric_left_tail_inverse(k, m, delta, M, start='above') == hypergeometric_left_tail_inverse(k, m, delta, M, start='below')


def test_log_hypergeometric_left_tail_inverse_above_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    assert log_hypergeometric_left_tail_inverse(k, m, np.log(hypergeometric_left_tail(k,m,K,M)), M) == K
    assert log_hypergeometric_left_tail_inverse(k, m, np.log(hypergeometric_left_tail(k,m,K,M)-0.001), M) == K+1
    assert log_hypergeometric_left_tail_inverse(k, m, np.log(hypergeometric_left_tail(k,m,K,M)+0.001), M) == K

    for delta in [0.05, 0.1, 0.25]:
        assert np.log(hypergeometric_left_tail(k, m, log_hypergeometric_left_tail_inverse(k,m,np.log(delta),M), M)) <= np.log(delta)


def test_log_hypergeometric_left_tail_inverse_above_is_same_as_log_below():
    k, m, K, M = 5, 13, 16, 30
    for delta in [0.05, 0.1, 0.25]:
        assert log_hypergeometric_left_tail_inverse(k, m, np.log(delta), M, start='above') == log_hypergeometric_left_tail_inverse(k, m, np.log(delta), M, start='below')
    k, m, K, M = 7, 50, 40, 200
    for delta in [0.05, 0.1, 0.25]:
        assert log_hypergeometric_left_tail_inverse(k, m, np.log(delta), M, start='above') == log_hypergeometric_left_tail_inverse(k, m, np.log(delta), M, start='below')


def test_naive_hypergeometric_left_tail_inverse_above_is_inverse():
    k, m, K, M = 5, 13, 16, 30
    assert naive_hypergeometric_left_tail_inverse(k, m, hypergeometric_left_tail(k,m,K,M), M) == K
    assert naive_hypergeometric_left_tail_inverse(k, m, hypergeometric_left_tail(k,m,K,M)-0.001, M) == K+1
    assert naive_hypergeometric_left_tail_inverse(k, m, hypergeometric_left_tail(k,m,K,M)+0.001, M) == K

    for delta in [0.05, 0.1, 0.25]:
        assert hypergeometric_left_tail(k, m, naive_hypergeometric_left_tail_inverse(k,m,delta,M), M) <= delta


def test_naive_hypergeometric_left_tail_inverse_above_is_same_as_naive_below():
    k, m, K, M = 5, 13, 16, 30
    for delta in [0.05, 0.1, 0.25]:
        assert naive_hypergeometric_left_tail_inverse(k, m, delta, M, start='above') == naive_hypergeometric_left_tail_inverse(k, m, delta, M, start='below')
    k, m, K, M = 7, 50, 40, 200
    for delta in [0.05, 0.1, 0.25]:
        assert naive_hypergeometric_left_tail_inverse(k, m, delta, M, start='above') == naive_hypergeometric_left_tail_inverse(k, m, delta, M, start='below')