import numpy as np
from numpy.core.numeric import isclose
from scipy.special import binom, comb, gammaln
from scipy.stats import hypergeom
import math

import warnings
warnings.filterwarnings('error')


from source.utils import close_to, close_to_or_less_than


def binomln(m, k):
    """
    Log of the binomial coefficient. Is approximated via a logarithmic implementation of the gamma function.

    Args:
        m (int): Sample size.
        k (int): Number of elements.
    """
    return gammaln(m+1) - gammaln(k+1) - gammaln(m-k+1)


def hypergeometric_pmf(k, m, K, M):
    """
    Hypergeometric distribution probability mass function.

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        K (int): Number of errors in the whole population.
        M (int): Population size.
    """
    # return comb(K, k, exact=True)*comb(M-K, m-k, exact=True)/comb(M, m, exact=True)
    return hypergeom.pmf(k, M, K, m)


def hypergeometric_left_tail(k, m, K, M):
    """
    Hypergeometric distribution left tail, AKA cumulative distribution function.

        Hyp(k, m, K, M) = Σ_(j<=k) hyp(j, m, K, M),

    where hyp(j, m, K, M) is the probability mass function.

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        K (int): Number of errors in the whole population.
        M (int): Population size.
    """
    # return sum(hypergeometric_pmf(j, m, K, M) for j in range(max(0, m-M+K), k+1))
    return hypergeom.cdf(k, M, K, m)
    # return sum(hypergeom.pmf(j, M, K, m) for j in range(max(0, m-M+K), k+1))

def berkopec_single_term(k, m, K, M):
    """
    Computes a single term of Berkopec's formula for the hypergeometric cumulative distribution function. Berkopec's formula is:
        Hyp(k, m, K, M) = Σ_{J=K}^{M-m+k} binom(J, k) * binom(M-J-1, M-K-m+k) / binom(M, m),
    where binom(m, k) is the binomial coefficient, with the convention that binom(-1, 0) = 1.

    NOTE: This function divides two very large numbers, which introduces numerical errors of the order of 10-16. These errors can pile up when summing the terms, which can change significantly the result of the algorithm for computing the hypergeometric tail inverse with Berkopec's forumla when delta is smaller than the numerical error. A work around is to normalize at the end only. See for example 'hypergeometric_left_tail_inverse'.

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        K (int): Number of errors in the whole population.
        M (int): Population size.
    """
    if M == K and m == k: # Case with binom(-n, 0) = 1 (scipy's notation is 0).
        return 1
    else:
        return comb(K, k, exact=True) * comb(M-K-1, M-K-m+k, exact=True) / comb(M, m, exact=True)


def berkopec_unnormalized_single_term(k, m, K, M):
    """
    Computes a unnormalized single term of Berkopec's formula for the hypergeometric cumulative distribution function. Berkopec's formula is:
        Hyp(k, m, K, M) = Σ_{J=K}^{M-m+k} binom(J, k) * binom(M-J-1, M-K-m+k) / binom(M, m),
    where binom(m, k) is the binomial coefficient, with the convention that binom(-1, 0) = 1.

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        K (int): Number of errors in the whole population.
        M (int): Population size.
    """
    if M == K and m == k: # Case with binom(-n, 0) = 1 (scipy's notation is 0).
        return comb(K, k, exact=True)
    else:
        return comb(K, k, exact=True) * comb(M-K-1, M-K-m+k, exact=True)


def hypergeometric_berkopec_left_tail(k, m, K, M):
    """
    Hypergeometric distribution left tail, AKA cumulative distribution function using Berkopec's formula.

        Hyp(k, m, K, M) = Σ_{J=K}^{M-m+k} binom(J, k) * binom(M-J-1, M-J-m+k) / binom(M, m)

    where hyp(j, m, K, M) is the probability mass function.

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        K (int): Number of errors in the whole population.
        M (int): Population size.
    """
    return sum(berkopec_unnormalized_single_term(k, m, J, M) for J in range(K, M-m+k+1)) / comb(M, m, exact=True)


def hypergeometric_left_tail_inverse(k, m, delta, M, start='below'):
    """
    Computes the pseudo-inverse of the hypergeometric distribution left tail:
        HypInv(k, m, delta, M) = min{ K : Hyp(j, m, K, M) <= delta },
    where Hyp(k, m, K, M) is the cumulative distribution function (CDF).

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        delta (float in (0,1)): Confidence parameter threshold.
        M (int): Population size.
        start (string, 'above' or 'below'): Specifies if the algorithm should approach delta from above or from below. Use 'above' if k << M - m and below otherwise. See the note below for more info.

    Note: We use Berkopec's formula for the CDF to find the minimum value. The formula is a sum over the parameter K. One can approach delta from below by adding the terms of the summation sequentially, or from above by computing the whole sum and substracting terms sequentially. Both ways will yield the same answer, but one will require more operations than the other. The efficiency of the approach depends on the parameters.

    The 'above' algorithm is optimized for the regime where k << M - m and delta near 1, because we compute the CDF using the sum over 'k' but we substract terms using Berkopec's sum over 'K'.

    The 'below' algorithm is optimized for the regime where m - k << M and delta near 0, because it ensures that the number of terms to be summed in Berkopec's formula is small and we expect K to be large so that the minimum will be found quickly.

    Returns K the number of errors in the whole population with probability 1 - delta.
    """
    norm_factor = comb(M, m, exact=True)
    if start == 'above':
        K = k
        term = berkopec_unnormalized_single_term(k, m, K, M)
        hyp_cdf = norm_factor # Unnormalized CDF, necessary to limit numerical errors
        while hyp_cdf/norm_factor > delta and not close_to(hyp_cdf/norm_factor, delta, atol=0, rtol=10e-16) and K < M-m+k:
            hyp_cdf -= term
            K += 1
            term *= K*(M-K+1-m+k)
            term //= (K-k)*(M-K)
        return K

    elif start == 'below':
        K = M - m + k
        term = berkopec_unnormalized_single_term(k, m, K, M)
        hyp_cdf = term
        while close_to_or_less_than(hyp_cdf/norm_factor, delta, atol=0, rtol=10e-16) and K >= k:
            term *= (K-k)*(M-K)
            term //= K*(M-K+1-m+k)
            # term = berkopec_unnormalized_single_term(k, m, K-1, M)
            hyp_cdf += term
            K -= 1
        return K + 1


def log_hypergeometric_left_tail_inverse(k, m, log_delta, M, start='below'):
    """
    Computes the pseudo-inverse of the hypergeometric distribution left tail for a logarithmic delta term and with a logarithmic algorithm to avoid under- and overflows and less memory usage.

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        log_delta (negative float): Logarithm of the confidence parameter threshold.
        M (int): Population size.
        start (string, 'above' or 'below'): Specifies if the algorithm should approach log_delta from above or from below. Use 'above' if k << M - m and below otherwise.

    See the doc of the function 'hypergeometric_left_tail_inverse' for more info.

    NOTE: This implementation is very fast, but it is prone to numerical errors. In particular, the 'above' approach is unstable for small deltas of order 10e-14 (i.e. very negative log_deltas) and will NOT yield the correct result. Such cases are handled easily and quite fast by using the 'below' approach. A good way to know if the result if correct is to check if both approaches agree together.

    This algorithm is the same as in 'hypergeometric_left_tail_inverse', but in logarithmic form to limit memory usage and to increase efficiency using optimized algorithms to compute the binomial coefficients. We use the identity
        log(a + b) = log(a) + log(1 + b/a) = log(a) + log(1 + exp(log(b) - log(a)))
    to compute only the change to the logarithmic CDF at each step, with the fac that log(a) is the quantity to update and log(b) is quick to compute.

    Returns K the number of errors in the whole population with probability 1 - delta.
    """
    log_delta += binomln(M, m)
    if start == 'above':
        K = k
        log_hyp_cdf = math.log(comb(M, m, exact=True))
        while log_hyp_cdf > log_delta and not close_to(log_hyp_cdf, log_delta, atol=0, rtol=10e-16) and K < M-m+k:
            log_hyp_cdf += np.log1p(-np.exp(binomln(K, k) + binomln(M-K-1, M-K-m+k) - log_hyp_cdf))
            K += 1
        return K

    elif start == 'below':
        K = M - m + k
        log_hyp_cdf = binomln(K, k) + binomln(M-K-1, M-K-m+k)
        while close_to_or_less_than(log_hyp_cdf, log_delta, atol=0, rtol=10e-16) and K >= k:
            K -= 1
            log_hyp_cdf += np.log1p(np.exp(binomln(K, k) + binomln(M-K-1, M-K-m+k) - log_hyp_cdf))
        return K + 1


def naive_hypergeometric_left_tail_inverse(k, m, delta, M, start='below'):
    """
    NOTE: This implementation is much slower than the others.

    Computes the pseudo-inverse of the hypergeometric distribution left tail:
        HypInv(k, m, delta, M) = min{ K : Hyp(j, m, K, M) <= delta },
    where Hyp(k, m, K, M) is the cumulative distribution function (CDF).

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        delta (float in (0,1)): Confidence parameter threshold.
        M (int): Population size.
        start (string, 'above' or 'below'): Specifies if the algorithm should approach delta from above or from below. Use 'above' if k << M - m and below otherwise. See the doc of the function 'hypergeometric_left_tail_inverse' for more info.

    Implements a naive version of the algorithm not using Berkopec's formula, which implies recomputing the entire CDF at each step, hence slowing down considerably the algorithm.

    Returns K the number of errors in the whole population with probability 1 - delta.
    """
    if start == 'above':
        K = k
        hyp_cdf = hypergeometric_left_tail(k, m, K, M)
        while hyp_cdf > delta and not close_to(hyp_cdf, delta, atol=0, rtol=10e-16) and K < M-m+k:
            K += 1
            hyp_cdf = hypergeometric_left_tail(k, m, K, M)
        return K

    elif start == 'below':
        K = M - m + k
        hyp_cdf = hypergeometric_left_tail(k, m, K, M)
        while close_to_or_less_than(hyp_cdf, delta, atol=0, rtol=10e-16) and K >= k:
            K -= 1
            hyp_cdf = hypergeometric_left_tail(k, m, K, M)
        return K + 1