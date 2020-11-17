import numpy as np
from numpy.core.numeric import isclose
from scipy.special import binom, gammaln
from scipy.stats import hypergeom


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
    return hypergeom.cdf(k, M, K, m)


def berkopec_single_term(k, m, K, M):
    """
    Computes a single term of Berkopec's formula for the hypergeometric cumulative distribution function. Berkopec's formula is:
        Hyp(k, m, K, M) = Σ_{J=K}^{M-m+k} binom(J, k) * binom(M-J-1, M-K-m+k) / binom(M, m),
    where binom(m, k) is the binomial coefficient.

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        K (int): Number of errors in the whole population.
        M (int): Population size.
    """
    return np.exp(binomln(K, k) + binomln(M-K-1, M-K-m+k) - binomln(M, m))


def hypergeometric_left_tail_inverse(k, m, delta, M, start='above'):
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
    if start == 'above':
        K = k
        hyp_cdf = hypergeom.cdf(k, M, K, m)
        while hyp_cdf > delta and K <= M-m+k:
            hyp_cdf -= berkopec_single_term(k, m, K, M)
            K += 1
        return K

    elif start == 'below':
        K = M - m + k
        hyp_cdf = berkopec_single_term(k, m, K, M)
        while (hyp_cdf <= delta or np.isclose(hyp_cdf, delta, atol=10e-16)) and K >= k:
            K -= 1
            hyp_cdf += berkopec_single_term(k, m, K, M)
        return K + 1


def log_hypergeometric_left_tail_inverse(k, m, log_delta, M, start='above'):
    """
    Computes the pseudo-inverse of the hypergeometric distribution left tail for a logarithmic delta term and with a logarithmic algorithm to avoid under- and overflows.

    Args:
        k (int): Number of errors observed.
        m (int): Sample size.
        log_delta (negative float): Logarithm of the confidence parameter threshold.
        M (int): Population size.
        start (string, 'above' or 'below'): Specifies if the algorithm should approach delta from above or from below. Use 'above' if k << M - m and below otherwise.

    See the doc of the function 'hypergeometric_left_tail_inverse' for more info.

    Returns K the number of errors in the whole population with probability 1 - delta.
    """
    log_delta += binomln(M, m)
    if start == 'above':
        K = k
        log_hyp_cdf = np.log(hypergeom.cdf(k, M, K, m)) + binomln(M, m)
        while log_hyp_cdf > log_delta and K <= M-m+k:
            log_hyp_cdf += np.log(1 - np.exp(binomln(K, k) + binomln(M-K-1, M-K-m+k) - log_hyp_cdf))
            K += 1
        return K

    elif start == 'below':
        K = M - m + k
        log_hyp_cdf = binomln(K, k) + binomln(M-K-1, M-K-m+k)
        while log_hyp_cdf <= log_delta and K >= k:
            K -= 1
            log_hyp_cdf += np.log(1 + np.exp(log_hyp_cdf - binomln(K, k) + binomln(M-K-1, M-K-m+k)))
        return K + 1
