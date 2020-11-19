import numpy as np

from source.hypergeometric_distribution import hypergeometric_left_tail_inverse


def hypinv_upperbound(k, m, growth_function, delta=0.05, mprime=None, max_mprime=None):
    """
    Implements the bound of Theorem 8.

    Args:
        k (int): Number of errors of the classifier on the sample.
        m (int): Number of examples of the sample.
        growth_function (callable): Growth function of the hypothesis class. Will receive m+mprime as input and should output a number.
        delta (float between 0 and 1): Confidence paramter.
        mprime (int or None): Ghost sample size. If None, will be optimized for the given inputs. This requires calling growth_function 'max_mprime' times. If too slow, one can use the heuristic value of 4*m as a good guess.
        max_mprime (int): Used when optimizing mprime. Will evaluate the best value of mprime within 1 and 'max_mprime'. If None, defaults to 15*m.

    Returns epsilon, the upper bound between 0 and 1.
    """
    if k == m:
        return 1

    if mprime is None:
        if max_mprime is None:
            max_mprime = 15*m
        mprime = _optimize_mprime(k, m, growth_function, delta, max_mprime)

    return max(1, hypergeometric_left_tail_inverse(k, m, delta/4/growth_function(m+mprime), m+mprime) - 1 - k)/mprime


def _optimize_mprime(k, m, growth_function, delta, max_mprime):
    bounds = np.ones(max_mprime)
    best_bound = 1
    for mprime in range(1, max_mprime+1):
        bound = hypinv_upperbound(k, m, growth_function, delta, mprime)
        bounds[mprime-1] = bound
        if bound <= best_bound:
            best_bound = bound

    best_mprimes = []
    for mp, bound in enumerate(bounds, start=1):
        if np.isclose(bound, best_bound, rtol=10e-12, atol=0):
            best_mprimes.append(mp)

    return best_mprimes[len(best_mprimes)//2] # Median of the best mprimes

