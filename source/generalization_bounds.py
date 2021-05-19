import numpy as np
from scipy.special import binom, erfc

from source.hypergeometric_distribution import hypergeometric_tail_inverse
from source.binomial_distribution import binomial_tail_inverse


def hypinv_upperbound(k, m, growth_function, delta=0.05, mprime=None, max_mprime=None):
    """
    Implements the bound of Theorem 8.

    Args:
        k (int): Number of errors of the classifier on the sample.
        m (int): Number of examples of the sample.
        growth_function (callable):
            Growth function of the hypothesis class. Will receive m+mprime as input and should output a number.
        delta (float between 0 and 1): Confidence parameter.
        mprime (int or None):
            Ghost sample size. If None, will be optimized for the given inputs. This requires calling growth_function 'max_mprime' times. If too slow, one can use the heuristic value of 4*m as a good guess.
        max_mprime (int):
            Used when optimizing mprime. Will evaluate the best value of mprime within 1 and 'max_mprime'. If None, defaults to 15*m.

    Returns epsilon, the upper bound between 0 and 1.
    """
    if k == m:
        return 1

    if mprime is None:
        if max_mprime is None:
            max_mprime = 15*m
        mprime = optimize_mprime(k, m, growth_function, delta, max_mprime)

    return max(1, hypergeometric_tail_inverse(k, m, delta/4/growth_function(m+mprime), m+mprime) - 1 - k)/mprime


def hypinv_reldev_upperbound(k, m, growth_function, delta=0.05, mprime=None, max_mprime=None):
    """
    Implements the bound of Theorem 9.

    Args:
        k (int): Number of errors of the classifier on the sample.
        m (int): Number of examples of the sample.
        growth_function (callable):
            Growth function of the hypothesis class. Will receive m+mprime as input and should output a number.
        delta (float between 0 and 1): Confidence parameter.
        mprime (int or None):
            Ghost sample size. If None, will be optimized for the given inputs. This requires calling growth_function 'max_mprime' times. If too slow, one can use the heuristic value of 4*m as a good guess.
        max_mprime (int):
            Used when optimizing mprime. Will evaluate the best value of mprime within 1 and 'max_mprime'. If None, defaults to 15*m.

    Returns epsilon, the upper bound between 0 and 1.
    """
    if k == m:
        return 1

    if mprime is None:
        if max_mprime is None:
            max_mprime = 15*m
        mprime = optimize_mprime(k, m, growth_function, delta, max_mprime, hypinv_reldev_upperbound)

    M = m + mprime

    tau = growth_function(M)
    eta = 0
    if k != m:
        u = hypergeometric_tail_inverse(k, m, delta/(4*tau), M) - 1
        eta = max(1/np.sqrt(mprime), (M)/mprime*np.sqrt(u/M - 2*k/m + k**2/m**2/u*M))
    return k/m + eta**2/2 + eta/2 * np.sqrt(eta**2 + 4*k/m)


def optimize_mprime(k,
                    m,
                    growth_function,
                    delta,
                    max_mprime=10_000,
                    min_mprime=1,
                    bound=hypinv_upperbound,
                    early_stopping=np.inf,
                    return_bound=False):
    steps_since_last_best = 0
    bounds = np.ones(max_mprime - min_mprime + 1)
    best_bound = 1
    best_mprime = min_mprime
    for i, mprime in enumerate(range(min_mprime, max_mprime+1)):
        bound_value = bound(k, m, growth_function, delta, mprime)
        bounds[i] = bound_value
        if bound_value <= best_bound:
            best_bound = bound_value
            best_mprime = mprime
            steps_since_last_best = 0
        steps_since_last_best += 1
        if steps_since_last_best >= early_stopping:
            # print(f'early stopped after {i} iterations')
            break

    if not return_bound:
        return best_mprime
    else:
        return best_mprime, best_bound


def vapnik_pessismistic_bound(k, m, growth_function, delta):
    """
    Implements the Vapnik's pessimistic bound.

    Args:
        k (int): Number of errors of the classifier on the sample.
        m (int): Number of examples of the sample.
        growth_function (callable):
            Growth function of the hypothesis class. Will receive 2m as input and should output a number.
        delta (float between 0 and 1): Confidence parameter.

    Returns epsilon, the upper bound between 0 and 1.
    """
    e = (np.log(4) + np.log(growth_function(2*m)) - np.log(delta))/m
    return (k+1)/m + np.sqrt(e)


def vapnik_relative_deviation_bound(k, m, growth_function, delta):
    """
    Implements the Vapnik's relative deviation bound.

    Args:
        k (int): Number of errors of the classifier on the sample.
        m (int): Number of examples of the sample.
        growth_function (callable):
            Growth function of the hypothesis class. Will receive 2m as input and should output a number.
        delta (float between 0 and 1): Confidence parameter.

    Returns epsilon, the upper bound between 0 and 1.
    """
    e = (np.log(4) + np.log(growth_function(2*m)) - np.log(delta))/m
    r = k/m
    return r + 2*e*(1 + np.sqrt(1 + r/e))


def sample_compression_bound(k, m, d, delta, compression_scheme_prob=None):
    """
    Implements the sample compression bound.

    Args:
        k (int): Number of errors of the classifier on the sample.
        m (int): Number of examples of the sample.
        d (int): Number of examples in the compressed sample.
        delta (float between 0 and 1): Confidence parameter.
        compression_scheme_prob (float):
            Probability assigned to the compression scheme. Defaults to uniform distribution over compression sample sizes, i.e. 1/(m*binom(m, d)).

    Returns epsilon, the upper bound between 0 and 1.
    """
    if compression_scheme_prob is None:
        compression_scheme_prob = 1/(m*binom(m, d))
    if k >= m-d:
        return 1
    return binomial_tail_inverse(k, m-d, delta*compression_scheme_prob)


def catoni_4_2(k, m, d, delta, mprime):
    """
    Implements Theorem 4.2 of Catoni (2004) - Improved VC Bounds

    Args:
        k (int): Number of errors of the classifier on the sample.
        m (int): Number of examples of the sample.
        d (int): Number of examples in the compressed sample.
        delta (float between 0 and 1): Confidence parameter.
        mprime (int): Ghost sample size.

    Returns epsilon, the upper bound between 0 and 1.
    """

    # Converting to Catoni's notation
    r1 = k/m # Empirical risk
    N = m # Number of examples
    h = d # VC dimension
    k = mprime/m
    epsilon = delta

    d_star = h * np.log(np.e*(k+1)*N/h) - np.log(epsilon)
    dprime = d_star*(1+1/k)**2 * (1 + np.log(2*np.e*np.sqrt(d_star/np.pi))/(2*d_star))**2

    B = 1/(1+2*dprime/N) * \
        (r1 + dprime/N + np.sqrt(
            2*dprime*r1*(1-r1)/N + dprime**2/N**2
        ))

    for i, r in enumerate(B):
        if r > 1/2 or B[i] > 1/2:
            B[i] = 1
    return B


def catoni_4_6(k, m, d, delta, mprime=None, max_mprime=None):
    """Theorem 4.6 of Catoni (2004) - Improved VC Bounds

    Args:
        k (int): Number of errors of the classifier on the sample.
        m (int): Number of examples of the sample.
        d (int): Number of examples in the compressed sample.
        delta (float between 0 and 1): Confidence parameter.
        mprime (int): Ghost sample size.

    Returns epsilon, the upper bound between 0 and 1.
    """
    if mprime is None: # Must optimize
        if max_mprime is None:
            max_mprime = 100*m
        best_bound = 1
        current_bound = 1
        mprime = m
        while best_bound >= current_bound and mprime <= max_mprime:
            current_bound = catoni_4_6(k, m, d, delta, mprime)
            if current_bound < best_bound:
                best_bound = current_bound
            mprime += m

        return best_bound

    # Converting to Catoni's notation
    r1 = k/m # Empirical risk
    N = m # Number of examples
    h = d # VC dimension
    k = mprime/m
    epsilon = delta

    d_star = h * np.log(np.e*(k+1)*N/h) - np.log(epsilon)
    d_prime = d_star * (1+1/k)**2

    B = 1/(1+2*d_prime/N) * \
        (r1 + d_prime/N + 1/N*np.sqrt(
            2*d_prime*r1*(1-r1)*N + d_prime**2
        ))

    if r1 > 1/2 or B > 1/2:
        B = 1
    return B
    # for i, r in enumerate(B):
    #     if r > 1/2 or B[i] > 1/2:
    #         B[i] = 1
    # return B


def lugosi_chaining(k, m, d, delta):
    """Theorem 1.16 of Lugosi (2002) - Pattern classification and learning theory

    Args:
        k (int): Number of errors of the classifier on the sample.
        m (int): Number of examples of the sample.
        d (int): Number of examples in the compressed sample.
        delta (float between 0 and 1): Confidence parameter.
        mprime (int): Ghost sample size.

    Returns epsilon, the upper bound between 0 and 1.
    """

    a = (d+1)*(np.log(2)+2)/(2*d)

    epsilon = 24 / np.sqrt(m) * np.sqrt(2*d) * (np.sqrt(a) + np.sqrt(np.pi)*np.exp(a)/2 * erfc(np.sqrt(a)))

    return k/m + epsilon + np.sqrt(-np.log(delta)/(2*m))
