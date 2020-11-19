import numpy as np
import csv
from graal_utils import Timer

from source import hypinv_upperbound, hypergeometric_left_tail_inverse, berkopec_hypergeometric_left_tail_inverse


def compute_bound_data(k, m, delta=0.05, d=10, max_mprime=10_000):
    growth_function = lambda M: (np.e*M/d)**d

    bounds = np.ones(max_mprime)
    best_bound = 1
    for mprime in range(1, max_mprime+1):
        bound = hypinv_upperbound(k, m, growth_function, delta, mprime)
        bounds[mprime-1] = bound
        if bound <= best_bound:
            best_bound = bound

    best_mprimes = []
    for mp, bound in enumerate(bounds, start=1):
        if np.isclose(bound, best_bound, rtol=10e-8, atol=0):
            best_mprimes.append(mp)

    return bounds, best_mprimes


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    risks = [0]
    # risks = np.linspace(0, .1, 2)
    ms = [200]

    for m in ms:
        for risk in risks:
            k = int(m*risk)
            with Timer(f'm={m}, k={k}'):
                bounds, mprimes = compute_bound_data(k, m)
                print(m, k, mprimes, bounds[mprimes[0]-1])
                plt.plot(bounds)
                plt.scatter(mprimes, [bounds[mprimes[0]-1]])
    plt.ylim((0,1))
    plt.xscale('log')
    plt.grid()
    plt.show()
