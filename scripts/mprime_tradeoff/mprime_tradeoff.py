import numpy as np
import csv
from graal_utils import Timer
import os

from source import hypinv_upperbound, hypergeometric_left_tail_inverse, berkopec_hypergeometric_left_tail_inverse


def compute_bound_data(k, m, delta=0.05, d=10, max_mprime=3_00):
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

    max_mprime = 10_000
    risks = np.linspace(0, .5, 11)
    ms = [100, 200, 300, 500, 1000]
    ds = [10]

    for m in ms:
        for d in ds:
            for risk in risks:
                k = int(m*risk)
                path = './scripts/mprime_tradeoff/data/'
                os.makedirs(path, exist_ok=True)
                filename = f'mprime_tradeoff-m={m}_k={k}_d={d}'
                with open(path + filename + '.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['mprime', 'bound'])
                    with Timer(f'm={m}, k={k}, d={d}'):
                        bounds, mprimes = compute_bound_data(k, m, d=d, max_mprime=max_mprime)
                        for mp, bound in enumerate(bounds, start=1):
                            writer.writerow([mp, bound])
                        print(m, k, mprimes, bounds[mprimes[0]-1])
