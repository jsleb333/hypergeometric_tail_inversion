import numpy as np
import csv
from itertools import product
import os, sys
sys.path.append(os.getcwd())
from graal_utils import Timer
import pandas as pd
import xarray as xr

from source import hypinv_upperbound


def compute_bound_data(k, m, delta=0.05, d=10, max_mprime=300):
    growth_function = lambda M: (np.e*M/d)**d

    bounds = np.ones(max_mprime)
    best_bound = 1
    for mprime in range(1, max_mprime+1):
        bound = hypinv_upperbound(k, m, growth_function, delta, mprime)
        bounds[mprime-1] = bound
        if bound <= best_bound:
            best_bound = bound

    return bounds


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    max_mprime = 10_000
    risks = np.linspace(0, .5, 11)
    ms = [100, 200, 300, 500, 1000]
    ds = [5, 10, 20, 35, 50]
    deltas = [0.0001, 0.0025, 0.05, 0.1]

    path = './scripts/mprime_tradeoff/data/'
    os.makedirs(path, exist_ok=True)

    # Generates all the data and saves it
    for m, risk, d, delta in product(ms, risks, ds, deltas):
        k = int(m*risk)

        filename = f'mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}'
        if os.path.exists(path + filename + '.csv'):
            continue

        with open(path + filename + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['mprime', 'bound'])
            with Timer(f'm={m}, k={k}, d={d}, delta={delta}'):
                bounds = compute_bound_data(k, m, delta=delta, d=d, max_mprime=max_mprime)
                for mp, bound in enumerate(bounds, start=1):
                    writer.writerow([mp, bound])

    # Computes the optimal value of mprime and saves it
    best_mprimes = np.zeros((len(ms), len(risks), len(ds), len(deltas)))
    best_bounds = np.zeros((len(ms), len(risks), len(ds), len(deltas)))
    bounds_at_mprime_equals_m = np.zeros((len(ms), len(risks), len(ds), len(deltas)))

    for m_idx, risk_idx, d_idx, delta_idx in product(range(len(ms)),
                                                     range(len(risks)),
                                                     range(len(ds)),
                                                     range(len(deltas))
                                                     ):
        m, risk, d, delta = ms[m_idx], risks[risk_idx], ds[d_idx], deltas[delta_idx]
        k = int(m*risk)

        filepath = path + f'mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
        df = pd.read_csv(filepath, sep=',', header=0)

        mprimes, bounds = df['mprime'], df['bound']
        min_idx = np.argmin(bounds)
        best_mprimes[m_idx, risk_idx, d_idx, delta_idx] = mprimes[min_idx]
        best_bounds[m_idx, risk_idx, d_idx, delta_idx] = bounds[min_idx]
        bounds_at_mprime_equals_m[m_idx, risk_idx, d_idx, delta_idx] = bounds[m-1]

    data = xr.Dataset(
        data_vars={
            'bound': (['m', 'risk', 'd', 'delta'], best_bounds),
            'bound_at_mprime=m': (['m', 'risk', 'd', 'delta'], bounds_at_mprime_equals_m),
            'mprime': (['m', 'risk', 'd', 'delta'], best_mprimes)
        },
        coords={
            'm': ms,
            'risk': risks,
            'd': ds,
            'delta': deltas
        }
    )
    data.to_netcdf(path + 'optimal_bound.nc')