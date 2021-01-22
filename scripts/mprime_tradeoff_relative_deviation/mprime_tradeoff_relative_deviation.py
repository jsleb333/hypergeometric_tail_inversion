import numpy as np
import csv
from itertools import product
import os, sys
sys.path.append(os.getcwd())
from graal_utils import Timer
import pandas as pd
import xarray as xr

from source import hypinv_reldev_upperbound
from source.utils import sauer_shelah


if __name__ == "__main__":

    k, m, d, delta = 0, 200, 10, 0.05
    max_mprime = 20_000

    ks = [0, 10, 30, 50, 100]
    ms = [100, 300, 1000]
    ds = [5, 10, 20, 35, 50]
    deltas = [0.0001, 0.1]

    params = [(x,m,d,delta) for x in ks] +\
             [(k,x,d,delta) for x in ms] +\
             [(k,m,x,delta) for x in ds] +\
             [(k,m,d,x) for x in deltas]

    path = './scripts/mprime_tradeoff_relative_deviation/data/'
    os.makedirs(path, exist_ok=True)

    # Generates all the data and saves it
    for k, m, d, delta in params:
        filename = f'mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}'
        # if os.path.exists(path + filename + '.csv'):
        #     continue

        with open(path + filename + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['mprime', 'bound'])
            with Timer(f'm={m}, k={k}, d={d}, delta={delta}'):
                bounds = np.zeros(max_mprime)
                for mp in range(1, max_mprime+1):
                    bounds[mp-1] = hypinv_reldev_upperbound(k, m, sauer_shelah(d), delta, mprime=mp)
                    print(f'Computing bounds: {mp}/{max_mprime}', end='\r')
                for mp, bound in enumerate(bounds, start=1):
                    writer.writerow([mp, bound])

    # Computes the optimal value of mprime and saves it
    best_mprimes = np.zeros((len(ms), len(ks), len(ds), len(deltas)))
    best_bounds = np.zeros((len(ms), len(ks), len(ds), len(deltas)))
    bounds_at_mprime_equals_m = np.zeros((len(ms), len(ks), len(ds), len(deltas)))

    for m_idx, k_idx, d_idx, delta_idx in product(range(len(ms)),
                                                  range(len(ks)),
                                                  range(len(ds)),
                                                  range(len(deltas))
                                                  ):
        m, k, d, delta = ms[m_idx], ks[k_idx], ds[d_idx], deltas[delta_idx]

        filepath = path + f'mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
        df = pd.read_csv(filepath, sep=',', header=0)

        mprimes, bounds = df['mprime'], df['bound']
        min_idx = np.argmin(bounds)
        best_mprimes[m_idx, k_idx, d_idx, delta_idx] = mprimes[min_idx]
        best_bounds[m_idx, k_idx, d_idx, delta_idx] = bounds[min_idx]
        bounds_at_mprime_equals_m[m_idx, k_idx, d_idx, delta_idx] = bounds[m-1]

    data = xr.Dataset(
        data_vars={
            'bound': (['m', 'k', 'd', 'delta'], best_bounds),
            'bound_at_mprim=m': (['m', 'k', 'd', 'delta'], bounds_at_mprime_equals_m),
            'mprime': (['m', 'k', 'd', 'delta'], best_mprimes)
        },
        coords={
            'm': ms,
            'k': ks,
            'd': ds,
            'delta': deltas
        }
    )
    data.to_netcdf(path + 'optimal_bound_relative_deviation.nc')