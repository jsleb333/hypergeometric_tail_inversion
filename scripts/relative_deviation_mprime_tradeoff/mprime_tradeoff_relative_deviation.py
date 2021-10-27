import numpy as np
import csv
from graal_utils import Timer

from hypergeo import hypinv_reldev_upperbound
from hypergeo.utils import sauer_shelah

import os
path = os.path.dirname(__file__) + '/data/'


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

    os.makedirs(path, exist_ok=True)

    # Generates all the data and saves it
    for k, m, d, delta in params:
        filename = f'mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}'

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
