import numpy as np
import os, sys
sys.path.append(os.getcwd())
import csv
import pandas as pd

from graal_utils import Timer
from time import time

from source import optimize_mprime, hypinv_reldev_upperbound
from source.utils import sauer_shelah


risk, d, delta = 0.1, 20, 0.05
ms = np.array(list(range(20, 400, 10))
              + list(range(400, 1000, 50))
              + list(range(1000, 10_000, 100))
              + list(range(10_000, 40_001, 1000))
              )
os.chdir('./scripts/bounds_comparison/')

path = './data/'
os.makedirs(path, exist_ok=True)

filename = f'best_mprimes{risk=}_{d=}_{delta=}.csv'

if os.path.exists(path + filename):
    df = pd.read_csv(path + filename, sep=',', header=0)
    current_m = df['m'].iloc[-1]
    hti_mp = df['HTI-mprime'].iloc[-1]
    htird_mp = df['HTI-RD-mprime'].iloc[-1]
    file = open(path + filename, 'a', newline='')
    csvwriter = csv.writer(file)
else:
    current_m = 0
    hti_mp = 0
    htird_mp = 0
    file = open(path + filename, 'w', newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['m', 'HTI-mprime', 'HTI-bound', 'HTI-RD-mprime', 'HTI-RD-bound'])
    file.flush()

with Timer():
    for i, m in enumerate(ms):
        if m > current_m:
            if m <= 100:
                hti_mp_init = m
                htird_mp_init = m
            else:
                hti_mp_init = hti_mp - 100
                htird_mp_init = htird_mp - 100

            t0 = time()

            hti_mp, hti_bound = optimize_mprime(
                                        int(risk*m),
                                        m,
                                        sauer_shelah(d),
                                        delta,
                                        max_mprime=hti_mp_init*100,
                                        min_mprime=hti_mp_init,
                                        early_stopping=200,
                                        return_bound=True,
                                        )
            htird_mp, htird_bound = optimize_mprime(
                                        int(risk*m),
                                        m,
                                        sauer_shelah(d),
                                        delta,
                                        max_mprime=htird_mp_init*100,
                                        min_mprime=htird_mp_init,
                                        bound=hypinv_reldev_upperbound,
                                        early_stopping=200,
                                        return_bound=True,
                                        )

            row = [m, hti_mp, hti_bound, htird_mp, hti_bound]
            csvwriter.writerow(row)
            file.flush()
            print(f'Optimized {m=} in {time()-t0:.3f}s, {i+1}/{len(ms)} done.', end='\r')

    file.close()