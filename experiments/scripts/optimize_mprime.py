import os, sys
sys.path.append(os.getcwd())
import numpy as np

from graal_utils import Timer

from source.generalization_bounds import optimize_mprime
from source.utils import sauer_shelah


n_examples = 250
min_degree = 1
max_degree = 10
delta = 0.05
delta_d = lambda d: delta * 6 / (np.pi**2 * d**2)

mprimes = {}
for n in range(min_degree, max_degree+1):
    with Timer(f'Classifier degree: {n}'):
        mprime = optimize_mprime(0, n_examples, sauer_shelah(n+1), delta_d(n+1), min_mprime=2*n_examples, max_mprime=6*n_examples)

        mprimes[n] = mprime
        print(mprime)

filename = './experiments/scripts/best_mprimes.py'
with open(filename, 'w') as file:
    file.write(f'{mprimes=}')
