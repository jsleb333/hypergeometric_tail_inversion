import os, sys
sys.path.append(os.getcwd())
import python2latex as p2l
import numpy as np
import pandas as pd

from source.generalization_bounds import hypinv_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, sample_compression_bound
from source.utils import sauer_shelah

exp_name = 'test'
n_examples = 100
min_true_degree = 2
max_true_degree = 4
n_runs = 10
noise = 1.5
C = 10e6
delta = 0.05
delta_d = lambda d: delta * 6 / (np.pi**2 * d**2)
from experiments.scripts.best_mprimes import mprimes

bounds = {
    'HTI': lambda k, d, mp: hypinv_upperbound(k, n_examples, sauer_shelah(d), delta_d(d), mprime=mp),
    'SC': lambda k, d, mp: sample_compression_bound(k, n_examples, d, delta_d(d)*n_examples),
    'VP': lambda k, d, mp: vapnik_pessismistic_bound(k, n_examples, sauer_shelah(d), delta_d(d)),
    'VRD': lambda k, d, mp: vapnik_relative_deviation_bound(k, n_examples, sauer_shelah(d), delta_d(d)),
}

path = f'./experiments/results/{exp_name}/'
doc = p2l.Document('tables', filepath=path)

n_bounds = len(bounds)
n_cols = 3 + n_bounds*2
n_degrees = 10
n_rows = n_degrees + 2

for true_degree in range(min_true_degree, max_true_degree+1):
    filename = f'n={true_degree}-m={n_examples}-noise={noise}-runs={n_runs}-C={C}'
    df = pd.read_csv(path + filename + '.csv', sep=',', header=0)

    table = doc.new(p2l.Table(shape=(n_rows, n_cols)))

    table[0:2,0].multicell('$n$', v_shift='-2pt')
    table[0:2,1].multicell('Train risk', v_shift='-2pt')
    table[0:2,2].multicell('Test risk', v_shift='-2pt')

    for bound_name, i in zip(bounds.keys(), range(3, n_cols, 2)):
        table[0, i:i+2] = bound_name
        table[0, i:i+2].add_rule(trim_left=True, trim_right=True)
        table[1, i:i+2] = ['Bound', 'Rate']

    table[1].add_rule()

    for bound, i in zip(bounds.values(), range(3, n_cols, 2)):
        bound_values = np.zeros((n_degrees, n_runs))
        for n in range(1, n_degrees+1):
            tr_risk, ts_risk = df[f'{n=}-tr-risk'], df[f'{n=}-ts-risk']
            table[n+1, 0:3] = n, np.mean(tr_risk), np.mean(ts_risk)
            n_errors = tr_risk*n_examples
            bound_values[n-1] = [bound(int(k), n+1, mprimes[n]) for k in n_errors]
            table[n+1, i] = np.mean(bound_values[n-1])

        for n in range(1, n_degrees+1):
            success_rate = 0
            for run in range(n_runs):
                if np.argmin(bound_values[:,run])+1 == n:
                    success_rate += 1/n_runs
            table[n+1, i+1] = success_rate

    table[true_degree+1].apply_command(p2l.bold)


doc.build()