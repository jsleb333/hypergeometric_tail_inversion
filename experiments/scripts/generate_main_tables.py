import os, sys
from numpy.lib.function_base import trim_zeros
sys.path.append(os.getcwd())
import python2latex as p2l
import numpy as np
import pandas as pd

from source.generalization_bounds import hypinv_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, sample_compression_bound
from source.utils import sauer_shelah

exp_name = '2021-01-07'
n_examples = 250
min_true_degree = 2
max_true_degree = 7
n_runs = 100
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
doc = p2l.Document('main_tables', filepath=path)

n_bounds = len(bounds)
n_cols = 1 + n_bounds
n_degrees = 10
n_true_degrees = max_true_degree - min_true_degree + 1
n_rows = n_true_degrees + 3

table1 = doc.new(p2l.Table(shape=(n_rows, n_cols)))
table2 = doc.new(p2l.Table(shape=(n_rows-1, n_cols)))

table1[0:2,0].multicell('$n^*$', v_shift='0pt')
table1[0,1:] = 'Success rate'

table2[0:2,0].multicell('$n^*$', v_shift='0pt')
table2[0,1:] = 'Bound value'

for bound_name, i in zip(bounds.keys(), range(1, n_cols, 1)):
    table1[1, i] = bound_name
    table2[1, i] = bound_name

table1[1].add_rule()
table2[1].add_rule()

bound_values = np.zeros((n_degrees, n_runs, n_true_degrees, n_bounds))

for i, true_degree in enumerate(range(min_true_degree, max_true_degree+1)):
    filename = f'n={true_degree}-m={n_examples}-noise={noise}-runs={n_runs}-C={C}'
    df = pd.read_csv(path + filename + '.csv', sep=',', header=0)
    for n in range(1, n_degrees+1):
        n_errors = df[f'{n=}-tr-risk']*n_examples
        for j, bound in enumerate(bounds.values()):
            bound_values[n-1, :, i, j] = [bound(int(k), n+1, mprimes[n]) for k in n_errors]


success_rate = np.zeros((n_true_degrees, n_bounds))
for i, true_degree in enumerate(range(min_true_degree, max_true_degree+1)):
    table1[i+2, 0] = true_degree
    table2[i+2, 0] = true_degree
    for j in range(n_bounds):
        for run in range(n_runs):
            if np.argmin(bound_values[:, run, i, j])+1 == true_degree:
                success_rate[i, j] += 1/n_runs
        table1[i+2, j+1] = success_rate[i, j]
        table2[i+2, j+1] = np.mean(np.min(bound_values[:,:,i,j], axis=0))

    table1[i+2,1:].highlight_best()
    table2[i+2,1:].highlight_best(mode='low')

table1[-2].add_rule()
table1[-1, 0] = 'All'
table1[-1, 1:] = np.mean(success_rate, axis=0)
table1[-1, 1:].highlight_best()

print('Building...')
doc.build()