import numpy as np
from time import sleep

from source import *
from graal_utils import Timer

d = 10
k, m, M = 10, 200, 1000
delta = .05
# delta = .05/(4*np.e*1200/d)**d

N = 10

with Timer('berkopec above') as t:
    for _ in range(N):
        K = hypergeometric_left_tail_inverse(k, m, delta, M, 'above')
    print(K)
with Timer('berkopec below') as t:
    for _ in range(N):
        K = hypergeometric_left_tail_inverse(k, m, delta, M, 'below')
    print(K)

with Timer('log above') as t:
    for _ in range(N):
        K = log_hypergeometric_left_tail_inverse(k, m, np.log(delta), M, 'above')
    print(K)
with Timer('log below') as t:
    for _ in range(N):
        K = log_hypergeometric_left_tail_inverse(k, m, np.log(delta), M, 'below')
    print(K)

with Timer('naive above') as t:
    for _ in range(N):
        K = naive_hypergeometric_left_tail_inverse(k, m, delta, M, 'above')
    print(K)
with Timer('naive below') as t:
    for _ in range(N):
        K = naive_hypergeometric_left_tail_inverse(k, m, delta, M, 'below')
    print(K)

