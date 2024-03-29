import numpy as np

from hypergeo import *
from graal_utils import Timer

d = 10
k, m, M = 20, 2000, 10000
delta = .05/(4*(np.e*M/d)**d)

N = 10

with Timer('bisect') as t:
    for _ in range(N):
        K = hypergeometric_tail_inverse(k, m, delta, M)
    print(K)

with Timer('berkopec above') as t:
    for _ in range(N):
        K = berkopec_hypergeometric_tail_inverse(k, m, delta, M, 'above')
    print(K)
with Timer('berkopec below') as t:
    for _ in range(N):
        K = berkopec_hypergeometric_tail_inverse(k, m, delta, M, 'below')
    print(K)

with Timer('log berkopec above') as t:
    for _ in range(N):
        K = logberkopec_hypergeometric_tail_inverse(k, m, np.log(delta), M, 'above')
    print(K)
with Timer('log berkopec below') as t:
    for _ in range(N):
        K = logberkopec_hypergeometric_tail_inverse(k, m, np.log(delta), M, 'below')
    print(K)

with Timer('naive above') as t:
    for _ in range(N):
        K = naive_hypergeometric_tail_inverse(k, m, delta, M, 'above')
    print(K)
with Timer('naive below') as t:
    for _ in range(N):
        K = naive_hypergeometric_tail_inverse(k, m, delta, M, 'below')
    print(K)
