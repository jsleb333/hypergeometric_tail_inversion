import numpy as np
from scipy.special import binom, betaincinv
import os, sys
sys.path.append(os.getcwd())

import python2latex as p2l
from graal_utils import Timer

from source import optimize_mprime, hypinv_upperbound, hypinv_reldev_upperbound


def sauer_shelah(d):
    def _sauer_shelah(m):
        return (np.e*m/d)**d
    return _sauer_shelah


def vapnik_pessismistic_bound(k, m, d, delta):
    e = (np.log(4) + d*np.log(2*m*np.e/d) - np.log(delta))/m
    return (k+1)/m + np.sqrt(e)


def vapnik_relative_deviation_bound(k, m, d, delta):
    e = (np.log(4) + d*np.log(2*m*np.e/d) - np.log(delta))/m
    r = k/m
    return r + 2*e*(1 + np.sqrt(1 + r/e))


def bininv(k, m, delta):
    return 1-betaincinv(m-k, k+1, delta)

def sample_compression_bound(k, m, d, delta):
    return bininv(k, m-d, delta/(m*binom(m, d)))


m, d, delta = 1500, 15, 0.05
# mp = optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=8000, min_mprime=4000)
# mp = 678 # Previous line returns this number when run with m=200, d=15 and delta=0.05
mp = 5937 # Previous line returns this number when run with m=1500, d=15 and delta=0.05
# print(mp)

# mp_rd = optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=22_000, min_mprime=18_000, bound=hypinv_reldev_upperbound)
# mp_rd = 2535 # Previous line returns this number when run with m=200, d=15 and delta=0.05
mp_rd = 20587 # Previous line returns this number when run with m=1500, d=15 and delta=0.05
# print(mp_rd)

plot = p2l.Plot(plot_name=f'bounds_comp_{m=}_{d=}_{delta=}',
                plot_path='figures',
                as_float_env=False,
                width='13cm',
                height='7cm',
                lines='1pt',)
plot.axis.kwoptions['legend style'] = r'{font=\tiny}'
# plot.axis.kwoptions['restrict y to domain'] = r'{0:1}'

bounds = [
    (
        "Sample compression",
        sample_compression_bound,
        "dotted"
    ),
    (
        "HypInv",
        lambda k,m,d,delta: hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mp),
        ""
    ),
    (
        "RD HypInv",
        lambda k,m,d,delta: hypinv_reldev_upperbound(k, m, sauer_shelah(d), delta, mprime=mp_rd),
        ""
    ),
    (
        "Vapnik's pessimistic",
        vapnik_pessismistic_bound,
        "dashed"
    ),
    (
        "Vapnik's RD",
        vapnik_relative_deviation_bound,
        "dashed"
    ),
]

ks = np.arange(0, m, 1)
for name, bound, style in bounds:
    with Timer(name):
        plot.add_plot(ks/m, [bound(k, m, d, delta) for k in ks], style, legend=name)

plot.add_plot([0,1], [0,1], color='black')
plot.x_min = 0
plot.y_min = 0
plot.x_label = "Empirical risk"
plot.y_label = 'Upper bound on the true risk'

plot.legend_position = 'north west'

os.chdir('./scripts/bounds_comparison/')

doc = p2l.Document(f'bounds_comparison', doc_type='standalone')
doc += plot

print('Building...')
doc.build()