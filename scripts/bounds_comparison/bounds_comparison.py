import numpy as np
import os, sys
sys.path.append(os.getcwd())

import python2latex as p2l
from graal_utils import Timer

from source import optimize_mprime
from source import hypinv_upperbound, hypinv_reldev_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, sample_compression_bound
from source.utils import sauer_shelah


m, d, delta = 500, 20, 0.05
# mp = optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=8000, min_mprime=4000)
# mp = 678 # Optimization returns this number when run with m=200, d=15 and delta=0.05
mp = 4001 # Optimization returns this number when run with m=500, d=20 and delta=0.05
# mp = 4071 # Optimization returns this number when run with m=1000, d=10 and delta=0.05
# mp = 5937 # Optimization returns this number when run with m=1500, d=15 and delta=0.05
# print(mp)

# mp_rd = optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=22_000, min_mprime=18_000, bound=hypinv_reldev_upperbound)
# mp_rd = 2535 # Optimization returns this number when run with m=200, d=15 and delta=0.05
mp_rd = 18005 # Optimization returns this number when run with m=500, d=20 and delta=0.05
# mp_rd = 18012 # Optimization returns this number when run with m=1000, d=10 and delta=0.05
# mp_rd = 20587 # Optimization returns this number when run with m=1500, d=15 and delta=0.05
# print(mp_rd)

plot = p2l.Plot(plot_name=f'bounds_comp_{m=}_{d=}_{delta=}',
                plot_path='figures',
                as_float_env=False,
                width='14cm',
                height='8cm',
                lines='1pt',)
plot.axis.kwoptions['legend style'] = r'{font=\footnotesize}'
plot.axis.kwoptions['y label style'] = r'{yshift=-.3cm}'
plot.axis.kwoptions['legend cell align'] = '{left}'

bounds = [
    (
        "SC",
        # "Sample compression",
        lambda k: sample_compression_bound(k, m, d, delta),
        "dotted"
    ),
    (
        "HTI",
        lambda k: hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mp),
        ""
    ),
    (
        "HTI-RD",
        lambda k: hypinv_reldev_upperbound(k, m, sauer_shelah(d), delta, mprime=mp_rd),
        ""
    ),
    (
        "VP",
        # "Vapnik's pessimistic",
        lambda k: vapnik_pessismistic_bound(k, m, sauer_shelah(d), delta),
        "dash dot"
    ),
    (
        "VRD",
        # "Vapnik's RD",
        lambda k: vapnik_relative_deviation_bound(k, m, sauer_shelah(d), delta),
        "dash dot"
    ),
]

ks = np.arange(0, m, 1)
for name, bound, style in bounds:
    with Timer(name):
        plot.add_plot(ks/m, [bound(k) for k in ks], style, legend=name)

plot.add_plot([0,1], [0,1], 'dashed', color='gray')

plot.x_min = 0
plot.x_max = 1.02
plot.y_min = 0
plot.y_max = 1.6
plot.x_label = "Empirical risk"
plot.y_label = 'Upper bound on the true risk'
plot.legend_position = 'south east'


os.chdir('./scripts/bounds_comparison/')

filename = 'bounds_comparison'

doc = p2l.Document(filename, doc_type='standalone')
doc += plot

print('Building...')
doc.build(delete_files='all')
