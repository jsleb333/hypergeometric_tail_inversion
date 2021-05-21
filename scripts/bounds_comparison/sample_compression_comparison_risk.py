import numpy as np
import os, sys
sys.path.append(os.getcwd())

import python2latex as p2l
from graal_utils import Timer

from source import optimize_mprime
from source import hypinv_upperbound, hypinv_reldev_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, sample_compression_bound, catoni_4_6
from source.utils import sauer_shelah


m, d, delta = 2000, 50, 0.05
# mp = optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=10*m, min_mprime=3*m, early_stopping=1000)
# print(mp)
# mp = 678 # Optimization returns this number when run with m=200, d=15 and delta=0.05
# mp = 1693 # Optimization returns this number when run with m=500, d=20 and delta=0.05
# mp = 4071 # Optimization returns this number when run with m=1000, d=10 and delta=0.05
# mp = 3611 # Optimization returns this number when run with m=1000, d=20 and delta=0.05
# mp = 5937 # Optimization returns this number when run with m=1500, d=15 and delta=0.05
mp = 6970 # Optimization returns this number when run with m=2000, d=50 and delta=0.05
# mp = m


plot = p2l.Plot(plot_name=f'sc_comparison_risk_{m=}_{d=}_{delta=}',
                plot_path='figures',
                as_float_env=False,
                width='7.45cm',
                height='6cm',
                lines='1pt',)
plot.axis.kwoptions['legend style'] = r'{font=\scriptsize}'
plot.axis.kwoptions['y label style'] = r'{yshift=-.3cm}'
plot.axis.kwoptions['legend cell align'] = '{left}'

palette = p2l.holi(2)

bounds = [
    (
        "SC",
        lambda k: sample_compression_bound(k, m, d, delta),
        '',
        palette[1]
    ),
    (
        "HTI",
        lambda k: hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mp),
        "",
        palette[0]
    )
]
bounds.reverse()

ks = np.arange(0, m, 5)
for name, bound, style, color in bounds:
    with Timer(name):
        bs = [bound(k) for k in ks]
        plot.add_plot(ks/m, bs, style, color=color, legend=name)
        print(name, bs[0])

plot.add_plot([0,1], [0,1], line_width='.5pt', color='gray!50')

plot.x_min = 0
plot.x_max = 1.02
plot.y_min = 0
plot.y_max = 1.02
plot.y_ticks = np.linspace(0, 1.25, 6)

plot.x_label = "Empirical risk $R_S(h)$"
plot.y_label = 'Upper bound on the true risk'
plot.legend_position = 'south east'


os.chdir('./scripts/bounds_comparison/')

filename = 'sc_comparison_risk'

doc = p2l.Document(filename, doc_type='standalone')
doc.add_package('times')
doc += plot

print('Building...')
doc.build(delete_files='all')
