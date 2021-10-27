import numpy as np
from graal_utils import Timer
import python2latex as p2l

from hypergeo import hypinv_upperbound, hypinv_reldev_upperbound
from hypergeo.utils import sauer_shelah

import os
path = os.path.dirname(__file__)


m, d, delta = 1000, 20, 0.05
# mp = optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=20*m, min_mprime=3*m, early_stopping=200)
# print(mp)
# mp = 678 # Optimization returns this number when run with k=0, m=200, d=15 and delta=0.05
# mp = 1693 # Optimization returns this number when run with k=0, m=500, d=20 and delta=0.05
# mp = 4071 # Optimization returns this number when run with k=0, m=1000, d=10 and delta=0.05
# mp = 3611 # Optimization returns this number when run with k=0, m=1000, d=20 and delta=0.05
mp = 12_851 # Optimization returns this number when run with k=800, m=1000, d=20 and delta=0.05

# mp_rd = optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=20*m, min_mprime=3*m, bound=hypinv_reldev_upperbound, early_stopping=200)
# print(mp_rd)
# mp_rd = 2535 # Optimization returns this number when run with k=0, m=200, d=15 and delta=0.05
# mp_rd = 6007 # Optimization returns this number when run with k=0, m=500, d=20 and delta=0.05
# mp_rd = 18_012 # Optimization returns this number when run with k=0, m=1000, d=10 and delta=0.05
# mp_rd = 12_462 # Optimization returns this number when run with k=0, m=1000, d=20 and delta=0.05
mp_rd = 15_766 # Optimization returns this number when run with k=800, m=1000, d=20 and delta=0.05

plot = p2l.Plot(plot_name=f'bounds_comp_{m=}_{d=}_{delta=}',
                plot_path=path+'/figures',
                as_float_env=False,
                width='7.45cm',
                height='6cm',
                lines='1pt',)
plot.axis.kwoptions['legend style'] = r'{font=\scriptsize}'
plot.axis.kwoptions['y label style'] = r'{yshift=-.3cm}'
plot.axis.kwoptions['legend cell align'] = '{left}'

ks = np.arange(0, m, 5)
colors = p2l.holi(5)

# HTI
with Timer('HTI opti'):
    plot.add_plot(ks/m,
              [hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mp) for k in ks],
              color=colors[0],
              line_width='.7pt',
              legend='HTI\\textsubscript{opti}',
              )

with Timer("HTI m=m'"):
    plot.add_plot(ks/m,
              [hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=m) for k in ks],
              color=colors[1],
              line_width='.7pt',
              legend="HTI$_{m'=m}$",
              )

# HTI-RD
with Timer('HTI-RD opti'):
    plot.add_plot(ks/m,
              [hypinv_reldev_upperbound(k, m, sauer_shelah(d), delta, mprime=mp_rd) for k in ks],
              'dashed',
              color=colors[-2],
              legend='HTI-RD\\textsubscript{opti}',
              )

with Timer("HTI-RD m=m'"):
    plot.add_plot(ks/m,
              [hypinv_reldev_upperbound(k, m, sauer_shelah(d), delta, mprime=m) for k in ks],
              'dashed',
              color=colors[-1],
              legend="HTI-RD$_{m'=m}$",
              )

plot.add_plot([0,1], [0,1], line_width='.5pt', color='gray!50')

plot.x_min = 0
plot.x_max = 1.02
plot.y_min = 0
plot.y_max = 1.02
plot.y_ticks = np.linspace(0, 1.25, 6)

plot.x_label = "Empirical risk"
plot.y_label = 'Upper bound on the true risk'
plot.legend_position = 'south east'

doc = p2l.Document(f'rd_comp_risk_{m=}', filepath=path, doc_type='standalone')
doc += plot

print('Building...')
doc.build(delete_files='all', show_pdf=False)
