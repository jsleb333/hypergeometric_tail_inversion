import numpy as np
import os, sys
sys.path.append(os.getcwd())
import pandas as pd

import python2latex as p2l
from graal_utils import Timer, timed

from source import optimize_mprime
from source import hypinv_upperbound, hypinv_reldev_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, sample_compression_bound
from source.utils import sauer_shelah


risk, d, delta = 0.1, 20, 0.05
ms = np.array(list(range(20, 400, 10))
              + list(range(400, 1000, 50))
              + list(range(1000, 10_000, 100))
              + list(range(10_000, 40_001, 1000))
              )
ks = np.array([int(risk*m) for m in ms])

os.chdir('./scripts/bounds_comparison/')
path = './data/'
os.makedirs(path, exist_ok=True)
filename = f'best_mprimes_{risk=}_{d=}_{delta=}.csv'
df = pd.read_csv(path + filename, sep=',', header=0)


plot = p2l.Plot(plot_name=f'growth_rates_{d=}_{delta=}',
                plot_path='figures',
                as_float_env=False,
                width='7.45cm',
                height='6cm',
                lines='1pt',
                xmode='log',
                ymode='log',
                )
plot.axis.kwoptions['legend style'] = r'{font=\scriptsize}'
plot.axis.kwoptions['y label style'] = r'{yshift=-.2cm}'
plot.axis.kwoptions['legend cell align'] = '{left}'

plot.x_min = 20
plot.x_max = max(ms)
plot.y_min = 0
plot.y_max = 1.5
plot.x_label = "Sample size $m$"
plot.y_label = 'Upper bound on $R_\mathcal{D}(h) - R_S(h)$'
# plot.y_label = 'Upper bound on the excess risk'
plot.legend_position = 'south west'
# plot.legend_position = 'north east'

# VRD
plot.add_plot(ms,
              vapnik_relative_deviation_bound(ks, ms, sauer_shelah(d), delta)-risk,
              legend='VRD')
# VP
plot.add_plot(ms,
              vapnik_pessismistic_bound(ks, ms, sauer_shelah(d), delta)-risk,
              legend='VP')
# HTI-RD
mps = df['HTI-RD-mprime']
bounds = [hypinv_reldev_upperbound(k, m, sauer_shelah(d), delta, mprime=mp) for k, m, mp in zip(ks, ms, mps)]
bounds = np.array([b if b < 1 else 1 for b in bounds])
plot.add_plot(ms,
              bounds-risk,
            #   'dashed',
              legend='HTI-RD')
# HTI
plot.add_plot(ms,
              df['HTI-bound']-risk,
              legend='HTI')
# SC
plot.add_plot(ms,
              [sample_compression_bound(k, m, d, delta)-risk for k, m in zip(ks, ms)],
            #   'dotted',
              legend='SC')

plot.add_plot(ms, np.sqrt(d/ms), color='gray', line_width='.5pt', legend=r'\scalebox{.8}{\normalsize $\sqrt{d/m}$}')

# Empirical risk
# plot.add_plot([ms[0], ms[-1]], [risk, risk], 'dashed', color='gray', line_width='.5pt')


filename = 'bounds_comparison_m_opti'
doc = p2l.Document(filename, doc_type='standalone')
doc.add_package('mathalfa', cal='dutchcal', scr='boondox')
doc += plot
print('Building...')
doc.build(delete_files='all')
