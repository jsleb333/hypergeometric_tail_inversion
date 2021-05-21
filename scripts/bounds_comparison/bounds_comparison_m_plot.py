import numpy as np
import os, sys
sys.path.append(os.getcwd())
import pandas as pd

import python2latex as p2l
from graal_utils import Timer, timed

from source import optimize_mprime, optimize_catoni
from source import hypinv_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, catoni_4_6
from source.utils import sauer_shelah


risk, d, delta = 0.05, 50, 0.05
print(f'{risk=}, {d=}, {delta=}')
ms = np.array(
    list(range(d, 500, 10))
    + list(range(500, 1000, 50))
    + [int(m) for m in np.logspace(10, 20, num=40, base=2)]
)
ks = np.array([int(risk*m) for m in ms])

os.chdir('./scripts/bounds_comparison/')
# path = './data/'
# os.makedirs(path, exist_ok=True)
# filename = f'best_mprimes_{risk=}_{d=}_{delta=}.csv'
# df = pd.read_csv(path + filename, sep=',', header=0)


plot = p2l.Plot(plot_name=f'bounds_comp_m_{d=}_{delta=}',
                plot_path='figures',
                as_float_env=False,
                width='7.45cm',
                height='6cm',
                lines='1pt',
                xmode='log',
                ymode='log',
                palette=reversed(p2l.holi(4))
                )
plot.axis.kwoptions['legend style'] = r'{font=\scriptsize}'
plot.axis.kwoptions['y label style'] = r'{yshift=-.2cm}'
plot.axis.kwoptions['legend cell align'] = '{left}'

plot.x_min = d
plot.x_max = 1e6
plot.y_min = 0
plot.y_max = 1.1
plot.x_label = "Sample size $m$"
plot.y_label = 'Upper bound on $R_\mathcal{D}(h) - R_S(h)$'
plot.legend_position = 'south west'
# plot.legend_position = 'north east'

# # Lugosi
# print('Lugosi')
# plot.add_plot(ms,
#               lugosi_chaining(ks, ms, d, delta)-risk,
#             #   'dotted',
#               legend='Lugosi')

# VRD
with Timer('VRD'):
    plot.add_plot(ms,
                vapnik_relative_deviation_bound(ks, ms, sauer_shelah(d), delta)-risk,
                legend='VRD')

# VP
with Timer('VP'):
    plot.add_plot(ms,
                vapnik_pessismistic_bound(ks, ms, sauer_shelah(d), delta)-risk,
                legend='VP')

# Catoni
with Timer('C4.6'):
    plot.add_plot(ms,
                [catoni_4_6(k, m, d, delta, mprime=optimize_catoni(k, m, d, delta)[1])-risk for k, m in zip(ks, ms)],
                legend='C4.6')

# HTI
with Timer('HTI'):
    def mprime(k, m):
        # return sum(coef*param for coef, param in zip(np.polyfit([d, 1e6], [3.5*m, 9*m], 1), [m, 1]))
        best_mp = int(3.25*m)
        best_bound = 1
        for mp in np.linspace(3.25*m, 10*m, num=(10-3)*4):
            mp = int(mp)
            bound = hypinv_upperbound(k, m, sauer_shelah(d), delta, mp)
            if bound < best_bound:
                best_bound = bound
                best_mp = mp
            # elif bound > best_bound:
            #     break
        # print(m, best_mp)
        return best_mp

    plot.add_plot(ms,
                  [hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mprime(k, m))-risk for k, m in zip(ks, ms)],
                  #   df['HTI-bound']-risk,
                  legend='HTI')


plot.add_plot(ms, np.sqrt(d/ms), color='gray', line_width='.5pt', legend=r'\scalebox{.8}{\normalsize $\sqrt{d/m}$}')

filename = 'bounds_comparison_m'
doc = p2l.Document(filename, doc_type='standalone')
doc.add_package('mathalfa', cal='dutchcal', scr='boondox')
doc.add_package('times')
doc += plot
print('Building...')
doc.build(delete_files='all')
