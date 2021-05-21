import numpy as np
import os, sys
sys.path.append(os.getcwd())

import python2latex as p2l
from graal_utils import Timer, timed

from source import optimize_mprime
from source import hypinv_upperbound, hypinv_reldev_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, sample_compression_bound, catoni_4_6, lugosi_chaining
from source.utils import sauer_shelah


risk, d, delta = 0.1, 50, 0.05
print(f'{risk=}, {d=}, {delta=}')

bounds = [
    (
        "HTI",
        lambda k, m, mp: hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mp),
        ""
    ),
    (
        "SC",
        lambda k, m, mp: sample_compression_bound(k, m, d, delta),
        '',
    )
]
bounds.reverse()

ms = np.array(list(range(d,1000,10)) + [int(m) for m in np.logspace(10, 20, num=50, base=2)])

bound_values = {name:np.zeros_like(ms, dtype=float) for name, *_ in bounds}

mp = ms[0]
for name, bound, _ in bounds:
    with Timer(name):
        for i, m in enumerate(ms):
            k = int(risk*m)
            if m <= 5000:
                mp = 4*m
            elif m <= 10_000:
                mp = 6*m
            else:
                mp = 9*m
            bound_values[name][i] = bound(k, m, mp) - risk
        nonvacuous_i = np.argmin((bound_values[name] + risk - .5)**2)
        print(bound_values[name][nonvacuous_i], ms[nonvacuous_i])

plot = p2l.Plot(plot_name=f'sc_comp_m_{d=}_{delta=}',
                plot_path='figures',
                as_float_env=False,
                # width='15cm',
                # height='15cm',
                width='7.45cm',
                height='6cm',
                lines='1pt',
                xmode='log',
                ymode='log',
                palette=reversed(p2l.holi(len(bounds)))
                )
plot.axis.kwoptions['legend style'] = r'{font=\scriptsize}'
plot.axis.kwoptions['y label style'] = r'{yshift=-.2cm}'
plot.axis.kwoptions['legend cell align'] = '{left}'

plot.x_min = d
plot.x_max = 1e6
plot.y_min = 0
# plot.y_max = 10
plot.y_max = 1.1
plot.x_label = "Sample size $m$"
plot.y_label = 'Upper bound on $R_\mathcal{D}(h) - R_S(h)$'
plot.legend_position = 'south west'
# plot.legend_position = 'north east'


for name, _, style in bounds:
    plot.add_plot(ms, bound_values[name], style, legend=name)

os.chdir('./scripts/bounds_comparison/')

plot.add_plot(ms, np.sqrt(d/ms), color='gray', line_width='.5pt', legend=r'\scalebox{.8}{\normalsize $\sqrt{d/m}$}')

filename = 'sc_comparison_m'
doc = p2l.Document(filename, doc_type='standalone')
doc.add_package('times')
doc += plot
print('Building...')
doc.build(delete_files='all')
