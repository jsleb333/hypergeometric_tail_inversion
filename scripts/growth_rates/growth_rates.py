import numpy as np
import os, sys
sys.path.append(os.getcwd())

import python2latex as p2l
from graal_utils import Timer, timed

from source import optimize_mprime
from source import hypinv_upperbound, hypinv_reldev_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, sample_compression_bound
from source.utils import sauer_shelah


os.chdir('./scripts/growth_rates/')

risk, d, delta = 0.1, 20, 0.05

bounds = [
    (
        "SC",
        # "Sample compression",
        lambda k, m, mp: sample_compression_bound(k, m, d, delta),
        "dotted"
    ),
    (
        "HTI",
        lambda k, m, mp: hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mp),
        ""
    ),
    (
        "HTI-RD",
        lambda k, m, mp: hypinv_reldev_upperbound(k, m, sauer_shelah(d), delta, mprime=12*m),
        ""
    ),
    (
        "VP",
        # "Vapnik's pessimistic",
        lambda k, m, mp: vapnik_pessismistic_bound(k, m, sauer_shelah(d), delta),
        "dash dot"
    ),
    (
        "VRD",
        # "Vapnik's RD",
        lambda k, m, mp: vapnik_relative_deviation_bound(k, m, sauer_shelah(d), delta),
        "dash dot"
    ),
]
bounds.reverse()

ms = np.array(list(range(20, 400, 10))
              + list(range(400, 1000, 50))
              + list(range(1000, 10_000, 100))
              + list(range(10_000, 40_000, 1000))
              )

bound_values = {name:np.zeros_like(ms, dtype=float) for name, *_ in bounds}

mp = ms[0]
for i, m in enumerate(ms):
    k = int(risk*m)
    # with Timer(f'{m=}'):
        # if m < 100:
        #     min_mprime = m
        # else:
        #     min_mprime = mp-100
        # mp = optimize_mprime(k, m, sauer_shelah(d), delta, min_mprime=min_mprime, max_mprime=100*mp, early_stopping=200)
        # print(mp)
    mp = 9*m
    for name, bound, _ in bounds:
        bound_values[name][i] = bound(k, m, mp)

plot = p2l.Plot(plot_name=f'growth_rates_{d=}_{delta=}',
                plot_path='figures',
                as_float_env=False,
                width='7.45cm',
                height='6cm',
                lines='1pt',
                xmode='log',
                )
plot.axis.kwoptions['legend style'] = r'{font=\scriptsize}'
plot.axis.kwoptions['y label style'] = r'{yshift=-.3cm}'
plot.axis.kwoptions['legend cell align'] = '{left}'

for name, _, style in bounds:
    plot.add_plot(ms, bound_values[name], style, legend=name)

# plot.add_plot(ms, np.sqrt(d/ms)+risk, 'dashed', line_width='.5pt')
plot.add_plot([ms[0], ms[-1]], [risk, risk], 'dashed', color='gray', line_width='.5pt')

plot.x_min = 20
plot.x_max = max(ms)
plot.y_min = 0
plot.y_max = 1.5
plot.x_label = "Sample size $m$"
plot.y_label = 'Upper bound on the true risk'
plot.legend_position = 'north east'

filename = 'growth_rates'
doc = p2l.Document(filename, doc_type='standalone')
doc += plot
print('Building...')
doc.build()


plot = p2l.Plot(plot_name=f'relative_gain_{d=}_{delta=}',
                plot_path='figures',
                as_float_env=False,
                width='7.45cm',
                height='6cm',
                lines='1pt',
                xmode='log',
                )
plot.axis.kwoptions['legend style'] = r'{font=\scriptsize}'
# plot.axis.kwoptions['y label style'] = r'{yshift=-.3cm}'
plot.axis.kwoptions['legend cell align'] = '{left}'

for name, _, style in bounds:
    gain = 1 - bound_values['HTI']/bound_values[name]
    plot.add_plot(ms, gain, style, legend=name)

plot.x_min = 20
plot.x_max = max(ms)
plot.y_min = -0.5
plot.y_max = 1.25

plot.y_ticks = np.linspace(-0.5, 1.25, 8)

plot.x_label = "Sample size $m$"
plot.y_label = 'HTI relative gain'
plot.legend_position = 'north east'

filename = 'relative_gain'
doc = p2l.Document(filename, doc_type='standalone')
doc += plot
print('Building...')
doc.build(delete_files='all')