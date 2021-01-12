import numpy as np
import xarray as xr
import os
from itertools import product
from graal_utils import Timer

from scipy.optimize import curve_fit

import python2latex as p2l


def plot_mprime_best(ms, risks, d, delta):
    ms, risks = np.array(ms), np.array(risks)

    plot = p2l.Plot(plot_name=f'mprime_best{d=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='7.45cm',
                    height='7.45cm',
                    marks='1.5pt',
                    lines=False,
                    palette='holi5')

    best_mprimes = xr.open_dataset('data/optimal_bound.nc')['mprime'].drop_sel(d=50)

    for m in ms:
        plot_mprimes = best_mprimes.sel(m=m, d=d, delta=delta)
        plot.add_plot(risks, plot_mprimes/m, legend=f'\\scriptsize $m={m}$')

    plot.x_min = -0.05
    plot.x_max = 0.55
    plot.x_ticks = 0, .1, .2, .3, .4, .5
    plot.x_label = r"$\displaystyle\frac{k}{m}$"
    plot.y_label = r"$\displaystyle\frac{m'_{\textrm{\scriptsize best}}}{m}$"
    plot.legend_position = 'north west'
    # plot.legend_position = 'south east'
    plot.axis.kwoptions['ylabel style'] = r'{rotate=-90, xshift=.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    # plot.y_max = 10

    return plot


if __name__ == "__main__":
    os.chdir('./scripts/mprime_tradeoff/')

    d = 10
    ms = [100, 200, 300, 500, 1000]
    risks =  np.linspace(0, .5, 11)
    delta = 0.05

    filename = f'mprime_best_{d=}'
    doc = p2l.Document(filename, doc_type='standalone')
    # doc.add_package('xcolor', 'dvipsnames')

    doc += plot_mprime_best(ms, risks, d, delta)

    print('Building...')
    doc.build()

    os.remove(filename + '.log')
    os.remove(filename + '.tex')
    os.remove(filename + '.aux')