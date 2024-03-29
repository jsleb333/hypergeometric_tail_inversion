import numpy as np
import xarray as xr

import python2latex as p2l

import os
path = os.path.dirname(__file__)


def plot_mprime_best(ms, risks, d, delta):
    ms, risks = np.array(ms), np.array(risks)

    plot = p2l.Plot(plot_name=f'mprime_best{d=}',
                    plot_path=path+'/figures',
                    as_float_env=False,
                    width='7.45cm',
                    height='5.45cm',
                    marks='1.5pt',
                    lines=False,
                    palette='holi5')

    best_mprimes = xr.open_dataset(path+'/data/optimal_bound.nc')['mprime'].drop_sel(d=50)

    for m in ms:
        plot_mprimes = best_mprimes.sel(m=m, d=d, delta=delta)
        plot.add_plot(risks, plot_mprimes/m, legend=f'\\scriptsize $m={m}$')

    plot.x_min = -0.05
    plot.x_max = 0.55
    plot.x_ticks = 0, .1, .2, .3, .4, .5
    plot.x_label = r"$k/m$"
    plot.y_label = r"$\displaystyle\frac{m'_{\textrm{\scriptsize best}}}{m}$"
    plot.legend_position = 'north west'
    plot.axis.kwoptions['ylabel style'] = r'{rotate=-90, xshift=.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    return plot


if __name__ == "__main__":
    d = 10
    # d = 35
    ms = [100, 200, 300, 500, 1000]
    risks =  np.linspace(0, .5, 11)
    delta = 0.05

    filename = f'mprime_best_{d=}'
    doc = p2l.Document(filename, filepath=path, doc_type='standalone')

    doc += plot_mprime_best(ms, risks, d, delta)

    print('Building...')
    doc.build(delete_files='all', show_pdf=False)
