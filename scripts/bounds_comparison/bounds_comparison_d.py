import numpy as np

import python2latex as p2l
from graal_utils import Timer

from hypergeo import hypinv_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, catoni_4_6
from hypergeo.utils import sauer_shelah

import os
path = os.path.dirname(__file__)


def plot_comp_d(risk, m, delta=0.05):
    ds = np.array(
        list(range(1, 20, 1))
        + list(range(20, 101, 5))
    )
    k = int(risk*m)

    plot = p2l.Plot(plot_name=f'bounds_comp_d_{risk=}_{m=}_{delta=}',
                    plot_path=path+'/figures',
                    as_float_env=False,
                    width='7.45cm',
                    height='6cm',
                    lines='1pt',
                    palette=reversed(p2l.holi(4))
                    )
    plot.axis.kwoptions['legend style'] = r'{font=\scriptsize}'
    plot.axis.kwoptions['y label style'] = r'{yshift=-.2cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.x_min = min(ds)
    plot.x_max = max(ds)
    plot.y_min = 0
    plot.y_max = 1.05
    plot.x_label = "VC dimension $d$"
    plot.y_label = 'Upper bound on the true risk $R_\mathcal{D}(h)$'
    plot.y_ticks = np.linspace(0, 1, 5)
    plot.legend_position = 'north west'

    # VRD
    with Timer('VRD'):
        bound_values = np.array([vapnik_relative_deviation_bound(k, m, sauer_shelah(d), delta) for d in ds])
        print(ds[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ds, bound_values, legend='VRD')

    # VP
    with Timer('VP'):
        bound_values = np.array([vapnik_pessismistic_bound(k, m, sauer_shelah(d), delta) for d in ds])
        print(ds[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ds, bound_values, legend='VP')

    # Catoni
    with Timer('C4.6'):
        bound_values = np.array([catoni_4_6(k, m, d, delta, mprime=None) for d in ds])
        print(ds[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ds, bound_values, legend='C4.6')

    # HTI
    with Timer('HTI'):
        def mprime(d):
            best_mp = int(3.25*m)
            best_bound = 1
            for ratio in np.arange(3.25, 13, step=.25):
                mp = int(m*ratio)
                bound = hypinv_upperbound(k, m, sauer_shelah(d), delta, mp)
                if bound < best_bound:
                    best_bound = bound
                    best_mp = mp
                elif bound > best_bound:
                    break
            return best_mp

        bound_values = np.array([hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mprime(d)) for d in ds])
        print(ds[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ds, bound_values, legend='HTI')

    filename = f'bounds_comparison_d_{risk=}_{m=}'
    doc = p2l.Document(filename, filepath=path, doc_type='standalone')
    doc.add_package('mathalfa', cal='dutchcal', scr='boondox')
    doc += plot
    print('Building...')
    doc.build(delete_files='all', show_pdf=False)
    print('Building completed.')


if __name__ == '__main__':
    risk, m, delta = .05, 2000, 0.05
    plot_comp_d(risk, m, delta)

    # for risk in [0, 0.1, 0.2, 0.3]:
