import numpy as np
import os, sys
sys.path.append(os.getcwd())
os.chdir('./scripts/bounds_comparison/')

import python2latex as p2l
from graal_utils import Timer

from source import hypinv_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, catoni_4_6
from source.utils import sauer_shelah, log_sauer_shelah


def plot_comp_d(risk, m, delta=0.05):
    ds = np.array(
        list(range(1, 20, 1))
        + list(range(20, 201, 5))
    )
    k = int(risk*m)
    log_delta = np.log(delta)

    plot = p2l.Plot(plot_name=f'bounds_comp_d_{risk=}_{m=}_{delta=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='7.45cm',
                    height='6cm',
                    lines='1pt',
                    # xmode='log',
                    # ymode='log',
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
        bound_values = np.array([vapnik_relative_deviation_bound(k, m, log_sauer_shelah(d), log_delta, True) for d in ds])
        print(ds[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ds, bound_values, legend='VRD')

    # VP
    with Timer('VP'):
        bound_values = np.array([vapnik_pessismistic_bound(k, m, log_sauer_shelah(d), log_delta, True) for d in ds])
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
            # return sum(coef*param for coef, param in zip(np.polyfit([d, 1e6], [3.5*m, 9*m], 1), [m, 1]))
            # best_mp = int(1*m)
            # best_bound = 1
            # for ratio in np.arange(1, 13, step=.1):
            #     mp = int(m*ratio)
            #     bound = hypinv_upperbound(k, m, log_sauer_shelah(d), log_delta, mp, log_delta=True)
            #     if bound < best_bound:
            #         best_bound = bound
            #         best_mp = mp
            #     elif bound > best_bound:
            #         break
            # print(d, best_mp, best_mp/m)
            # return best_mp
            return 4*m

        bound_values = np.array([hypinv_upperbound(k, m, log_sauer_shelah(d), log_delta, mprime=mprime(d), log_delta=True) for d in ds])
        print(ds[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ds, bound_values, legend='HTI')

    filename = f'bounds_comparison_d_log_{risk=}_{m=}'
    doc = p2l.Document(filename, doc_type='standalone')
    doc.add_package('mathalfa', cal='dutchcal', scr='boondox')
    doc.add_package('times')
    doc += plot
    print('Building...')
    doc.build(delete_files='all', show_pdf=True)
    print('Building completed.')


if __name__ == '__main__':
    risk, m, delta = .05, 1000, 0.05
    plot_comp_d(risk, m, delta)

