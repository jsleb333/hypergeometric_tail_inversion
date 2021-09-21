import numpy as np
import os, sys
sys.path.append(os.getcwd())
os.chdir('./scripts/bounds_comparison/')

import python2latex as p2l
from graal_utils import Timer

from hypergeo import hypinv_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, catoni_4_6
from hypergeo.utils import sauer_shelah


def plot_comp_m(risk, d, delta=0.05):
    ms = np.array(
        list(range(d, 1000, 10))
        + list(range(1000, 10_000, 100))
        + [int(m) for m in np.logspace(13.5, 20, num=10, base=2)]
    )
    ks = np.array([int(risk*m) for m in ms])

    plot = p2l.Plot(plot_name=f'bounds_comp_m_{risk=}_{d=}_{delta=}',
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

    # VRD
    with Timer('VRD'):
        bound_values = vapnik_relative_deviation_bound(ks, ms, sauer_shelah(d), delta)
        print(ms[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ms, bound_values - risk, legend='VRD')

    # VP
    with Timer('VP'):
        bound_values = vapnik_pessismistic_bound(ks, ms, sauer_shelah(d), delta)
        print(ms[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ms, bound_values - risk, legend='VP')

    # Catoni
    with Timer('C4.6'):
        bound_values = np.array([catoni_4_6(k, m, d, delta, mprime=None) for k, m in zip(ks, ms)])
        print(ms[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ms, bound_values - risk, legend='C4.6')

    # HTI
    with Timer('HTI'):
        def mprime(k, m):
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

        bound_values = np.array([hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mprime(k, m)) for k, m in zip(ks, ms)])
        print(ms[np.argmin((bound_values - .5)**2)])
        plot.add_plot(ms, bound_values - risk, legend='HTI')


    if risk > 0:
        plot.add_plot(ms, np.sqrt(d/ms), color='gray', line_width='.5pt', legend=r'\scalebox{.8}{\normalsize $\sqrt{d/m}$}')
    else:
        plot.add_plot(ms, d/ms, color='gray', line_width='.5pt', legend='$d/m$')

    filename = f'bounds_comparison_m_{risk=}_{d=}'
    doc = p2l.Document(filename, doc_type='standalone')
    doc.add_package('mathalfa', cal='dutchcal', scr='boondox')
    doc.add_package('times')
    doc += plot
    print('Building...')
    doc.build(delete_files='all', show_pdf=False)
    print('Building completed.')


if __name__ == '__main__':
    risk, d, delta = 0.05, 50, 0.05
    print(f'{risk=}, {d=}, {delta=}')

    # for risk in [0, 0.1, 0.2, 0.3]:
    #     plot_comp_m(risk, d, delta)

    plot_comp_m(risk, 10, delta)
