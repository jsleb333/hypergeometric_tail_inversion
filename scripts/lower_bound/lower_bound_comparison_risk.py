import numpy as np

import python2latex as p2l
from graal_utils import Timer

from hypergeo import optimize_mprime, hypinv_upperbound, hypinv_lowerbound
from hypergeo.utils import sauer_shelah

import os
path = os.path.dirname(__file__)


# Saved values of optimized m' for parameters (m, d, delta) using 'optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=13*m, min_mprime=3*m, early_stopping=1000)'
mp_dict = {
    (2000, 20, 0.2): 7896,
    (2000, 20, 0.05): 7987,
}


def plot_risk_comp(m, d, delta=0.05):

    if (m, d, delta) in mp_dict:
        mp = mp_dict[(m, d, delta)]
    else:
        mp = Timer(optimize_mprime)(0, m, sauer_shelah(d), delta, max_mprime=13*m, min_mprime=3*m, early_stopping=1000, bound=hypinv_upperbound)
        print(f'Optimal mprime for params ({m=}, {d=}, {delta=}): {mp=}')

    plot = p2l.Plot(plot_name=f'bounds_comp_{m=}_{d=}_{delta=}',
                    plot_path=path+'/figures',
                    as_float_env=False,
                    width='7.45cm',
                    height='6cm',
                    lines='1pt',)
    plot.axis.kwoptions['legend style'] = r'{font=\scriptsize}'
    plot.axis.kwoptions['y label style'] = r'{yshift=-.2cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    palette = p2l.holi(2)

    bounds = [
        (
            "Upper HTI",
            lambda k: hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mp),
            "",
            palette[0]
        ),
        (
            "Lower HTI",
            lambda k: hypinv_lowerbound(k, m, sauer_shelah(d), delta, mprime=mp),
            "",
            palette[1]
        ),
    ]

    ks = np.array([int(k) for k in np.linspace(0, m, num=200)])
    for name, bound, style, color in bounds:
        with Timer(name):
            bs = [bound(k) for k in ks]
            plot.add_plot(ks/m, bs, style, color=color, legend=name)
            print(name, bs[0])

    plot.add_plot([0,1], [0,1], line_width='.5pt', color='gray!50')

    plot.x_min = 0
    plot.x_max = 1.02
    plot.y_min = 0
    plot.y_max = 1.02
    plot.y_ticks = np.linspace(0, 1.25, 6)

    plot.x_label = "Empirical risk $R_S(h)$"
    plot.y_label = r'Bound on the true risk $R_\mathcal{D}(h)$'
    plot.legend_position = 'south east'

    filename = f'lower_bound_comparison_risk_{m=}_{d=}'

    doc = p2l.Document(filename, filepath=path, doc_type='standalone')

    doc.add_package('mathalfa', cal='dutchcal', scr='boondox')
    doc += plot

    print('Building...')
    doc.build(delete_files='all', show_pdf=False)
    print('Building completed.')


if __name__ == '__main__':
    d, delta = 20, 0.05
    # for m in [100, 500, 2000, 20_000]:
    #     with Timer(f'{m=}'):
    #         plot_risk_comp(m, d, delta)

    plot_risk_comp(2000, d, delta)
