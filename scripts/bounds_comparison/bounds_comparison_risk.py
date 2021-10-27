import numpy as np

import python2latex as p2l
from graal_utils import Timer

from hypergeo import optimize_mprime, optimize_catoni
from hypergeo import hypinv_upperbound, vapnik_pessismistic_bound, vapnik_relative_deviation_bound, catoni_4_6
from hypergeo.utils import sauer_shelah

from scripts.utils import main_path as path

# Saved values of optimized m' for parameters (m, d, delta) using 'optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=13*m, min_mprime=3*m, early_stopping=1000)'
mp_dict = {
    (100, 50, 0.05): 388,
    (200, 15, 0.05): 678,
    (400, 10, 0.05): 1463,
    (500, 20, 0.05): 1673,
    (500, 50, 0.05): 1578,
    (1000, 10, 0.05): 4071,
    (1000, 20, 0.05): 3611,
    (1500, 15, 0.05): 5937,
    (2000, 10, 0.05): 8762,
    (2000, 50, 0.05): 6970,
    (20_000, 50, 0.05): 91594,
}

def plot_risk_comp(m, d, delta=0.05, show_non_opti=False):
    catoni_mp = optimize_catoni(0, m, d, delta)[1]
    print('Optimal m_prime for C4.6:', catoni_mp/m)

    if (m, d, delta) in mp_dict:
        mp = mp_dict[(m, d, delta)]
    else:
        mp = optimize_mprime(0, m, sauer_shelah(d), delta, max_mprime=13*m, min_mprime=3*m, early_stopping=1000)
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

    palette = p2l.holi(4)

    bounds = [
        (
            "HTI\\textsubscript{opti}",
            lambda k: hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=mp),
            "",
            palette[0]
        ),
        (
            "HTI$_{m'=m}$",
            lambda k: hypinv_upperbound(k, m, sauer_shelah(d), delta, mprime=m),
            "dashed",
            palette[0]
        ),
        (
            "C4.6", # Catoni (2004)'s Theorem 4.6
            lambda k: catoni_4_6(k, m, d, delta, mprime=catoni_mp),
            '',
            palette[1]
        ),
        (
            "VP",
            lambda k: vapnik_pessismistic_bound(k, m, sauer_shelah(d), delta),
            "",
            palette[2]
        ),
        (
            "VRD",
            lambda k: vapnik_relative_deviation_bound(k, m, sauer_shelah(d), delta),
            "",
            palette[3]
        ),
    ]
    if not show_non_opti:
        del bounds[1]
    bounds.reverse()

    ks = np.array([int(k) for k in np.linspace(0, m, num=100)])
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
    plot.y_label = r'Upper bound on the true risk $R_\mathcal{D}(h)$'
    plot.legend_position = 'south east'



    filename = f'bounds_comparison_risk_{m=}_{d=}'
    if show_non_opti:
        filename += '_with_non_opti'

    doc = p2l.Document(filename, filepath=path, doc_type='standalone')
    doc.add_package('mathalfa', cal='dutchcal', scr='boondox')
    doc += plot

    print('Building...')
    doc.build(delete_files='all', show_pdf=False)
    print('Building completed.')


if __name__ == '__main__':
    d, delta = 10, 0.05
    # for m in [100, 500, 2000, 20_000]:
    #     with Timer(f'{m=}'):
    #         plot_risk_comp(m, d, delta)

    plot_risk_comp(400, d, delta, show_non_opti=True)
