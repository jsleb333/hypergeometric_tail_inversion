import numpy as np
from python2latex import Document, Plot, holi
from itertools import chain

from hypergeo import hypergeometric_tail_inverse

import os
path = os.path.dirname(__file__)


def plot_hyp_tail_inv_k(m, deltas, M):
    plot = Plot(plot_name='plot_hyp_tail_inv_k',
                plot_path=path+'/figures',
                width='7.45cm',
                height='7.45cm',
                lines='.9pt',
                marks='1.35pt',
                as_float_env=False)

    for delta, color in zip(deltas, holi()):
        ks = np.arange(0, m+1)
        if delta == 1e-6:
            legend = '10^{-6}'
        else:
            legend = str(delta)

        plot.add_plot(ks, [hypergeometric_tail_inverse(k,m,delta,M) for k in ks], color=color, legend=f'\\scriptsize $\\delta={legend}$')

    plot.legend_position = 'south east'
    plot.x_label = '$k$'
    plot.y_label = r'$\overline{\textnormal{Hyp}}$'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.x_min = 0
    plot.x_max = 21

    plot.y_min = 0
    plot.y_max = M+2

    return plot


def plot_hyp_tail_inv_m(k, delta, Ms):
    plot = Plot(plot_name='plot_hyp_tail_inv_m',
                plot_path=path+'/figures',
                width='7.45cm',
                height='7.45cm',
                lines='.9pt',
                marks='1.35pt',
                as_float_env=False)

    for M, color in zip(Ms, holi()):
        ms = np.arange(k, 20+1)
        plot.add_plot(ms, [hypergeometric_tail_inverse(k,m,delta,M) for m in ms], color=color, legend=f'\\scriptsize ${M=}$')

    plot.legend_position = 'north east'
    plot.x_label = '$m$'
    plot.y_label = r'$\overline{\textnormal{Hyp}}$'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.x_min = k
    plot.x_max = 21

    plot.y_min = 0
    plot.y_max = Ms[-1]+k+1

    return plot


def plot_hyp_tail_inv_delta(ks, m, M):
    plot = Plot(plot_name='plot_hyp_tail_delta',
                plot_path=path+'/figures',
                width='7.45cm',
                height='7.45cm',
                lines='.9pt',
                marks='1.35pt',
                as_float_env=False)

    for k, color in zip(ks, holi()):

        steps = []
        delta_start, hypinv_current = 0, hypergeometric_tail_inverse(k,m,0,M)
        deltas = [d for d in chain(np.logspace(-16, -1, 101),
                                   np.linspace(.1, .9, 101),
                                   reversed(1 - np.logspace(-16, -1, 101)))]
        for delta in deltas:
            hypinv = hypergeometric_tail_inverse(k,m,delta,M)
            if hypinv < hypinv_current:
                steps.append((delta_start, delta, hypinv_current))
                hypinv_current = hypinv
                delta_start = delta
        else:
            steps.append((delta_start, delta, hypinv_current))

        for delta_start, delta_end, hypinv in steps:
            plot.add_plot([delta_start, delta_end], [hypinv, hypinv], color=color)
            plot.add_plot([delta_end], [hypinv], color=color, mark_options='{fill=white}')

        plot.axis += rf'\addlegendentry{{\scriptsize ${k=}$}}'
        plot.axis += fr'\addlegendimage{{color={str(color)}, line width=1.35pt, mark=none}}'

    plot.legend_position = 'north east'
    plot.x_label = '$\\delta$'
    plot.y_label = r'$\overline{\textnormal{Hyp}}$'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.x_min = 0
    plot.x_max = 1.05

    plot.y_min = 0
    plot.y_max = 41

    return plot


def plot_hyp_tail_inv_M(k, m, deltas):
    plot = Plot(plot_name='plot_hyp_tail_inv_M',
                plot_path=path+'/figures',
                width='7.45cm',
                height='7.45cm',
                lines='.9pt',
                marks='1.35pt',
                as_float_env=False)

    Ms = np.arange(m, 3*m+1)
    for delta, color in zip(deltas, holi()):
        if delta == 1e-6:
            legend = '10^{-6}'
        else:
            legend = str(delta)
        plot.add_plot(Ms, [hypergeometric_tail_inverse(k,m,delta,M) for M in Ms], color=color, legend=f'\\scriptsize $\\delta={legend}$')

    plot.legend_position = 'north west'
    plot.x_label = '$M$'
    plot.y_label = r'$\overline{\textnormal{Hyp}}$'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.x_min = m
    plot.x_max = 3*m+1

    plot.y_min = 0

    return plot


if __name__ == "__main__":

    m = 20
    M = 40
    k = 3
    delta = 0.05

    # filename = 'hyp_tail_plot_inv_k'
    # plot = plot_hyp_tail_inv_k(m, [1e-6, 1e-3, 0.05, .2, .5], M)

    # filename = 'hyp_tail_plot_inv_m'
    # plot = plot_hyp_tail_inv_m(k, delta, [25, 30, 40, 55, 75])

    filename = 'hyp_tail_plot_inv_delta'
    plot = plot_hyp_tail_inv_delta([0, 3, 6, 9, 12], m, M)

    # filename = 'hyp_tail_plot_inv_capital_M'
    # plot = plot_hyp_tail_inv_M(k, m, [1e-6, 1e-3, 0.05, .2, .5])

    doc = Document(filename, filepath=path, doc_type='standalone')
    doc.add_package('amsmath')
    doc += plot

    print('Building...')
    doc.build(delete_files='all', show_pdf=False)
