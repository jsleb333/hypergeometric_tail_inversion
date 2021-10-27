import numpy as np
from python2latex import Document, Plot, holi

from hypergeo import hypergeometric_tail

import os
path = os.path.dirname(__file__)


def plot_hyp_tail_k(m, Ks, M):
    plot = Plot(plot_name='plot_hyp_tail_k',
                width='7.45cm',
                height='7.45cm',
                lines='.9pt',
                marks='1.35pt',
                as_float_env=False)

    for K, color in zip(Ks, holi()):
        ks = np.arange(0, m+1)
        plot.add_plot(ks, [hypergeometric_tail(k,m,K,M) for k in ks], color=color, legend=f'\\scriptsize ${K=}$')

    plot.legend_position = 'south east'
    plot.x_label = '$k$'
    plot.y_label = 'Hyp'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.x_min = 0
    plot.x_max = 21

    plot.y_min = 0
    plot.y_max = 1.05

    return plot


def plot_hyp_tail_m(k, K, Ms):
    plot = Plot(plot_name='plot_hyp_tail_m',
                width='7.45cm',
                height='7.45cm',
                lines='.9pt',
                marks='1.35pt',
                as_float_env=False)

    for M, color in zip(Ms, holi()):
        ms = np.arange(20+1)
        plot.add_plot(ms, [hypergeometric_tail(k,m,K,M) for m in ms], color=color, legend=f'\\scriptsize ${M=}$')

    plot.legend_position = 'south west'
    plot.x_label = '$m$'
    plot.y_label = 'Hyp'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.x_min = 0
    plot.x_max = 21

    plot.y_min = 0
    plot.y_max = 1.05

    return plot


def plot_hyp_tail_K(ks, m, M):
    plot = Plot(plot_name='plot_hyp_tail_K',
                width='7.45cm',
                height='7.45cm',
                lines='.9pt',
                marks='1.35pt',
                as_float_env=False)

    for k, color in zip(ks, holi()):
        Ks = np.arange(0, M+1)
        plot.add_plot(Ks, [hypergeometric_tail(k,m,K,M) for K in Ks], color=color, legend=f'\\scriptsize ${k=}$')

    plot.legend_position = 'north east'
    plot.x_label = '$K$'
    plot.y_label = 'Hyp'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.x_min = 0
    plot.x_max = M+1

    plot.y_min = 0
    plot.y_max = 1.05

    return plot


def plot_hyp_tail_M(k, m, Ks):
    plot = Plot(plot_name='plot_hyp_tail_M',
                width='7.45cm',
                height='7.45cm',
                lines='.9pt',
                marks='1.35pt',
                as_float_env=False)

    Ms = np.arange(m, 3*m+1)
    for K, color in zip(Ks, holi()):
        plot.add_plot(Ms, [hypergeometric_tail(k,m,K,M) for M in Ms], color=color, legend=f'\\scriptsize ${K=}$')

    plot.legend_position = 'north west'
    plot.x_label = '$M$'
    plot.y_label = 'Hyp'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.3cm}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.x_min = m
    plot.x_max = 3*m+1

    plot.y_min = 0
    plot.y_max = 1.05

    return plot


if __name__ == "__main__":
    m = 20
    M = 40
    k = 3
    K = 10

    filename = 'hyp_tail_plot_k'
    plot = plot_hyp_tail_k(m, [1, 5, 10, 17, 25], M)

    # filename = 'hyp_tail_plot_capital_K'
    # plot = plot_hyp_tail_K([0, 3, 6, 9, 12], m, M)

    # filename = 'hyp_tail_plot_m'
    # plot = plot_hyp_tail_m(k, K, [25, 30, 40, 55, 75])

    # filename = 'hyp_tail_plot_capital_M'
    # plot = plot_hyp_tail_M(k, m, [6, 9, 12, 15, 18])

    doc = Document(filename, filepath=path, doc_type='standalone')
    doc += plot

    print('Building...')
    doc.build(delete_files='all', show_pdf=False)

