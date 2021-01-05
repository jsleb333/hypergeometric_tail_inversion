import numpy as np
from python2latex import Document, Plot, holi
import os, sys
sys.path.append(os.getcwd())

from source import hypergeometric_tail


def plot_hyp_tail_k(ms, Ks, Ms):
    plot = Plot(plot_name='plot_hyp_tail_k',
                width='7.5cm',
                height='7.5cm',
                as_float_env=False)

    palette = iter(holi())
    for m in ms:
        ks = np.arange(0, m+3) - 1
        for K in Ks:
            M = 3*m
            color = next(palette)
            plot.add_plot(ks, [hypergeometric_tail(k,m,K,M) for k in ks], color=color, label=f'{m=} {K=} {M=}')
            k0 = max(K+m-M,0)
            plot.add_plot([k0], [hypergeometric_tail(k0,m,K,M)], 'only marks', 'mark size=2pt', color=color)
            km = min(K,m)
            plot.add_plot([km], [hypergeometric_tail(km,m,K,M)], 'only marks', 'mark size=2pt', color=color)
            for i in range(3):
                print(hypergeometric_tail(k0+i-1,m,K,M))

    return plot

def plot_hyp_tail_K(ms, ks, Ms):
    plot = Plot(plot_name='plot_hyp_tail_K',
                width='13cm',
                height='9cm',
                as_float_env=False)

    palette = iter(holi())
    for m in ms:
        M = 3*m
        for k in ks:
            Ks = np.arange(k, M+3) - 1
            color = next(palette)
            plot.add_plot(Ks, [hypergeometric_tail(k,m,K,M) for K in Ks], color=color, label=f'{m=} {k=} {M=}')
            # k0 = max(K+m-M,0)
            # plot.add_plot([k0], [hypergeometric_tail(k0,m,K,M)], 'only marks', 'mark size=2pt', color=color)
            # km = min(K,m)
            # plot.add_plot([km], [hypergeometric_tail(km,m,K,M)], 'only marks', 'mark size=2pt', color=color)
            # for i in range(3):
            #     print(hypergeometric_tail(k0+i-1,m,K,M))

    return plot

def plot_hyp_tail_M(ks, ms, Ks, max_M=300):
    plot = Plot(plot_name='plot_hyp_tail_M',
                width='13cm',
                height='9cm',
                as_float_env=False)

    for k in ks:
        for m in ms:
            for K in Ks:
                Ms = np.arange(m, max_M)
                plot.add_plot(Ms, [hypergeometric_tail(k, m, K, M) for M in Ms],
                              label=f'{k=} {m=} {K=}')

    return plot

def plot_hyp_tail_delta(ks, ms, Ms):
    plot = Plot(plot_name='plot_hyp_tail_delta',
                width='13cm',
                height='9cm',
                as_float_env=False)

    for k in ks:
        for m in ms:
            for M in Ms:
                deltas = np.linspace(10e-16, .5, 100)
                plot.add_plot(deltas, [hypergeometric_tail(k, m, delta, M) for delta in deltas],
                              label=f'{k=} {m=} {M=}')

    return plot


if __name__ == "__main__":
    import os
    os.chdir('./scripts/hypergeometric_tail/')

    doc = Document('hyp_tail_plot', doc_type='article', margin='2.5cm')
    ks = [0, 3, 5, 10]
    ms = [20]
    Ks = [25]
    Ms = [50]
    # doc += plot_hyp_tail_K(ms, ks, Ms)
    doc += plot_hyp_tail_M(ks, ms, Ks)

    # Ms = [600]
    # doc += plot_hyp_tail_delta(ks, ms, Ms)

    print('Building...')
    doc.build()

