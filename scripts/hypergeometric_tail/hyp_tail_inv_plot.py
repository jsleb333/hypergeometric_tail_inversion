import numpy as np
from python2latex import Document, Plot
import os, sys
sys.path.append(os.getcwd())

from source import hypergeometric_left_tail_inverse


def plot_hyp_tail_inv_M(ks, ms, deltas, max_M=300):
    plot = Plot(plot_name='plot_hyp_tail_inv_M',
                width='13cm',
                height='9cm',
                as_float_env=False)

    for k in ks:
        for m in ms:
            Ms = np.arange(m+1, max_M+1)
            for delta in deltas:
                tau = lambda M: (np.e*M/25)**25
                plot.add_plot(Ms, [hypergeometric_left_tail_inverse(k, m, delta/4/tau(M), M)-k for M in Ms],
                              label=f'{k=} {m=} {delta=}')

            plot.add_plot([Ms[0], Ms[-1]], [Ms[0]-m, Ms[-1]-m])

    return plot

def plot_hyp_tail_inv_delta(ks, ms, Ms):
    plot = Plot(plot_name='plot_hyp_tail_inv_delta',
                width='13cm',
                height='9cm',
                as_float_env=False)

    for k in ks:
        for m in ms:
            for M in Ms:
                deltas = np.linspace(10e-16, .5, 100)
                plot.add_plot(deltas, [hypergeometric_left_tail_inverse(k, m, delta, M) for delta in deltas],
                              label=f'{k=} {m=} {M=}')

    return plot


if __name__ == "__main__":
    import os
    os.chdir('./scripts/hypergeometric_tail/')

    doc = Document('hyp_tail_inv', doc_type='standalone')
    ks = [0, 30, 100]
    ms = [200]
    deltas = [.05]
    doc += plot_hyp_tail_inv_M(ks, ms, deltas)

    # Ms = [600]
    # doc += plot_hyp_tail_inv_delta(ks, ms, Ms)

    print('Building...')
    doc.build()

