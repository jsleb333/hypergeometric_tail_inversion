import numpy as np
import pandas as pd
import os

import python2latex as p2l


def plot_comp_k(m, ks, d, delta=0.05):

    plot = p2l.Plot(plot_name=f'tradeoff_comp_k_{m=}_{d=}_{delta=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='13cm',
                    height='7cm',
                    xmode='log',
                    lines='1pt',)
    plot.axis.kwoptions['legend style'] = r'{font=\tiny}'
    plot.axis.kwoptions['restrict y to domain'] = r'{0:1}'

    palette = p2l.holi(6)
    plot.add_plot([m,m], [0,1], color=palette[0], line_width='1pt', opacity='.5', label="\\footnotesize $m'=m$", label_anchor='south')

    for k, color in zip(ks, palette[1:]):
        datafile = f'./data/mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
        df = pd.read_csv(datafile, sep=',', header=0)
        idx = list(range(0, 2000, 100)) + list(range(2000, 10_000, 1000)) + [9_999]
        mprimes = [df['mprime'][i] for i in idx]
        bounds = [df['bound'][i] for i in idx]


        plot.add_plot(mprimes, bounds, color, label=f"\\footnotesize $k={k}$", line_join='round')

        min_idx = np.argmin(bounds)
        min_bound = bounds[min_idx]
        min_mprime = mprimes[min_idx]
        plot.add_plot([min_mprime], [min_bound], color, 'only marks', mark_size='1.5pt', label=f'${min_mprime}$', label_anchor='south')

    plot.x_min = 5
    plot.x_max = 30_000
    plot.x_label = "$m'$"
    plot.y_min = 0
    plot.y_max = 1.08
    plot.y_label = 'Upper bound'

    plot.caption = f"Value of the upper bound as a function of $m'$ for various numbers of errors $k$. The number of examples is fixed to $m={m}$ and the VCdim is set to $d={d}$."

    return plot


def plot_comp_d(m, k, ds, delta=0.05):

    plot = p2l.Plot(plot_name=f'tradeoff_comp_d_{m=}_{k=}_{delta=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='13cm',
                    height='7cm',
                    xmode='log',
                    lines='1pt')
    plot.axis.kwoptions['legend style'] = r'{font=\tiny}'
    plot.axis.kwoptions['restrict y to domain'] = r'{0:1}'

    palette = p2l.holi(6)
    plot.add_plot([m,m], [0,1], palette[0], line_width='1pt', opacity='.5', label="\\footnotesize $m'=m$", label_anchor='south')

    for d, color in zip(ds, palette[1:]):
        datafile = f'./data/mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
        df = pd.read_csv(datafile, sep=',', header=0)
        idx = list(range(0, 2000)) + list(range(2000, 10_000, 500)) + [9_999]
        mprimes = [df['mprime'][i] for i in idx]
        bounds = [df['bound'][i] for i in idx]


        plot.add_plot(mprimes, bounds, color, label=f"\\footnotesize $d = {d}$", line_join='round')

        min_idx = np.argmin(bounds)
        min_bound = bounds[min_idx]
        min_mprime = mprimes[min_idx]
        plot.add_plot([min_mprime], [min_bound], color, 'only marks', mark_size='1.5pt', label=f'${min_mprime}$', label_anchor='south')

    plot.x_min = 5
    plot.x_max = 30_000
    plot.x_label = "$m'$"
    plot.y_min = 0
    plot.y_max = 1.08
    plot.y_label = 'Upper bound'

    plot.caption = f"Value of the upper bound as a function of $m'$ for various numbers of VC dimension $d$. The number of examples is fixed to $m={m}$ and the number of errors is set to $d={k}$."

    return plot


def plot_comp_delta(m, k, d, deltas):

    plot = p2l.Plot(plot_name=f'tradeoff_comp_delta_{m=}_{k=}_{d=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='13cm',
                    height='7cm',
                    xmode='log',
                    lines='1pt')
    plot.axis.kwoptions['legend style'] = r'{font=\tiny}'
    plot.axis.kwoptions['restrict y to domain'] = r'{0:1}'

    palette = p2l.holi(len(deltas)+1)
    plot.add_plot([m,m], [0,1], palette[0], line_width='1pt', opacity='.5', label="\\footnotesize $m'=m$", label_anchor='south')

    for delta, color in zip(deltas, palette[1:]):
        datafile = f'./data/mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
        df = pd.read_csv(datafile, sep=',', header=0)
        idx = list(range(0, 2000)) + list(range(2000, 10_000, 500)) + [9_999]
        mprimes = [df['mprime'][i] for i in idx]
        bounds = [df['bound'][i] for i in idx]


        plot.add_plot(mprimes, bounds, color, label=f"\\footnotesize $\\delta = {delta}$", line_join='round')

        min_idx = np.argmin(bounds)
        min_bound = bounds[min_idx]
        min_mprime = mprimes[min_idx]
        plot.add_plot([min_mprime], [min_bound], color, 'only marks', mark_size='1.5pt', label=f'${min_mprime}$', label_anchor='south')

    plot.x_min = 5
    plot.x_max = 30_000
    plot.x_label = "$m'$"
    plot.y_min = 0
    plot.y_max = 1.08
    plot.y_label = 'Upper bound'

    plot.caption = f"Value of the upper bound as a function of $m'$ for various numbers of VC dimension $d$. The number of examples is fixed to $m={m}$ and the number of errors is set to $d={k}$."

    return plot

def plot_comp_m(ms, k, d, delta):

    plot = p2l.Plot(plot_name=f'tradeoff_comp_m_{k=}_{d=}_{delta=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='13cm',
                    height='7cm',
                    xmode='log',
                    lines='1pt')
    plot.axis.kwoptions['legend style'] = r'{font=\tiny}'
    plot.axis.kwoptions['restrict y to domain'] = r'{0:1}'

    palette = p2l.holi(len(ms)+1)

    for m, color in zip(ms, palette[1:]):
        plot.add_plot([m,m], [0,1], color=color, line_width='1pt', opacity='.5', label=f"\\footnotesize ${m}$", label_anchor='south')
        datafile = f'./data/mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
        df = pd.read_csv(datafile, sep=',', header=0)
        idx = list(range(0, 2000)) + list(range(2000, 10_000, 500)) + [9_999]
        mprimes = df['mprime']
        bounds = df['bound']

        mprimes_plot = [mprimes[i] for i in idx]
        bounds_plot = [bounds[i] for i in idx]
        plot.add_plot(mprimes_plot, bounds_plot, color, label=f"\\footnotesize $m = {m}$", line_join='round')

        min_idx = np.argmin(bounds)
        min_bound = bounds[min_idx]
        min_mprime = mprimes[min_idx]
        plot.add_plot([min_mprime], [min_bound], color, 'only marks', mark_size='1.5pt', label=f'\\footnotesize ${min_mprime}$', label_anchor='south')

    plot.x_min = 5
    plot.x_max = 30_000
    plot.x_label = "$m'$"
    plot.y_min = 0
    plot.y_max = 1.08
    plot.y_label = 'Upper bound'

    return plot

if __name__ == "__main__":
    ms = [100, 200, 300, 500, 1000]
    deltas = [0.0001, 0.0025, 0.05, 0.1]
    ks = [0, 10, 30, 50, 100]
    ds = [5, 10, 20, 35, 50]

    os.chdir('./scripts/mprime_tradeoff/')

    doc = p2l.Document(f'mprime_tradeoff_comp_m', doc_type='standalone')

    # doc = p2l.Document(f'mprime_tradeoff_comp_d', doc_type='standalone')
    # doc += plot_comp_d(ms[1], ks[0], ds, deltas[2])

    doc = p2l.Document(f'mprime_tradeoff_comp_k', doc_type='standalone')
    doc += plot_comp_k(ms[1], ks, ds[1], deltas[2])

    # doc += plot_comp_delta(ms[1], ks[0], ds[1], deltas)

    # doc += plot_comp_m(ms, ks[0], ds[1], deltas[2])

    doc.add_package('xcolor', 'dvipsnames')
    print('Building...')
    doc.build()