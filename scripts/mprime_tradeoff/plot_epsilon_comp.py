import numpy as np
import pandas as pd
import os

import python2latex as p2l


def plot_comp_k(m, ks, d, delta=0.05):

    plot = p2l.Plot(plot_name=f'tradeoff_comp_k_{m=}_{d=}_{delta=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='7.45cm',
                    height='7.45cm',
                    xmode='log',
                    lines='1pt',)
    plot.axis.kwoptions['legend style'] = r'{font=\tiny}'
    plot.axis.kwoptions['restrict y to domain'] = r'{0:1}'

    palette = p2l.holi(6)
    plot.add_plot([m,m], [0,1], color=palette[0], line_width='1pt', opacity='.5', label="\\footnotesize $m'=m$", label_anchor='south')

    for k, color in zip(ks, palette[1:]):
        datafile = f'./data/mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
        df = pd.read_csv(datafile, sep=',', header=0)
        idx = list(range(0, 2000, 1)) + list(range(2000, 10_000, 500)) + [9_999]
        mprimes = [df['mprime'][i] for i in idx]
        bounds = [df['bound'][i] for i in idx]

        plot.add_plot(mprimes, bounds, color, label=f"\\footnotesize $k={k}$", line_join='round')

        min_idx = np.argmin(bounds)
        min_bound = bounds[min_idx]
        min_mprime = mprimes[min_idx]
        plot.add_plot([min_mprime], [min_bound], color, 'only marks', mark_size='1.5pt', label=f'\\footnotesize ${min_mprime}$', label_anchor='south')

    return plot


def plot_comp_d(m, k, ds, delta=0.05):

    plot = p2l.Plot(plot_name=f'tradeoff_comp_d_{m=}_{k=}_{delta=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='7.45cm',
                    height='7.45cm',
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


        plot.add_plot(mprimes, bounds, color, label=f"\\footnotesize ${d=}$", line_join='round')

        min_idx = np.argmin(bounds)
        min_bound = bounds[min_idx]
        min_mprime = mprimes[min_idx]
        plot.add_plot([min_mprime], [min_bound], color, 'only marks', mark_size='1.5pt', label=f'\\footnotesize ${min_mprime}$', label_anchor='south')

    return plot


def plot_comp_delta(m, k, d, deltas):

    plot = p2l.Plot(plot_name=f'tradeoff_comp_delta_{m=}_{k=}_{d=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='7.45cm',
                    height='7.45cm',
                    xmode='log',
                    lines='1pt')
    plot.axis.kwoptions['legend style'] = r'{font=\tiny}'
    plot.axis.kwoptions['restrict y to domain'] = r'{0:1}'

    palette = p2l.holi(2*len(deltas)+1)
    plot.add_plot([m,m], [0,1], palette[0], line_width='1pt', opacity='.5', label="\\footnotesize $m'=m$", label_anchor='south')

    # First line plot
    delta = deltas[0]
    color = palette[2]
    datafile = f'./data/mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
    df = pd.read_csv(datafile, sep=',', header=0)
    idx = list(range(0, 2000)) + list(range(2000, 10_000, 500)) + [9_999]
    mprimes = [df['mprime'][i] for i in idx]
    bounds = [df['bound'][i] for i in idx]

    plot.add_plot(mprimes, bounds, color, label=f"\\footnotesize $\\delta={delta}$", line_join='round', label_anchor='-135')

    min_idx = np.argmin(bounds)
    min_bound = bounds[min_idx]
    min_mprime = mprimes[min_idx]
    plot.add_plot([min_mprime], [min_bound], color, 'only marks', mark_size='1.5pt', label=f'\\footnotesize ${min_mprime}$', label_anchor='south')


    # Second line plot
    delta = deltas[1]
    color = palette[4]
    datafile = f'./data/mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
    df = pd.read_csv(datafile, sep=',', header=0)
    idx = list(range(0, 2000)) + list(range(2000, 10_000, 500)) + [9_999]
    mprimes = [df['mprime'][i] for i in idx]
    bounds = [df['bound'][i] for i in idx]

    plot.add_plot(mprimes, bounds, color, label=f"\\footnotesize $\\delta={delta}$", line_join='round', label_anchor='135')

    min_idx = np.argmin(bounds)
    min_bound = bounds[min_idx]
    min_mprime = mprimes[min_idx]
    plot.add_plot([min_mprime], [min_bound], color, 'only marks', mark_size='1.5pt', label=f'\\footnotesize ${min_mprime}$', label_anchor='north')

    return plot

def plot_comp_m(ms, k, d, delta):

    plot = p2l.Plot(plot_name=f'tradeoff_comp_m_{k=}_{d=}_{delta=}',
                    plot_path='figures',
                    as_float_env=False,
                    width='7.45cm',
                    height='7.45cm',
                    xmode='log',
                    lines='1pt')
    plot.axis.kwoptions['legend style'] = r'{font=\tiny}'
    plot.axis.kwoptions['restrict y to domain'] = r'{0:1}'

    palette = p2l.holi(2*len(ms)-1)

    for m, color in zip(ms, palette[::2]):
        plot.add_plot([m,m], [0,1], color=color, line_width='1pt', opacity='.5', label=f"\\footnotesize ${m}$", label_anchor='south')
        datafile = f'./data/mprime_tradeoff-{m=}_{k=}_{d=}_{delta=}.csv'
        df = pd.read_csv(datafile, sep=',', header=0)
        idx = list(range(0, 2000)) + list(range(2000, 10_000, 500)) + [9_999]
        mprimes = df['mprime']
        bounds = df['bound']

        mprimes_plot = [mprimes[i] for i in idx]
        bounds_plot = [bounds[i] for i in idx]
        plot.add_plot(mprimes_plot, bounds_plot, color, label=f"\\footnotesize ${m=}$", line_join='round', label_anchor='-155')

        min_idx = np.argmin(bounds)
        min_bound = bounds[min_idx]
        min_mprime = mprimes[min_idx]
        plot.add_plot([min_mprime], [min_bound], color, 'only marks', mark_size='1.5pt', label=f'\\footnotesize ${min_mprime}$', label_anchor='south')

    return plot

if __name__ == "__main__":
    os.chdir('./scripts/mprime_tradeoff/')

    ms = [100, 300, 1000]
    # ms = [100, 200, 300, 500, 1000]
    deltas = [0.0001, 0.1]
    ks = [0, 10, 30, 50, 100]
    ds = [5, 10, 20, 35, 50]

    filename = f'mprime_tradeoff_comp_k'
    plot = plot_comp_k(200, ks, 10, 0.05)

    # filename = f'mprime_tradeoff_comp_d'
    # plot = plot_comp_d(200, 0, ds, 0.05)

    # filename = f'mprime_tradeoff_comp_delta'
    # plot = plot_comp_delta(200, 0, 10, deltas)

    # filename = f'mprime_tradeoff_comp_m'
    # plot = plot_comp_m(ms, 0, 10, 0.05)

    doc = p2l.Document(filename, doc_type='standalone')

    plot.x_min = 5
    plot.x_max = 65_000
    plot.x_label = "$m'$"
    plot.y_min = 0
    plot.y_max = 1.08
    plot.y_label = '$\\epsilon$'
    plot.axis.kwoptions['ylabel style'] = '{yshift=-.4cm}'

    doc += plot

    doc.add_package('xcolor', 'dvipsnames')

    print('Building...')
    doc.build()

    os.remove(filename + '.log')
    os.remove(filename + '.tex')
    os.remove(filename + '.aux')