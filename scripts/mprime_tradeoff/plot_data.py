import numpy as np
import csv
import pandas as pd
import os

import python2latex as p2l


palette = {"Bottle Green":"006652","True Blue":"1b65c5","Royal Purple":"7a4fba","Amaranth":"dc4159","Orange Peel":"ff9e1f"}

colors = [p2l.Color(color_hex, color_name=color_name, color_model='HTML') for color_name, color_hex in palette.items()]

def plot_comp_k(m, ks, d):

    plot = p2l.Plot(plot_name=f'tradeoff_comp_k_{m=}_{d=}', plot_path='figures', xmode='log', lines='1pt')
    plot.axis.kwoptions['legend style'] = r'{font=\tiny}'
    plot.axis.kwoptions['restrict y to domain'] = r'{0:1}'

    plot.add_plot([m,m], [0,1], 'CarnationPink', line_width='1pt', opacity='.5', label="\\footnotesize $m'=m$", label_anchor='south')

    for k, color in zip(ks, colors):
        datafile = f'./data/mprime_tradeoff-{m=}_{k=}_{d=}.csv'
        df = pd.read_csv(datafile, sep=',', header=0)
        idx = list(range(0, 2000)) + list(range(2000, 10_000, 500)) + [9_999]
        mprimes = [df['mprime'][i] for i in idx]
        bounds = [df['bound'][i] for i in idx]


        plot.add_plot(mprimes, bounds, color, label=f"\\footnotesize $k = {k}$", line_join='round')

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
    # plot.legend_position = 'south west'

    plot.caption = f"Value of the upper bound as a function of $m'$ for various numbers of errors $k$. The number of examples is fixed to $m={m}$ and the VCdim is set to $d={d}$."

    return plot


if __name__ == "__main__":
    m = 200
    ks = [0, 10, 30, 50, 100]
    d = 10

    os.chdir('./scripts/mprime_tradeoff/')

    doc = p2l.Document(f'mprime_tradeoff_comp_k_{m=}_{d=}')
    doc.add_package('xcolor', 'dvipsnames')

    doc += plot_comp_k(m, ks, d)

    doc.build()