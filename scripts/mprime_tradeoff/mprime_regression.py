import numpy as np
import xarray as xr
import os
from itertools import product
from graal_utils import Timer

from scipy.optimize import curve_fit

import python2latex as p2l


def plot_best_mprime(ms, risks, ds, deltas):
    ms, risks, ds, deltas = map(np.array, (ms, risks, ds, deltas))

    plot = p2l.Plot(plot_name=f'tradeoff_best_mprime',
                    plot_path='figures',
                    as_float_env=False,
                    width='13cm',
                    height='7cm',
                    marks='2pt',
                    lines=False,
                    palette='holi5')

    best_mprimes = xr.open_dataset('data/optimal_bound.nc')['mprime'].drop_sel(d=50)
    # best_mprimes = best_mprimes.drop_sel(d=50) # d=50 has problem with m=100, so we drop it
    # print(best_mprimes.drop_sel(d=50))

    best_mprimes_stacked = best_mprimes.stack(mp=['m', 'risk', 'd', 'delta'])

    mprimes_reg = best_mprimes_stacked.values
    ms_reg, risks_reg, ds_reg, deltas_reg = (best_mprimes_stacked[coord].values for coord in ['m', 'risk', 'd', 'delta'])
    ks_reg = np.array([int(m*risk) for m, risk in zip(ms_reg, risks_reg)])

    def reg(k, a, b, c, d, e, f, g, m=ms_reg, dim=ds_reg, delta=deltas_reg):
        # return a/m + b*np.log(m) + c*np.log(dim) + d +  e*(k/m*f)**(g*dim/m)
        return a*(0+k/m*b)**(c*dim/m) + e*np.log(dim)/np.log(m) + f*np.log(delta)/np.log(m) + d/(m*np.log(m))

    # def reg(k, a, b, c, d, e, m=ms_reg):
    #     return a*np.log(1+k/b-m/c)/m + d*np.log(m)/m + e


    ys = mprimes_reg/(ms_reg*np.log(ms_reg))

    popt, pcov = curve_fit(reg, ks_reg, ys,
                           p0=(100, 1, 1, 0, 1, 1, 1))
                        #    p0=(1, 1, 1, .1, 10, 1000))
                        #    p0=(1, 1, 100000, .1, .4))

    print(popt)
    residuals = ys - reg(ks_reg, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ys - np.mean(ys))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(r_squared)


    d = 35
    delta = 0.05
    for m in ms:
        plot_mprimes = best_mprimes.sel(m=m, d=d, delta=delta)
        plot.add_plot(risks, plot_mprimes/(m), legend=f'$m={m}$')

    # for m in ms:
    #     risks = np.linspace(0,.5,50)
    #     plot.add_plot(risks, reg(risks*m, *popt, m=m, dim=d, delta=delta), 'sharp plot')


    plot.x_label = r"$\displaystyle\frac{k}{m}$"
    plot.y_label = r"$\displaystyle\frac{m'_{\textrm{\footnotesize best}}}{m}$"
    plot.legend_position = 'north west'
    # plot.legend_position = 'south east'
    plot.axis.kwoptions['ylabel style'] = r'{rotate=-90}'
    plot.axis.kwoptions['legend cell align'] = '{left}'

    plot.caption = f"$m'/m$ in function of $k/m$ with the VCdim set to $d={d}$ and $\delta={delta}$."
    plot.y_max = 10

    return plot


if __name__ == "__main__":
    ms = [100, 200, 300, 500, 1000]
    risks =  np.linspace(0, .5, 11)
    ds = [5, 10, 20, 35]
    deltas = [0.0001, 0.0025, 0.05, 0.1]


    os.chdir('./scripts/mprime_tradeoff/')

    doc = p2l.Document(f'mprime_regression', doc_type='standalone')
    doc.add_package('xcolor', 'dvipsnames')

    doc += plot_best_mprime(ms, risks, ds, deltas)

    doc.build()