import numpy as np
import xarray as xr
import os

import os
os.chdir(os.path.dirname(__file__))


data = xr.open_dataset('data/optimal_bound.nc').drop_sel(d=50)
best_mprimes = data['mprime'].values
best_bounds = data['bound'].values
bound_at_mprime_equal_m = data['bound_at_mprime=m'].values

gain = (bound_at_mprime_equal_m - best_bounds)/bound_at_mprime_equal_m
print(np.mean(gain), np.std(gain))

gain_at_zero = gain[:, 0, :, :]
print(np.mean(gain_at_zero), np.std(gain_at_zero))

