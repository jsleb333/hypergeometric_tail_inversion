from sklearn.preprocessing import StandardScaler

import numpy as np
import numpy.random as random
from numpy.polynomial import polynomial as poly

from graal_utils import Timer

from source.generalization_bounds import hypinv_upperbound, hypinv_reldev_upperbound,     vapnik_pessismistic_bound, vapnik_relative_deviation_bound, sample_compression_bound


def make_dataset_from_polynomial(n_examples, polynomial, xmin=0, xmax=1, noise=0):
    X = random.uniform(xmin, xmax, (n_examples,1))
    X.sort(axis=0)
    Y = polynomial(X[:,0])
    Y += random.normal(scale=noise, size=(n_examples,))
    return X, Y


def make_polynomial_dataset(n_examples,
                            degree,
                            noise=.5,
                            return_poly=False,
                            root_dist=(.1, .5),
                            root_margin=2,
                            poly_scale=1):
    # Random polynomial
    ## Generate random roots with good spacing
    roots = np.zeros((degree,))
    for i, step in enumerate(random.uniform(size=degree, low=root_dist[0], high=root_dist[1])):
        roots[i] = step + roots[i-1] if i > 0 else step

    ## Make polynomial and normalize it
    polynomial = poly.Polynomial(poly.polyfromroots(roots))

    optima_x = polynomial.deriv().roots()
    optima_y = polynomial(np.array(optima_x))
    scale_inv = min(abs(optima_y))/poly_scale
    polynomial = polynomial/scale_inv # Set smallest optima at .25 (arbitrary value necessary for scaled noise)

    # Generate X and Y noisy values
    xmin = roots[0] - root_margin
    xmax = roots[-1] + root_margin
    Xtr, Ytr = make_dataset_from_polynomial(n_examples, polynomial, xmin=xmin, xmax=xmax, noise=noise)
    Xts, Yts = make_dataset_from_polynomial(n_examples, polynomial, xmin=xmin, xmax=xmax, noise=noise)

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xts = scaler.transform(Xts)

    if not return_poly:
        return (Xtr, Ytr), (Xts, Yts)
    else:
        return (Xtr, Ytr), (Xts, Yts), polynomial, scaler


def make_polynomial_features(X, degree=1):
    new_X = []
    for d in range(degree+1):
        new_X.append(X**(d))
    return np.concatenate(new_X, axis=1)