from hypergeo.generalization_bounds import *


def test_hypinv_upperbound():
    k, m = 5, 50
    d = 5
    growth_function = lambda M: (np.e*M/d)**d

    best_bound = hypinv_upperbound(k, m, growth_function, max_mprime=5*m)

    assert best_bound <= 1
    assert hypinv_upperbound(k, m, growth_function, mprime=m) > best_bound


def test_hypinv_lowerbound():
    k, m = 100, 500
    d = 10
    growth_function = lambda M: (np.e*M/d)**d

    best_bound = hypinv_lowerbound(k, m, growth_function, mprime=4*m)

    assert best_bound <= 1
    assert best_bound >= 0
    assert hypinv_lowerbound(k, m, growth_function, mprime=3*m) < best_bound
