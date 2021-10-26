from hypergeo.generalization_bounds import *


def test_optimize_mprime_max_mode():
    k, m, delta, d = 5, 50, 0.05, 5
    growth_function = lambda M: (np.e*M/d)**d
    mprime = optimize_mprime(
        k=k,
        m=m,
        growth_function=growth_function,
        delta=delta,
        max_mprime=5*m,
        bound=hypinv_lowerbound,
        optimization_mode='max',
    )
    print(mprime)
    raise


def test_hypinv_upperbound():
    k, m = 5, 50
    d = 5
    growth_function = lambda M: (np.e*M/d)**d

    best_bound = hypinv_upperbound(k, m, growth_function, max_mprime=5*m)

    assert best_bound <= 1
    assert hypinv_upperbound(k, m, growth_function, mprime=m) > best_bound


def test_hypinv_lowerbound():
    k, m = 5, 50
    d = 5
    growth_function = lambda M: (np.e*M/d)**d

    best_bound = hypinv_lowerbound(k, m, growth_function, max_mprime=5*m)

    assert best_bound <= 1
    assert best_bound >= 0
    assert hypinv_lowerbound(k, m, growth_function, mprime=m) < best_bound
