import numpy as np


def close_to(a, b, atol=0, rtol=10e-16):
    return abs(a - b) <= atol + rtol*abs(b)


def close_to_or_less_than(a, b, atol=0, rtol=10e-16):
    return a <= b or close_to(a, b, atol, rtol)


def sauer_shelah(d):
    return lambda m: (np.e*m/d)**d

def log_sauer_shelah(d):
    return lambda m: d*np.log(np.e*m/d)
