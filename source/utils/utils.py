

def close_to(a, b, atol=0, rtol=10e-16):
    return abs(a - b) <= atol + rtol*abs(b)


def close_to_or_less_than(a, b, atol=0, rtol=10e-16):
    return a <= b or close_to(a, b, atol, rtol)