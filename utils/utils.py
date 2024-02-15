def affine(x, s0, t0, s1, t1):
    return ((x-s0)*t1 + (t0-x)*s1)/(t0-s0)