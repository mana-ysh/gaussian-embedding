
import numpy as np


def circular_correlation(v1, v2):
    raise NotImplementedError()


def all_inner_product(mat, vec):
    vals = np.dot(mat.real, vec.real) + np.dot(mat.imag, vec.imag)
    return vals / vec.shape[0]


def product_freq(v1, v2):
    return v1 * v2


def inner_product(v1, v2):
    dim = v1.shape[0]
    # val = np.inner(v1, v2)
    # assert np.absolute(val.imag) < 1.0e-10, "v1: {} \nv2: {} \nval: {}\nval.imag: {}".format(v1, v2, val, float(val.imag))
    # print('v1: {}, v2: {}, val: {}'.format(v1, v2, val))
    # print('val: {}'.format(val))
    # print('imag in inner: {}'.format(val.imag))
    val = np.inner(v1.real, v2.real) + np.inner(v1.imag, v2.imag)
    return val / dim

def norm(v):
    return np.sqrt(inner_product(v, v))
