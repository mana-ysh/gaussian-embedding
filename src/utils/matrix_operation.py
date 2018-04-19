
import numpy as np


def to_tridiag(mat):
    return np.diag(np.diag(mat, 1), 1) + np.diag(np.diag(mat)) + np.diag(np.diag(mat, -1), -1)