
import numpy as np


def gaussian_kl(mu1, sigma1, mu2, sigma2):
    """
    calcurating KL-divergence between two gaussian distributions
    (It is necessary covariance matrix is diagonal, so sigmas in arguments are 1-dimentional array)
    """
    d = mu1.shape[0]
    det_fac = np.sum(np.log(sigma2)) - np.sum(np.log(sigma1))
    trace_fac = np.sum(sigma2 / sigma1)
    return 0.5 * float(trace_fac + np.sum((mu1 - mu2)**2 / sigma1) - d - det_fac)
