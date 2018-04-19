"""
TODO
- change the Variable name "n_sample"
"""

import numpy as np


class UniformNegativeSampler(object):
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def sample(self):
        return np.random.randint(self.n_sample)
