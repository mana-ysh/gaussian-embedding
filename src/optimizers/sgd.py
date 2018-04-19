
import sys

sys.path.append('../')
from utils.signal_operation import norm


class SGD(object):
    def __init__(self, lr, gradclip):
        self.lr = lr
        self.gradclip = gradclip

    def update(self, params, grads):
        for p, g in zip(params, grads):
            norm_g = norm(g)
            if norm_g > self.gradclip:
                g *= self.gradclip / norm_g
            p -= self.lr * g
