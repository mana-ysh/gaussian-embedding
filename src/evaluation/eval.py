"""
TODO:

"""

import sys

sys.path.append('../')
from evaluation.mrr import multi_cal_mrr
from evaluation.hits import multi_cal_hits


class Evaluator(object):
    def __init__(self, metric, nbest=None):
        assert metric in ['mrr', 'hits'], 'Invalid metric: {}'.format(metric)
        if metric == 'hits':
            assert nbest, 'Please indecate n-best in using hits'

        self.metric = metric
        self.nbest = nbest

    def run(self, model, dataset):
        if self.metric == 'mrr':
            return multi_cal_mrr(model, dataset)
        elif self.metric == 'hits':
            return multi_cal_hits(model, dataset, self.nbest)
        else:
            raise ValueError


def evaluation(model, dataset, metric, nbest=None):
    assert metric in ['mrr', 'hits'], 'Invalid metric: {}'.format(metric)
    if metric == 'hits':
        assert nbest, 'Please indecate n-best in using hits'

    if metric == 'mrr':
        from evaluation.mrr import multi_cal_mrr
        res = multi_cal_mrr(model, dataset)
    elif metric == 'hits':
        from evaluation.hits import multi_cal_hits
        res = multi_cal_hits(model, dataset, nbest)
    else:
        raise ValueError('Invalid: {}'.format(metric))
    return res
