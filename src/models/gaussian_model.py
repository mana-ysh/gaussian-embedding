"""
TODO:

"""


import numpy as np
import math
import pickle
import sys
import time

sys.path.append('../')
from utils.signal_operation import *
from utils.dist_operation import *


np.random.seed(46)

MEAN_NORM = 1.0  # mu should be lower than 1.0 for regularizarion
SIGMA_MEAN = 10


class GaussianModel(object):
    def __init__(self, n_entity, n_relation, dist_dim, c_min, c_max, opt):
        # initialize params
        self.dist_dim = dist_dim
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.c_min = c_min
        self.c_max = c_max
        self.init_params(c_min, c_max)
        self.opt = opt

    # initialize parameters (covariance matricise are diagonal)
    def init_params(self, c_min, c_max):
        bound = 6 / math.sqrt(self.dist_dim)
        self.entity_mu = np.random.uniform(-bound, bound, (self.n_entity, self.dist_dim))
        self.entity_sigma = np.random.randn(self.n_entity, self.dist_dim)
        self.entity_sigma += SIGMA_MEAN
        self.entity_sigma = np.maximum(c_min, np.minimum(self.entity_sigma, c_max))
        self.relation_mu = np.random.uniform(-bound, bound, (self.n_entity, self.dist_dim))
        self.relation_sigma = np.random.randn(self.n_relation, self.dist_dim)
        self.relation_sigma += SIGMA_MEAN
        self.relation_sigma = np.maximum(c_min, np.minimum(self.relation_sigma, c_max))

    def cal_margin_loss(self, pos_sample, neg_sample):
        pos_score = self._cal_score(*pos_sample)
        neg_score = self._cal_score(*neg_sample)
        return max(0, 1. - (pos_score - neg_score))

    # calcurating negative KL-divergence(obj -> sub + rel)
    def _cal_score(self, sub, rel, obj):
        sub_mu, sub_sigma = self.entity_mu[sub], self.entity_sigma[sub]
        rel_mu, rel_sigma = self.relation_mu[rel], self.relation_sigma[rel]
        obj_mu, obj_sigma = self.entity_mu[obj], self.entity_sigma[obj]
        # compose two distributions
        comp_mu = sub_mu + rel_mu
        comp_sigma = sub_sigma + rel_sigma
        det_fac = np.sum(np.log(comp_sigma)) - np.sum(np.log(obj_sigma))
        trace_fac = np.sum(comp_sigma / obj_sigma)
        return -0.5 * float(trace_fac + np.sum((obj_mu - comp_mu)**2 / obj_sigma) - self.dist_dim - det_fac)


    def update(self, pos_sample, neg_sample):
        assert pos_sample[:2] == neg_sample[:2]
        # pick up all params for updating
        sub, rel, pos_obj = pos_sample
        neg_obj = neg_sample[2]
        sub_mu, sub_sigma = self.entity_mu[sub], self.entity_sigma[sub]
        rel_mu, rel_sigma = self.relation_mu[rel], self.relation_mu[rel]
        pos_obj_mu, pos_obj_sigma = self.entity_mu[pos_obj], self.entity_sigma[pos_obj]
        neg_obj_mu, neg_obj_sigma = self.entity_mu[neg_obj], self.entity_sigma[neg_obj]
        comp_mu = sub_mu + rel_mu
        comp_sigma = sub_sigma + rel_sigma

        loss = self.cal_margin_loss(pos_sample, neg_sample)

        if loss != 0:
            # compute gradients
            grad_pos_obj_mu = (pos_obj_mu - comp_mu) / pos_obj_sigma
            grad_neg_obj_mu = -(neg_obj_mu - comp_mu) / neg_obj_sigma
            grad_sub_mu = grad_rel_mu = - (grad_pos_obj_mu + grad_neg_obj_mu)

            grad_pos_obj_sigma = 0.5 * (-comp_sigma - grad_pos_obj_mu*grad_pos_obj_mu.T + (1/pos_obj_sigma))
            grad_neg_obj_sigma = -0.5 * (-comp_sigma - grad_neg_obj_mu*grad_neg_obj_mu.T + (1/neg_obj_sigma))
            grad_sub_sigma = grad_rel_sigma = 0.5 * (1/pos_obj_sigma - 1/neg_obj_sigma)

            # update
            sub_mu -= self.opt.lr * grad_sub_mu
            rel_mu -= self.opt.lr * grad_rel_mu
            pos_obj_mu -= self.opt.lr * grad_pos_obj_mu
            neg_obj_mu -= self.opt.lr * grad_neg_obj_mu

            sub_sigma -= self.opt.lr * grad_sub_sigma
            rel_sigma -= self.opt.lr * grad_rel_sigma
            pos_obj_sigma -= self.opt.lr * grad_pos_obj_sigma
            neg_obj_sigma -= self.opt.lr * grad_neg_obj_sigma

            # compute norm
            sub_mu_norm = np.linalg.norm(sub_mu)
            rel_mu_norm = np.linalg.norm(rel_mu)
            pos_obj_mu_norm = np.linalg.norm(pos_obj_mu)
            neg_obj_mu_norm = np.linalg.norm(neg_obj_mu)

            # regularize
            if sub_mu_norm > MEAN_NORM:
                sub_mu *= (MEAN_NORM / sub_mu_norm)
            if rel_mu_norm > MEAN_NORM:
                rel_mu *= (MEAN_NORM / rel_mu_norm)
            if pos_obj_mu_norm > MEAN_NORM:
                pos_obj_mu *= (MEAN_NORM / pos_obj_mu_norm)
            if neg_obj_mu_norm > MEAN_NORM:
                neg_obj_mu *= (MEAN_NORM / neg_obj_mu_norm)

            sub_sigma = np.maximum(self.c_min, np.minimum(sub_sigma, self.c_max))
            rel_sigma = np.maximum(self.c_min, np.minimum(rel_sigma, self.c_max))
            pos_obj_sigma = np.maximum(self.c_min, np.minimum(pos_obj_sigma, self.c_max))
            neg_obj_sigma = np.maximum(self.c_min, np.minimum(neg_obj_sigma, self.c_max))

        return loss

    def most_similar(self, sub, rel, topn):
        raise NotImplementedError
        scores = self.cal_all_scores(sub, rel)
        result = np.argsort(scores)[::-1]
        return result[:topn]

    def most_similar_multi(self, args):
        raise NotImplementedError
        sub, rel, topn = args
        scores = self.cal_all_scores(sub, rel)
        result = np.argsort(scores)[::-1]
        return result[:topn]

    def cal_rank(self, sub, rel, obj):
        scores = self.cal_all_scores(sub, rel)
        result = np.argsort(scores)[::-1]
        rank = np.where(result==obj)[0][0] + 1
        return rank

    def cal_rank_multi(self, sample):
        scores = self.cal_all_scores(sample[0], sample[1])
        result = np.argsort(scores)[::-1]
        rank = np.where(result==sample[2])[0][0] + 1
        return rank

    # TODO: fast implementation
    def cal_all_scores(self, sub, rel):
        sub_mu, sub_sigma = self.entity_mu[sub], self.entity_sigma[sub]
        rel_mu, rel_sigma = self.relation_mu[rel], self.relation_sigma[rel]
        comp_mu = sub_mu + rel_mu
        comp_sigma = sub_sigma + rel_sigma
        scores = []
        for i in range(self.n_entity):
            obj_mu, obj_sigma = self.entity_mu[i], self.entity_sigma[i]
            det_fac = np.sum(np.log(comp_sigma)) - np.sum(np.log(obj_sigma))
            trace_fac = np.sum(comp_sigma / obj_sigma)
            score =  -0.5 * float(trace_fac + np.sum((obj_mu - comp_mu)**2 / obj_sigma) - self.dist_dim - det_fac)
            scores.append(score)
        return scores

    def save_model(self, model_path):
        with open(model_path, 'wb') as fw:
            pickle.dump(self, fw)

    @classmethod
    def load_model(cls, model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if np.where(model.entity_sigma<0)[0].shape[0] != 0:
            print('===CAUTION: covariance matrix of entity include negative values===')
            model.entity_sigma = np.maximum(model.c_min, np.minimum(model.entity_sigma, model.c_max))
        if np.where(model.relation_sigma<0)[0].shape[0] != 0:
            print('===CAUTION: covariance matrix of relation include negative values===')
            model.relation_sigma = np.maximum(model.c_min, np.minimum(model.relation_sigma, model.c_max))
        return model
