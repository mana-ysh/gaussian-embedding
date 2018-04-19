
import numpy as np
import math
import pickle
import sys

sys.path.append('../')
from utils.dist_operation import *
from utils.matrix_operation import *

np.random.seed(46)

MEAN_NORM = 1.0  # mu should be lower than 1.0 for regularizarion
SIGMA_MEAN = 10


class GaussianAffineModel(object):
    def __init__(self, n_entity, n_relation, dim, ):
        # initialize params
        self.dim = dim
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.c_min = c_min
        self.c_max = c_max
        self.opt = opt
        self.tri_flg = tri_flg
        self.sigma_mean = sigma_mean

        self.init_params()

    # initialize parameters (covariance matricise for each entities are diagonal)
    def init_params(self):
        bound = 6 / math.sqrt(self.dim)
        self.entity_mu = np.random.uniform(-bound, bound, (self.n_entity, self.dim))
        self.entity_sigma = np.random.randn(self.n_entity, self.dim)
        self.entity_sigma += self.sigma_mean
        self.entity_sigma = np.maximum(self.c_min, np.minimum(self.entity_sigma, self.c_max))
        self.relation_mats = np.random.uniform(-bound, bound, (self.n_relation, self.dim, self.dim))
        if self.tri_flg:
            print('relation matrix is tri-diagonal')
            for i in range(self.n_relation):
                self.relation_mats[i] = to_tridiag(self.relation_mats[i])

    # initilize representations of inverse relations by inverse matrix
    def init_inverse(self):
        inv_relation_mats = np.linalg.inv(self.relation_mats)
        self.relation_mats = np.r_[self.relation_mats, inv_relation_mats]

    def cal_margin_loss(self, pos_sample, neg_sample):
        pos_score = self._cal_score(*pos_sample)
        neg_score = self._cal_score(*neg_sample)
        return max(0, 1. - (pos_score - neg_score))

    def fast_cal_margin_loss(self, pos_sample, neg_sample):
        sub, rel = pos_sample[0], pos_sample[1]
        sub_mu, sub_sigma = self.entity_mu[sub], self.entity_sigma[sub]
        rel_mat = self.relation_mats[rel]

        # affin transform
        comp_mu = rel_mat.dot(sub_mu)
        comp_sigma = rel_mat.dot(np.diag(sub_sigma)).dot(rel_mat.T)

        pos_score = self._cal_fast_score(comp_mu, comp_sigma, pos_sample[2])
        neg_score = self._cal_fast_score(comp_mu, comp_sigma, neg_sample[2])
        return max(0, 1. - (pos_score - neg_score))

    # calcurating negative KL-divergence(obj -> sub + rel)
    # TODO: decompose for faster
    def _cal_score(self, sub, rel, obj):
        # get params
        sub_mu, sub_sigma = self.entity_mu[sub], self.entity_sigma[sub]
        rel_mat = self.relation_mats[rel]
        obj_mu, obj_sigma = self.entity_mu[obj], self.entity_sigma[obj]

        # affin transform
        comp_mu = rel_mat.dot(sub_mu)
        comp_sigma = rel_mat.dot(np.diag(sub_sigma)).dot(rel_mat.T)

        # KL-divergence
        det_fac = np.sum(np.log(abs(np.linalg.det(comp_sigma)))) - np.sum(np.log(obj_sigma))
        trace_fac = np.sum(np.diag(comp_sigma) / obj_sigma)
        return -0.5 * float(trace_fac + np.sum((obj_mu - comp_mu)**2 / obj_sigma) - self.dim - det_fac)

    # not calculating det(rel_mat) and some unnessesary terms (in many cases, these terms can be considered as constants)
    def _cal_fast_score(self, comp_mu, comp_sigma, obj):
        obj_mu, obj_sigma = self.entity_mu[obj], self.entity_sigma[obj]
        trace_fac = np.sum(np.diag(comp_sigma) / obj_sigma)
        exp_fac = np.sum((obj_mu - comp_mu)**2 / obj_sigma)
        return -0.5 * float(trace_fac + exp_fac + np.sum(np.log(obj_sigma)))

    def cal_all_scores(self, sub, rel):
        sub_mu, sub_sigma = self.entity_mu[sub], self.entity_sigma[sub]
        if type(rel) == int:
            rel_mat = self.relation_mats[rel]
            comp_mu = rel_mat.dot(sub_mu)
            comp_sigma = rel_mat.dot(np.diag(sub_sigma)).dot(rel_mat.T)
        elif type(rel) == list:
            comp_mu = sub_mu
            comp_sigma = np.diag(sub_sigma)
            for r in rel:
                rel_mat = self.relation_mats[r]
                comp_mu = rel_mat.dot(comp_mu)
                comp_sigma = rel_mat.dot(comp_sigma).dot(rel_mat.T)
        det_facs = - np.sum(np.log(self.entity_sigma), axis=1)  # determinants of comp_sigma is not needed for scores
        trace_facs = np.sum(np.tile(np.diag(comp_sigma), (self.n_entity, 1)) / self.entity_sigma, axis=1)
        exp_facs = np.sum((self.entity_mu - comp_mu)**2 / self.entity_sigma, axis=1)
        scores = -0.5 * (trace_facs + exp_facs - det_facs)
        return scores

    def update(self, pos_sample, neg_sample):
        assert pos_sample[:2] == neg_sample[:2]
        # pick up all params for updating
        sub, rel, pos_obj = pos_sample
        neg_obj = neg_sample[2]
        sub_mu, sub_sigma = self.entity_mu[sub], self.entity_sigma[sub]
        rel_mat = self.relation_mats[rel]
        pos_obj_mu, pos_obj_sigma = self.entity_mu[pos_obj], self.entity_sigma[pos_obj]
        neg_obj_mu, neg_obj_sigma = self.entity_mu[neg_obj], self.entity_sigma[neg_obj]

        loss = self.fast_cal_margin_loss(pos_sample, neg_sample)

        if loss != 0:
            # compute gradients
            inv_pos_obj_sigma = 1/pos_obj_sigma
            inv_neg_obj_sigma = 1/neg_obj_sigma
            comp_mu = rel_mat.dot(sub_mu)
            comp_sigma = rel_mat.dot(np.diag(sub_sigma)).dot(rel_mat.T)

            grad_pos_obj_mu = inv_pos_obj_sigma * (pos_obj_mu - comp_mu)
            grad_neg_obj_mu = - inv_neg_obj_sigma * (neg_obj_mu - comp_mu)
            grad_sub_mu = - rel_mat.T.dot(np.diag(1/pos_obj_sigma)).dot(pos_obj_mu - comp_mu) + rel_mat.T.dot(np.diag(1/neg_obj_sigma)).dot(neg_obj_mu - comp_mu)
            grad_pos_obj_sigma = 0.5 * (- np.diag(comp_sigma) * inv_pos_obj_sigma**2 - grad_pos_obj_mu**2 + inv_pos_obj_sigma)
            grad_neg_obj_sigma = -0.5 * (- np.diag(comp_sigma) * inv_neg_obj_sigma**2 - grad_neg_obj_mu**2 + inv_neg_obj_sigma)
            grad_sub_sigma = 0.5 * (rel_mat.T.dot(np.diag(inv_pos_obj_sigma - inv_neg_obj_sigma)).dot(rel_mat))
            grad_rel_mat = np.tile(inv_pos_obj_sigma, (self.dim, 1)).T * rel_mat * np.tile(sub_sigma, (self.dim, 1)) - np.outer(grad_pos_obj_mu, sub_mu)
            grad_rel_mat -= np.tile(inv_neg_obj_sigma, (self.dim, 1)).T * rel_mat * np.tile(sub_sigma, (self.dim, 1)) - np.outer( -grad_neg_obj_mu, sub_mu)

            if self.tri_flg:
                grad_rel_mat = to_tridiag(grad_rel_mat)

            # update (CAUTION: assuming call by reference)
            sub_mu -= self.opt.lr * grad_sub_mu
            pos_obj_mu -= self.opt.lr * grad_pos_obj_mu
            neg_obj_mu -= self.opt.lr * grad_neg_obj_mu

            sub_sigma -= self.opt.lr * np.diag(grad_sub_sigma)
            pos_obj_sigma -= self.opt.lr * grad_pos_obj_sigma
            neg_obj_sigma -= self.opt.lr * grad_neg_obj_sigma

            rel_mat -= self.opt.lr * grad_rel_mat

            # compute norm
            sub_mu_norm = np.linalg.norm(sub_mu)
            pos_obj_mu_norm = np.linalg.norm(pos_obj_mu)
            neg_obj_mu_norm = np.linalg.norm(neg_obj_mu)

            # regularize
            if sub_mu_norm > MEAN_NORM:
                sub_mu *= (MEAN_NORM / sub_mu_norm)
            if pos_obj_mu_norm > MEAN_NORM:
                pos_obj_mu *= (MEAN_NORM / pos_obj_mu_norm)
            if neg_obj_mu_norm > MEAN_NORM:
                neg_obj_mu *= (MEAN_NORM / neg_obj_mu_norm)

            sub_sigma = np.maximum(self.c_min, np.minimum(sub_sigma, self.c_max))
            pos_obj_sigma = np.maximum(self.c_min, np.minimum(pos_obj_sigma, self.c_max))
            neg_obj_sigma = np.maximum(self.c_min, np.minimum(neg_obj_sigma, self.c_max))

        return loss

    def most_similar(self, sub, rel, topn):
        scores = self.cal_all_scores(sub, rel)
        result = np.argsort(scores)[::-1]
        return result[:topn]

    def most_similar_multi(self, args):
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
        if not hasattr(model, 'tri_flg'):
            model.tri_flg = False
        return model
