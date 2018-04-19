"""
TODO:
- 最適化の部分をモデルクラスから切り離す
- 勾配クリップのとこすっきりさせる
"""


import numpy as np
import math
import pickle
import sys
import time

sys.path.append('../')
from utils.signal_operation import *

np.random.seed(46)


class ComplexModel(object):
    def __init__(self, n_entity, n_relation, vec_dim, opt):
        # initialize params
        bound = - math.sqrt(6) / math.sqrt(2*vec_dim)
        self.entity_embeds = np.fft.fft(np.random.uniform(-bound, bound, (n_entity, vec_dim)))
        self.relation_embeds = np.fft.fft(np.random.uniform(-bound, bound, (n_relation, vec_dim)))
        # if vec_dim%2 != 0:
        #     raise NotImplementedError()
        # self.entity_embeds = self._init_complex_embs((n_entity, vec_dim))
        # self.relation_embeds = self._init_complex_embs((n_relation, vec_dim))
        # self.lr = lr
        # self.gradclip = gradclip
        self.opt = opt

    def cal_margin_loss(self, pos_sample, neg_sample):
        pos_score = self._cal_score(*pos_sample)
        neg_score = self._cal_score(*neg_sample)
        return max(0, 1. - (pos_score - neg_score))

    def _cal_score(self, sub, rel, obj):
        sub_vec, rel_vec, obj_vec = self.entity_embeds[sub], self.relation_embeds[rel], self.entity_embeds[obj]
        self.compose_vec = product_freq(sub_vec, rel_vec)
        return inner_product(self.compose_vec, obj_vec)

    # initialize embeddings from gaussian distribution (mu=0, sigma=1)
    def _init_complex_embs(self, size):
        mu = 0
        sigma = 1
        num, dim_vec = size
        share_val = np.ndarray((num, dim_vec//2-1), dtype=np.complex)
        share_val.real = np.random.normal(mu, sigma, (num, dim_vec//2-1))
        share_val.imag = np.random.normal(mu, sigma, (num, dim_vec//2-1))
        return np.c_[np.random.normal(mu, sigma, (num, 1)), share_val, np.random.normal(mu, sigma, (num, 1)), np.fliplr(share_val.conj())]

    def update(self, pos_sample, neg_sample):
        assert pos_sample[:2] == neg_sample[:2]
        sub, rel, pos_obj = pos_sample
        neg_obj = neg_sample[2]
        sub_vec = self.entity_embeds[sub]
        rel_vec = self.relation_embeds[rel]
        pos_obj_vec = self.entity_embeds[pos_obj]
        neg_obj_vec = self.entity_embeds[neg_obj]

        loss = self.cal_margin_loss(pos_sample, neg_sample)

        if loss != 0:
            # compute gradients
            grad_pos_obj = - self.compose_vec
            grad_neg_obj = self.compose_vec
            grad_sub = - product_freq(rel_vec.conj(), pos_obj_vec) + product_freq(rel_vec.conj(), neg_obj_vec)
            grad_rel = - product_freq(sub_vec.conj(), pos_obj_vec) + product_freq(sub_vec.conj(), neg_obj_vec)

            self.opt.update([sub_vec, rel_vec, pos_obj_vec, neg_obj_vec], [grad_sub, grad_rel, grad_pos_obj, grad_neg_obj])

            # # compute norms
            # norm_pos_obj = norm(grad_pos_obj)
            # norm_neg_obj = norm(grad_neg_obj)
            # norm_sub = norm(grad_sub)
            # norm_rel = norm(grad_rel)

            # # gradient clipping
            # if norm_pos_obj > self.gradclip:
            #     grad_pos_obj *= self.gradclip / norm_pos_obj
            # if norm_neg_obj > self.gradclip:
            #     grad_neg_obj *= self.gradclip / norm_neg_obj
            # if norm_sub > self.gradclip:
            #     grad_sub *= self.gradclip / norm_sub
            # if norm_rel > self.gradclip:
            #     grad_rel *= self.gradclip / norm_rel

            # # update
            # sub_vec -= self.lr * grad_sub
            # rel_vec -= self.lr * grad_rel
            # pos_obj_vec -= self.lr * grad_pos_obj
            # neg_obj_vec -= self.lr * grad_neg_obj

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

    def cal_all_scores(self, sub, rel):
        sub_vec, rel_vec = self.entity_embeds[sub], self.relation_embeds[rel]
        compose_vec = product_freq(sub_vec, rel_vec)
        scores = all_inner_product(self.entity_embeds, compose_vec)
        return scores

    def save_model(self, model_path):
        with open(model_path, 'wb') as fw:
            pickle.dump(self, fw)

    @classmethod
    def load_model(cls, model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def test_scores(self, sub, rel):
        s = time.time()
        scores1 = self.cal_all_scores(sub, rel)
        print(time.time()-s)

        s = time.time()
        sub_vec, rel_vec = self.entity_embeds[sub], self.relation_embeds[rel]
        compose_vec = product_freq(sub_vec, rel_vec)
        scores2 = all_inner_product(self.entity_embeds, compose_vec)
        print(time.time()-s)

        print(np.array(scores1))
        print(scores2)
