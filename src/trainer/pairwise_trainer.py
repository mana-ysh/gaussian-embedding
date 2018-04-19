
import copy
import os
import time
import sys


sys.path.append('../')
from utils.dataset import data_iter
from trainer.sampler import UniformNegativeSampler


class PairwiseSimpleTrainer(object):
    def __init__(self, model, evaluator, n_epoch, n_negative, logger):
        self.model = model
        self.n_epoch = n_epoch
        self.n_negative = n_negative
        self.logger = logger
        self.evaluator = evaluator

        self.log_dir = os.path.dirname(logger.handlers[0].baseFilename)
        self.sampler = UniformNegativeSampler(model.n_entity)

    def run(self, train_dat, valid_dat=None, eval_step=1):
        best_model = None
        best_val = -1
        best_epoch = 0
        for epoch in range(self.n_epoch):
            self.logger.info('start {} epoch'.format(epoch+1))
            sum_loss = 0
            start = time.time()
            for i, pos_sample in enumerate(data_iter(train_dat)):
                neg_samples = [(pos_sample[0], pos_sample[1], self.sampler.sample()) for _ in range(self.n_negative)]
                for neg_sample in neg_samples:
                    loss = self.model.updata(pos_sample, neg_sample)
                    sum_loss += loss
                print('processing {} samples in this epoch'.format(i+1))
            self.logger.info('sum_loss: {}'.format(sum_loss))
            self.logger.info('{} sec/epoch for training'.format(time.time()-start))
            model_path = os.path.join(self.log_dir, 'model{}'.format(epoch+1))
            self.model.save_model(model_path)

            if valid_dat and (epoch+1) % eval_step == 0:  # evaluation
                val = self.evaluator.run(self.model, valid_dat)
                self.logger.info('validation: {}'.format(val))
                if val > best_val:
                    best_model = copy.deepcopy(self.model)
                    best_val = val
                    best_epoch = epoch + 1

        if valid_dat:
            self.logger.info('best model is {} epoch'.format(best_epoch))
            model_path = os.path.join(self.log_dir, 'bestmodel')
            best_model.save_model(model_path)

        self.logger.info('done all')
