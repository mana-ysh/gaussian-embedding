
import argparse
import copy
from datetime import datetime
import logging
import numpy as np
import os
import sys
import time

sys.path.append('../')
from evaluation.eval import evaluation
from models.gaussian_model import GaussianModel
from utils.dataset import TripletDataset, data_iter
from utils.vocab import Vocab
from optimizers.sgd import SGD

np.random.seed(46)


def train(args):

    if args.log:
        log_dir = args.log
    else:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '{}'.format(datetime.now().strftime('%Y%m%d_%H:%M')))

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # setting for logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(log_dir, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info('Arguments...')
    for arg, val in vars(args).items():
        logger.info('{} : {}'.format(arg, val))

    logger.info('Preparing dataset...')
    if not args.entity or not args.relation:
        # make vocab from train set
        logger.info('Making entity/relation vocab from train data...')
        raise NotImplementedError()
    else:
        ent_vocab = Vocab.load(args.entity)
        rel_vocab = Vocab.load(args.relation)

    n_entity, n_relation = len(ent_vocab), len(rel_vocab)
    train_dat = TripletDataset.load(args.train, ent_vocab, rel_vocab)
    logger.info('')
    if args.valid:
        assert args.metric in ['mrr', 'hits'], 'Invalid evaluation metric: {}'.format(args.metric)
        assert args.metric, 'Please indecate evaluation metric for validation'
        if args.metric == 'hits':
            assert args.nbest, 'Please indecate nbest for hits'
        valid_dat = TripletDataset.load(args.valid, ent_vocab, rel_vocab)

    if args.restart:
        logger.info('Restarting training: {}'.format(args.restart))
        model = GaussianModel.load_model(args.restart)
    else:
        logger.info('Building new model')
        opt = SGD(args.lr, args.gradclip)
        model = GaussianModel(n_entity, n_relation, args.dim, args.cmin, args.cmax, opt)

    best_model = None
    best_val = -1
    for epoch in range(args.epoch):
        logger.info('start {} epoch'.format(epoch+1))
        sum_loss = 0
        start = time.time()
        for i, pos_sample in enumerate(data_iter(train_dat)):
            neg_samples = [(pos_sample[0], pos_sample[1], np.random.randint(n_entity)) for _ in range(args.num_negative)]
            for neg_sample in neg_samples:
                loss = model.update(pos_sample, neg_sample)
                sum_loss += loss
                # logger.info('loss: {}'.format(loss))
            # logger.info('processing {} samples in this epoch'.format(i+1))
            print('processing {} samples in this epoch'.format(i+1))
        logger.info('sum loss: {}'.format(sum_loss))
        logger.info('{} sec/epoch for training'.format(time.time()-start))
        model_path = os.path.join(log_dir, 'model{}'.format(epoch+1))
        model.save_model(model_path)
        if args.valid and (epoch+1)%args.evalstep == 0:
            val = evaluation(model, valid_dat, args.metric, args.nbest)
            logger.info('{} in validation: {}'.format(args.metric, val))
            if val > best_val:
                best_model = copy.deepcopy(model)
                best_val = val
                best_epoch = epoch+1

    if args.valid:
        logger.info('best model is {} epoch'.format(best_epoch))
        model_path = os.path.join(log_dir, 'bestmodel')
        best_model.save_model(model_path)

    logger.info('done all')


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument('--train')
    p.add_argument('--valid')
    p.add_argument('--entity')
    p.add_argument('--relation')

    # model config
    p.add_argument('--epoch', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--num_negative', type=int)
    p.add_argument('--cmin', type=float)
    p.add_argument('--cmax', type=float)
    p.add_argument('--gradclip', type=float)
    # p.add_argument('--l2', type=float)
    p.add_argument('--dim', type=int)
    p.add_argument('--restart')

    # others
    p.add_argument('--metric')
    p.add_argument('--nbest')
    p.add_argument('--evalstep', type=int, default=1)
    p.add_argument('--log')

    train(p.parse_args())
