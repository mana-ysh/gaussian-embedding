
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
from evaluation.eval import Evaluator
from utils.dataset import TripletDataset, PathQueryDataset
from utils.vocab import Vocab
from optimizers.sgd import SGD
from trainer.pairwise_trainer import PairwiseSimpleTrainer

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
        raise NotImplementedError()
    else:
        ent_vocab = Vocab.load(args.entity)
        rel_vocab = Vocab.load(args.relation)

    n_entity, n_relation = len(ent_vocab), len(rel_vocab)

    if args.task == 'kbc':
        logger.info('KBC Task setting...')
        train_dat = TripletDataset.load(args.train, ent_vocab, rel_vocab)
    elif args.task == 'pq':
        logger.info('Path Query setting...')
        rel_vocab.add_inverse()
        train_dat = PathQueryDataset.load(args.train, ent_vocab, rel_vocab)
    else:
        raise ValueError

    if args.valid:
        assert args.metric in ['mrr', 'hits']
        assert args.metric, 'Please indecate evaluation metric for validation'
        if args.metric == 'hits':
            assert args.nbest, 'Please indecate nbest for hits'
        if args.task == 'kbc':
            valid_dat = TripletDataset.load(args.valid, ent_vocab, rel_vocab)
        else:  # path query
            valid_dat = PathQueryDataset.load(args.valid, ent_vocab, rel_vocab)
        evaluator = Evaluator(args.metric, args.nbest)
    else:
        evaluator = None
        valid_dat = None

    if args.model == 'gb':
        from models.gaussian_bilinear_model import GaussianBilinearModel as Model
    else:
        raise ValueError

    if args.restart:
        logger.info('Restarting training: {}'.format(args.restart))
        model = Model.load_model(args.restart)
    else:
        logger.info('Building new model')
        opt = SGD(args.lr, args.gradclip)
        model = Model(n_entity, n_relation, opt, args)

    trainer = PairwiseSimpleTrainer(model, evaluator, args.n_epoch, args.num_negative, logger)
    trainer.run(train_dat, valid_dat)


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # model
    p.add_argument('--model', default='gb')

    # task
    p.add_argument('--task', default='kbc')

    # dataset
    p.add_argument('--train')
    p.add_argument('--valid')
    p.add_argument('--entity')
    p.add_argument('--relation')

    # model config
    p.add_argument('--epoch', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--num_negative', type=int, default=10)
    p.add_argument('--dim', type=int, default=100)
    p.add_argument('--restart')

    # others
    p.add_argument('--metric')
    p.add_argument('--nbest')
    p.add_argument('--evalstep', type=int, default=1)
    p.add_argument('--log')

    train(p.parse_args())
