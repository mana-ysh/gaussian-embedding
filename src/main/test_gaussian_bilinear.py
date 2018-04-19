
import argparse
import sys

sys.path.append('../')
from models.gaussian_bilinear_model import GaussianBilinearModel
from utils.dataset import TripletDataset, PathQueryDataset, data_iter
from utils.vocab import Vocab, RelationVocab


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--task', default='kbc')
    p.add_argument('--model')
    p.add_argument('--data')
    p.add_argument('--metric')
    p.add_argument('--nbest', type=int)
    p.add_argument('--entity')
    p.add_argument('--relation')

    args = p.parse_args()

    assert args.task in ['kbc', 'pq'], 'Invalid task: {}'.format(args.task)
    assert args.metric in ['mrr', 'hits'], 'Invalid metric: {}'.format(args.metric)
    if args.metric == 'hits':
        assert args.nbest, 'Please indecate n-best in using hits'

    model = GaussianBilinearModel.load_model(args.model)

    print('Preparing dataset...')
    if args.task == 'kbc':
        ent_vocab = Vocab.load(args.entity)
        rel_vocab = Vocab.load(args.relation)
        dataset = TripletDataset.load(args.data, ent_vocab, rel_vocab)
    elif args.task == 'pq':
        ent_vocab = Vocab.load(args.entity)
        rel_vocab = RelationVocab.load(args.relation, inv_flg=True)
        dataset = PathQueryDataset.load(args.data, ent_vocab, rel_vocab)
        if not hasattr(model, 'inv_flg') or not model.inv_flg:
            print('initializing inverse relation representations...')
            model.init_inverse()

    print('Start evaluation...')
    if args.metric == 'mrr':
        from evaluation import mrr
        # res = mrr.cal_mrr(model, dataset)
        res = mrr.multi_cal_mrr(model, dataset)
        print('MRR: {}'.format(res))

    elif args.metric == 'hits':
        from evaluation import hits
        # res = hits.cal_hits(model, dataset, args.nbest)
        res = hits.multi_cal_hits(model, dataset, args.nbest)
        print('HITS@{}: {}'.format(args.nbest, res))

    else:
        raise ValueError('Invalid')
