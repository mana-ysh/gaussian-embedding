
import argparse
import numpy as np
import sys

sys.path.append('../')
from models.gaussian_bilinear_model import GaussianBilinearModel
from utils.vocab import Vocab

def covar_analysis(args):
    model = GaussianBilinearModel.load_model(args.model)
    rel_vocab = Vocab.load(args.relation)
    rel_mats = model.relation_mats
    scores = [abs(np.linalg.det(mat)) for mat in rel_mats]

    sort_idxs = np.argsort(scores)[::-1]
    for idx in sort_idxs:
        print('{} : {}'.format(rel_vocab.get_word(idx), scores[idx]))



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('--relation')

    covar_analysis(p.parse_args())
