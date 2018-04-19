
import argparse
import numpy as np
import sys

sys.path.append('../')
from lib.viz.wrapper_plotly import scatter2d


N_SAMPLE = 1000


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--emb')
    p.add_argument('--label')
    p.add_argument('--out', default='tmp')

    args = p.parse_args()

    embs = np.loadtxt(args.emb)
    labels = np.array([l.strip() for l in open(args.label)])

    rand_idxs = np.random.permutation(len(labels))[:N_SAMPLE+1]

    scatter2d(embs[rand_idxs, 0], embs[rand_idxs, 1], labels[rand_idxs], args.out)
