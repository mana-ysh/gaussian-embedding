
import argparse
import copy
from datetime import datetime
import logging
import numpy as np
import os
import sys
import time

sys.path.append('../')
from analysis.graph import LabeledDiGraph
from utils.dataset import TripletDataset, PathQueryDataset
from utils.vocab import Vocab, RelationVocab


def path_analysis(args):
    ent_vocab = Vocab.load(args.entity)
    rel_vocab = RelationVocab.load(args.relation, inv_flg=True)
    triple_dat = TripletDataset.load(args.triple, ent_vocab, rel_vocab)
    pq_dat = PathQueryDataset.load(args.query, ent_vocab, rel_vocab)
    g = LabeledDiGraph(triple_dat, inv_flg=True)

    # traversal path querys
    n_rel = []
    n_tail = []
    for (sub, rels, _) in pq_dat.samples:
        cur_ents = set([sub])
        for r in rels:
            next_ents = set()
            for e in cur_ents:
                new_ents = g.walk(e, r)
                next_ents.update(new_ents)
            cur_ents = next_ents
        n_rel.append(len(rels))
        n_tail.append(len(cur_ents))
    print(n_rel)
    print(n_tail)
    print('Correlation Coefficient: {}'.format(np.corrcoef(n_rel, n_tail)[0, 1]))





if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--triple')
    p.add_argument('--entity')
    p.add_argument('--relation')
    p.add_argument('--query')

    path_analysis(p.parse_args())
