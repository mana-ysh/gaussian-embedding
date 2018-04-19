
import numpy as np

np.random.seed(46)


class TripletDataset(object):
    def __init__(self, samples, n_entity, n_relation):
        """
        Args:
            samples (list): each element has (sub, rel, obj) tuple
        """
        self.samples = samples
        self.n_entity = n_entity
        self.n_relation = n_relation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        samples = []
        with open(data_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                samples.append((ent_vocab[sub], rel_vocab[rel], ent_vocab[obj]))
        return TripletDataset(samples, len(ent_vocab), len(rel_vocab))


class PathQueryDataset(object):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        samples = []
        n_rel = len(rel_vocab)
        with open(data_path) as f:
            for line in f:
                sub, rels, obj = line.strip().split('\t')
                rels = [rel_vocab[r] for r in rels.split(',')]
                samples.append((ent_vocab[sub], rels, ent_vocab[obj]))
        return PathQueryDataset(samples)


# TODO: enabling mini-batch
def data_iter(dataset, rand_flg=True):
    n_sample = len(dataset)
    idxs = np.random.permutation(n_sample) if rand_flg else range(n_sample)
    for idx in idxs:
        yield dataset[idx]
