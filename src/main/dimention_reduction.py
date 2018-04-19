
import argparse
import bhtsne
import matplotlib.pylab as plt
import numpy as np
import sklearn.base
from sklearn.decomposition import PCA
import sys

sys.path.append('../')
from models.gaussian_bilinear_model import GaussianBilinearModel
from utils.vocab import Vocab


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed

    def fit_transform(self, x):
        return bhtsne.tsne(
            x.astype(np.float64), dimensions=self.dimensions, perplexity=self.perplexity, theta=self.theta,
            rand_seed=self.rand_seed)


def visualize(embeds, id2word):
    assert len(id2word) == embeds.shape[0]
    print('converting vectors into 2-dimension...')
    embeds_2d = tsne(embeds)
    print('plotting...')
    plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1])
    for w, x, y in zip(id2word, embeds_2d[:, 0], embeds_2d[:, 1]):
        plt.annotate(w, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.savefig('out.png')


def tsne(embeds):
    tsne_model = BHTSNE()
    # inter_embeds = PCA(n_components=15).fit_transform(embeds)
    return tsne_model.fit_transform(embeds)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('--out')
    # p.add_argument('--entity')

    args = p.parse_args()

    print('preparation...')
    m = GaussianBilinearModel.load_model(args.model)
    # v = Vocab.load(args.entity)

    embeds = tsne(m.entity_mu)
    np.savetxt(args.out, embeds)
