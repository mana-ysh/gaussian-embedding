
import multiprocessing
from multiprocessing import Pool
import sys

sys.path.append('../')
from utils.dataset import TripletDataset, data_iter

n_cpu = multiprocessing.cpu_count()


def cal_hits(model, dataset, nbest):
    n_sample = len(dataset)
    n_corr = 0
    for sample in data_iter(dataset, rand_flg=False):
        res = model.most_similar(sample[0], sample[1], nbest)
        if sample[2] in res:
            n_corr += 1
    return float(n_corr/n_sample)


def multi_cal_hits(model, dataset, nbest):
    n_sample = len(dataset)
    pool = Pool(n_cpu)
    args_list = [[sample[0], sample[1], nbest] for sample in data_iter(dataset, rand_flg=False)]
    res = pool.map(model.most_similar_multi, args_list)
    n_corr = sum(1 for i in range(n_sample) if dataset.samples[i][2] in res[i])
    return float(n_corr/n_sample)
