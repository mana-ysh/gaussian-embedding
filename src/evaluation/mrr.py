
import multiprocessing
from multiprocessing import Pool
import sys

sys.path.append('../')
from utils.dataset import TripletDataset, data_iter

n_cpu = multiprocessing.cpu_count()


def cal_mrr(model, dataset):
    n_sample = len(dataset)
    sum_rr = 0
    for sample in data_iter(dataset, rand_flg=False):
        rank = model.cal_rank(sample[0], sample[1], sample[2])
        sum_rr += float(1/rank)
    return float(sum_rr/n_sample)


def multi_cal_mrr(model, dataset):
    n_sample = len(dataset)
    pool = Pool(n_cpu)
    ranks = pool.map(model.cal_rank_multi, dataset.samples)
    sum_rr = sum(float(1/rank) for rank in ranks)
    return float(sum_rr/n_sample)
