
from collections import defaultdict
import networkx as nx
import numpy as np


# class LabeledDiGraph(object):
#     def __init__(self, triple_data, inv_flg=False):
#         """
#         Args:
#           - triples: TripletDataset
#         """
#
#         print('Building Graph......')
#         self.adj_mat = np.zeros((triple_data.n_entity, triple_data.n_entity), dtype=np.int32) - 1
#         for (s, r, o) in triple_data.samples:
#             self.adj_mat[s][o] = r
#             if inv_flg:
#                 self.adj_mat[o][s] = r + triple_data.n_relation/2
#         print('done')
#
#     def walk(self, node, edge):
#         return np.where(self.adj_mat[node]==edge)[0]

class LabeledDiGraph(object):
    def __init__(self, triple_data, inv_flg=False):
        """
        Args:
          - triples: TripletDataset
        """
        n_entity = triple_data.n_entity
        n_relation = triple_data.n_relation

        neighbors = defaultdict(lambda: defaultdict(set))
        relation_args = defaultdict(lambda: defaultdict(set))

        print('building graph...aaaa')
        for (s, r, t) in triple_data.samples:
            relation_args[r]['s'].add(s)
            relation_args[r]['t'].add(t)
            neighbors[s][r].add(t)
            if inv_flg:
                neighbors[t][r+(n_relation/2)].add(s)

        def freeze(d):
            frozen = {}
            for key, subdict in d.items():
                frozen[key] = {}
                for subkey, set_val in subdict.items():
                    frozen[key][subkey] = tuple(set_val)
            return frozen

        # WARNING: both neighbors and relation_args must not have default initialization.
        # Default init is dangerous, because we sometimes perform uniform sampling over
        # all keys in the dictionary. This distribution will get altered if a user asks about
        # entities or relations that weren't present.

        # self.neighbors[start][relation] = (end1, end2, ...)
        #na self.relation_args[relation][position] = (ent1, ent2, ...)
        # position is either 's' (domain) or 't' (range)
        self.neighbors = freeze(neighbors)
        self.relation_args = freeze(relation_args)

        print('done')

    def walk(self, node, edge):
        return self.neighbors[node][edge]
