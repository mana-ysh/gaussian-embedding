
import argparse

from neo4jrestclient.client import GraphDatabase


def build_graphdb(args):

    url = 'http://{}:{}@localhost:7474/db/data/'.format(args.user, args.pw)
    db = GraphDatabase(url)

    # delete
    db.query("MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r", data_contents=True)

    print('adding nodes...')
    node_dict = {}
    for line in open(args.node):
        name = line.strip()
        node_dict[name] = db.nodes.create(name=name)
        node_dict[name].labels.add('entity')

    print('adding edges...')
    for line in open(args.triple):
        sub, rel, obj = line.strip().split('\t')
        node_dict[sub].relationships.create(rel, node_dict[obj])

    print('DONE')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--user', default='neo4j')
    p.add_argument('--pw')
    p.add_argument('--node')
    p.add_argument('--triple')

    build_graphdb(p.parse_args())
