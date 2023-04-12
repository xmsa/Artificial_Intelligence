
import os
import sys
import random
import numpy as np
import pandas as pd
from network import Network, Node
from query import Queries, Query


def real_value(nw: Network, query: Query):
    jt: pd.DataFrame = nw.joint_table.copy()

    ev = query.evidence_variables
    qv = query.query_variable
    
    for name, value in ev.items():
        jt.query(f"{name}=={value}", inplace=True)
        
    # normalize
    jt["value"] = jt["value"]/jt["value"].sum()

    for name, value in qv.items():
        jt.query(f"{name}=={value}", inplace=True)
        
    return round(jt.sum(axis=0)["value"], 5)


def gibbs_sampling():
    return 0


def likelihood_sampling():
    return 0


def rejection_sampling(network, query, size=1000, seed=101):

    def generate_sample(network):
        order = network.order
        names = network.names
        randoms_number = []
        names = network.names
        visited = set()
        sample = []
        for i in range(len(network)):
            tbl = network[i].tabel.copy()
            name = network[i].name
            cols = set(tbl.columns).intersection(visited)
            for col in cols:
                ind = names.index(col)
                tbl.query(f"{col}=={sample[ind]}", inplace=True)
            rnd = random.random()
            value = tbl["value"][tbl["value"].cumsum() > rnd].index.min()
            sample.append(tbl[name].loc[value])
            visited.add(name)

        return sample

    def is_reject(randoms_number, evidence_variables, names):
        for name, value in evidence_variables.items():
            idx = names.index(name)
            if randoms_number[idx] != value:
                return True
        return False

    def generate_list_sample(network, size, seed, evidence_variables):
        random.seed(seed)
        samples = []
        names = network.names
        # while len(samples) < size:
        for _ in range(size):
            randoms_number = generate_sample(network)
            if is_reject(randoms_number, evidence_variables, names):
                continue
            samples.append(randoms_number)
        return pd.DataFrame(samples, columns=names)

    ev = query.evidence_variables
    qv = query.query_variable
    sample = generate_list_sample(network, size, seed, ev)

    for name, value in qv.items():
        sample.query(f"{name}=={value}", inplace=True)

    size_sample = sample.shape[0]
    return size_sample / size


def prior_sampling(network, query, size=1000, seed=100):

    def generate_sample(network):
        order = network.order
        names = network.names
        randoms_number = []
        names = network.names
        visited = set()
        sample = []
        for i in range(len(network)):
            tbl = network[i].tabel.copy()
            name = network[i].name
            cols = set(tbl.columns).intersection(visited)
            for col in cols:
                ind = names.index(col)
                tbl.query(f"{col}=={sample[ind]}", inplace=True)
            rnd = random.random()
            value = tbl["value"][tbl["value"].cumsum() > rnd].index.min()
            sample.append(tbl[name].loc[value])
            visited.add(name)

        return sample

    def generate_list_sample(network, size, seed):
        random.seed(seed)
        samples = []
        for _ in range(size):
            randoms_number = generate_sample(network)
            samples.append(randoms_number)
        return pd.DataFrame(samples, columns=network.names)

    sample = generate_list_sample(network, size, seed)

    ev = query.evidence_variables
    qv = query.query_variable

    for name, value in ev.items():
        sample.query(f"{name}=={value}", inplace=True)
    size_ = sample.shape[0]
    for name, value in qv.items():
        sample.query(f"{name}=={value}", inplace=True)

    size_sample = sample.shape[0]

    return round(size_sample/size_,5)


if __name__ == '__main__':
    filename = 'input.txt'
    nw, queries = Network.read_file(filename=filename)
    query = queries[1]
    # real_value(nw, query)
    print(prior_sampling(nw, queries[1]))
    print(prior_sampling(nw, queries[2]))
    print(prior_sampling(nw, queries[3]))
    print(prior_sampling(nw, queries[4]))
    # print(rejection_sampling(nw, query))
