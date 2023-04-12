
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


def gibbs_sampling(network, query, size=1000, seed=103):
    def generate_sample(network: network, name_change, last_sample, last_change):
        order = network.order
        names = network.names
        last_change = (last_change+1) % len(name_change)
        index = network.index(name_change[last_change])
        tbl: pd.DataFrame = network.joint_table.copy()
        name = network[index].name
        cols = set(names) - set(name)

        for col in cols:
            ind = names.index(col)
            tbl.query(f"{col}=={last_sample[ind]}", inplace=True)
        tbl.reset_index(inplace=True)
        tbl["value"] = tbl["value"] / tbl["value"].sum()
        rnd = random.random()
        value = tbl["value"][tbl["value"].cumsum() > rnd].index.min()
        sample = last_sample.copy()
        sample[index] = tbl[name].loc[value]

        return sample, last_change, last_sample

    def generate_list_sample(network, size, seed, evidence_variables, start_sample=None):
        if start_sample is None:
            start_sample = [0]*len(network)
            for name, value in query.evidence_variables.items():
                start_sample[network.index(name)] = value
        random.seed(seed)
        samples = [start_sample]
        last_change = -1
        names = set(network.names)
        name_change = list(set(names) - set(evidence_variables.keys()))
        while len(samples) < size:
            last_sample = samples[-1]
            sample, last_change, last_sample = generate_sample(
                network, name_change, last_sample, last_change)
            samples.append(sample)

        df = pd.DataFrame(samples, columns=network.names)

        return df

    ev = query.evidence_variables
    qv = query.query_variable
    sample = generate_list_sample(network, size, seed, ev)

    for name, value in qv.items():
        sample.query(f"{name}=={value}", inplace=True)
    return round(sample.shape[0]/size, 5)


def likelihood_weight_sampling(network, query, size=1000, seed=102):

    def generate_sample(network, evidence_variables):
        order = network.order
        names = network.names
        randoms_number = []
        names = network.names
        visited = set()
        sample = []
        weight = np.array([1.0])
        for i in range(len(network)):
            tbl = network[i].tabel.copy()

            name = network[i].name

            cols = set(tbl.columns).intersection(visited)
            for col in cols:
                ind = names.index(col)
                tbl.query(f"{col}=={sample[ind]}", inplace=True)

            if name in evidence_variables.keys():
                result = evidence_variables[name]
                weight *= tbl[tbl[name] == result]["value"].values
            else:
                rnd = random.random()
                value = tbl["value"][tbl["value"].cumsum() > rnd].index.min()
                result = tbl[name].loc[value]
                # weight*=tbl[tbl[name]==result]["value"].values

            sample.append(result)
            visited.add(name)

        sample.append(weight[0])
        return sample

    def generate_list_sample(network, size, seed, evidence_variables):
        random.seed(seed)
        samples = []
        for _ in range(size):
            randoms_number = generate_sample(network, evidence_variables)
            samples.append(randoms_number)

        df = pd.DataFrame(samples, columns=network.names+["weight"])

        return df

    ev = query.evidence_variables
    qv = query.query_variable
    sample = generate_list_sample(network, size, seed, ev)
    total = sample["weight"].sum()

    for name, value in qv.items():
        sample.query(f"{name}=={value}", inplace=True)
    return round(sample["weight"].sum()/total, 5)


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
    size_ = sample.shape[0]
    for name, value in qv.items():
        sample.query(f"{name}=={value}", inplace=True)

    size_sample = sample.shape[0]
    return round(size_sample / size_, 5)


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

    return round(size_sample/size_, 5)


if __name__ == '__main__':
    filename = 'input.txt'
    nw, queries = Network.read_file(filename=filename)
    query = queries[1]
    # real_value(nw, query)
    print(gibbs_sampling(nw, queries[1]))
    print(gibbs_sampling(nw, queries[2]))
    print(gibbs_sampling(nw, queries[3]))
    print(gibbs_sampling(nw, queries[4]))
