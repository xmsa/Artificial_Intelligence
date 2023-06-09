import os
import sys
import random
import numpy as np
import pandas as pd
from network import Network, Node
from query import Queries, Query
from sampling import real_value, gibbs_sampling, likelihood_weight_sampling, rejection_sampling, prior_sampling


def sample(network, queries):
    for i, query in enumerate(queries):
        rv = real_value(network, query)
        ps = prior_sampling(network, query)
        rs = rejection_sampling(network, query)
        ls = likelihood_weight_sampling(network, query)
        gs = gibbs_sampling(network, query)
        print(i+1, rv, ps, rs, ls, gs)

def main():
    filename = 'input.txt'
    network, queries = Network.read_file(filename=filename)
    # network, queries = Network.read_file()
    sample(network, queries)

if __name__ == '__main__':
    main()
