
import os
import sys
import random
import numpy as np
import pandas as pd
from network import Network, Node
from query import Queries, Query


def real_value(nw: Network, query: Query):
    jt:pd.DataFrame = nw.joint_table.copy()
    
    if len(query.evidence_variables) != 0:
        for i in query.evidence_variables.keys():
            q = f"{i} == {query.evidence_variables[i]}"
            jt.query(q, inplace=True)
    for i in query.query_variable.keys():
        q = f"{i} == {query.query_variable[i]}"
        jt.query(q, inplace=True)

    return round(jt.sum(axis=0)["value"], 5)


def gibbs_sampling():
    return 0


def likelihood_sampling():
    return 0


def rejection_sampling():
    return 0


def prioir_sampling():
    return 0


if __name__ == '__main__':
    filename = 'input.txt'
    nw, queries = Network.read_file(filename=filename)
    query = queries[1]
    real_value(nw, query)
