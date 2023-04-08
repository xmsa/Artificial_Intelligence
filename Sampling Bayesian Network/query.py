

import re
import numpy as np
import pandas as pd


class Query:
    def __init__(self, query_variable, evidence_variables):
        self.__query_variable = query_variable
        self.__evidence_variables = evidence_variables

    @property
    def query_variable(self):
        return self.__query_variable

    @property
    def evidence_variables(self):
        return self.__evidence_variables


class Queries:
    def __init__(self):
        self.__list_of_query = list()

    def append(self, query):
        self.__list_of_query.append(query)

    def __getitem__(self, index):
        return self.__list_of_query[index]

    @staticmethod
    def split_query(query):
        def split_variable(string):
            lst = dict()
            if len(string) == 0:
                return lst
            string = string.split(',')
            pattern = r'\"(\w+)\":(\d*)'
            for s in string:
                q = re.findall(pattern, s)[0]
                lst[q[0]] = int(q[1])
            return lst

        queries = Queries()

        pattern = r'\[\{(.*?)\},\{(.*?)\}\]'
        for match in re.findall(pattern, query):
            qv = split_variable(match[0])
            ev = split_variable(match[1])
            q = Query(qv, ev)
            queries.append(q)
        return queries
